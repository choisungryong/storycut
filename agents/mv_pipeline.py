"""
Music Video Pipeline - 뮤직비디오 생성 파이프라인

Phase 1: 기본 MV 생성
- 음악 분석 → 씬 분할 → 이미지 생성 → 영상 합성
"""

import os
import json
import re
import uuid
import time
from typing import Dict, Any, List, Optional, Callable
from pathlib import Path
from datetime import datetime

# 섹션 마커 패턴: [Chorus], [Verse 1], [Pre-Chorus], [Intro] 등
_SECTION_MARKER_RE = re.compile(r'^\[.*?\]$')

from schemas.mv_models import (
    MVProject, MVScene, MVProjectStatus, MVSceneStatus,
    MVGenre, MVMood, MVStyle, MusicAnalysis, MVProjectRequest,
    VisualBible, MVCharacter, MVSceneBlocking, MVNarrativeArc
)
from agents.music_analyzer import MusicAnalyzer
from agents.image_agent import ImageAgent
from agents.subtitle_utils import (
    ffprobe_duration_sec, split_lyrics_lines,
    clamp_timeline_anchored, detect_anchors, write_srt,
    SubtitleLine, AlignedSubtitle, align_lyrics_with_stt, write_ass,
)
from utils.ffmpeg_utils import FFmpegComposer


class MVPipeline:
    """
    뮤직비디오 생성 파이프라인

    Phase 1 기능:
    - 음악 업로드 & 분석
    - 수동/자동 씬 분할
    - 씬별 이미지 생성
    - 음악 + 영상 합성
    """

    def __init__(self, output_base_dir: str = "outputs"):
        self.output_base_dir = output_base_dir
        self.music_analyzer = MusicAnalyzer()
        self.image_agent = ImageAgent()
        self.ffmpeg_composer = FFmpegComposer()
        self._genre_profiles = None  # lazy cache

    @property
    def genre_profiles(self) -> Dict[str, Any]:
        if self._genre_profiles is None:
            from config import load_genre_profiles
            self._genre_profiles = load_genre_profiles()
        return self._genre_profiles

    # ================================================================
    # Step 1: 음악 업로드 & 분석
    # ================================================================

    def upload_and_analyze(
        self,
        music_file_path: str,
        project_id: Optional[str] = None,
        user_lyrics: Optional[str] = None
    ) -> MVProject:
        """
        음악 파일 업로드 및 분석

        Args:
            music_file_path: 업로드된 음악 파일 경로
            project_id: 프로젝트 ID (없으면 자동 생성)
            user_lyrics: 사용자가 직접 입력한 가사 (있으면 Gemini 추출 대신 타이밍만 싱크)

        Returns:
            MVProject 객체
        """
        # 프로젝트 ID 생성
        if not project_id:
            project_id = f"mv_{uuid.uuid4().hex[:8]}"

        print(f"\n{'='*60}")
        print(f"[MV Pipeline] Starting project: {project_id}")
        print(f"{'='*60}")

        # 프로젝트 디렉토리 생성
        project_dir = f"{self.output_base_dir}/{project_id}"
        os.makedirs(project_dir, exist_ok=True)
        os.makedirs(f"{project_dir}/media/images", exist_ok=True)
        os.makedirs(f"{project_dir}/media/video", exist_ok=True)

        # 음악 파일 복사 (원본 보존)
        music_filename = os.path.basename(music_file_path)
        music_dest = f"{project_dir}/music/{music_filename}"
        os.makedirs(os.path.dirname(music_dest), exist_ok=True)

        import shutil
        if music_file_path != music_dest:
            shutil.copy2(music_file_path, music_dest)
            stored_music_path = music_dest
        else:
            stored_music_path = music_file_path

        # 프로젝트 생성
        project = MVProject(
            project_id=project_id,
            status=MVProjectStatus.ANALYZING,
            music_file_path=stored_music_path,
            created_at=datetime.now()
        )

        # 음악 분석
        try:
            print(f"\n[Step 1] Analyzing music...")
            analysis_result = self.music_analyzer.analyze(stored_music_path)

            # 가사 처리: 사용자 가사 있으면 저장만 (STT 싱크는 compose 시점으로 지연)
            if user_lyrics and user_lyrics.strip():
                print(f"\n[Step 1.5] User lyrics received ({len(user_lyrics.strip())} chars) - timing sync deferred to compose")
                analysis_result["extracted_lyrics"] = user_lyrics.strip()
                extracted_lyrics = user_lyrics.strip()
            else:
                print(f"\n[Step 1.5] Extracting lyrics with Gemini...")
                extracted_lyrics = self.music_analyzer.extract_lyrics_with_gemini(stored_music_path)
                if extracted_lyrics:
                    analysis_result["extracted_lyrics"] = extracted_lyrics
                    timed_lyrics = getattr(self.music_analyzer, '_last_timed_lyrics', None)
                    if timed_lyrics:
                        analysis_result["timed_lyrics"] = timed_lyrics
                # Raw STT 문장 보존 (타이밍 에디터용)
                stt_sentences = getattr(self.music_analyzer, '_last_stt_sentences', None)
                if stt_sentences:
                    analysis_result["stt_sentences"] = stt_sentences

            project.music_analysis = MusicAnalysis(**analysis_result)
            project.lyrics = extracted_lyrics or ""
            project.status = MVProjectStatus.READY
            project.progress = 10

            print(f"  Duration: {project.music_analysis.duration_sec:.1f}s")
            print(f"  BPM: {project.music_analysis.bpm or 'N/A'}")
            print(f"  Segments: {len(project.music_analysis.segments)}")
            print(f"  Lyrics: {'YES (' + str(len(extracted_lyrics)) + ' chars)' if extracted_lyrics else 'NONE'}")

        except Exception as e:
            project.status = MVProjectStatus.FAILED
            project.error_message = f"Music analysis failed: {str(e)}"
            print(f"  [ERROR] {project.error_message}")

        # 매니페스트 저장
        self._save_manifest(project, project_dir)

        return project

    # ================================================================
    # Step 2: 씬 생성
    # ================================================================

    def generate_scenes(
        self,
        project: MVProject,
        request: MVProjectRequest,
        on_scene_complete: Optional[Callable] = None
    ) -> MVProject:
        """
        씬 구성 및 이미지 프롬프트 생성

        Args:
            project: MVProject 객체
            request: MVProjectRequest (가사, 컨셉, 스타일 등)
            on_scene_complete: 씬 완료 콜백

        Returns:
            업데이트된 MVProject
        """
        print(f"\n[Step 2] Generating scenes...")

        project_dir = f"{self.output_base_dir}/{project.project_id}"

        # 요청 정보 저장
        project.lyrics = request.lyrics
        print(f"  Lyrics received: {'YES (' + str(len(request.lyrics)) + ' chars)' if request.lyrics else 'EMPTY'}")
        project.concept = request.concept
        project.character_setup = request.character_setup
        project.character_ethnicity = request.character_ethnicity
        project.genre = request.genre
        project.mood = request.mood
        project.style = request.style
        project.subtitle_enabled = request.subtitle_enabled
        project.watermark_enabled = request.watermark_enabled
        project.status = MVProjectStatus.GENERATING
        project.current_step = "씬 구성 중..."

        # 수동 씬 분할이 있으면 사용, 없으면 자동 분할
        if request.manual_scenes:
            scenes = self._create_manual_scenes(request.manual_scenes, project)
        else:
            scenes = self._create_auto_scenes(project, request)

        # max_scenes 제한 (퀵 테스트 모드: 균등 샘플링)
        max_scenes = getattr(request, 'max_scenes', None)
        if max_scenes and len(scenes) > max_scenes:
            step = len(scenes) / max_scenes
            sampled = [scenes[int(i * step)] for i in range(max_scenes)]
            # scene_id 재배정
            for idx, s in enumerate(sampled):
                s.scene_id = idx + 1
            print(f"  Quick test: {len(scenes)} -> {len(sampled)} scenes (sampled)")
            scenes = sampled

        project.scenes = scenes
        project.progress = 20

        print(f"  Total scenes: {len(scenes)}")

        # 프롬프트는 Visual Bible 이후 generate_scene_prompts()에서 생성
        # 여기서는 씬 골격(타이밍+가사)만 준비
        for i, scene in enumerate(project.scenes):
            print(f"  [Scene {scene.scene_id}] {scene.start_sec:.1f}s - {scene.end_sec:.1f}s "
                  f"lyrics={'YES' if scene.lyrics_text else 'none'}")

        self._save_manifest(project, project_dir)

        return project

    # ================================================================
    # Step 2.1: Story Analysis (가사 전체 → 씬별 서사 분석)
    # ================================================================

    def analyze_story(self, project: MVProject) -> MVProject:
        """
        가사 전체를 읽고 씬별 서사 이벤트 + 드라마틱 중요도를 분석.
        Visual Bible과 씬 프롬프트 생성에 앞서 '이 노래가 하는 이야기'를 구조화.
        """
        import os as _os
        api_key = _os.getenv("GOOGLE_API_KEY")
        full_lyrics = project.lyrics or ""
        if not api_key or not full_lyrics.strip() or not project.scenes:
            print("  [Story Analysis] Skipped (no API key or lyrics)")
            return project

        project.current_step = "가사 서사 분석 중..."
        project_dir = f"{self.output_base_dir}/{project.project_id}"
        self._save_manifest(project, project_dir)

        try:
            from google import genai
            from google.genai import types

            client = genai.Client(api_key=api_key)

            # 씬 타임라인 + 전체 가사 구성
            scene_lines = []
            for s in project.scenes:
                lyrics_part = s.lyrics_text or "(instrumental)"
                scene_lines.append(
                    f"Scene {s.scene_id} [{s.start_sec:.1f}s-{s.end_sec:.1f}s] "
                    f"({s.visual_description or 'unknown'}):\n  lyrics: \"{lyrics_part}\""
                )
            scenes_block = "\n".join(scene_lines)

            system_prompt = (
                "You are a story analyst for music videos. Your job is to read the COMPLETE lyrics "
                "of a song and understand the NARRATIVE STORY being told.\n\n"
                "Then, for each scene (which has specific lyrics assigned), you must identify:\n"
                "1. WHAT HAPPENS in this scene narratively (the event, not the mood)\n"
                "2. HOW IMPORTANT this scene is dramatically (1-5 scale)\n\n"
                "=== IMPORTANCE SCALE ===\n"
                "1 = Atmospheric/transitional (intro establishing mood, outro fading)\n"
                "2 = Supporting (reinforcing mood, showing daily life, setting context)\n"
                "3 = Developing (advancing the story, showing change or movement)\n"
                "4 = Key moment (revelation, confrontation, significant emotional shift)\n"
                "5 = Climactic/turning point (the single most dramatic moment, death, reunion, betrayal)\n\n"
                "=== RULES ===\n"
                "- Read ALL lyrics first, understand the full story arc BEFORE analyzing individual scenes\n"
                "- narrative_event must describe a CONCRETE EVENT, not a vague mood\n"
                "  BAD: 'sadness deepens' / 'emotional moment' / 'melancholic atmosphere'\n"
                "  GOOD: 'A messenger arrives on horseback to deliver news of the husband's death'\n"
                "  GOOD: 'The wife collapses to the ground upon hearing the news'\n"
                "  GOOD: 'Flashback to happier times - the couple walks through a garden'\n"
                "- For instrumental sections with no lyrics, infer from context what visual would serve the story\n"
                "- key_visual must describe THE SINGLE MOST IMPORTANT THING to show in this scene's image\n"
                "- Only 1-2 scenes should be importance=5 (the true climax)\n\n"
                "Return ONLY valid JSON array, no markdown:\n"
                "[{\"scene_id\": 1, \"narrative_event\": \"...\", \"importance\": 3, \"key_visual\": \"...\"}]"
            )

            user_prompt = (
                f"Full lyrics:\n{full_lyrics}\n\n"
                f"Scene timeline:\n{scenes_block}\n\n"
                f"Analyze the story in these {len(project.scenes)} scenes."
            )

            print(f"  [Story Analysis] Analyzing {len(project.scenes)} scenes with full lyrics ({len(full_lyrics)} chars)...")

            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=user_prompt,
                config=types.GenerateContentConfig(
                    system_instruction=system_prompt,
                    temperature=0.4,
                    response_mime_type="application/json",
                )
            )

            result_text = response.text.strip()
            story_beats = json.loads(result_text)

            if isinstance(story_beats, list):
                beat_map = {b["scene_id"]: b for b in story_beats if "scene_id" in b}
                for scene in project.scenes:
                    beat = beat_map.get(scene.scene_id)
                    if beat:
                        scene.story_event = beat.get("narrative_event", "")
                        scene.story_importance = beat.get("importance", 2)
                        print(
                            f"    Scene {scene.scene_id} [importance={scene.story_importance}]: "
                            f"{(scene.story_event or '')[:60]}"
                        )

                # 클라이맥스 씬 강조
                climax_scenes = [s for s in project.scenes if (s.story_importance or 0) >= 4]
                if climax_scenes:
                    print(f"  [Story Analysis] Key moments: {[s.scene_id for s in climax_scenes]}")

            self._save_manifest(project, project_dir)
            print(f"  [Story Analysis] Complete")

        except Exception as e:
            print(f"  [Story Analysis] Failed (non-fatal): {e}")

        return project

    # ================================================================
    # Step 2.5: Visual Bible 생성 (Pass 1)
    # ================================================================

    def generate_visual_bible(self, project: MVProject) -> MVProject:
        """
        Pass 1: 2단 생성 — GenreProfile 기반 + LLM 콘텐츠만

        1단계: GenreProfile 로드 → 팔레트/조명/모티프/금지키워드 확정
        2단계: LLM에게 characters + scene_blocking + narrative_arc 만 요청
        3단계: 병합 — GenreProfile 필드 → VisualBible 기본값, LLM 결과 레이어링
        """
        import os as _os
        api_key = _os.getenv("GOOGLE_API_KEY")
        if not api_key:
            print("  [Visual Bible] No API key, skipping")
            return project

        try:
            from google import genai
            from google.genai import types

            client = genai.Client(api_key=api_key)

            # ── 1단계: GenreProfile 로드 (장르 + 스타일 병합) ──
            genre_gp = self.genre_profiles.get(project.genre.value, {})
            style_gp = self.genre_profiles.get(project.style.value, {}) if project.style else {}

            # 스타일 프로필이 별도로 존재하면 (예: hoyoverse) 장르 프로필에 병합
            if genre_gp and style_gp and project.genre.value != project.style.value:
                gp = dict(genre_gp)  # 장르 기반 복사
                # 스타일 프로필의 핵심 비주얼 요소를 오버라이드
                for key in ("palette_base", "palette_mode", "lighting", "motif_library",
                            "avoid_keywords", "atmosphere", "composition_guide", "prompt_lexicon"):
                    if style_gp.get(key):
                        gp[key] = style_gp[key]
                print(f"  [Visual Bible] GenreProfile merged: genre={project.genre.value} + style={project.style.value}")
            elif genre_gp:
                gp = genre_gp
                print(f"  [Visual Bible] GenreProfile loaded: {project.genre.value}")
            elif style_gp:
                gp = style_gp
                print(f"  [Visual Bible] GenreProfile loaded (style): {project.style.value}")
            else:
                gp = {}

            has_genre_profile = bool(gp)
            if has_genre_profile:
                print(f"    Palette mode: {gp.get('palette_mode', 'guide')}")
            else:
                print(f"  [Visual Bible] No GenreProfile for genre='{project.genre.value}' or style='{project.style.value}', using full-LLM mode")

            # 가사 전체 전달 (이전: 500자 절삭 → 개선: 전체)
            lyrics_summary = ""
            if project.lyrics:
                lyrics_summary = project.lyrics
            elif project.music_analysis and project.music_analysis.extracted_lyrics:
                lyrics_summary = project.music_analysis.extracted_lyrics

            # 씬 타임라인 + 가사 전체 + Story Analysis 결과 포함
            scene_timeline = ""
            if project.scenes:
                timeline_parts = []
                for s in project.scenes:
                    lyrics_full = (s.lyrics_text or "").replace("\n", " ").strip()
                    line = f"  Scene {s.scene_id}: {s.start_sec:.1f}s-{s.end_sec:.1f}s"
                    if lyrics_full:
                        line += f' lyrics: "{lyrics_full}"'
                    # Story Analysis 결과가 있으면 포함
                    if s.story_event:
                        line += f" | STORY: {s.story_event}"
                    if s.story_importance and s.story_importance >= 4:
                        line += f" | *** KEY MOMENT (importance={s.story_importance}) ***"
                    timeline_parts.append(line)
                scene_timeline = "\n".join(timeline_parts)

            total_scenes = len(project.scenes) if project.scenes else 8

            # ── 2단계: LLM 시스템 프롬프트 (narrow/full 분기) ──
            if has_genre_profile:
                # Narrow mode: GenreProfile이 비주얼 아이덴티티를 확정, LLM은 콘텐츠만
                gp_context = (
                    f"\n[GENRE PROFILE - PRE-DEFINED, DO NOT OVERRIDE]\n"
                    f"Palette: {', '.join(gp.get('palette_base', []))}\n"
                    f"Lighting: {gp.get('lighting', '')}\n"
                    f"Motifs: {', '.join(gp.get('motif_library', []))}\n"
                    f"Atmosphere: {gp.get('atmosphere', '')}\n"
                    f"Composition: {gp.get('composition_guide', '')}\n"
                    f"Avoid: {', '.join(gp.get('avoid_keywords', []))}\n"
                    f"Reference: {', '.join(gp.get('reference_artists', []))}\n"
                )

                system_prompt = (
                    "You are a music video visual director. A GenreProfile has already defined the "
                    "visual identity (palette, lighting, motifs, atmosphere). Your job is to create "
                    "ONLY the characters, scene blocking, and narrative arc.\n\n"
                    "=== YOUR CORE MISSION ===\n"
                    "Read the FULL lyrics first. Understand the STORY the song is telling.\n"
                    "Then design a visual narrative so that a viewer watching ONLY the images (no audio) "
                    "can understand the story. Every scene image must serve a PURPOSE in the narrative.\n"
                    "Do NOT randomly assign pretty images. Ask yourself: 'Why does this scene NEED this image?'\n\n"
                    "Return ONLY valid JSON with these fields:\n\n"
                    "=== CHARACTERS (Director's Brief) ===\n"
                    "- characters: array of character objects (as many as the story needs), each with:\n"
                    "  - role: string (e.g. 'protagonist', 'love_interest', 'antagonist')\n"
                    "  - description: SPECIFIC appearance -- ethnicity, age range, hair color/length/style, "
                    "facial hair (beard/mustache/clean-shaven + length if applicable), "
                    "eye shape, face shape, skin tone, body type. Be CONCRETE "
                    f"{self._get_ethnicity_description_example(project)}\n"
                    "  - outfit: clothing description\n"
                    f"  - appears_in: array of scene IDs (1-{total_scenes}) where this character appears\n\n"
                    "=== SCENE BLOCKING ===\n"
                    f"- scene_blocking: array of {total_scenes} objects (one per scene), each with:\n"
                    "  - scene_id: int (1-based)\n"
                    "  - shot_type: 'wide'/'medium'/'close-up'/'extreme-close-up'\n"
                    "  - narrative_beat: WHY this scene exists in the story. What story information does the "
                    "viewer learn from this image? (e.g. 'Establish protagonist alone in empty apartment — "
                    "viewer learns she lives alone after breakup')\n"
                    "  - visual_continuity: how this scene visually connects to the PREVIOUS scene. Use one of:\n"
                    "    'same_location_time_change' / 'same_character_state_change' / 'cause_and_effect' / "
                    "'motif_callback' / 'contrast_cut' / 'establishing' (first scene only)\n"
                    "  - characters: array of role names appearing in this scene\n"
                    "  - expression: REQUIRED string - the character's facial expression matching the lyrics emotion "
                    "(e.g. 'tearful with trembling lips', 'bright joyful smile with sparkling eyes', "
                    "'clenched jaw with furious glare', 'gentle melancholic gaze', 'wide-eyed shock', "
                    "'peaceful closed-eye serenity', 'bitter smirk holding back tears')\n"
                    "    IMPORTANT: Expression MUST change scene-by-scene following the song's emotional arc. "
                    "A sad verse needs grief, a powerful chorus needs intensity, a calm bridge needs peace.\n"
                    "  - lighting: scene-specific lighting (or null)\n"
                    "  - action_pose: REQUIRED string describing what the character is PHYSICALLY DOING "
                    "(e.g. 'leaning against wall with arms crossed', 'running through rain', "
                    "'sitting at piano playing keys', 'dancing with arms raised')\n\n"
                    "=== VISUAL CONTINUITY RULES (CRITICAL — NO RANDOM IMAGE LISTING) ===\n"
                    "Adjacent scenes MUST be visually connected. Methods:\n"
                    "- same_location_time_change: Same place shown at different time (day→night, sunny→rainy)\n"
                    "- same_character_state_change: Same person's emotional/physical change (smiling→crying)\n"
                    "- cause_and_effect: Action in scene N leads to result in scene N+1\n"
                    "- motif_callback: A prop/color/symbol from earlier reappears in transformed form\n"
                    "- contrast_cut: Deliberate visual contrast to highlight emotional shift\n"
                    "BAD example: city→flower field→space→ocean (random unrelated backgrounds)\n"
                    "GOOD example: empty room→picks up photo→flashback: walking together→same street alone\n\n"
                    "=== ACTION_POSE RULES ===\n"
                    "Lyrics are often METAPHORICAL. You MUST interpret them as realistic human actions:\n"
                    "- 'spitting fire' -> 'rapping intensely into microphone, leaning forward aggressively'\n"
                    "- 'flying high' -> 'standing on rooftop with arms spread wide, wind in hair'\n"
                    "- 'heart is breaking' -> 'sitting alone, head down, hands covering face'\n"
                    "- 'burning up' -> 'dancing passionately, sweat glistening, dynamic mid-motion pose'\n"
                    "- 'drowning' -> 'curled up on floor in dark room, overwhelmed posture'\n"
                    "NEVER depict metaphors literally (no actual fire-breathing, no literal flying, no supernatural).\n"
                    "Vary poses: sitting, walking, running, dancing, leaning, crouching, reaching, turning away, etc.\n\n"
                    "=== NARRATIVE ARC ===\n"
                    "- narrative_arc: object with:\n"
                    "  - acts: array of 3 act objects, each with 'scenes' (range like '1-4'), "
                    "'description', and 'tone'\n\n"
                    "CRITICAL RULES:\n"
                    "- ONLY the defined characters may appear. NEVER introduce unnamed people.\n"
                    "- NEVER change a character's ethnicity, age, facial hair, or core features between scenes.\n"
                    "- Each character's 'appears_in' must match the scenes where they appear in scene_blocking.\n"
                    "- The characters and blocking must strongly reflect the genre, mood, and style choices.\n"
                    "- *** TIME PERIOD LOCK ***: If the concept specifies a historical period (medieval, ancient, etc.), "
                    "ALL character outfits, props, locations, and actions must be era-appropriate. "
                    "NEVER give characters modern items (earphones, smartphones, coffee cups, modern clothes). "
                    "Design outfits and settings that match the specified time period.\n"
                    "- Do NOT include color_palette, lighting_style, recurring_motifs, atmosphere, "
                    "avoid_keywords, composition_notes, or reference_artists in your JSON -- those are pre-defined."
                    f"{gp_context}"
                )
            else:
                # Full mode: 기존 동작 (LLM이 모든 것 생성)
                system_prompt = (
                    "You are a music video visual director. Create a Director's Brief (JSON) that defines "
                    "the entire visual identity AND character/scene direction for this music video.\n\n"
                    "=== YOUR CORE MISSION ===\n"
                    "Read the FULL lyrics first. Understand the STORY the song is telling.\n"
                    "Then design a visual narrative so that a viewer watching ONLY the images (no audio) "
                    "can understand the story. Every scene image must serve a PURPOSE in the narrative.\n"
                    "Do NOT randomly assign pretty images. Ask: 'Why does this scene NEED this image?'\n\n"
                    "Return ONLY valid JSON with these fields:\n\n"
                    "=== VISUAL IDENTITY ===\n"
                    "- color_palette: array of 5 hex color codes\n"
                    "- lighting_style: string describing lighting approach\n"
                    "- recurring_motifs: array of 3-4 visual motifs (reuse these across scenes for continuity)\n"
                    "- character_archetypes: array of character type names (brief)\n"
                    "- atmosphere: string describing overall atmosphere\n"
                    "- avoid_keywords: array of visual elements to AVOID\n"
                    "- composition_notes: string with cinematography guidance\n"
                    "- reference_artists: array of 2-3 reference artists/films\n\n"
                    "=== CHARACTERS (Director's Brief) ===\n"
                    "- characters: array of character objects (as many as the story needs), each with:\n"
                    "  - role: string (e.g. 'protagonist', 'love_interest', 'antagonist')\n"
                    "  - description: SPECIFIC appearance -- ethnicity, age range, hair color/length/style, "
                    "facial hair (beard/mustache/clean-shaven + length if applicable), "
                    "eye shape, face shape, skin tone, body type. Be CONCRETE "
                    f"{self._get_ethnicity_description_example(project)}\n"
                    "  - outfit: clothing description\n"
                    f"  - appears_in: array of scene IDs (1-{total_scenes}) where this character appears\n\n"
                    "=== SCENE BLOCKING ===\n"
                    f"- scene_blocking: array of {total_scenes} objects (one per scene), each with:\n"
                    "  - scene_id: int (1-based)\n"
                    "  - shot_type: 'wide'/'medium'/'close-up'/'extreme-close-up'\n"
                    "  - narrative_beat: WHY this scene exists in the story. What story information does the "
                    "viewer learn from this image? (e.g. 'Establish protagonist alone in empty apartment — "
                    "viewer learns she lives alone after breakup')\n"
                    "  - visual_continuity: how this scene visually connects to the PREVIOUS scene. Use one of:\n"
                    "    'same_location_time_change' / 'same_character_state_change' / 'cause_and_effect' / "
                    "'motif_callback' / 'contrast_cut' / 'establishing' (first scene only)\n"
                    "  - characters: array of role names appearing in this scene\n"
                    "  - expression: REQUIRED string - the character's facial expression matching the lyrics emotion "
                    "(e.g. 'tearful with trembling lips', 'bright joyful smile with sparkling eyes', "
                    "'clenched jaw with furious glare', 'gentle melancholic gaze', 'wide-eyed shock', "
                    "'peaceful closed-eye serenity', 'bitter smirk holding back tears')\n"
                    "    IMPORTANT: Expression MUST change scene-by-scene following the song's emotional arc. "
                    "A sad verse needs grief, a powerful chorus needs intensity, a calm bridge needs peace.\n"
                    "  - lighting: scene-specific lighting (or null)\n"
                    "  - action_pose: REQUIRED string describing what the character is PHYSICALLY DOING "
                    "(e.g. 'leaning against wall with arms crossed', 'running through rain', "
                    "'sitting at piano playing keys', 'dancing with arms raised')\n\n"
                    "=== VISUAL CONTINUITY RULES (CRITICAL — NO RANDOM IMAGE LISTING) ===\n"
                    "Adjacent scenes MUST be visually connected. Methods:\n"
                    "- same_location_time_change: Same place shown at different time (day→night, sunny→rainy)\n"
                    "- same_character_state_change: Same person's emotional/physical change (smiling→crying)\n"
                    "- cause_and_effect: Action in scene N leads to result in scene N+1\n"
                    "- motif_callback: A prop/color/symbol from earlier reappears in transformed form\n"
                    "- contrast_cut: Deliberate visual contrast to highlight emotional shift\n"
                    "BAD example: city→flower field→space→ocean (random unrelated backgrounds)\n"
                    "GOOD example: empty room→picks up photo→flashback: walking together→same street alone\n\n"
                    "=== ACTION_POSE RULES ===\n"
                    "Lyrics are often METAPHORICAL. You MUST interpret them as realistic human actions:\n"
                    "- 'spitting fire' -> 'rapping intensely into microphone, leaning forward aggressively'\n"
                    "- 'flying high' -> 'standing on rooftop with arms spread wide, wind in hair'\n"
                    "- 'heart is breaking' -> 'sitting alone, head down, hands covering face'\n"
                    "- 'burning up' -> 'dancing passionately, sweat glistening, dynamic mid-motion pose'\n"
                    "- 'drowning' -> 'curled up on floor in dark room, overwhelmed posture'\n"
                    "NEVER depict metaphors literally (no actual fire-breathing, no literal flying, no supernatural).\n"
                    "Vary poses: sitting, walking, running, dancing, leaning, crouching, reaching, turning away, etc.\n\n"
                    "=== NARRATIVE ARC ===\n"
                    "- narrative_arc: object with:\n"
                    "  - acts: array of 3 act objects, each with 'scenes' (range like '1-4'), "
                    "'description', and 'tone'\n\n"
                    "CRITICAL RULES:\n"
                    "- ONLY the defined characters may appear. NEVER introduce unnamed people.\n"
                    "- NEVER change a character's ethnicity, age, facial hair, or core features between scenes.\n"
                    "- Each character's 'appears_in' must match the scenes where they appear in scene_blocking.\n"
                    "- The Visual Bible must strongly reflect the genre, mood, and style choices."
                )

            # 캐릭터 구성 지시 생성
            char_setup = getattr(project, 'character_setup', None)
            char_setup_val = char_setup.value if hasattr(char_setup, 'value') else str(char_setup or 'auto')
            _CHAR_SETUP_INSTRUCTIONS = {
                "male_female": "MANDATORY: Characters must be a MALE-FEMALE romantic couple. The lead must be male, the love interest must be female.",
                "female_female": "MANDATORY: Characters must be a FEMALE-FEMALE romantic couple. Both leads must be women.",
                "male_male": "MANDATORY: Characters must be a MALE-MALE romantic couple. Both leads must be men.",
                "solo_male": "MANDATORY: Only ONE male character. No romantic partner. Solo story.",
                "solo_female": "MANDATORY: Only ONE female character. No romantic partner. Solo story.",
                "group": "MANDATORY: Create a GROUP of 3+ characters (mixed genders allowed).",
            }
            char_instruction = _CHAR_SETUP_INSTRUCTIONS.get(char_setup_val, "")

            # 캐릭터 인종/외형 지시 생성
            char_ethnicity = getattr(project, 'character_ethnicity', None)
            char_eth_val = char_ethnicity.value if hasattr(char_ethnicity, 'value') else str(char_ethnicity or 'auto')
            _ETHNICITY_INSTRUCTIONS = {
                "korean": "All characters MUST be Korean. Describe them with Korean facial features, skin tone, and names.",
                "japanese": "All characters MUST be Japanese. Describe them with Japanese facial features, skin tone, and names.",
                "chinese": "All characters MUST be Chinese. Describe them with Chinese facial features, skin tone, and names.",
                "southeast_asian": "All characters MUST be Southeast Asian. Describe them with Southeast Asian facial features and skin tone.",
                "european": "All characters MUST be European/Caucasian. Describe them with European facial features, light skin, and Western names.",
                "black": "All characters MUST be Black/African. Describe them with African facial features, dark skin tone, and appropriate names.",
                "hispanic": "All characters MUST be Hispanic/Latino. Describe them with Latin American features and Spanish names.",
                "mixed": "Characters should be a MIX of different ethnicities. Make each character a different race for diversity.",
            }
            eth_instruction = _ETHNICITY_INSTRUCTIONS.get(char_eth_val, "")

            user_prompt = (
                f"Genre: {project.genre.value}\n"
                f"Mood: {project.mood.value}\n"
                f"Style: {project.style.value}\n"
                f"Concept: {project.concept or 'free'}\n"
                f"Total scenes: {total_scenes}\n"
                f"Lyrics excerpt: {lyrics_summary or '(instrumental)'}\n"
            )
            if char_instruction or eth_instruction:
                user_prompt += "\n*** CHARACTER SETUP (NON-NEGOTIABLE) ***\n"
                if char_instruction:
                    user_prompt += f"{char_instruction}\n"
                if eth_instruction:
                    user_prompt += f"{eth_instruction}\n"
            if scene_timeline:
                user_prompt += f"\nScene timeline:\n{scene_timeline}\n"
            user_prompt += "\nCreate the Director's Brief JSON:"

            print(f"  [Visual Bible] Generating with Gemini...")

            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=user_prompt,
                config=types.GenerateContentConfig(
                    system_instruction=system_prompt,
                    temperature=0.7,
                    response_mime_type="application/json",
                )
            )

            result_text = response.text.strip()
            # JSON 파싱
            import json as _json
            vb_data = _json.loads(result_text)

            # Director's Brief 확장 필드 추출 및 변환
            characters_data = vb_data.pop("characters", [])
            scene_blocking_data = vb_data.pop("scene_blocking", [])
            narrative_arc_data = vb_data.pop("narrative_arc", None)

            mv_characters = [MVCharacter(**c) for c in characters_data] if characters_data else []
            mv_blocking = [MVSceneBlocking(**b) for b in scene_blocking_data] if scene_blocking_data else []
            mv_arc = MVNarrativeArc(**narrative_arc_data) if narrative_arc_data else None

            # ── 3단계: GenreProfile + LLM 결과 병합 ──
            if has_genre_profile:
                palette_mode = gp.get("palette_mode", "guide")
                llm_palette = vb_data.get("color_palette", [])

                if palette_mode == "lock":
                    # Lock: GenreProfile 팔레트 강제
                    vb_data["color_palette"] = gp.get("palette_base", [])
                elif palette_mode == "guide" and llm_palette:
                    # Guide + LLM이 팔레트를 반환 → LLM 채택
                    vb_data["color_palette"] = llm_palette
                else:
                    # Guide + LLM이 팔레트를 반환하지 않음 → GenreProfile fallback
                    vb_data["color_palette"] = gp.get("palette_base", [])

                # GenreProfile 필드를 기본값으로 설정 (LLM이 반환하지 않은 필드)
                vb_data.setdefault("lighting_style", gp.get("lighting", ""))
                vb_data.setdefault("recurring_motifs", gp.get("motif_library", []))
                vb_data.setdefault("atmosphere", gp.get("atmosphere", ""))
                vb_data.setdefault("avoid_keywords", gp.get("avoid_keywords", []))
                vb_data.setdefault("composition_notes", gp.get("composition_guide", ""))
                vb_data.setdefault("reference_artists", gp.get("reference_artists", []))
                vb_data.setdefault("character_archetypes", [])

                print(f"    Palette mode: {palette_mode}, final palette: {vb_data.get('color_palette', [])}")

            vb_data["characters"] = mv_characters
            vb_data["scene_blocking"] = mv_blocking
            vb_data["narrative_arc"] = mv_arc

            project.visual_bible = VisualBible(**vb_data)

            # MVScene.characters_in_scene + camera_directive 자동 채우기 (scene_blocking 기반)
            for blocking in mv_blocking:
                idx = blocking.scene_id - 1
                if 0 <= idx < len(project.scenes):
                    project.scenes[idx].characters_in_scene = blocking.characters
                    # shot_type → camera_directive 매핑
                    cam_parts = []
                    if blocking.shot_type:
                        cam_parts.append(blocking.shot_type)
                    if blocking.expression:
                        cam_parts.append(f"expression: {blocking.expression}")
                    if blocking.lighting:
                        cam_parts.append(f"lighting: {blocking.lighting}")
                    if cam_parts:
                        project.scenes[idx].camera_directive = ", ".join(cam_parts)

            print(f"  [Director's Brief] Generated successfully:")
            print(f"    Colors: {project.visual_bible.color_palette}")
            print(f"    Lighting: {project.visual_bible.lighting_style}")
            print(f"    Characters: {[c.role for c in mv_characters]}")
            print(f"    Scene Blocking: {len(mv_blocking)} scenes")
            print(f"    Narrative Arc: {len(mv_arc.acts) if mv_arc else 0} acts")
            print(f"    Avoid: {project.visual_bible.avoid_keywords}")

        except Exception as e:
            print(f"  [Visual Bible] Generation failed (non-fatal): {e}")
            # Non-fatal: 기존 파이프라인 동작 유지

        project_dir = f"{self.output_base_dir}/{project.project_id}"
        self._save_manifest(project, project_dir)
        return project

    # ================================================================
    # Step 2.6: 전용 스타일 앵커 생성 (Pass 2)
    # ================================================================

    def generate_style_anchor(self, project: MVProject) -> MVProject:
        """
        Pass 2: VisualBible 기반 전용 스타일 앵커 이미지 생성

        scene_01 이미지 의존을 제거하고 독립적인 스타일 레퍼런스를 생성합니다.
        """
        project_dir = f"{self.output_base_dir}/{project.project_id}"
        anchor_dir = f"{project_dir}/media/images"
        os.makedirs(anchor_dir, exist_ok=True)
        anchor_path = f"{anchor_dir}/style_anchor.png"

        # VisualBible 기반 앵커 프롬프트 구성
        vb = project.visual_bible
        if vb:
            anchor_parts = [
                vb.atmosphere if vb.atmosphere else "cinematic atmosphere",
                f"color palette: {', '.join(vb.color_palette[:5])}" if vb.color_palette else "",
                f"lighting: {vb.lighting_style}" if vb.lighting_style else "",
                f"motifs: {', '.join(vb.recurring_motifs[:3])}" if vb.recurring_motifs else "",
                f"inspired by {', '.join(vb.reference_artists[:2])}" if vb.reference_artists else "",
            ]
            anchor_prompt = ", ".join([p for p in anchor_parts if p])
        else:
            # VisualBible 없이 기본 앵커 생성
            anchor_prompt = f"{project.style.value} style, {project.mood.value} mood, {project.genre.value} genre, establishing shot, cinematic"

        anchor_prompt += ", no text, no letters, no words, no watermark, landscape scene, 16:9"

        print(f"  [Style Anchor] Generating dedicated anchor image...")
        print(f"    Prompt: {anchor_prompt[:100]}...")

        try:
            vb_dict = vb.model_dump() if vb else None
            image_path, _ = self.image_agent.generate_image(
                scene_id=0,
                prompt=anchor_prompt,
                style=project.style.value,
                output_dir=anchor_path,
                genre=project.genre.value,
                mood=project.mood.value,
                visual_bible=vb_dict,
            )
            project.style_anchor_path = image_path
            print(f"  [Style Anchor] Created: {image_path}")
        except Exception as e:
            print(f"  [Style Anchor] Generation failed (non-fatal): {e}")
            # Non-fatal: generate_images()가 scene_01 fallback 사용

        self._save_manifest(project, project_dir)
        return project

    # ================================================================
    # Step 2.7: 캐릭터 앵커 이미지 생성 (Pass 2.5)
    # ================================================================

    def generate_character_anchors(self, project: MVProject) -> MVProject:
        """
        Pass 2.5: VisualBible 캐릭터별 앵커 포트레이트 생성

        CharacterManager의 scored multi-candidate 시스템을 사용하여
        각 캐릭터의 앵커 이미지를 생성합니다. (후보 2장 + Gemini Vision 채점)
        실패 시 기존 단순 방식(_generate_character_anchors_simple)으로 폴백.
        """
        if not project.visual_bible or not project.visual_bible.characters:
            print("  [Character Anchors] No characters defined, skipping")
            return project

        project_dir = f"{self.output_base_dir}/{project.project_id}"
        characters = project.visual_bible.characters
        print(f"\n  [Character Anchors] Generating {len(characters)} character portrait(s) via CharacterManager...")

        try:
            from agents.character_manager import CharacterManager
            char_manager = CharacterManager(image_agent=self.image_agent)
            role_to_path, role_to_poses = char_manager.cast_mv_characters(
                characters=characters,
                project=project,
                project_dir=project_dir,
                candidates_per_pose=2,
            )

            # 결과를 MVCharacter.anchor_image_path + anchor_poses에 저장
            for character in characters:
                path = role_to_path.get(character.role)
                if path:
                    character.anchor_image_path = path
                    print(f"    [OK] {character.role} -> {path}")
                poses = role_to_poses.get(character.role)
                if poses:
                    character.anchor_poses = poses
                    print(f"         poses: {list(poses.keys())}")

        except Exception as e:
            print(f"  [WARNING] CharacterManager failed, falling back to simple method: {e}")
            self._generate_character_anchors_simple(project, project_dir)

        # CharacterQA: 앵커 임베딩 추출 + 등록
        try:
            from agents.character_qa import CharacterQA
            self._character_qa = CharacterQA(threshold=0.45)
            for character in characters:
                if character.anchor_image_path and os.path.exists(character.anchor_image_path):
                    emb = self._character_qa.register_anchor(character.role, character.anchor_image_path)
                    if emb:
                        character.face_embedding = emb
        except Exception as e:
            print(f"  [WARNING] CharacterQA init failed: {e}")
            self._character_qa = None

        self._save_manifest(project, project_dir)
        return project

    def _generate_character_anchors_simple(self, project: MVProject, project_dir: str) -> None:
        """
        폴백: 기존 단순 앵커 생성 (캐릭터당 1장, 채점 없음)
        """
        char_dir = f"{project_dir}/media/characters"
        os.makedirs(char_dir, exist_ok=True)

        characters = project.visual_bible.characters

        # 스타일별 전용 캐릭터 앵커 directive
        _style_directives = {
            "cinematic": "cinematic film still, dramatic chiaroscuro lighting, color graded like a Hollywood blockbuster",
            "anime": "Japanese anime cel-shaded illustration, bold black outlines, vibrant saturated colors, anime proportions, NOT a photograph",
            "webtoon": "Korean webtoon manhwa digital art, clean sharp lines, flat color blocks, NOT a photograph",
            "realistic": "hyperrealistic photograph, DSLR quality, natural lighting, sharp focus, visible skin texture, natural imperfections, NOT anime, NOT cartoon, NOT AI look, NOT plastic skin",
            "illustration": "digital painting illustration, painterly brushstrokes, concept art quality, NOT a photograph",
            "abstract": "abstract expressionist art, surreal dreamlike, bold geometric shapes",
            "hoyoverse": "anime game cinematic illustration, HoYoverse Genshin Impact quality, cel-shaded with dramatic lighting, vibrant saturated colors, NOT photorealistic, NOT western cartoon",
        }
        style_context = _style_directives.get(project.style.value, f"{project.style.value} style")
        style_context += f", {project.mood.value} mood"

        for i, character in enumerate(characters):
            import re as _re
            safe_role = _re.sub(r'[^\w\-]', '_', character.role)[:20]
            anchor_path = f"{char_dir}/{safe_role}.png"

            print(f"    [Fallback {i+1}/{len(characters)}] Generating anchor: {character.role}")

            portrait_prompt = (
                f"Character portrait, upper body, face clearly visible, centered composition, "
                f"{character.description}"
            )
            if character.outfit:
                portrait_prompt += f", wearing {character.outfit}"
            portrait_prompt += (
                f", {style_context}, neutral background, studio lighting, "
                f"no text, no letters, no words, no watermark"
            )

            try:
                vb_dict = project.visual_bible.model_dump() if project.visual_bible else None
                image_path, _ = self.image_agent.generate_image(
                    scene_id=0,
                    prompt=portrait_prompt,
                    style=project.style.value,
                    output_dir=anchor_path,
                    genre=project.genre.value,
                    mood=project.mood.value,
                    visual_bible=vb_dict,
                )
                character.anchor_image_path = image_path
                print(f"      Created: {image_path}")
            except Exception as e:
                print(f"      [WARNING] Character anchor failed (non-fatal): {e}")

    def _create_manual_scenes(
        self,
        manual_scenes: List[Dict],
        project: MVProject
    ) -> List[MVScene]:
        """수동 씬 분할"""
        scenes = []

        for i, ms in enumerate(manual_scenes):
            scene = MVScene(
                scene_id=i + 1,
                start_sec=ms.get("start_sec", 0),
                end_sec=ms.get("end_sec", 0),
                duration_sec=ms.get("end_sec", 0) - ms.get("start_sec", 0),
                visual_description=ms.get("description", ""),
                lyrics_text=ms.get("lyrics", ""),
                image_prompt=""  # 나중에 생성
            )
            scenes.append(scene)

        return scenes

    def _create_auto_scenes(
        self,
        project: MVProject,
        request: MVProjectRequest
    ) -> List[MVScene]:
        """자동 씬 분할 (음악 분석 기반)

        모든 씬이 고유 이미지를 생성합니다.
        """
        scenes = []

        if not project.music_analysis:
            raise ValueError("Music analysis required for auto scene creation")

        segments = project.music_analysis.segments
        segment_types = [seg.segment_type for seg in segments]
        timed_lyrics = project.music_analysis.timed_lyrics  # STT/Gemini 타임스탬프

        for i, segment in enumerate(segments):
            # timed_lyrics가 있으면 시간 기반 배정, 없으면 균등 분배
            if timed_lyrics:
                lyrics_text = self._extract_lyrics_by_time(
                    timed_lyrics, segment.start_sec, segment.end_sec
                )
            else:
                lyrics_text = self._extract_lyrics_for_segment(
                    request.lyrics, i, len(segments),
                    segment_types=segment_types,
                )

            scene = MVScene(
                scene_id=i + 1,
                start_sec=segment.start_sec,
                end_sec=segment.end_sec,
                duration_sec=segment.duration_sec,
                visual_description=f"{segment.segment_type} section",
                lyrics_text=lyrics_text,
                image_prompt="",
            )
            scenes.append(scene)

        method = "timed_lyrics" if timed_lyrics else "proportional"
        print(f"  Scene split: {len(scenes)} scenes (lyrics: {method})")

        return scenes

    @staticmethod
    def _extract_lyrics_by_time(timed_lyrics: list, start_sec: float, end_sec: float) -> str:
        """timed_lyrics에서 해당 시간대에 불리는 가사 추출 (이미지+자막 타이밍 일치)"""
        lines = []
        for entry in timed_lyrics:
            t = float(entry.get("t", 0))
            text = entry.get("text", "").strip()
            if not text:
                continue
            if start_sec <= t < end_sec:
                lines.append(text)
        return "\n".join(lines)

    # 보컬이 있는 세그먼트 타입 (가사 할당 대상)
    _VOCAL_SEGMENT_TYPES = {
        "verse", "chorus", "pre_chorus", "pre-chorus",
        "post_chorus", "post-chorus", "hook", "rap", "refrain",
    }

    def _extract_lyrics_for_segment(
        self,
        lyrics: Optional[str],
        segment_index: int,
        total_segments: int,
        segment_types: Optional[List[str]] = None,
    ) -> str:
        """가사에서 해당 구간 텍스트 추출 (보컬 세그먼트만 할당)

        intro, bridge, outro, instrumental 등 비보컬 구간에는 가사를 할당하지 않고,
        보컬 세그먼트(verse, chorus 등)에만 균등 분배합니다.
        """
        if not lyrics:
            return ""

        lines = [l.strip() for l in lyrics.strip().split('\n')
                 if l.strip() and not _SECTION_MARKER_RE.match(l.strip())]
        if not lines:
            return ""

        # segment_types가 있으면 보컬 세그먼트만 가사 할당
        if segment_types:
            vocal_indices = [
                i for i, t in enumerate(segment_types)
                if t.lower() in self._VOCAL_SEGMENT_TYPES
            ]
            # 보컬 세그먼트가 하나도 없으면 전체를 보컬로 간주 (fallback)
            if not vocal_indices:
                vocal_indices = list(range(total_segments))

            if segment_index not in vocal_indices:
                return ""  # 비보컬 세그먼트 → 가사 없음

            vocal_pos = vocal_indices.index(segment_index)
            num_vocal = len(vocal_indices)
        else:
            vocal_pos = segment_index
            num_vocal = total_segments

        # 보컬 세그먼트 수 기준으로 균등 분할
        lines_per_segment = max(1, len(lines) // num_vocal)
        start_idx = vocal_pos * lines_per_segment
        end_idx = min(start_idx + lines_per_segment, len(lines))

        # 마지막 보컬 세그먼트는 남은 가사 전부 할당
        if vocal_pos == num_vocal - 1:
            end_idx = len(lines)

        return '\n'.join(lines[start_idx:end_idx])

    # ================================================================
    # Step 2.8: 씬별 이미지 프롬프트 생성 (Visual Bible 이후)
    # ================================================================

    def generate_scene_prompts(
        self,
        project: MVProject,
        request: MVProjectRequest,
    ) -> MVProject:
        """
        Visual Bible + Story Analysis가 완료된 후 씬별 이미지 프롬프트를 생성.
        이전에는 generate_scenes()에서 VB 없이 프롬프트를 생성했으나,
        이제는 VB의 캐릭터/블로킹/서사 아크와 Story Analysis를 모두 반영.
        """
        print(f"\n[Step 2.8] Generating scene prompts (with Visual Bible + Story Analysis)...")
        project.current_step = "씬 프롬프트 생성 중..."
        project_dir = f"{self.output_base_dir}/{project.project_id}"
        self._save_manifest(project, project_dir)

        gemini_prompts = self._generate_prompts_with_gemini(project, request)

        for i, scene in enumerate(project.scenes):
            if gemini_prompts and i < len(gemini_prompts):
                scene.image_prompt = gemini_prompts[i]
            else:
                scene.image_prompt = self._generate_image_prompt(
                    scene=scene,
                    project=project,
                    request=request,
                    scene_index=i,
                    total_scenes=len(project.scenes)
                )

            print(f"  [Scene {scene.scene_id}] "
                  f"{'*** ' if (scene.story_importance or 0) >= 4 else ''}"
                  f"{scene.image_prompt[:80]}...")

        self._save_manifest(project, project_dir)
        print(f"  [Step 2.8] {len(project.scenes)} prompts generated")
        return project

    def _generate_prompts_with_gemini(
        self,
        project: MVProject,
        request: MVProjectRequest
    ) -> Optional[List[str]]:
        """
        Gemini LLM으로 씬별 이미지 프롬프트를 한 번에 생성

        Returns:
            씬별 프롬프트 리스트 (실패 시 None → fallback 사용)
        """
        import os as _os
        api_key = _os.getenv("GOOGLE_API_KEY")
        if not api_key:
            return None

        try:
            from google import genai
            from google.genai import types

            client = genai.Client(api_key=api_key)

            # 씬 정보 구성
            # B-roll 대상 세그먼트 (Pexels 스톡 영상 사용)
            _BROLL_SEGMENTS = {"intro", "outro", "bridge"}
            _has_pexels = bool(_os.environ.get("PEXELS_API_KEY"))

            scene_descriptions = []
            for i, scene in enumerate(project.scenes):
                lyrics_part = f'  가사: "{scene.lyrics_text}"' if scene.lyrics_text else "  가사: (없음)"
                # B-roll 마킹: intro/outro/bridge + 캐릭터 미등장
                broll_tag = ""
                if _has_pexels:
                    desc_lower = (scene.visual_description or "").lower()
                    is_broll_seg = any(s in desc_lower for s in _BROLL_SEGMENTS)
                    if is_broll_seg and not scene.characters_in_scene:
                        broll_tag = " [B-ROLL: 스톡 영상 사용]"
                # Story Analysis 결과 포함
                story_line = ""
                if scene.story_event:
                    story_line = f"\n  *** STORY EVENT: {scene.story_event}"
                    if (scene.story_importance or 0) >= 4:
                        story_line += f" [DRAMATIC IMPORTANCE: {scene.story_importance}/5 - THIS IS A KEY MOMENT. The image MUST depict this event specifically.]"
                scene_descriptions.append(
                    f"Scene {i+1} [{scene.start_sec:.1f}s-{scene.end_sec:.1f}s] "
                    f"({scene.visual_description}){broll_tag}\n{lyrics_part}{story_line}"
                )

            scenes_text = "\n".join(scene_descriptions)

            # 스타일별 구체적 비주얼 가이드
            style_guide = {
                "cinematic": "cinematic film still, dramatic chiaroscuro lighting, shallow depth of field, anamorphic lens flare, color graded like a Hollywood blockbuster",
                "anime": "Japanese anime cel-shaded illustration, bold outlines, vibrant saturated colors, anime eyes and proportions, manga-inspired composition",
                "webtoon": "Korean webtoon digital art style, clean sharp lines, flat color blocks, manhwa character design, vertical scroll composition",
                "realistic": "hyperrealistic photograph, DSLR quality, natural lighting, photojournalistic, sharp focus, real-world textures, visible skin pores, natural asymmetry, candid feel, 35mm film grain",
                "illustration": "digital painting illustration, painterly brushstrokes, concept art quality, rich color palette, artstation trending",
                "abstract": "abstract expressionist art, surreal dreamlike imagery, bold geometric shapes, color field painting, non-representational",
                "hoyoverse": "anime game cinematic, cel-shaded illustration with dramatic lighting, character action pose, elemental effects, fantasy weapon glow, flowing hair and fabric, epic sky background, HoYoverse quality, vibrant saturated colors"
            }.get(request.style.value, "cinematic film still, dramatic lighting")

            # GenreProfile: 스타일 프로필이 있으면 장르 프로필에 병합
            genre_gp = self.genre_profiles.get(request.genre.value, {})
            style_gp = self.genre_profiles.get(request.style.value, {}) if request.style else {}
            if genre_gp and style_gp and request.genre.value != request.style.value:
                gp = dict(genre_gp)
                for key in ("prompt_lexicon", "lighting", "motif_library", "avoid_keywords",
                            "atmosphere", "composition_guide"):
                    if style_gp.get(key):
                        gp[key] = style_gp[key]
            elif genre_gp:
                gp = genre_gp
            elif style_gp:
                gp = style_gp
            else:
                gp = {}
            genre_guide = gp.get("prompt_lexicon", {}).get("positive", "")
            if not genre_guide:
                # Fallback for unknown genres without GenreProfile
                genre_guide = {
                    "fantasy": "magical fantasy world, enchanted forests, glowing runes, ethereal creatures, mythical landscapes, moonlit atmosphere",
                    "romance": "intimate romantic scenes, warm golden hour lighting, soft bokeh, couples in tender moments, rain reflections, night city warmth",
                    "action": "high-energy action scenes, dynamic motion blur, explosive effects, intense close-ups, sparks and debris",
                    "horror": "dark horror atmosphere, unsettling shadows, eerie fog, distorted perspectives, muted desaturated colors, narrow light sources",
                    "scifi": "futuristic sci-fi environment, neon holographics, cyberpunk city, advanced technology, chrome surfaces, data streams",
                    "drama": "dramatic emotional scenes, naturalistic lighting, expressive faces, strong contrast, muted tones",
                    "comedy": "bright cheerful scenes, exaggerated expressions, warm vivid colors, playful compositions, comedic timing",
                    "abstract": "surreal abstract visuals, impossible geometry, color explosions, dreamscape, ink in water",
                }.get(request.genre.value, "")

            mood_guide = {
                "epic": "grand epic scale, sweeping wide shots, majestic skylines, golden hour, heroic poses",
                "dreamy": "soft dreamy atmosphere, pastel haze, lens diffusion, floating particles, ethereal glow",
                "energetic": "high energy dynamic composition, vivid neon colors, speed lines, angular framing",
                "calm": "serene peaceful mood, soft natural light, muted earth tones, minimalist composition",
                "dark": "dark moody atmosphere, deep shadows, cool blue-black tones, film noir inspired",
                "romantic": "warm romantic ambiance, rose and amber tones, soft candlelight, intimate framing",
                "melancholic": "melancholic bittersweet tone, rain and mist, faded desaturated colors, solitary figures",
                "uplifting": "bright uplifting feeling, sun rays breaking through, warm golden tones, upward angles"
            }.get(request.mood.value, "")

            # Visual Bible 컨텍스트 (있으면 포함)
            vb_context = ""
            if project.visual_bible:
                vb = project.visual_bible
                vb_parts = []
                if vb.color_palette:
                    vb_parts.append(f"Color palette: {', '.join(vb.color_palette)}")
                if vb.lighting_style:
                    vb_parts.append(f"Lighting: {vb.lighting_style}")
                if vb.recurring_motifs:
                    vb_parts.append(f"Recurring motifs: {', '.join(vb.recurring_motifs)}")
                if vb.atmosphere:
                    vb_parts.append(f"Atmosphere: {vb.atmosphere}")
                if vb.avoid_keywords:
                    vb_parts.append(f"AVOID these visuals: {', '.join(vb.avoid_keywords)}")
                if vb.composition_notes:
                    vb_parts.append(f"Composition guide: {vb.composition_notes}")
                if vb.reference_artists:
                    vb_parts.append(f"Reference artists: {', '.join(vb.reference_artists)}")
                vb_context = "\n[VISUAL BIBLE - 반드시 반영]\n" + "\n".join(vb_parts) + "\n"

            # Director's Brief 컨텍스트: 캐릭터, 씬 블로킹, 서사 아크
            directors_context = ""
            if project.visual_bible:
                vb = project.visual_bible
                dc_parts = []

                # 캐릭터 목록
                if vb.characters:
                    dc_parts.append("[CHARACTERS - 이 인물만 등장 가능]")
                    for c in vb.characters:
                        dc_parts.append(
                            f"- {c.role}: {c.description}"
                            + (f" | outfit: {c.outfit}" if c.outfit else "")
                            + f" | appears in scenes: {c.appears_in}"
                        )
                    dc_parts.append("!! 위 캐릭터 외 이름 없는 인물을 절대 등장시키지 마세요.")

                # 씬별 블로킹
                if vb.scene_blocking:
                    dc_parts.append("\n[SCENE BLOCKING - 씬별 연출 (이 장면이 이야기에서 왜 필요한지 반영)]")
                    for b in vb.scene_blocking:
                        parts_str = f"Scene {b.scene_id}: shot={b.shot_type}"
                        if getattr(b, 'narrative_beat', None):
                            parts_str += f", STORY_ROLE={b.narrative_beat}"
                        if getattr(b, 'visual_continuity', None):
                            parts_str += f", CONTINUITY={b.visual_continuity}"
                        if b.characters:
                            parts_str += f", characters={b.characters}"
                        if b.expression:
                            parts_str += f", EXPRESSION={b.expression}"
                        if b.action_pose:
                            parts_str += f", ACTION={b.action_pose}"
                        if b.lighting:
                            parts_str += f", lighting={b.lighting}"
                        dc_parts.append(f"- {parts_str}")

                # 서사 아크
                if vb.narrative_arc and vb.narrative_arc.acts:
                    dc_parts.append("\n[NARRATIVE ARC]")
                    for act in vb.narrative_arc.acts:
                        dc_parts.append(
                            f"- Scenes {act.get('scenes', '?')}: "
                            f"{act.get('description', '')} (tone: {act.get('tone', '')})"
                        )

                if dc_parts:
                    directors_context = "\n" + "\n".join(dc_parts) + "\n"

            # GenreProfile 컨텍스트 (씬 프롬프트 생성에 활용)
            genre_profile_context = ""
            if gp:
                gp_parts = []
                if gp.get("composition_guide"):
                    gp_parts.append(f"Composition: {gp['composition_guide']}")
                if gp.get("lighting"):
                    gp_parts.append(f"Lighting: {gp['lighting']}")
                if gp.get("shot_bias"):
                    bias_str = ", ".join(f"{k}: {v:.0%}" for k, v in gp["shot_bias"].items())
                    gp_parts.append(f"Shot distribution hint: {bias_str}")
                if gp.get("arc_tone_sequence"):
                    gp_parts.append(f"Arc tone sequence: {' -> '.join(gp['arc_tone_sequence'])}")
                if gp_parts:
                    genre_profile_context = "\n[GENRE PROFILE]\n" + "\n".join(gp_parts) + "\n"

            system_prompt = (
                "당신은 뮤직비디오 감독(Visual Director)입니다.\n"
                "당신의 임무는 가사 전체를 하나의 '시각적 이야기'로 만드는 것입니다.\n"
                "시청자가 이미지만 순서대로 봐도 '아, 이런 이야기구나'를 이해할 수 있어야 합니다.\n\n"

                "=== 핵심 원칙: 서사적 개연성 ===\n"
                "1. 먼저 가사 전체를 읽고 '이 노래가 말하는 이야기'를 파악하세요.\n"
                "2. 그 이야기를 시각적으로 전달하기 위해 '꼭 필요한 장면'이 무엇인지 판단하세요.\n"
                "3. 각 씬의 이미지는 이야기 속에서 '역할'이 있어야 합니다:\n"
                "   - 도입부: 주인공과 세계관을 소개하는 설정 샷 (누구인지, 어디인지)\n"
                "   - 전개부: 사건/감정의 변화를 보여주는 장면 (무슨 일이 일어나는지)\n"
                "   - 클라이맥스: 감정의 정점을 시각적으로 폭발시키는 장면\n"
                "   - 마무리: 결말 또는 여운을 남기는 장면\n\n"

                "=== STORY EVENT 반영 (최우선) ===\n"
                "- 각 씬에 'STORY EVENT'가 주어져 있으면, 그 이벤트를 이미지의 핵심으로 반영하세요.\n"
                "- DRAMATIC IMPORTANCE가 4-5인 씬은 이야기의 전환점입니다. "
                "이 씬의 프롬프트는 해당 이벤트를 구체적이고 강렬하게 묘사해야 합니다.\n"
                "- 예: STORY EVENT='전령이 부고를 전달한다' → 프롬프트에 반드시 전령+부고 전달 장면 포함\n"
                "- 예: STORY EVENT='아내가 무너져 내린다' → 프롬프트에 반드시 쓰러지는/무릎 꿇는 장면 포함\n"
                "- STORY EVENT가 없는 씬은 가사와 분위기를 기반으로 자유롭게 작성\n\n"

                "=== 씬 간 연결 규칙 (나열 금지) ===\n"
                "- 연속된 씬은 시각적으로 연결되어야 합니다. 방법:\n"
                "  a) 같은 장소를 다른 시간대에 보여주기 (낮→밤, 맑음→비)\n"
                "  b) 같은 인물의 상태 변화 (웃는 얼굴→눈물, 함께→혼자)\n"
                "  c) 인과관계가 있는 행동 (편지를 쓰는 손→우체통에 넣기→상대방이 읽기)\n"
                "  d) 시각적 모티프 반복 (특정 소품, 색상, 장소가 변형되며 반복)\n"
                "- 절대 금지: 씬마다 아무 관계 없는 랜덤 이미지를 나열하는 것\n"
                "- 예시 (나쁜 예): 도시 야경 → 꽃밭 → 우주 → 바다 (관계 없는 배경 나열)\n"
                "- 예시 (좋은 예): 빈 방에 혼자 앉아있음 → 사진을 꺼내 바라봄 → "
                "회상: 둘이 같이 걷던 거리 → 같은 거리를 혼자 걸음 (인과관계+장소 반복)\n\n"

                "=== 캐릭터/동작 규칙 ===\n"
                "- CHARACTERS 섹션의 인물만 등장. 정의되지 않은 인물 절대 추가 금지.\n"
                "- *** NO UNNAMED PEOPLE ***: If a scene has NO characters assigned (instrumental/intro/outro), "
                "the prompt MUST describe ONLY scenery, environment, objects, or atmosphere. "
                "NEVER add random people (old man, stranger, bystander, crowd, silhouette of person) "
                "to fill empty scenes. Use landscape, architecture, nature, or abstract visuals instead.\n"
                "- 캐릭터 외형(인종, 나이, 헤어, 수염/턱수염, 체형)을 프롬프트에 구체적으로 포함하세요.\n"
                "- *** ETHNICITY IS MANDATORY IN EVERY PROMPT ***: Every prompt MUST explicitly state "
                "the character's ethnicity/race. "
                f"{self._get_ethnicity_prompt_example(request)} "
                "NEVER omit ethnicity. If you skip it, the image model will generate random races.\n"
                "- *** SETTING / TIME PERIOD LOCK ***: The '컨셉' field defines the world and era. "
                "ALL scene prompts MUST be consistent with that setting. "
                "If the concept mentions a historical period (medieval, ancient, 1800s, etc.), "
                "NEVER include modern objects (smartphones, earphones, headphones, laptops, cars, café, "
                "modern furniture, streetlights, neon signs, contemporary clothing). "
                "Use ONLY era-appropriate props, architecture, clothing, and technology. "
                "If concept is 'medieval' → stone castles, candlelight, swords, robes, taverns. "
                "If concept is 'ancient' → temples, torches, clay pots, linen garments. "
                "Violating the time period is as serious as changing a character's ethnicity.\n"
                "- SCENE BLOCKING의 shot_type, expression, lighting, ACTION을 반드시 반영하세요.\n"
                "- EXPRESSION 필드의 표정을 프롬프트 앞부분에 강하게 포함. 무표정 금지.\n"
                "  표정은 가사의 감정을 직접 반영해야 합니다. 슬픈 가사=슬픈 표정, 격한 가사=격한 표정.\n"
                "  예시: 'tearful eyes looking down', 'joyful bright smile', 'fierce determined expression', "
                "'peaceful serene face with closed eyes', 'anguished screaming expression'\n"
                "- ACTION 필드의 동작/포즈를 프롬프트에 반드시 포함. 단순 서있기 금지.\n"
                "  예시: 'walking down rainy street', 'sitting on bench looking up at sky', "
                "'dancing mid-spin with flowing dress', 'leaning on railing gazing at city lights'\n"
                "- 가사의 은유적 표현을 절대 문자 그대로 묘사하지 마세요. "
                "현실적이고 자연스러운 인간 동작으로 변환하세요.\n\n"

                "=== B-ROLL 씬 규칙 ===\n"
                "- [B-ROLL: 스톡 영상 사용]으로 표시된 씬은 Pexels 스톡 영상으로 대체됩니다.\n"
                "- B-ROLL 씬의 프롬프트는 스톡 영상 검색에 적합하게 작성하세요:\n"
                "  - 분위기, 풍경, 환경 묘사 위주 (atmospheric, cinematic landscape, mood)\n"
                "  - 구체적 캐릭터 묘사 불필요 (캐릭터가 등장하지 않는 씬)\n"
                "  - 검색 키워드로 쓸 수 있는 일반적이고 보편적인 비주얼 단어 사용\n"
                "  - 예: 'rainy city night, neon reflections, cinematic establishing shot'\n"
                "  - 예: 'golden sunset over ocean, peaceful ending, warm atmosphere'\n"
                "- B-ROLL이 아닌 씬은 기존대로 구체적인 AI 이미지 프롬프트로 작성하세요.\n\n"

                "=== 출력 형식 ===\n"
                "- 정확히 씬 개수만큼 줄을 출력 (한 줄에 하나의 프롬프트)\n"
                "- 각 프롬프트는 영어로 1-2문장, 쉼표로 구분된 키워드 형태\n"
                "- 절대 씬 번호나 설명 없이 프롬프트만 출력\n"
                "- 모든 프롬프트 끝에 반드시 'no text, no letters, no words, no writing, no watermark' 포함\n"
                "- 고급: 각 프롬프트 끝에 '|' 구분자로 추가 지시를 포함하세요:\n"
                "  형식: positive prompt | negative keywords | color mood | camera directive\n"
                "  예: cinematic hero shot... | gore, blood | warm golden | low angle tracking\n"
                "  negative/color/camera가 불필요하면 비워두되 구분자는 유지: prompt | | |\n\n"

                f"[필수 비주얼 스타일]\n"
                f"모든 씬에 다음 스타일을 강하게 적용하세요:\n"
                f"- 아트 스타일: {style_guide}\n"
                f"- 장르 비주얼: {genre_guide}\n"
                f"- 분위기/톤: {mood_guide}\n"
                f"각 프롬프트의 첫 부분에 스타일 키워드를 반드시 포함하세요."
                f"{genre_profile_context}"
                f"{vb_context}"
                f"{directors_context}\n\n"

                f"[NEGATIVE LOCK]\n"
                f"{'no cartoon, no anime, no illustration, ' if request.style not in (MVStyle.ANIME, MVStyle.WEBTOON) else ''}"
                f"{'no photorealistic, no photograph, ' if request.style not in (MVStyle.REALISTIC, MVStyle.CINEMATIC) else ''}"
                f"no different ethnicity, no different skin tone, "
                f"no different outfit colors, keep signature outfit, "
                f"no extra characters, no random faces"
            )

            # 인종 지시: system_prompt의 "ETHNICITY IS MANDATORY" + Visual Bible 지시로 충분.
            # user_prompt에서 중복 제거 (LLM이 "Korean Korean Korean" 과잉 반복하는 문제 방지)
            # 최종 방어선은 generate_images()의 런타임 주입이 담당.

            # --- 시대/배경 키워드 추출 (concept에서) ---
            _concept = (request.concept or "").lower()
            _era_warning = ""
            if any(kw in _concept for kw in [
                "medieval", "중세", "ancient", "고대", "조선", "joseon",
                "victorian", "빅토리아", "1800", "1700", "1600", "1500",
                "renaissance", "르네상스", "전쟁", "war", "삼국", "고려",
                "roman", "로마", "greek", "그리스", "edo", "에도",
            ]):
                _era_warning = (
                    "\n*** TIME PERIOD WARNING ***\n"
                    f"The concept '{request.concept}' specifies a HISTORICAL setting.\n"
                    "You MUST ensure ALL scene prompts use ONLY era-appropriate elements.\n"
                    "ABSOLUTELY FORBIDDEN in prompts: café, coffee shop, earphones, headphones, "
                    "smartphone, laptop, computer, modern car, streetlight, neon sign, "
                    "modern clothing (hoodie, jeans, sneakers, t-shirt), glasses (modern frames), "
                    "electric guitar, microphone stand, modern building, concrete, asphalt road.\n"
                    "USE INSTEAD: tavern, candle, torch, horse, sword, bow, quill, parchment, "
                    "stone walls, wooden furniture, linen/silk/leather garments, medieval armor.\n"
                )

            user_prompt = (
                f"컨셉: {request.concept or '자유'}\n"
                f"총 씬 수: {len(project.scenes)}\n\n"
            )
            user_prompt += (
                f"[중요] 먼저 아래 가사 전체를 읽고 '이 노래의 이야기'를 파악한 뒤,\n"
                f"각 씬이 그 이야기의 어느 부분을 담당하는지 정한 후 프롬프트를 작성하세요.\n"
                f"시청자가 이미지 순서만 봐도 이야기를 따라갈 수 있어야 합니다.\n"
                f"{_era_warning}\n"
                f"씬 정보:\n{scenes_text}\n\n"
                f"위 {len(project.scenes)}개 씬에 대해 각각 이미지 생성 프롬프트를 한 줄씩 출력하세요."
            )

            print(f"  [Gemini] Generating {len(project.scenes)} scene prompts...")

            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=user_prompt,
                config=types.GenerateContentConfig(
                    system_instruction=system_prompt,
                    temperature=0.8,
                )
            )

            result_text = response.text.strip()
            lines = [l.strip() for l in result_text.split('\n') if l.strip()]

            # 씬 수와 프롬프트 수가 일치하는지 확인
            if len(lines) < len(project.scenes):
                print(f"  [Gemini] Warning: got {len(lines)} prompts for {len(project.scenes)} scenes, padding with last")
                while len(lines) < len(project.scenes):
                    lines.append(lines[-1] if lines else "cinematic scene, dramatic lighting")
            elif len(lines) > len(project.scenes):
                lines = lines[:len(project.scenes)]

            # Pass 3: 구조화된 프롬프트 파싱 (positive | negative | color_mood | camera)
            prompts = []
            for i, line in enumerate(lines):
                parts = [p.strip() for p in line.split('|')]
                positive = parts[0] if len(parts) > 0 else line
                negative = parts[1] if len(parts) > 1 and parts[1] else None
                color_m = parts[2] if len(parts) > 2 and parts[2] else None
                camera = parts[3] if len(parts) > 3 and parts[3] else None

                prompts.append(positive)

                # MVScene에 구조화된 데이터 저장
                if i < len(project.scenes):
                    if negative:
                        project.scenes[i].negative_prompt = negative
                    if color_m:
                        project.scenes[i].color_mood = color_m
                    if camera:
                        project.scenes[i].camera_directive = camera

            print(f"  [Gemini] Generated {len(prompts)} unique prompts (structured)")
            return prompts

        except Exception as e:
            print(f"  [Gemini] Prompt generation failed: {e}, using template fallback")
            return None

    def _generate_image_prompt(
        self,
        scene: MVScene,
        project: MVProject,
        request: MVProjectRequest,
        scene_index: int,
        total_scenes: int
    ) -> str:
        """씬별 이미지 프롬프트 생성 (템플릿 fallback)"""

        # 스타일 프리픽스
        style_map = {
            MVStyle.CINEMATIC: "cinematic film still, dramatic lighting, high contrast",
            MVStyle.ANIME: "Japanese anime cel-shaded illustration, bold black outlines, NOT a photograph",
            MVStyle.WEBTOON: "Korean manhwa webtoon digital art, clean sharp lines, NOT a photograph",
            MVStyle.REALISTIC: "hyperrealistic photograph, DSLR quality, natural lighting, visible skin texture, natural imperfections, NOT anime, NOT cartoon, NOT illustration, NOT AI-generated look, NOT plastic skin",
            MVStyle.ILLUSTRATION: "digital painting illustration, painterly brushstrokes, concept art quality",
            MVStyle.ABSTRACT: "abstract expressionist art, surreal dreamlike, non-representational",
            MVStyle.HOYOVERSE: "anime game cinematic, cel-shaded illustration, dramatic lighting, elemental effects, HoYoverse Genshin Impact quality, vibrant saturated colors, NOT photorealistic",
        }
        style_prefix = style_map.get(request.style, "cinematic")

        # 장르 키워드 (GenreProfile + 스타일 병합)
        genre_gp = self.genre_profiles.get(request.genre.value, {})
        style_gp = self.genre_profiles.get(request.style.value, {}) if request.style else {}
        if genre_gp and style_gp and request.genre.value != request.style.value:
            gp = dict(genre_gp)
            for key in ("prompt_lexicon", "lighting", "motif_library", "avoid_keywords"):
                if style_gp.get(key):
                    gp[key] = style_gp[key]
        elif genre_gp:
            gp = genre_gp
        elif style_gp:
            gp = style_gp
        else:
            gp = {}
        genre_keywords = gp.get("prompt_lexicon", {}).get("positive", "")
        if not genre_keywords:
            genre_map = {
                MVGenre.FANTASY: "fantasy world, magical, epic, moonlit",
                MVGenre.ROMANCE: "romantic atmosphere, emotional, warm",
                MVGenre.ACTION: "dynamic action, intense, explosive",
                MVGenre.HORROR: "dark, horror atmosphere, eerie, dread",
                MVGenre.SCIFI: "futuristic, sci-fi, technology, neon",
                MVGenre.DRAMA: "dramatic, emotional depth, grounded",
                MVGenre.COMEDY: "bright, cheerful, fun, playful",
                MVGenre.ABSTRACT: "abstract, artistic, surreal, dreamscape",
            }
            genre_keywords = genre_map.get(request.genre, "")

        # 분위기 키워드
        mood_map = {
            MVMood.EPIC: "epic, grand scale, majestic",
            MVMood.DREAMY: "dreamy, soft focus, ethereal",
            MVMood.ENERGETIC: "energetic, dynamic, vibrant",
            MVMood.CALM: "calm, peaceful, serene",
            MVMood.DARK: "dark, moody, atmospheric",
            MVMood.ROMANTIC: "romantic, warm, intimate",
            MVMood.MELANCHOLIC: "melancholic, bittersweet, nostalgic",
            MVMood.UPLIFTING: "uplifting, bright, hopeful"
        }
        mood_keywords = mood_map.get(request.mood, "")

        # 씬 위치별 연출 힌트
        if scene_index == 0:
            position_hint = "opening scene, establishing shot"
        elif scene_index == total_scenes - 1:
            position_hint = "finale, climactic moment, closing shot"
        elif scene_index == total_scenes // 2:
            position_hint = "climax, peak emotion, turning point"
        else:
            position_hint = "continuation, building tension"

        # 가사 기반 비주얼 힌트
        lyrics_hint = ""
        if scene.lyrics_text:
            # 가사 첫 줄을 힌트로
            first_line = scene.lyrics_text.split('\n')[0][:50]
            lyrics_hint = f"visual representation of: {first_line}"

        # 컨셉 힌트
        concept_hint = request.concept if request.concept else ""

        # 프롬프트 조합
        prompt_parts = [
            style_prefix,
            genre_keywords,
            mood_keywords,
            position_hint,
            lyrics_hint,
            concept_hint,
            "music video scene",
            "16:9 aspect ratio",
            "professional quality",
            "no text, no letters, no words, no writing, no watermark"
        ]

        # 빈 문자열 제거하고 조합
        prompt = ", ".join([p for p in prompt_parts if p])

        return prompt

    def _get_character_anchors_for_scene(self, project: MVProject, scene: MVScene) -> List[str]:
        """씬에 등장하는 캐릭터의 앵커 이미지 경로 반환 (최대 3개, shot_type 기반 포즈 선택)"""
        if not project.visual_bible or not project.visual_bible.characters:
            return []

        # shot_type → 최적 포즈 매핑
        _SHOT_TO_POSE = {
            "close-up": "front",
            "extreme-close-up": "front",
            "medium": "front",
            "wide": "full_body",
            "full": "full_body",
        }
        # 씬 블로킹에서 shot_type 추출 (VisualBible.scene_blocking 리스트에서 scene_id로 조회)
        shot_type = "medium"
        if project.visual_bible and project.visual_bible.scene_blocking:
            blocking_map = {b.scene_id: b for b in project.visual_bible.scene_blocking}
            blocking = blocking_map.get(scene.scene_id)
            if blocking:
                shot_type = getattr(blocking, 'shot_type', 'medium') or 'medium'
        target_pose = _SHOT_TO_POSE.get(shot_type.lower(), "front")

        results = []
        for role in (scene.characters_in_scene or [])[:3]:
            char = next((c for c in project.visual_bible.characters if c.role == role), None)
            if not char:
                continue
            # 포즈별 앵커에서 최적 포즈 선택
            if char.anchor_poses and target_pose in char.anchor_poses:
                pose_path = char.anchor_poses[target_pose]
                if os.path.exists(pose_path):
                    results.append(pose_path)
                    continue
            # 폴백: 기본 anchor_image_path
            if char.anchor_image_path and os.path.exists(char.anchor_image_path):
                results.append(char.anchor_image_path)
        return results

    def _extract_era_setting(self, concept: str) -> tuple:
        """concept에서 시대/배경 키워드를 추출. (era_prefix, era_negative) 반환."""
        if not concept:
            return ("", "")
        _c = concept.lower()
        _ERA_MAP = {
            "medieval": ("medieval setting, period-accurate props and architecture", "modern objects, café, earphones, headphones, smartphone, laptop, contemporary clothing, hoodie, jeans, sneakers, neon, streetlight, asphalt"),
            "중세": ("medieval setting, period-accurate props and architecture", "modern objects, café, earphones, headphones, smartphone, laptop, contemporary clothing, hoodie, jeans, sneakers, neon, streetlight, asphalt"),
            "ancient": ("ancient era setting, historical architecture", "modern objects, café, earphones, smartphone, contemporary clothing, neon, electric light"),
            "고대": ("ancient era setting, historical architecture", "modern objects, café, earphones, smartphone, contemporary clothing, neon, electric light"),
            "조선": ("Joseon dynasty Korea, traditional hanbok, wooden architecture", "modern objects, café, earphones, smartphone, western clothing, neon, electric light, concrete"),
            "joseon": ("Joseon dynasty Korea, traditional hanbok, wooden architecture", "modern objects, café, earphones, smartphone, western clothing, neon, electric light, concrete"),
            "victorian": ("Victorian era setting, 19th century props and fashion", "modern objects, smartphone, earphones, neon, contemporary clothing, electric devices"),
            "빅토리아": ("Victorian era setting, 19th century props and fashion", "modern objects, smartphone, earphones, neon, contemporary clothing, electric devices"),
            "renaissance": ("Renaissance era setting, classical architecture", "modern objects, smartphone, earphones, neon, contemporary clothing"),
            "르네상스": ("Renaissance era setting, classical architecture", "modern objects, smartphone, earphones, neon, contemporary clothing"),
        }
        for keyword, (prefix, negative) in _ERA_MAP.items():
            if keyword in _c:
                return (prefix, negative)
        return ("", "")

    def _inject_character_descriptions(self, project: MVProject, scene: MVScene, prompt: str) -> str:
        """씬 프롬프트에 Visual Bible 캐릭터의 외형+의상 설명 및 action_pose를 주입.

        LLM이 생성한 프롬프트는 캐릭터 외형을 생략하거나 변경할 수 있음.
        이 메서드가 원본 Visual Bible 설명을 강제 주입하여 일관성 확보.
        또한 action_pose를 주입하여 앵커 이미지의 직립 자세 복제를 방지.
        """
        if not project.visual_bible or not project.visual_bible.characters:
            return prompt
        if not scene.characters_in_scene:
            return prompt

        char_map = {c.role: c for c in project.visual_bible.characters}
        char_descs = []
        for role in scene.characters_in_scene[:3]:
            char = char_map.get(role)
            if not char:
                continue
            parts = [char.description] if char.description else []
            if char.outfit:
                parts.append(f"wearing {char.outfit}")
            if parts:
                char_descs.append(f"[{role.upper()}] {', '.join(parts)}")

        if not char_descs:
            return prompt

        # 씬 블로킹에서 action_pose, expression 추출
        action_pose = ""
        expression = ""
        if project.visual_bible and project.visual_bible.scene_blocking:
            blocking_map = {b.scene_id: b for b in project.visual_bible.scene_blocking}
            blocking = blocking_map.get(scene.scene_id)
            if blocking:
                if blocking.action_pose:
                    action_pose = blocking.action_pose
                if blocking.expression:
                    expression = blocking.expression

        # 렌즈 프롬프트 락: 캐릭터 샷의 얼굴 왜곡 방지
        # 씬 블로킹의 shot_type에 따라 렌즈 토큰 결정
        shot_type_raw = ""
        if project.visual_bible and project.visual_bible.scene_blocking:
            blocking_map = {b.scene_id: b for b in project.visual_bible.scene_blocking}
            b = blocking_map.get(scene.scene_id)
            if b:
                shot_type_raw = (b.shot_type or "").lower()
        is_closeup = "close" in shot_type_raw
        lens_token = "cinematic 50mm lens, natural facial proportions" if is_closeup else "portrait 85mm lens, natural facial proportions"

        # 손/접촉 토큰: 2인 이상 씬에서 손 품질 강화
        hand_token = ""
        if len(scene.characters_in_scene) >= 2:
            hand_token = "natural hands, correct fingers, anatomically correct hands"

        # 프롬프트 조립: action_pose를 최우선에 배치 (앵커 이미지의 직립 자세 덮어쓰기)
        prefix_parts = []
        if action_pose:
            prefix_parts.append(f"POSE: {action_pose}")
        if expression:
            prefix_parts.append(f"EXPRESSION: {expression}")
        prefix_parts.append(lens_token)
        if hand_token:
            prefix_parts.append(hand_token)
        char_block = " | ".join(char_descs)
        prefix_parts.append(char_block)

        return f"{'. '.join(prefix_parts)}. {prompt}"

    # 인종 키워드 매핑 (중복 정의 방지용 공용 상수)
    _ETH_KEYWORD_MAP = {
        "korean": "Korean", "japanese": "Japanese", "chinese": "Chinese",
        "southeast_asian": "Southeast Asian", "european": "European",
        "black": "Black", "hispanic": "Hispanic",
    }

    def _get_ethnicity_prompt_example(self, request) -> str:
        """시스템 프롬프트용 인종 예시 문구 생성 (하드코딩 Korean 방지)."""
        _eth = getattr(request, 'character_ethnicity', None)
        _eth_v = _eth.value if hasattr(_eth, 'value') else str(_eth or 'auto')
        _EXAMPLES = {
            "korean":         "(e.g. 'a Korean man', 'a Korean woman').",
            "japanese":       "(e.g. 'a Japanese man', 'a Japanese woman').",
            "chinese":        "(e.g. 'a Chinese man', 'a Chinese woman').",
            "southeast_asian":"(e.g. 'a Southeast Asian man', 'a Southeast Asian woman').",
            "european":       "(e.g. 'a European man', 'a European woman').",
            "black":          "(e.g. 'a Black man', 'a Black woman').",
            "hispanic":       "(e.g. 'a Hispanic man', 'a Hispanic woman').",
            "mixed":          "State each character's SPECIFIC ethnicity explicitly.",
        }
        return _EXAMPLES.get(_eth_v, "(e.g. 'a [ethnicity] man', 'a [ethnicity] woman').")

    def _get_ethnicity_description_example(self, obj) -> str:
        """캐릭터 외형 예시 문구 생성 (Visual Bible 프롬프트용)."""
        _eth = getattr(obj, 'character_ethnicity', None)
        _eth_v = _eth.value if hasattr(_eth, 'value') else str(_eth or 'auto')
        _DESC = {
            "korean":         "(e.g. 'Korean woman in her mid-20s, long straight black hair, almond eyes, fair skin, slender build')",
            "japanese":       "(e.g. 'Japanese woman in her mid-20s, shoulder-length dark hair, soft features, fair skin, petite build')",
            "chinese":        "(e.g. 'Chinese man in his late-20s, short dark hair, oval face, fair skin, athletic build')",
            "southeast_asian":"(e.g. 'Southeast Asian woman in her mid-20s, wavy dark hair, warm brown skin, expressive eyes')",
            "european":       "(e.g. 'European woman in her mid-20s, wavy auburn hair, blue eyes, fair skin, slender build')",
            "black":          "(e.g. 'Black woman in her mid-20s, natural curly hair, warm brown skin, bright expressive eyes')",
            "hispanic":       "(e.g. 'Hispanic man in his late-20s, dark wavy hair, olive skin, strong jawline, athletic build')",
            "mixed":          "(e.g. 'a woman in her mid-20s with specific ethnicity stated, detailed hair/eye/skin description')",
        }
        return _DESC.get(_eth_v, "(e.g. 'a woman in her mid-20s, long straight hair, almond eyes, fair skin, slender build')")

    def _get_ethnicity_keyword(self, project: "MVProject") -> str:
        """프로젝트의 character_ethnicity에서 인종 키워드 추출."""
        _eth = getattr(project, 'character_ethnicity', None)
        _eth_v = _eth.value if hasattr(_eth, 'value') else str(_eth or 'auto')
        return self._ETH_KEYWORD_MAP.get(_eth_v, "")

    # ================================================================
    # Step 3: 이미지 생성
    # ================================================================

    def generate_images(
        self,
        project: MVProject,
        on_scene_complete: Optional[Callable] = None
    ) -> MVProject:
        """
        씬별 이미지 생성

        Args:
            project: MVProject 객체
            on_scene_complete: 씬 완료 콜백 (scene, index, total)

        Returns:
            업데이트된 MVProject
        """
        print(f"\n[Step 3] Generating images...")

        project_dir = f"{self.output_base_dir}/{project.project_id}"
        image_dir = f"{project_dir}/media/images"
        os.makedirs(image_dir, exist_ok=True)

        total_scenes = len(project.scenes)
        project.current_step = "이미지 생성 중..."

        # v3.0: 전용 스타일 앵커 사용 (scene_01 의존 제거)
        style_anchor_path = project.style_anchor_path
        if style_anchor_path and not os.path.exists(style_anchor_path):
            style_anchor_path = None
        # fallback: 전용 앵커 없으면 첫 번째 씬 이미지를 앵커로 사용
        fallback_anchor = style_anchor_path is None

        # Visual Bible dict (이미지 생성에 전달)
        vb_dict = project.visual_bible.model_dump() if project.visual_bible else None

        # 인종 키워드 (프롬프트에 없으면 자동 주입)
        _eth = getattr(project, 'character_ethnicity', None)
        _eth_v = _eth.value if hasattr(_eth, 'value') else str(_eth or 'auto')
        _ETH_KW = {
            "korean": "Korean", "japanese": "Japanese", "chinese": "Chinese",
            "southeast_asian": "Southeast Asian", "european": "European",
            "black": "Black", "hispanic": "Hispanic",
        }
        ethnicity_keyword = _ETH_KW.get(_eth_v, "")

        # 시대/배경 키워드 (concept에서 추출, 프롬프트에 자동 주입)
        era_prefix, era_negative = self._extract_era_setting(project.concept)
        if era_prefix:
            print(f"  [Era Lock] Detected historical setting: {era_prefix[:40]}...")

        # Pexels B-roll 에이전트 초기화
        pexels = None
        pexels_key = os.environ.get("PEXELS_API_KEY")
        if pexels_key:
            from agents.pexels_agent import PexelsAgent
            pexels = PexelsAgent(api_key=pexels_key)
            print(f"  [B-roll] Pexels agent enabled")

        BROLL_SEGMENTS = {"intro", "outro", "bridge"}

        # 모든 씬에 고유 이미지 생성
        print(f"  Generating {total_scenes} unique images")

        for i, scene in enumerate(project.scenes):
            # 취소 체크
            cancel_path = os.path.join(project_dir, ".cancel")
            if os.path.exists(cancel_path):
                try:
                    os.remove(cancel_path)
                except OSError:
                    pass
                print(f"\n  [CANCELLED] Generation stopped at scene {i+1}/{total_scenes}")
                project.status = MVProjectStatus.CANCELLED
                project.current_step = f"이미지 생성 중단됨 ({i}/{total_scenes})"
                self._save_manifest(project, project_dir)
                return project

            # B-roll 시도: 캐릭터 미등장 + 감정적으로 중요하지 않은 세그먼트만 스톡 사용
            # chorus/hook/pre_chorus는 감정 절정 구간이므로 AI 이미지 강제
            seg_type = self._extract_segment_type(scene)
            _EMOTIONAL_SEGMENTS = {"chorus", "hook", "pre_chorus"}
            is_emotional = seg_type in _EMOTIONAL_SEGMENTS
            if (pexels and not scene.characters_in_scene and not is_emotional):
                _vb = project.visual_bible
                queries = pexels.generate_stock_queries(
                    scene_prompt=scene.image_prompt,
                    lyrics_text=scene.lyrics_text,
                    segment_type=seg_type,
                    genre=project.genre.value,
                    mood=project.mood.value,
                    concept=project.concept or None,
                    era_setting=era_prefix or None,
                    color_palette=_vb.color_palette if _vb and _vb.color_palette else None,
                )
                broll_path = f"{project_dir}/media/video/broll_{scene.scene_id:02d}.mp4"
                os.makedirs(os.path.dirname(broll_path), exist_ok=True)
                video = pexels.fetch_broll(queries, scene.duration_sec, broll_path)
                if video:
                    scene.video_path = video
                    scene.is_broll = True
                    scene.status = MVSceneStatus.COMPLETED
                    # 첫 프레임을 썸네일로 추출
                    thumb_path = f"{image_dir}/scene_{scene.scene_id:02d}.png"
                    self._extract_thumbnail(video, thumb_path)
                    scene.image_path = thumb_path
                    print(f"\n  [Scene {scene.scene_id}/{total_scenes}] B-roll from Pexels ({seg_type})")
                    progress_per_scene = 50 / total_scenes
                    project.progress = int(20 + (i + 1) * progress_per_scene)
                    self._save_manifest(project, project_dir)
                    if on_scene_complete:
                        try:
                            on_scene_complete(scene, i + 1, total_scenes)
                        except Exception:
                            pass
                    continue
                else:
                    print(f"    [Pexels] B-roll failed for {seg_type}, falling back to image gen")

            print(f"\n  [Scene {scene.scene_id}/{total_scenes}] Generating image...")
            if style_anchor_path:
                print(f"    [Anchor] Using style anchor: {os.path.basename(style_anchor_path)}")
            scene.status = MVSceneStatus.GENERATING

            try:
                # 캐릭터 앵커 이미지 조회
                char_anchor_paths = self._get_character_anchors_for_scene(project, scene)
                if char_anchor_paths:
                    print(f"    [Characters] {len(char_anchor_paths)} anchor(s) for scene {scene.scene_id}")

                # 캐릭터 외형+의상 설명 주입 (Visual Bible 기준, LLM 생략/변경 방지)
                final_prompt = self._inject_character_descriptions(project, scene, scene.image_prompt)

                # 인종 키워드 자동 주입 (프롬프트에 없으면 앞에 추가)
                if ethnicity_keyword and ethnicity_keyword.lower() not in final_prompt.lower():
                    final_prompt = f"{ethnicity_keyword} characters, {final_prompt}"

                # 캐릭터 등장 씬: 렌즈 왜곡 + 손 기형 negative
                _scene_neg = scene.negative_prompt or ""
                if scene.characters_in_scene:
                    _lens_neg = "wide-angle distortion, fisheye, exaggerated facial features"
                    _hand_neg = "extra fingers, deformed hands, fused fingers, missing fingers"
                    _char_neg = f"{_lens_neg}, {_hand_neg}"
                    _scene_neg = f"{_char_neg}, {_scene_neg}" if _scene_neg else _char_neg

                # 캐릭터 미등장 씬: 정의 안 된 인물 등장 방지
                if not scene.characters_in_scene:
                    _no_people = "random person, unnamed person, elderly man, old man, bystander, stranger, human figure"
                    _scene_neg = f"{_no_people}, {_scene_neg}" if _scene_neg else _no_people

                # 시대/배경 키워드 자동 주입 (concept에서 추출)
                if era_prefix and era_prefix.lower() not in final_prompt.lower():
                    final_prompt = f"{era_prefix}, {final_prompt}"
                # 시대 부정 키워드 (negative_prompt에 추가)
                if era_negative:
                    _scene_neg = f"{era_negative}, {_scene_neg}" if _scene_neg else era_negative

                # Pass 4: 풀 컨텍스트 이미지 생성
                image_path, _ = self.image_agent.generate_image(
                    scene_id=scene.scene_id,
                    prompt=final_prompt,
                    style=project.style.value,
                    output_dir=image_dir,
                    style_anchor_path=style_anchor_path,
                    genre=project.genre.value,
                    mood=project.mood.value,
                    negative_prompt=_scene_neg or scene.negative_prompt,
                    visual_bible=vb_dict,
                    color_mood=scene.color_mood,
                    character_reference_paths=char_anchor_paths or None,
                    camera_directive=scene.camera_directive,
                )

                scene.image_path = image_path
                scene.status = MVSceneStatus.COMPLETED

                # CharacterQA: 생성 이미지 vs 앵커 임베딩 검증
                if hasattr(self, '_character_qa') and self._character_qa and scene.characters_in_scene:
                    try:
                        qa_results = self._character_qa.verify_scene_image(
                            image_path=image_path,
                            characters_in_scene=scene.characters_in_scene,
                            scene_id=scene.scene_id,
                        )
                        for role, qr in qa_results.items():
                            if not qr["passed"]:
                                print(f"    [CharacterQA] FAIL scene {scene.scene_id} role={role}: "
                                      f"sim={qr['similarity']:.3f} reason={qr['fail_reason']}")
                            else:
                                sim_str = f"{qr['similarity']:.3f}" if qr['similarity'] >= 0 else "n/a"
                                print(f"    [CharacterQA] OK scene {scene.scene_id} role={role}: sim={sim_str}")
                        # QA 결과를 씬에 저장 (compose_video에서 derived cut 제외 판단용)
                        scene.qa_results = qa_results
                    except Exception as qa_e:
                        print(f"    [CharacterQA] Error: {qa_e}")

                # fallback: 전용 앵커 없으면 첫 번째 성공 이미지를 스타일 앵커로 설정
                if fallback_anchor and style_anchor_path is None and image_path and os.path.exists(image_path):
                    style_anchor_path = image_path
                    print(f"    [Anchor] Fallback style anchor set: {os.path.basename(image_path)}")

                print(f"    Image saved: {image_path}")

                # 진행률 업데이트
                progress_per_scene = 50 / total_scenes
                project.progress = int(20 + (i + 1) * progress_per_scene)

            except Exception as e:
                scene.status = MVSceneStatus.FAILED
                print(f"    [ERROR] Image generation failed: {e}")

                # 실패해도 진행률은 업데이트 (멈춤 방지)
                progress_per_scene = 50 / total_scenes
                project.progress = int(20 + (i + 1) * progress_per_scene)

            # 콜백 호출
            if on_scene_complete:
                try:
                    on_scene_complete(scene, i + 1, total_scenes)
                except Exception as cb_e:
                    print(f"    [WARNING] Callback failed: {cb_e}")

            # 매니페스트 저장 (각 씬마다)
            self._save_manifest(project, project_dir)

        self._save_manifest(project, project_dir)

        # 성공한 이미지 수 확인
        success_count = sum(1 for s in project.scenes if s.image_path and os.path.exists(s.image_path))
        total_count = len(project.scenes)
        print(f"[MV] Image generation complete: {success_count}/{total_count} succeeded")

        if success_count == 0:
            project.status = MVProjectStatus.FAILED
            project.progress = 70
            project.current_step = f"이미지 생성 전체 실패 ({total_count}개 씬)"
            self._save_manifest(project, project_dir)
            raise RuntimeError(f"All {total_count} image generations failed")

        # 이미지 생성 완료 → IMAGES_READY 상태로 전환 (리뷰 대기)
        project.status = MVProjectStatus.IMAGES_READY
        project.progress = 70
        project.current_step = f"이미지 생성 완료 ({success_count}/{total_count}) - 리뷰 대기"
        self._save_manifest(project, project_dir)

        return project

    # ================================================================
    # Cut Plan: 파생 컷 플래닝 (24 scenes → 72~120 cuts)
    # ================================================================

    # 세그먼트별 리프레임 시퀀스 (글로벌 카운터 대신 세그먼트 컨텍스트 기반 순환)
    _SEGMENT_REFRAME = {
        "verse":       ["wide", "medium", "medium", "close"],
        "chorus":      ["medium", "close", "wide", "detail", "medium"],
        "hook":        ["close", "medium", "detail", "wide", "close"],
        "pre_chorus":  ["medium", "close", "medium", "wide"],
        "post_chorus": ["medium", "wide", "medium", "close"],
        "bridge":      ["wide", "medium", "wide"],
        "intro":       ["wide", "wide"],
        "outro":       ["wide", "medium"],
    }

    # 세그먼트별 모션 이펙트 시퀀스
    _SEGMENT_EFFECT = {
        "verse":       ["zoom_in", "pan_left", "zoom_in"],
        "chorus":      ["zoom_in", "pan_right", "diagonal", "zoom_out", "zoom_in"],
        "hook":        ["diagonal", "zoom_in", "pan_left", "zoom_out", "diagonal"],
        "pre_chorus":  ["zoom_in", "pan_left", "zoom_in", "diagonal"],
        "post_chorus": ["zoom_out", "pan_right", "zoom_in", "pan_left"],
        "bridge":      ["zoom_out", "pan_left", "zoom_out"],
        "intro":       ["zoom_in", "pan_right"],
        "outro":       ["zoom_out", "pan_left"],
    }

    # 세그먼트별 줌 강도 (calm=tight, energetic=wide range)
    _SEGMENT_ZOOM = {
        "verse":       (1.0, 1.06),
        "chorus":      (1.0, 1.12),
        "hook":        (1.0, 1.14),
        "pre_chorus":  (1.0, 1.08),
        "post_chorus": (1.0, 1.06),
        "bridge":      (1.0, 1.04),
        "intro":       (1.0, 1.05),
        "outro":       (1.0, 1.04),
    }

    # 얼굴 앵커 5개 (공간적 분리 >= 0.08로 ghosting 방지)
    _FACE_ANCHORS = [
        (0.50, 0.28), (0.42, 0.32), (0.58, 0.30),
        (0.48, 0.25), (0.52, 0.35),
    ]

    # 전환 타입별 duration (초)
    _TRANSITION_DURATIONS = {
        "cut": 0.0, "xfade": 0.3, "fadeblack": 0.5,
        "whiteflash": 0.25, "filmburn": 0.7, "glitch": 0.2,
    }

    def _generate_cut_plan(self, scenes: List[MVScene], project=None) -> List[dict]:
        """
        Subject-aware 컷 플래닝: bbox 기반 크롭 + 세그먼트별 시퀀스 + QA 검증.
        I2V(video_path) 씬은 분할하지 않고 단일 컷으로 통과.
        Ghost 방지 + Face-safe padding + Match-cut 연속성.

        Args:
            scenes: 완료된 MVScene 리스트
            project: MVProject (visual_continuity 참조용, optional)

        Returns:
            List of cut dicts with bbox/QA 로깅 메타데이터 포함
        """
        # 세그먼트 타입별 컷 설정
        cut_config = {
            "verse":       {"cuts": 3, "dur_range": (1.5, 3.0)},
            "chorus":      {"cuts": 5, "dur_range": (0.6, 1.2)},
            "hook":        {"cuts": 5, "dur_range": (0.6, 1.2)},
            "pre_chorus":  {"cuts": 4, "dur_range": (0.8, 1.5)},
            "post_chorus": {"cuts": 4, "dur_range": (0.8, 1.5)},
            "intro":       {"cuts": 2, "dur_range": (2.0, 4.0)},
            "outro":       {"cuts": 2, "dur_range": (2.0, 4.0)},
            "bridge":      {"cuts": 3, "dur_range": (1.0, 2.0)},
        }
        default_config = {"cuts": 3, "dur_range": (1.0, 2.5)}

        # shot_type별 줌 범위 (Rule 2) + Δscale 제한은 FFmpeg에서 추가 적용
        _SHOT_ZOOM = {
            "wide":   (1.00, 1.05),
            "medium": (1.05, 1.12),   # capped by Δscale 0.06
            "close":  (1.10, 1.18),   # capped; max 1.18 per Closeup_fix
            "detail": (1.10, 1.16),   # capped
        }

        # general anchor positions (bbox 없을 때 fallback)
        anchor_positions = [
            (0.33, 0.33), (0.5, 0.5), (0.67, 0.33),
            (0.33, 0.67), (0.67, 0.67), (0.5, 0.33),
        ]

        # ── bbox 캐시: 이미지당 1회만 추출 ──
        bbox_cache = {}
        unique_images = set()
        for scene in scenes:
            if scene.image_path and os.path.exists(scene.image_path):
                unique_images.add(scene.image_path)

        print(f"  [BBOX] Extracting subject bbox for {len(unique_images)} unique images...")
        for img_path in unique_images:
            try:
                bbox_cache[img_path] = self._extract_subject_bbox(img_path)
            except Exception as e:
                print(f"    [BBOX] Failed for {os.path.basename(img_path)}: {str(e)[:100]}")
                bbox_cache[img_path] = {
                    "face_bbox": None, "person_bbox": None,
                    "saliency_center": {"x": 0.5, "y": 0.4}, "bbox_source": "saliency"
                }

        bbox_sources = {}
        for v in bbox_cache.values():
            src = v.get("bbox_source", "unknown")
            bbox_sources[src] = bbox_sources.get(src, 0) + 1
        print(f"  [BBOX] Sources: {', '.join(f'{k}:{v}' for k, v in bbox_sources.items())}")

        # Face-safe padding (Rule 3)
        _PAD_TOP = 0.12
        _PAD_BOTTOM = 0.10
        _PAD_SIDES = 0.08
        _SUBTITLE_SAFE = 0.20  # Rule 6: 하단 20%

        cut_plan = []
        prev_reframe = None
        prev_center_x = None  # match-cut 연속성용
        motion_direction = "zoom_in"  # match-cut 모션 방향 연속성
        direction_counter = 0
        qa_stats = {"passed": 0, "retried": 0, "downgraded": 0}

        for scene in scenes:
            is_broll = getattr(scene, 'is_broll', False)
            has_chars = bool(scene.characters_in_scene)
            seg_type = self._extract_segment_type(scene)

            # I2V 씬은 분할하지 않음
            if scene.video_path and os.path.exists(scene.video_path):
                cut_plan.append({
                    "parent_scene_id": scene.scene_id,
                    "cut_index": 0,
                    "duration_sec": scene.duration_sec,
                    "reframe": "wide",
                    "crop_anchor": (0.5, 0.5),
                    "effect_type": "zoom_in",
                    "zoom_range": (1.0, 1.05),
                    "image_path": None,
                    "video_path": scene.video_path,
                    "is_broll": is_broll,
                    "shot_type": "wide",
                    "source": "stock" if is_broll else "ai",
                    "has_characters": has_chars,
                    "segment_type": seg_type,
                    "bbox_source": "none",
                })
                prev_reframe = "wide"
                continue

            if not scene.image_path or not os.path.exists(scene.image_path):
                continue

            config = cut_config.get(seg_type, default_config)
            num_cuts = config["cuts"]
            total_dur = scene.duration_sec
            cut_dur = total_dur / num_cuts

            # bbox 조회
            bbox_info = bbox_cache.get(scene.image_path, {
                "face_bbox": None, "person_bbox": None,
                "saliency_center": {"x": 0.5, "y": 0.4}, "bbox_source": "saliency"
            })
            face_bb = bbox_info.get("face_bbox")
            person_bb = bbox_info.get("person_bbox")
            saliency = bbox_info.get("saliency_center", {"x": 0.5, "y": 0.4})

            # 세그먼트별 시퀀스
            reframe_seq = self._SEGMENT_REFRAME.get(seg_type, self._SEGMENT_REFRAME["verse"])
            effect_seq = self._SEGMENT_EFFECT.get(seg_type, self._SEGMENT_EFFECT["verse"])

            # multi-face ambiguity flag
            is_multi_face_ambiguous = bbox_info.get("multi_face_ambiguous", False)
            face_count = bbox_info.get("face_count", 1 if face_bb else 0)

            for ci in range(num_cuts):
                reframe = reframe_seq[ci % len(reframe_seq)]
                effect = effect_seq[ci % len(effect_seq)]

                # Ghost 방지
                if reframe in ("close", "detail") and prev_reframe in ("close", "detail"):
                    reframe = "medium"

                # CharacterQA: 실패 씬은 close/detail 컷 제외 → medium으로 강등
                if has_chars and reframe in ("close", "detail"):
                    _qa_res = getattr(scene, 'qa_results', None)
                    if _qa_res and not all(r.get("passed", True) for r in _qa_res.values()):
                        reframe = "medium"

                # ── Closeup_fix #1: face_bbox 필수 (CLOSEUP/MEDIUM) ──
                if has_chars and reframe in ("close", "detail") and not face_bb:
                    # face_bbox 없으면 CLOSEUP/DETAIL은 WIDE로 강등
                    reframe = "wide"
                    qa_stats["downgraded"] += 1
                    print(f"    [QA] scene {scene.scene_id} cut {ci}: close->wide (no face_bbox)")

                if has_chars and reframe == "medium" and not face_bb:
                    # MEDIUM은 saliency fallback 허용하지만 로그 남김
                    print(f"    [QA] scene {scene.scene_id} cut {ci}: medium w/o face_bbox (saliency fallback)")

                # ── Closeup_fix #4: 다인 케이스 - ambiguous면 CLOSEUP 금지 ──
                if is_multi_face_ambiguous and reframe in ("close", "detail"):
                    reframe = "medium"
                    qa_stats["downgraded"] += 1
                    print(f"    [QA] scene {scene.scene_id} cut {ci}: close->medium (multi_face_ambiguous)")

                # ── Subject-aware crop_anchor (Rule 1) ──
                if reframe in ("close", "medium") and face_bb and has_chars:
                    # 얼굴 bbox 중심을 crop center로
                    anchor = (
                        face_bb["x"] + face_bb["w"] / 2,
                        face_bb["y"] + face_bb["h"] / 2,
                    )
                elif reframe == "wide" and person_bb:
                    anchor = (
                        person_bb["x"] + person_bb["w"] / 2,
                        person_bb["y"] + person_bb["h"] / 2,
                    )
                elif face_bb and has_chars:
                    anchor = (
                        face_bb["x"] + face_bb["w"] / 2,
                        face_bb["y"] + face_bb["h"] / 2,
                    )
                else:
                    anchor = (saliency["x"], saliency["y"])

                # ── Match-cut: 프레이밍 연속성 (Rule C-1) ──
                if has_chars and prev_center_x is not None:
                    dx = anchor[0] - prev_center_x
                    if abs(dx) > 0.05:
                        # ±5% 이내로 제한 (강박자에서만 반전 허용)
                        is_strong = seg_type in ("chorus", "hook")
                        if not is_strong:
                            anchor = (
                                prev_center_x + max(-0.05, min(0.05, dx)),
                                anchor[1],
                            )

                # ── Match-cut: 모션 방향 연속성 (Rule C-2) ──
                direction_counter += 1
                if direction_counter <= 6:
                    # 4-8컷 단위로 방향 유지
                    if effect in ("zoom_in", "zoom_out"):
                        if motion_direction in ("zoom_in", "zoom_out") and effect != motion_direction:
                            effect = motion_direction
                else:
                    # 전환 허용, 카운터 리셋
                    if effect in ("zoom_in", "zoom_out"):
                        motion_direction = effect
                    direction_counter = 0

                # shot_type별 줌 범위
                zoom_range = _SHOT_ZOOM.get(reframe, (1.0, 1.05))

                # ── QA Gate (Rule 1-6 + Closeup_fix #2 검증) ──
                qa_passed = True
                qa_fail_reason = None
                retry_count = 0
                original_reframe = reframe
                eyes_y_ratio = None
                chin_margin_ratio = None

                while retry_count < 4:
                    # reframe ratio
                    reframe_ratios = {"wide": 1.0, "medium": 0.75, "close": 0.5, "detail": 0.33}
                    ratio = reframe_ratios.get(reframe, 1.0)

                    qa_passed = True
                    qa_fail_reason = None
                    eyes_y_ratio = None
                    chin_margin_ratio = None

                    if face_bb and has_chars and reframe in ("close", "medium"):
                        face_cx = face_bb["x"] + face_bb["w"] / 2
                        face_cy = face_bb["y"] + face_bb["h"] / 2
                        face_top = face_bb["y"]
                        face_bottom = face_bb["y"] + face_bb["h"]

                        # 크롭 영역 계산 (정규화 좌표)
                        crop_left = anchor[0] - ratio / 2
                        crop_top = anchor[1] - ratio / 2
                        crop_right = anchor[0] + ratio / 2
                        crop_bottom = anchor[1] + ratio / 2

                        # Rule 3: Face-safe padding
                        face_top_in_crop = (face_top - crop_top) / ratio if ratio > 0 else 0
                        face_bottom_in_crop = (face_bottom - crop_top) / ratio if ratio > 0 else 1
                        if face_top_in_crop < _PAD_TOP:
                            qa_fail_reason = f"head_clip(top_pad={face_top_in_crop:.2f}<{_PAD_TOP})"
                            qa_passed = False
                        elif face_bottom_in_crop > (1.0 - _PAD_BOTTOM):
                            qa_fail_reason = f"chin_clip(bot_pad={1-face_bottom_in_crop:.2f}<{_PAD_BOTTOM})"
                            qa_passed = False

                        # Rule 6: Subtitle safe area
                        if qa_passed and face_bottom_in_crop > (1.0 - _SUBTITLE_SAFE):
                            qa_fail_reason = f"subtitle_overlap(face_bot={face_bottom_in_crop:.2f})"
                            qa_passed = False

                        # QA: 인물 중심 과도 치우침 (15% 이상)
                        face_cx_in_crop = (face_cx - crop_left) / ratio if ratio > 0 else 0.5
                        if qa_passed and abs(face_cx_in_crop - 0.5) > 0.15:
                            qa_fail_reason = f"off_center(cx={face_cx_in_crop:.2f})"
                            qa_passed = False

                        # ── Closeup_fix #2: 프레이밍 품질 (eyes_y / chin_y) ──
                        # eyes_y ≈ face_top + 0.35 * face_h
                        eyes_y = face_top + 0.35 * face_bb["h"]
                        eyes_y_in_crop = (eyes_y - crop_top) / ratio if ratio > 0 else 0.5
                        eyes_y_ratio = eyes_y_in_crop

                        # chin_y ≈ face_top + 0.90 * face_h
                        chin_y = face_top + 0.90 * face_bb["h"]
                        chin_y_in_crop = (chin_y - crop_top) / ratio if ratio > 0 else 0.5
                        chin_margin = 1.0 - chin_y_in_crop
                        chin_margin_ratio = chin_margin

                        if qa_passed and (eyes_y_in_crop < 0.22 or eyes_y_in_crop > 0.42):
                            qa_fail_reason = f"eyes_y_out(eyes_y={eyes_y_in_crop:.2f},need=[0.22,0.42])"
                            qa_passed = False

                        if qa_passed and chin_margin < 0.12:
                            qa_fail_reason = f"chin_margin_low(margin={chin_margin:.2f}<0.12)"
                            qa_passed = False

                        # ── Closeup_fix #3: CLOSEUP scale 상한 강제 ──
                        if reframe == "close" and zoom_range[1] > 1.18:
                            zoom_range = (zoom_range[0], 1.18)

                    if qa_passed:
                        qa_stats["passed"] += 1
                        break

                    # ── 재시도 (비용 0: 파라미터만 조정) ──
                    retry_count += 1
                    qa_stats["retried"] += 1
                    if retry_count == 1:
                        # 패딩 증가: anchor를 얼굴 위쪽으로 이동
                        anchor = (anchor[0], anchor[1] - 0.03)
                    elif retry_count == 2:
                        # 스케일 상한 감소
                        zoom_range = (zoom_range[0], zoom_range[0] + 0.04)
                    elif retry_count == 3:
                        # CLOSEUP → MEDIUM → WIDE 강등 (가장 안전)
                        if reframe in ("close", "detail"):
                            reframe = "medium"
                        else:
                            reframe = "wide"
                        zoom_range = _SHOT_ZOOM.get(reframe, (1.0, 1.05))
                        qa_stats["downgraded"] += 1
                        qa_passed = True
                        qa_fail_reason = f"downgraded_from_{original_reframe}"
                        break

                # crop_anchor 범위 클램핑
                anchor = (
                    max(0.1, min(0.9, anchor[0])),
                    max(0.1, min(0.9, anchor[1])),
                )

                prev_center_x = anchor[0] if has_chars else prev_center_x

                cut_plan.append({
                    "parent_scene_id": scene.scene_id,
                    "cut_index": ci,
                    "duration_sec": round(cut_dur, 2),
                    "reframe": reframe,
                    "crop_anchor": anchor,
                    "effect_type": effect,
                    "zoom_range": zoom_range,
                    "image_path": scene.image_path,
                    "video_path": None,
                    "shot_type": reframe,
                    "source": "stock" if is_broll else "ai",
                    "has_characters": has_chars,
                    "segment_type": seg_type,
                    # 로깅 메타데이터
                    "bbox_source": bbox_info.get("bbox_source", "none"),
                    "bbox_values": face_bb or person_bb,
                    "qa_fail_reason": qa_fail_reason,
                    # ── Closeup_fix: 확장 로깅 ──
                    "eyes_y_ratio": round(eyes_y_ratio, 3) if eyes_y_ratio is not None else None,
                    "chin_margin_ratio": round(chin_margin_ratio, 3) if chin_margin_ratio is not None else None,
                    "scale_from": zoom_range[0],
                    "scale_to": zoom_range[1],
                    "face_count": face_count,
                    "multi_face_ambiguous": is_multi_face_ambiguous,
                })
                prev_reframe = reframe

        # 로그
        type_counts = {}
        for scene in scenes:
            seg = self._extract_segment_type(scene)
            cfg = cut_config.get(seg, default_config)
            if scene.video_path and os.path.exists(scene.video_path):
                type_counts[seg] = type_counts.get(seg, 0) + 1
            else:
                type_counts[seg] = type_counts.get(seg, 0) + cfg["cuts"]
        type_summary = ", ".join(f"{k}:{v}" for k, v in type_counts.items())
        print(f"  [CUT PLAN] {len(scenes)} scenes -> {len(cut_plan)} cuts ({type_summary})")
        print(f"  [QA] passed:{qa_stats['passed']}, retried:{qa_stats['retried']}, downgraded:{qa_stats['downgraded']}")

        return cut_plan

    def _extract_segment_type(self, scene: MVScene) -> str:
        """씬의 visual_description에서 segment_type 추출."""
        desc = (scene.visual_description or "").lower()
        # "verse section" → "verse", "chorus section" → "chorus"
        for seg_type in ["pre_chorus", "post_chorus", "chorus", "hook", "verse", "bridge", "intro", "outro"]:
            if seg_type.replace("_", " ") in desc or seg_type.replace("_", "-") in desc or seg_type in desc:
                return seg_type
        return "verse"  # default

    # ================================================================
    # Subject BBox Extraction (Gemini Vision + OpenCV fallback)
    # ================================================================

    def _extract_subject_bbox(self, image_path: str) -> dict:
        """이미지에서 주 피사체 bbox 추출. Gemini Vision > OpenCV > saliency fallback.

        Returns:
            {
                "face_bbox": {"x": float, "y": float, "w": float, "h": float} or None,
                "person_bbox": {"x": float, "y": float, "w": float, "h": float} or None,
                "saliency_center": {"x": float, "y": float},
                "bbox_source": "gemini" | "opencv" | "saliency"
            }
        """
        # 1) Gemini Vision (recommended)
        try:
            result = self._gemini_bbox(image_path)
            if result and (result.get("face_bbox") or result.get("person_bbox")):
                result["bbox_source"] = "gemini"
                return result
        except Exception as e:
            print(f"    [bbox] Gemini Vision failed: {str(e)[:100]}")

        # 2) OpenCV fallback
        try:
            result = self._opencv_bbox(image_path)
            if result and result.get("face_bbox"):
                result["bbox_source"] = "opencv"
                return result
        except Exception as e:
            print(f"    [bbox] OpenCV failed: {str(e)[:100]}")

        # 3) Saliency center fallback
        result = self._saliency_center(image_path)
        result["bbox_source"] = "saliency"
        return result

    def _gemini_bbox(self, image_path: str) -> Optional[dict]:
        """Gemini Vision으로 face/person bbox 추출 (normalized 0~1 좌표).
        다인 케이스: face_count + multi_face_ambiguous 반환.
        """
        import os as _os
        api_key = _os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            return None

        from google import genai
        from google.genai import types
        from PIL import Image
        import io

        client = genai.Client(api_key=api_key)

        # 이미지를 작은 크기로 리사이즈 (비용/속도 절감)
        img = Image.open(image_path).convert("RGB")
        img_w, img_h = img.size
        img_small = img.resize((512, int(512 * img_h / img_w)))

        buf = io.BytesIO()
        img_small.save(buf, format="JPEG", quality=80)
        img_bytes = buf.getvalue()

        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=[
                types.Content(parts=[
                    types.Part.from_bytes(data=img_bytes, mime_type="image/jpeg"),
                    types.Part.from_text(
                        "Analyze this image. Return ONLY a JSON object (no markdown) with:\n"
                        "- face_bbox: bounding box of the DOMINANT (largest or most centered) face {x, y, w, h} in normalized 0-1 coordinates, or null if no face\n"
                        "- person_bbox: bounding box of the main person {x, y, w, h} in normalized 0-1 coordinates, or null if no person\n"
                        "- saliency_center: {x, y} center of visual interest in normalized 0-1 coordinates\n"
                        "- face_count: total number of faces detected in the image (integer)\n"
                        "Example: {\"face_bbox\": {\"x\": 0.35, \"y\": 0.15, \"w\": 0.3, \"h\": 0.25}, "
                        "\"person_bbox\": {\"x\": 0.2, \"y\": 0.05, \"w\": 0.6, \"h\": 0.9}, "
                        "\"saliency_center\": {\"x\": 0.5, \"y\": 0.3}, \"face_count\": 1}"
                    ),
                ]),
            ],
            config=types.GenerateContentConfig(
                temperature=0.1,
                max_output_tokens=400,
            ),
        )

        text = response.text.strip()
        # JSON 파싱 (markdown 코드블록 제거)
        if text.startswith("```"):
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
        text = text.strip()

        data = json.loads(text)
        face_count = int(data.get("face_count", 1))
        # face_count >= 2이고 dominant face가 명확하지 않으면 ambiguous
        multi_face_ambiguous = face_count >= 2

        result = {
            "face_bbox": None,
            "person_bbox": None,
            "saliency_center": {"x": 0.5, "y": 0.4},
            "face_count": face_count,
            "multi_face_ambiguous": multi_face_ambiguous,
        }

        if data.get("face_bbox"):
            fb = data["face_bbox"]
            if all(k in fb for k in ("x", "y", "w", "h")):
                result["face_bbox"] = {
                    "x": float(fb["x"]), "y": float(fb["y"]),
                    "w": float(fb["w"]), "h": float(fb["h"]),
                }
                # Gemini가 dominant face를 선택했으므로 단일 face는 ambiguous 아님
                if face_count == 1:
                    multi_face_ambiguous = False
                    result["multi_face_ambiguous"] = False

        if data.get("person_bbox"):
            pb = data["person_bbox"]
            if all(k in pb for k in ("x", "y", "w", "h")):
                result["person_bbox"] = {
                    "x": float(pb["x"]), "y": float(pb["y"]),
                    "w": float(pb["w"]), "h": float(pb["h"]),
                }

        if data.get("saliency_center"):
            sc = data["saliency_center"]
            result["saliency_center"] = {
                "x": float(sc.get("x", 0.5)),
                "y": float(sc.get("y", 0.4)),
            }

        if face_count >= 2:
            print(f"    [bbox] Gemini multi-face: {face_count} faces, ambiguous={multi_face_ambiguous}")

        return result

    def _opencv_bbox(self, image_path: str) -> Optional[dict]:
        """OpenCV Haar cascade로 얼굴 bbox 추출 (fallback).
        다인(multi-face) 케이스: dominant face 선택 + 불명확 시 ambiguous 플래그.
        """
        import cv2

        img = cv2.imread(image_path)
        if img is None:
            return None

        img_h, img_w = img.shape[:2]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(30, 30))

        if len(faces) == 0:
            return None

        face_count = len(faces)
        multi_face_ambiguous = False

        # 면적 내림차순 정렬
        sorted_faces = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)

        # dominant face 선택: 가장 큰 얼굴
        x, y, w, h = sorted_faces[0]

        # 다인 케이스: 가장 큰 얼굴이 차순위보다 1.5배 이상 크지 않으면 ambiguous
        if face_count >= 2:
            area_1st = sorted_faces[0][2] * sorted_faces[0][3]
            area_2nd = sorted_faces[1][2] * sorted_faces[1][3]
            if area_1st < area_2nd * 1.5:
                # 면적 비슷 → 프레임 중앙에 가장 가까운 얼굴 선택
                center_x, center_y = img_w / 2, img_h / 2
                def dist_to_center(f):
                    fx = f[0] + f[2] / 2
                    fy = f[1] + f[3] / 2
                    return ((fx - center_x) ** 2 + (fy - center_y) ** 2) ** 0.5
                closest = min(sorted_faces, key=dist_to_center)
                x, y, w, h = closest
                multi_face_ambiguous = True
            print(f"    [bbox] Multi-face: {face_count} faces, ambiguous={multi_face_ambiguous}")

        face_bbox = {
            "x": x / img_w, "y": y / img_h,
            "w": w / img_w, "h": h / img_h,
        }

        # 얼굴 중심 기반 person bbox 추정 (얼굴 높이의 ~4배)
        face_cx = face_bbox["x"] + face_bbox["w"] / 2
        face_cy = face_bbox["y"] + face_bbox["h"] / 2
        person_h = min(face_bbox["h"] * 4, 0.95)
        person_w = min(face_bbox["w"] * 2.5, 0.8)
        person_bbox = {
            "x": max(0, face_cx - person_w / 2),
            "y": max(0, face_cy - face_bbox["h"] * 0.5),
            "w": person_w,
            "h": person_h,
        }

        return {
            "face_bbox": face_bbox,
            "person_bbox": person_bbox,
            "saliency_center": {"x": face_cx, "y": face_cy},
            "face_count": face_count,
            "multi_face_ambiguous": multi_face_ambiguous,
        }

    def _saliency_center(self, image_path: str) -> dict:
        """PIL로 이미지의 시각적 관심 중심점 추출 (최종 fallback)."""
        try:
            from PIL import Image
            img = Image.open(image_path).convert("RGB").resize((64, 64))
            pixels = list(img.getdata())
            w, h = 64, 64
            # 밝기 가중 중심점 계산
            total_weight = 0
            cx, cy = 0, 0
            for i, (r, g, b) in enumerate(pixels):
                weight = 0.299 * r + 0.587 * g + 0.114 * b
                px = (i % w) / w
                py = (i // w) / h
                cx += px * weight
                cy += py * weight
                total_weight += weight
            if total_weight > 0:
                cx /= total_weight
                cy /= total_weight
            else:
                cx, cy = 0.5, 0.4
            return {
                "face_bbox": None,
                "person_bbox": None,
                "saliency_center": {"x": cx, "y": cy},
            }
        except Exception:
            return {
                "face_bbox": None,
                "person_bbox": None,
                "saliency_center": {"x": 0.5, "y": 0.4},
            }

    # ================================================================
    # Transition Planner: 씬 간 전환 타입 자동 결정
    # ================================================================

    def _save_cuts_and_render_log(self, project, project_dir: str, cut_plan: list,
                                   transition_plan: list, scenes: list) -> None:
        """cuts.json + render.log 저장. 디버깅/튜닝 산출물."""
        import json as _json
        from datetime import datetime as _dt

        # ── cuts.json ──
        cuts_path = f"{project_dir}/cuts.json"
        try:
            serializable_cuts = []
            for c in cut_plan:
                sc = {}
                for k, v in c.items():
                    if isinstance(v, tuple):
                        sc[k] = list(v)
                    elif isinstance(v, (str, int, float, bool, type(None), list, dict)):
                        sc[k] = v
                    else:
                        sc[k] = str(v)
                serializable_cuts.append(sc)
            with open(cuts_path, "w", encoding="utf-8") as f:
                _json.dump(serializable_cuts, f, indent=2, ensure_ascii=False)
            print(f"  [LOG] cuts.json saved ({len(cut_plan)} cuts)")
        except Exception as e:
            print(f"  [LOG] cuts.json save failed: {e}")

        # ── render.log ──
        log_path = f"{project_dir}/render.log"
        try:
            lines = []
            lines.append(f"# Render Log - {_dt.now().isoformat()}")
            lines.append(f"# Project: {project.project_id}")
            lines.append(f"# Scenes: {len(scenes)}, Cuts: {len(cut_plan)}, Transitions: {len(transition_plan)}")
            lines.append("")

            # Section 1: Cut Plan + BBox + QA + Framing Quality
            lines.append("## CUT PLAN")
            lines.append(f"{'scene':>5} {'ci':>3} {'reframe':>8} {'effect':>10} {'seg':>12} "
                         f"{'bbox_src':>8} {'eyes_y':>7} {'chin_m':>7} {'scale':>12} {'faces':>5} "
                         f"{'qa_fail':>30} {'anchor':>14}")
            lines.append("-" * 140)
            for c in cut_plan:
                anchor = c.get("crop_anchor", (0, 0))
                anchor_str = f"({anchor[0]:.2f},{anchor[1]:.2f})" if isinstance(anchor, (tuple, list)) else str(anchor)
                eyes_y = c.get("eyes_y_ratio")
                chin_m = c.get("chin_margin_ratio")
                s_from = c.get("scale_from", 0)
                s_to = c.get("scale_to", 0)
                fc = c.get("face_count", 0)
                lines.append(
                    f"{c.get('parent_scene_id', '?'):>5} "
                    f"{c.get('cut_index', 0):>3} "
                    f"{c.get('reframe', '?'):>8} "
                    f"{c.get('effect_type', '?'):>10} "
                    f"{c.get('segment_type', '?'):>12} "
                    f"{c.get('bbox_source', 'none'):>8} "
                    f"{(f'{eyes_y:.2f}' if eyes_y is not None else '-'):>7} "
                    f"{(f'{chin_m:.2f}' if chin_m is not None else '-'):>7} "
                    f"{f'{s_from:.2f}-{s_to:.2f}':>12} "
                    f"{fc:>5} "
                    f"{(c.get('qa_fail_reason') or 'OK'):>30} "
                    f"{anchor_str:>14}"
                )

            # Section 2: Transition Plan
            lines.append("")
            lines.append("## TRANSITION PLAN")
            lines.append(f"{'boundary':>8} {'type':>12} {'frames':>6} {'reason':>40}")
            lines.append("-" * 80)
            for ti, t in enumerate(transition_plan):
                lines.append(
                    f"{ti:>8} "
                    f"{t.get('transition_type', '?'):>12} "
                    f"{t.get('transition_frames', 0):>6} "
                    f"{t.get('reason', ''):>40}"
                )

            # Section 3: CharacterQA results
            qa = getattr(self, '_character_qa', None)
            if qa:
                qa_entries = qa.get_log_entries()
                if qa_entries:
                    lines.append("")
                    lines.append("## CHARACTER QA")
                    lines.append(f"{'scene':>5} {'role':>15} {'face':>5} {'sim':>6} {'pass':>5} {'reason':>25} {'provider':>20}")
                    lines.append("-" * 90)
                    for entry in qa_entries:
                        lines.append(
                            f"{entry.get('scene_id', '?'):>5} "
                            f"{entry.get('role', '?'):>15} "
                            f"{'Y' if entry.get('face_detected') else 'N':>5} "
                            f"{entry.get('similarity', -1):>6.3f} "
                            f"{'PASS' if entry.get('passed') else 'FAIL':>5} "
                            f"{(entry.get('fail_reason') or 'OK'):>25} "
                            f"{entry.get('provider', ''):>20}"
                        )

            # Section 4: Summary stats
            lines.append("")
            lines.append("## SUMMARY")
            bbox_sources = {}
            for c in cut_plan:
                src = c.get("bbox_source", "none")
                bbox_sources[src] = bbox_sources.get(src, 0) + 1
            lines.append(f"BBox sources: {bbox_sources}")

            qa_fail_counts = {}
            for c in cut_plan:
                reason = c.get("qa_fail_reason") or "OK"
                key = reason.split("(")[0]  # strip detail
                qa_fail_counts[key] = qa_fail_counts.get(key, 0) + 1
            lines.append(f"QA results: {qa_fail_counts}")

            t_counts = {}
            for t in transition_plan:
                tt = t.get("transition_type", "unknown")
                t_counts[tt] = t_counts.get(tt, 0) + 1
            lines.append(f"Transitions: {t_counts}")

            if qa:
                qa_entries = qa.get_log_entries()
                total_qa = len(qa_entries)
                passed_qa = sum(1 for e in qa_entries if e.get("passed"))
                lines.append(f"CharacterQA: {passed_qa}/{total_qa} passed (threshold={qa.threshold})")
                sims = [e.get("similarity", -1) for e in qa_entries if e.get("similarity", -1) >= 0]
                if sims:
                    lines.append(f"  Similarity distribution: min={min(sims):.3f}, max={max(sims):.3f}, "
                                 f"avg={sum(sims)/len(sims):.3f}, median={sorted(sims)[len(sims)//2]:.3f}")

            with open(log_path, "w", encoding="utf-8") as f:
                f.write("\n".join(lines))
            print(f"  [LOG] render.log saved ({len(lines)} lines)")
        except Exception as e:
            print(f"  [LOG] render.log save failed: {e}")

    def _get_palette_stats(self, image_path: str) -> dict:
        """이미지에서 평균 밝기(Y)와 채도(S) 추출. PIL 사용."""
        try:
            from PIL import Image
            img = Image.open(image_path).convert("RGB").resize((64, 64))
            pixels = list(img.getdata())
            avg_r = sum(p[0] for p in pixels) / len(pixels)
            avg_g = sum(p[1] for p in pixels) / len(pixels)
            avg_b = sum(p[2] for p in pixels) / len(pixels)
            avg_y = 0.299 * avg_r + 0.587 * avg_g + 0.114 * avg_b
            max_c = max(avg_r, avg_g, avg_b)
            min_c = min(avg_r, avg_g, avg_b)
            avg_s = (max_c - min_c) / max_c if max_c > 0 else 0
            return {"avgY": avg_y, "avgS": avg_s}
        except Exception:
            return {"avgY": 128.0, "avgS": 0.5}

    def _plan_transitions(self, cuts: List[dict], scenes: List[MVScene], project=None) -> List[dict]:
        """
        컷 경계마다 전환 타입을 결정. 9가지 우선순위 규칙 + 예산 제한.

        Args:
            cuts: _generate_cut_plan 출력 (shot_type, source, has_characters, segment_type 포함)
            scenes: 완료된 MVScene 리스트
            project: MVProject (visual_bible 참조용)

        Returns:
            List[dict] - 각 컷 경계(len = len(cuts)-1)마다:
                transition_type, transition_frames, overlay_asset_path
        """
        if len(cuts) <= 1:
            return []

        # visual_continuity 매핑 빌드 (scene_id -> continuity hint)
        vc_map = {}
        if project and hasattr(project, 'visual_bible') and project.visual_bible:
            vb = project.visual_bible
            if hasattr(vb, 'scene_blocking') and vb.scene_blocking:
                for sb in vb.scene_blocking:
                    if hasattr(sb, 'scene_id') and hasattr(sb, 'visual_continuity'):
                        vc_map[sb.scene_id] = getattr(sb, 'visual_continuity', '')

        # scene_id -> characters 매핑
        scene_chars = {}
        for s in scenes:
            scene_chars[s.scene_id] = set(s.characters_in_scene or [])

        # palette stats 캐시 (이미지 경로 -> stats)
        palette_cache = {}
        for cut in cuts:
            img = cut.get("image_path")
            if img and img not in palette_cache:
                palette_cache[img] = self._get_palette_stats(img)

        # 예산 계산
        total_scenes = len(set(c["parent_scene_id"] for c in cuts))
        glitch_budget = max(1, total_scenes // 15)
        filmburn_budget = max(1, total_scenes // 12)
        glitch_used = 0
        filmburn_used = 0

        fps = 30
        transitions = []

        for i in range(len(cuts) - 1):
            cur = cuts[i]
            nxt = cuts[i + 1]
            same_scene = cur["parent_scene_id"] == nxt["parent_scene_id"]

            # Rule 1: 같은 씬 내 컷 -> hard cut
            if same_scene:
                transitions.append({
                    "transition_type": "cut",
                    "transition_frames": 0,
                    "overlay_asset_path": None,
                })
                continue

            # 씬 경계 전환 결정
            cur_seg = cur.get("segment_type", "verse")
            nxt_seg = nxt.get("segment_type", "verse")
            cur_chars = scene_chars.get(cur["parent_scene_id"], set())
            nxt_chars = scene_chars.get(nxt["parent_scene_id"], set())
            shared_chars = cur_chars & nxt_chars
            nxt_vc = vc_map.get(nxt["parent_scene_id"], "")

            # palette 비교
            cur_palette = palette_cache.get(cur.get("image_path"), {"avgY": 128.0, "avgS": 0.5})
            nxt_palette = palette_cache.get(nxt.get("image_path"), {"avgY": 128.0, "avgS": 0.5})
            delta_y = abs(cur_palette["avgY"] - nxt_palette["avgY"])
            delta_s = abs(cur_palette["avgS"] - nxt_palette["avgS"])

            # shot role 판별 (hero = 캐릭터 AI 이미지, broll = 스톡/배경)
            cur_is_hero = cur.get("has_characters") and cur.get("source") == "ai"
            nxt_is_hero = nxt.get("has_characters") and nxt.get("source") == "ai"
            cur_is_broll = cur.get("source") == "stock" or not cur.get("has_characters")
            nxt_is_broll = nxt.get("source") == "stock" or not nxt.get("has_characters")

            transition_type = "xfade"  # default (Rule 9)
            custom_frames = None  # None이면 기본 duration 사용
            reason = "R9:default_xfade"

            # Rule 2: chorus/hook 진입 -> whiteflash
            if nxt_seg in ("chorus", "hook") and cur_seg not in ("chorus", "hook"):
                transition_type = "whiteflash"
                reason = f"R2:chorus_entry({cur_seg}->{nxt_seg})"
            # Rule 3: contrast_cut continuity -> fadeblack
            elif nxt_vc == "contrast_cut":
                transition_type = "fadeblack"
                reason = "R3:contrast_cut"
            # Rule 4: palette 충돌 -> fadeblack
            elif delta_y > 60 or delta_s > 0.3:
                transition_type = "fadeblack"
                reason = f"R4:palette_clash(dY={delta_y:.0f},dS={delta_s:.2f})"
            # Rule 5: 같은 캐릭터 연속 -> cut
            elif shared_chars and cur_seg == nxt_seg:
                transition_type = "cut"
                reason = f"R5:same_char({list(shared_chars)[:2]})"
            # xfade 제한: hero↔hero -> xfade 금지 (얼굴 겹침/ghosting)
            elif cur_is_hero and nxt_is_hero:
                transition_type = "fadeblack" if delta_y > 30 else "cut"
                reason = f"R-hero2hero(dY={delta_y:.0f})"
            # xfade 제한: hero↔broll -> fadeblack/filmburn 우선
            elif (cur_is_hero and nxt_is_broll) or (cur_is_broll and nxt_is_hero):
                if filmburn_used < filmburn_budget:
                    transition_type = "filmburn"
                    filmburn_used += 1
                    reason = "R-hero_broll:filmburn"
                else:
                    transition_type = "fadeblack"
                    reason = "R-hero_broll:fadeblack(budget)"
            # xfade 제한: broll↔broll -> short xfade 4-6 frames만
            elif cur_is_broll and nxt_is_broll:
                transition_type = "xfade"
                custom_frames = 5
                reason = "R-broll2broll:short_xfade"
            # Rule 7: chorus->chorus + glitch 예산 -> glitch
            elif cur_seg in ("chorus", "hook") and nxt_seg in ("chorus", "hook") and glitch_used < glitch_budget:
                transition_type = "glitch"
                glitch_used += 1
                reason = f"R7:chorus_glitch({glitch_used}/{glitch_budget})"
            # Rule 8: 완전 다른 캐릭터 + 다른 세그먼트 + filmburn 예산
            elif not shared_chars and cur_seg != nxt_seg and filmburn_used < filmburn_budget:
                transition_type = "filmburn"
                filmburn_used += 1
                reason = f"R8:diff_chars_seg({cur_seg}->{nxt_seg})"
            # Rule 9: default xfade (already set)

            duration_sec = self._TRANSITION_DURATIONS.get(transition_type, 0.3)
            t_frames = custom_frames if custom_frames is not None else int(duration_sec * fps)
            transitions.append({
                "transition_type": transition_type,
                "transition_frames": t_frames,
                "overlay_asset_path": None,
                "reason": reason,
            })

        # 로그
        type_counts = {}
        for t in transitions:
            tt = t["transition_type"]
            type_counts[tt] = type_counts.get(tt, 0) + 1
        summary = ", ".join(f"{k}:{v}" for k, v in sorted(type_counts.items()))
        print(f"  [TRANSITION PLAN] {len(transitions)} transitions: {summary}")

        return transitions

    def _extract_thumbnail(self, video_path: str, out_path: str):
        """B-roll 영상 첫 프레임을 썸네일로 추출"""
        import subprocess
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        cmd = ["ffmpeg", "-y", "-i", video_path, "-vframes", "1",
               "-vf", "scale=1920:1080", out_path]
        subprocess.run(cmd, capture_output=True, timeout=30)

    def _normalize_broll(self, video_path: str, target_duration: float, out_path: str) -> Optional[str]:
        """B-roll 영상을 Ken Burns 클립과 동일한 스펙으로 정규화

        - 해상도/FPS 통일 (ffmpeg_composer 기준)
        - 오디오 트랙 제거 (음악과 충돌 방지)
        - 너무 길면 target_duration으로 트림
        """
        import subprocess
        try:
            w = self.ffmpeg_composer.width
            h = self.ffmpeg_composer.height
            fps = self.ffmpeg_composer.fps
            cmd = [
                "ffmpeg", "-y", "-i", video_path,
                "-t", str(target_duration),
                "-an",  # 오디오 제거
                "-vf", f"scale={w}:{h}:force_original_aspect_ratio=decrease,"
                       f"pad={w}:{h}:(ow-iw)/2:(oh-ih)/2",
                "-r", str(fps),
                "-c:v", "libx264", "-preset", "ultrafast", "-crf", "23",
                "-pix_fmt", "yuv420p",
                out_path,
            ]
            result = subprocess.run(cmd, capture_output=True, timeout=60)
            if result.returncode == 0 and os.path.exists(out_path):
                return out_path
            return None
        except Exception:
            return None

    # ================================================================
    # Step 4: 영상 합성
    # ================================================================

    def compose_video(self, project: MVProject) -> MVProject:
        """
        이미지 + 음악 → 최종 뮤직비디오 합성

        Args:
            project: MVProject 객체

        Returns:
            업데이트된 MVProject
        """
        print(f"\n[Step 4] Composing final video...")

        project_dir = f"{self.output_base_dir}/{project.project_id}"
        project.status = MVProjectStatus.COMPOSING
        project.current_step = "영상 합성 중..."
        project.progress = 75
        self._save_manifest(project, project_dir)

        # 완료된 씬만 수집
        completed_scenes = [s for s in project.scenes if s.status == MVSceneStatus.COMPLETED and s.image_path]

        if not completed_scenes:
            project.status = MVProjectStatus.FAILED
            project.error_message = "No completed scenes to compose"
            return project

        print(f"  Completed scenes: {len(completed_scenes)}/{len(project.scenes)}")

        try:
            # 1. 파생 컷 플래닝 (세그먼트 인식) + 전환 플래닝
            cut_plan = self._generate_cut_plan(completed_scenes, project=project)
            transition_plan = self._plan_transitions(cut_plan, completed_scenes, project)

            # ── Phase 3: cuts.json + render.log 저장 ──
            self._save_cuts_and_render_log(project, project_dir, cut_plan, transition_plan, completed_scenes)

            video_clips = []
            # 씬 그룹 추적 (크로스페이드용): [[scene1_clips], [scene2_clips], ...]
            scene_groups = []
            current_scene_id = None

            for ci, cut in enumerate(cut_plan):
                clip_path = f"{project_dir}/media/video/cut_{cut['parent_scene_id']:02d}_{cut['cut_index']:02d}.mp4"

                # 씬 전환 감지 → 새 그룹 시작
                if cut['parent_scene_id'] != current_scene_id:
                    scene_groups.append([])
                    current_scene_id = cut['parent_scene_id']

                # I2V / B-roll 씬은 이미 비디오가 있음
                if cut.get("video_path"):
                    if cut.get("is_broll"):
                        # B-roll만 정규화 (해상도/FPS/코덱 통일 + 오디오 제거 + 트림)
                        norm_path = f"{project_dir}/media/video/norm_{cut['parent_scene_id']:02d}.mp4"
                        normalized = self._normalize_broll(cut["video_path"], cut["duration_sec"], norm_path)
                        resolved = normalized or cut["video_path"]
                    else:
                        # I2V: 이미 올바른 포맷이므로 그대로 사용
                        resolved = cut["video_path"]
                    video_clips.append(resolved)
                    scene_groups[-1].append(resolved)
                    print(f"    Cut {ci+1}/{len(cut_plan)}: scene {cut['parent_scene_id']} ({'B-roll' if cut.get('is_broll') else 'I2V'})")
                    continue

                if not cut.get("image_path"):
                    continue

                try:
                    self.ffmpeg_composer.derived_cut_clip(
                        image_path=cut["image_path"],
                        duration_sec=cut["duration_sec"],
                        out_path=clip_path,
                        reframe=cut["reframe"],
                        crop_anchor=cut["crop_anchor"],
                        effect_type=cut["effect_type"],
                        zoom_range=cut["zoom_range"],
                    )
                    video_clips.append(clip_path)
                    scene_groups[-1].append(clip_path)
                except Exception as cut_err:
                    print(f"    [WARNING] Derived cut failed (scene {cut['parent_scene_id']}, cut {cut['cut_index']}): {str(cut_err)[-200:]}")
                    # Fallback: 정적 이미지
                    try:
                        self.ffmpeg_composer._image_to_static_video(
                            image_path=cut["image_path"],
                            duration_sec=cut["duration_sec"],
                            output_path=clip_path
                        )
                        video_clips.append(clip_path)
                        scene_groups[-1].append(clip_path)
                    except Exception:
                        continue

                # 진행률: 컷 기준
                clip_progress = 75 + int((ci + 1) / len(cut_plan) * 10)
                project.progress = max(project.progress, clip_progress)

            project.progress = 85
            project.current_step = "영상 클립 생성 완료, 이어붙이는 중..."
            self._save_manifest(project, project_dir)

            # 비디오 클립이 없으면 실패
            if not video_clips:
                project.status = MVProjectStatus.FAILED
                project.error_message = "No video clips were created. Check image generation."
                return project

            print(f"  Video clips created: {len(video_clips)}")

            # 1.5. 씬 경계 밝기 점프 완화 (C-3: 3-6프레임 밝기 램프)
            scene_groups_clean = [g for g in scene_groups if g]
            if len(scene_groups_clean) >= 2:
                # 씬별 대표 이미지의 palette stats 수집
                scene_image_map = {}
                for scene in completed_scenes:
                    if scene.image_path and os.path.exists(scene.image_path):
                        scene_image_map[scene.scene_id] = scene.image_path

                scene_ids_ordered = []
                for scene in completed_scenes:
                    if scene.scene_id not in scene_ids_ordered:
                        scene_ids_ordered.append(scene.scene_id)

                ramp_count = 0
                for si in range(len(scene_ids_ordered) - 1):
                    prev_sid = scene_ids_ordered[si]
                    next_sid = scene_ids_ordered[si + 1]
                    prev_img = scene_image_map.get(prev_sid)
                    next_img = scene_image_map.get(next_sid)
                    if not prev_img or not next_img:
                        continue

                    prev_stats = self._get_palette_stats(prev_img)
                    next_stats = self._get_palette_stats(next_img)
                    delta_y = prev_stats["avgY"] - next_stats["avgY"]

                    # |deltaY| > 20이면 밝기 램프 적용 (과보정 방지: 50%만 보정)
                    if abs(delta_y) > 20:
                        offset = (delta_y / 255.0) * 0.5
                        # scene_groups에서 next scene의 첫 클립 찾기
                        if si + 1 < len(scene_groups_clean) and scene_groups_clean[si + 1]:
                            first_clip = scene_groups_clean[si + 1][0]
                            ramp_out = f"{project_dir}/media/video/ramp_{next_sid:02d}.mp4"
                            result = self.ffmpeg_composer.apply_brightness_ramp(
                                first_clip, ramp_out, brightness_offset=offset, ramp_frames=6
                            )
                            if result:
                                scene_groups_clean[si + 1][0] = ramp_out
                                ramp_count += 1

                if ramp_count > 0:
                    print(f"  [BRIGHTNESS] Applied {ramp_count} brightness ramps at scene boundaries")
                scene_groups = scene_groups_clean

            # 2. 클립들 이어붙이기 (전환 플랜 기반)
            concat_video = f"{project_dir}/media/video/concat.mp4"
            # 빈 그룹 제거
            scene_groups = [g for g in scene_groups if g]
            total_dur = sum(s.duration_sec for s in completed_scenes)
            print(f"  Concatenating {len(video_clips)} clips in {len(scene_groups)} scenes (total ~{total_dur:.0f}s)...")
            project.current_step = f"영상 {len(scene_groups)}개 씬 전환 합성 중..."
            self._save_manifest(project, project_dir)

            # 씬 경계 전환만 추출 (within-scene "cut" 제거)
            boundary_transitions = []
            if transition_plan:
                for ti, t in enumerate(transition_plan):
                    if ti < len(cut_plan) - 1:
                        if cut_plan[ti]["parent_scene_id"] != cut_plan[ti + 1]["parent_scene_id"]:
                            boundary_transitions.append(t)

            if len(scene_groups) >= 2:
                concat_result = self.ffmpeg_composer.concatenate_with_crossfade(
                    scene_groups=scene_groups,
                    output_path=concat_video,
                    fade_duration=0.3,
                    transition_plan=boundary_transitions if boundary_transitions else None,
                )
            else:
                # 씬이 1개면 크로스페이드 불필요
                concat_result = self.ffmpeg_composer.concatenate_videos(
                    video_paths=video_clips,
                    output_path=concat_video
                )

            if not concat_result or not os.path.exists(concat_video):
                raise RuntimeError("Video concatenation failed - output file not created")

            print(f"  Concatenation complete: {concat_video}")
            project.progress = 90
            project.current_step = "자막 처리 중..."
            self._save_manifest(project, project_dir)

            # 3. 가사 자막 생성 (STT 정렬 -> 균등 분배 fallback)
            user_lyrics_text = project.lyrics or ""
            subtitle_on = getattr(project, 'subtitle_enabled', True)
            has_lyrics = bool(user_lyrics_text.strip()) and subtitle_on

            srt_path = None
            n_lines = 0
            audio_duration = 0.0
            anchor_info = None
            timeline_mode = "none"

            if has_lyrics:
                print(f"  Generating lyrics subtitle from user input ({len(user_lyrics_text)} chars)...")
                os.makedirs(f"{project_dir}/media/subtitles", exist_ok=True)

                audio_duration = ffprobe_duration_sec(project.music_file_path)

                # 앵커 추정 (사용자 보정 > 세그먼트 > VAD > fallback)
                segments = None
                if project.music_analysis and project.music_analysis.segments:
                    segments = project.music_analysis.segments
                anchor_info = detect_anchors(
                    media_path=project.music_file_path,
                    segments=segments,
                    user_anchor_start=getattr(project, 'subtitle_anchor_start', None),
                    user_anchor_end=getattr(project, 'subtitle_anchor_end', None),
                )
                print(f"    Anchor: [{anchor_info.anchor_start:.1f}s ~ {anchor_info.anchor_end:.1f}s] method={anchor_info.method}")

                lines = split_lyrics_lines(user_lyrics_text)
                n_lines = len(lines)

                # STT 정렬 시도 (lazy: compose 시점에 Gemini Audio STT 호출)
                stt_segments = getattr(project, 'stt_segments', None)
                if not stt_segments:
                    print(f"    [Gemini-STT] Running audio STT for subtitle alignment...")
                    stt_segments = self.music_analyzer.transcribe_with_gemini_audio(
                        project.music_file_path
                    )
                    if stt_segments:
                        project.stt_segments = stt_segments
                        print(f"    [Gemini-STT] Got {len(stt_segments)} segments")
                    else:
                        print(f"    [Gemini-STT] Failed, will use uniform distribution")

                if stt_segments:
                    print(f"    [STT-Align] Using {len(stt_segments)} STT segments for alignment...")
                    aligned = align_lyrics_with_stt(
                        lines, stt_segments,
                        anchor_info.anchor_start, anchor_info.anchor_end,
                    )
                    timeline_mode = "stt_aligned"
                    avg_conf = sum(a.confidence for a in aligned) / len(aligned) if aligned else 0
                    print(f"    [STT-Align] {len(aligned)} lines aligned, avg confidence={avg_conf:.1f}")
                else:
                    print(f"    [Fallback] No STT segments, using uniform distribution...")
                    timeline = clamp_timeline_anchored(
                        lines, anchor_info.anchor_start, anchor_info.anchor_end
                    )
                    aligned = [AlignedSubtitle(s.start, s.end, s.text, 0.0) for s in timeline]
                    timeline_mode = "uniform_fallback"

                # ASS 출력 (pysubs2) -> SRT fallback
                try:
                    srt_path = f"{project_dir}/media/subtitles/lyrics.ass"
                    write_ass(aligned, srt_path)
                    print(f"    ASS: {n_lines} lines -> {srt_path}")
                except Exception as ass_err:
                    print(f"    [ASS Error] {ass_err}, falling back to SRT...")
                    srt_path = f"{project_dir}/media/subtitles/lyrics.srt"
                    srt_timeline = [SubtitleLine(a.start, a.end, a.text) for a in aligned]
                    write_srt(srt_timeline, srt_path)
                    timeline_mode += "+srt_fallback"
                    print(f"    SRT: {n_lines} lines -> {srt_path}")

                # alignment.json 저장 (디버깅용)
                project.aligned_lyrics = [
                    {"t": a.start, "end": a.end, "text": a.text, "confidence": a.confidence}
                    for a in aligned
                ]
                alignment_path = f"{project_dir}/media/subtitles/alignment.json"
                with open(alignment_path, "w", encoding="utf-8") as af:
                    json.dump(project.aligned_lyrics, af, ensure_ascii=False, indent=2)
            else:
                print(f"  [Lyrics Check] No user lyrics or subtitle disabled (subtitle_enabled={subtitle_on})")

            project.progress = 95
            project.current_step = "최종 렌더링 중..."
            self._save_manifest(project, project_dir)

            # 4. 최종 렌더링 (자막 + 음악 + 워터마크 통합 FFmpeg 패스)
            final_video = f"{project_dir}/final_mv.mp4"
            watermark = "Made with Klippa" if getattr(project, 'watermark_enabled', True) else None
            self._final_render(
                video_in=concat_video,
                audio_path=project.music_file_path,
                out_path=final_video,
                srt_path=srt_path,
                watermark_text=watermark,
            )

            # render.log 기록
            render_log_path = f"{project_dir}/render.log"
            with open(render_log_path, "a", encoding="utf-8") as log:
                log.write(f"--- compose_video {datetime.now().isoformat()} ---\n")
                log.write(f"lyrics_source=user_input_only\n")
                log.write(f"timeline_mode={timeline_mode}\n")
                log.write(f"N_lines={n_lines}\n")
                log.write(f"audio_duration={audio_duration:.2f}\n")
                if anchor_info:
                    log.write(f"anchor_start={anchor_info.anchor_start:.2f}\n")
                    log.write(f"anchor_end={anchor_info.anchor_end:.2f}\n")
                    log.write(f"anchor_method={anchor_info.method}\n")
                log.write(f"subtitle_path={srt_path}\n")
                log.write(f"\n")

            project.final_video_path = final_video
            project.status = MVProjectStatus.COMPLETED
            project.progress = 100
            project.current_step = "완료!"

            print(f"\n  Final video: {final_video}")

        except Exception as e:
            project.status = MVProjectStatus.FAILED
            project.error_message = f"Video composition failed: {str(e)}"
            print(f"  [ERROR] {project.error_message}")

        # 최종 매니페스트 저장
        project.updated_at = datetime.now()
        self._save_manifest(project, project_dir)

        return project

    # ================================================================
    # 자막 테스트 (이미지 생성 없이 음악 + 자막 프리뷰)
    # ================================================================

    def subtitle_test(self, project: MVProject) -> MVProject:
        """
        이미지 생성 없이 음악 + 자막만 프리뷰하는 테스트 모드.
        프로덕션 _final_render와 동일한 ASS(pysubs2) + ass= 필터 사용.
        """
        import subprocess

        project_dir = f"{self.output_base_dir}/{project.project_id}"
        os.makedirs(f"{project_dir}/media/subtitles", exist_ok=True)

        user_lyrics_text = project.lyrics or ""
        if not user_lyrics_text.strip():
            project.error_message = "No lyrics to test"
            return project

        # 디버그: 원본 가사 데이터 확인 (줄바꿈 포함 여부)
        print(f"  [SubTest] Raw lyrics length: {len(user_lyrics_text)} chars")
        print(f"  [SubTest] Raw lyrics newlines: {user_lyrics_text.count(chr(10))} LF, {user_lyrics_text.count(chr(13))} CR")
        print(f"  [SubTest] Raw lyrics preview (repr): {repr(user_lyrics_text[:300])}")

        project.current_step = "자막 테스트 생성 중..."
        self._save_manifest(project, project_dir)

        try:
            # ── Step 1: STT (캐시 또는 실행) ──
            stt_segments = getattr(project, 'stt_segments', None)
            if stt_segments:
                print(f"  [SubTest] Using cached STT ({len(stt_segments)} segments)")
            else:
                print(f"  [SubTest] Running Gemini Audio STT...")
                project.current_step = "음성 인식 중..."
                self._save_manifest(project, project_dir)
                stt_segments = self.music_analyzer.transcribe_with_gemini_audio(
                    project.music_file_path
                )
                if stt_segments:
                    project.stt_segments = stt_segments
                    print(f"  [SubTest] STT: {len(stt_segments)} segments")
                else:
                    print(f"  [SubTest] STT failed, using approximate timing")

            # ── Step 2: 앵커 추정 ──
            segments = None
            if project.music_analysis and project.music_analysis.segments:
                segments = project.music_analysis.segments
            anchor_info = detect_anchors(
                media_path=project.music_file_path,
                segments=segments,
                user_anchor_start=getattr(project, 'subtitle_anchor_start', None),
                user_anchor_end=getattr(project, 'subtitle_anchor_end', None),
            )
            print(f"  [SubTest] Anchor: [{anchor_info.anchor_start:.1f}s ~ {anchor_info.anchor_end:.1f}s]")

            # ── Step 3: 가사 정렬 ──
            lines = split_lyrics_lines(user_lyrics_text)
            print(f"  [SubTest] Lyrics: {len(lines)} lines (after filtering)")
            if not lines:
                project.error_message = "No lyrics lines after filtering"
                self._save_manifest(project, project_dir)
                return project

            use_uniform = False

            if stt_segments:
                aligned = align_lyrics_with_stt(
                    lines, stt_segments,
                    anchor_info.anchor_start, anchor_info.anchor_end,
                )
                avg_conf = sum(a.confidence for a in aligned) / len(aligned) if aligned else 0
                matched = sum(1 for a in aligned if a.confidence > 0)
                print(f"  [SubTest] STT alignment: {len(aligned)} lines, matched={matched}, avg_conf={avg_conf:.0f}")

                # STT 매칭이 나쁘면 균등 분배로 폴백
                if avg_conf < 40 or matched < len(aligned) * 0.3:
                    print(f"  [SubTest] Poor STT match (conf={avg_conf:.0f}, matched={matched}/{len(aligned)}) -> uniform fallback")
                    use_uniform = True
            else:
                use_uniform = True

            if use_uniform:
                timeline = clamp_timeline_anchored(
                    lines, anchor_info.anchor_start, anchor_info.anchor_end
                )
                aligned = [AlignedSubtitle(s.start, s.end, s.text, 0.0) for s in timeline]
                print(f"  [SubTest] Uniform distribution: {len(aligned)} lines over {anchor_info.anchor_start:.0f}s-{anchor_info.anchor_end:.0f}s")

            # 디버그: 전체 정렬 결과 출력
            for ai, a in enumerate(aligned):
                print(f"    [{ai:2d}] {a.start:6.1f}s - {a.end:6.1f}s  conf={a.confidence:3.0f}  '{a.text[:50]}'")

            # alignment.json 저장
            project.aligned_lyrics = [
                {"t": a.start, "end": a.end, "text": a.text, "confidence": a.confidence}
                for a in aligned
            ]
            alignment_path = f"{project_dir}/media/subtitles/alignment.json"
            with open(alignment_path, "w", encoding="utf-8") as af:
                json.dump(project.aligned_lyrics, af, ensure_ascii=False, indent=2)

            # ── Step 4: ASS 자막 파일 생성 (compose_video와 동일 방식) ──
            ass_path = f"{project_dir}/media/subtitles/lyrics_test.ass"
            write_ass(aligned, ass_path)
            ass_size = os.path.getsize(ass_path) if os.path.exists(ass_path) else 0
            print(f"  [SubTest] ASS file: {len(aligned)} events, {ass_size:,} bytes -> {ass_path}")

            # ASS 내용 미리보기 (디버그)
            try:
                with open(ass_path, 'r', encoding='utf-8-sig') as af:
                    ass_content = af.read()
                    dialogue_count = ass_content.count("Dialogue:")
                    print(f"  [SubTest] ASS Dialogue count: {dialogue_count}")
                    # 마지막 5줄 출력
                    ass_lines = ass_content.strip().split('\n')
                    for line in ass_lines[-5:]:
                        print(f"    ASS> {line[:120]}")
            except Exception as e:
                print(f"  [SubTest] ASS read error: {e}")

            # ── Step 5: 단일 패스 렌더링 (프로덕션 _final_render와 동일한 ass= 필터) ──
            audio_duration = ffprobe_duration_sec(project.music_file_path)
            audio_abs = os.path.abspath(project.music_file_path)
            out_path = f"{project_dir}/final_mv_subtitle_test.mp4"
            out_abs = os.path.abspath(out_path)

            ass_abs = os.path.abspath(ass_path)
            ass_escaped = ass_abs.replace("\\", "/").replace(":", "\\:")

            project.current_step = "자막 영상 렌더링 중..."
            self._save_manifest(project, project_dir)

            timeout = max(120, int(audio_duration * 3))

            # 프로덕션 _final_render와 동일: ASS 파일은 ass= 필터 사용 (force_style 없음)
            # subtitles= + force_style은 ASS 내장 스타일을 덮어써서 렌더링 오류 발생
            cmd = [
                "ffmpeg", "-y",
                "-f", "lavfi", "-i", f"color=c=black:s=1280x720:d={audio_duration}:r=24",
                "-i", audio_abs,
                "-filter_complex",
                f"[0:v]ass='{ass_escaped}'[v];[1:a]aresample=44100[a]",
                "-map", "[v]", "-map", "[a]",
                "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p",
                "-c:a", "aac", "-ar", "44100", "-b:a", "128k",
                "-shortest",
                out_abs,
            ]
            print(f"  [SubTest] Single-pass render ({audio_duration:.0f}s, ass= filter)...")
            print(f"  [SubTest] ASS: {ass_abs}")
            result = subprocess.run(cmd, capture_output=True, text=True,
                                    encoding='utf-8', errors='replace', timeout=timeout)

            if result.returncode == 0 and os.path.exists(out_abs) and os.path.getsize(out_abs) > 1024:
                print(f"  [SubTest] Render OK ({os.path.getsize(out_abs):,} bytes)")
                project.current_step = "자막 테스트 완료"
            else:
                stderr_tail = result.stderr[-500:] if result.stderr else "(empty)"
                print(f"  [SubTest] ass= filter FAILED (rc={result.returncode}): {stderr_tail}")

                # Fallback: SRT + subtitles= 필터로 재시도
                print(f"  [SubTest] Fallback: SRT + subtitles= filter...")
                srt_fallback = f"{project_dir}/media/subtitles/lyrics_test.srt"
                srt_timeline = [SubtitleLine(a.start, a.end, a.text) for a in aligned]
                write_srt(srt_timeline, srt_fallback)
                srt_abs = os.path.abspath(srt_fallback)
                srt_escaped = srt_abs.replace("\\", "/").replace(":", "\\:")

                cmd2 = [
                    "ffmpeg", "-y",
                    "-f", "lavfi", "-i", f"color=c=black:s=1280x720:d={audio_duration}:r=24",
                    "-i", audio_abs,
                    "-filter_complex",
                    f"[0:v]subtitles='{srt_escaped}'[v];[1:a]aresample=44100[a]",
                    "-map", "[v]", "-map", "[a]",
                    "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p",
                    "-c:a", "aac", "-ar", "44100", "-b:a", "128k",
                    "-shortest",
                    out_abs,
                ]
                r2 = subprocess.run(cmd2, capture_output=True, text=True,
                                    encoding='utf-8', errors='replace', timeout=timeout)
                if r2.returncode == 0 and os.path.exists(out_abs) and os.path.getsize(out_abs) > 1024:
                    print(f"  [SubTest] SRT fallback OK ({os.path.getsize(out_abs):,} bytes)")
                    project.current_step = "자막 테스트 완료"
                else:
                    print(f"  [SubTest] SRT fallback also FAILED: {r2.stderr[-300:] if r2.stderr else '(empty)'}")
                    # 최종 fallback: 자막 없이 오디오만
                    cmd3 = [
                        "ffmpeg", "-y",
                        "-f", "lavfi", "-i", f"color=c=black:s=1280x720:d={audio_duration}:r=24",
                        "-i", audio_abs,
                        "-map", "0:v", "-map", "1:a",
                        "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p",
                        "-c:a", "aac", "-ar", "44100", "-b:a", "128k",
                        "-shortest", out_abs,
                    ]
                    subprocess.run(cmd3, capture_output=True, text=True,
                                   encoding='utf-8', errors='replace', timeout=timeout)
                    project.current_step = "자막 테스트 완료"
                    project.error_message = "Subtitle rendering failed, audio-only output"

        except Exception as e:
            project.error_message = f"Subtitle test failed: {str(e)}"
            print(f"  [SubTest ERROR] {project.error_message}")

        self._save_manifest(project, project_dir)
        return project

    # ================================================================
    # 전체 파이프라인 실행
    # ================================================================

    def run(
        self,
        music_file_path: str,
        request: MVProjectRequest,
        on_progress: Optional[Callable] = None
    ) -> MVProject:
        """
        전체 MV 생성 파이프라인 실행

        Args:
            music_file_path: 음악 파일 경로
            request: MVProjectRequest
            on_progress: 진행 콜백 (project)

        Returns:
            완료된 MVProject
        """
        start_time = time.time()

        # Step 1: 업로드 & 분석
        project = self.upload_and_analyze(music_file_path, request.project_id)
        if project.status == MVProjectStatus.FAILED:
            return project

        if on_progress:
            on_progress(project)

        # Step 2: 씬 골격 생성
        project = self.generate_scenes(project, request)
        if on_progress:
            on_progress(project)

        # Step 2.1: 가사 서사 분석
        project = self.analyze_story(project)

        # Visual Bible + Style Anchor + Character Anchors
        project = self.generate_visual_bible(project)
        project = self.generate_style_anchor(project)
        project = self.generate_character_anchors(project)

        # Step 2.8: 씬 프롬프트 생성 (Visual Bible 이후)
        project = self.generate_scene_prompts(project, request)

        # Step 3: 이미지 생성
        def _on_scene_image(scene, idx, total):
            if on_progress:
                on_progress(project)

        project = self.generate_images(project, on_scene_complete=_on_scene_image)
        if on_progress:
            on_progress(project)

        # Step 4: 이미지 리뷰 대기 (compose_video는 별도 호출)
        # generate_images()에서 이미 IMAGES_READY로 설정됨

        elapsed = time.time() - start_time
        print(f"\n{'='*60}")
        print(f"[MV Pipeline] Images ready in {elapsed:.1f}s")
        print(f"  Status: {project.status}")
        print(f"  Waiting for user review before composing...")
        print(f"{'='*60}\n")

        return project

    # ================================================================
    # Step 3.5: 씬 이미지 재생성 / I2V 변환
    # ================================================================

    def regenerate_scene_image(self, project: MVProject, scene_id: int) -> MVScene:
        """
        특정 씬의 이미지를 재생성

        Args:
            project: MVProject 객체
            scene_id: 씬 ID (1-based)

        Returns:
            업데이트된 MVScene
        """
        if scene_id < 1 or scene_id > len(project.scenes):
            raise ValueError(f"Invalid scene_id: {scene_id}")

        scene = project.scenes[scene_id - 1]
        project_dir = f"{self.output_base_dir}/{project.project_id}"
        image_dir = f"{project_dir}/media/images"

        # B-roll 씬은 재생성 불가 (스톡 영상 보호)
        if getattr(scene, 'is_broll', False):
            print(f"[MV Regenerate] Scene {scene_id} is B-roll, skipping regeneration")
            return scene

        print(f"[MV Regenerate] Scene {scene_id} image regeneration...")

        # v3.0: 전용 스타일 앵커 우선 사용, 없으면 scene_01 fallback
        style_anchor_path = project.style_anchor_path
        if style_anchor_path and not os.path.exists(style_anchor_path):
            style_anchor_path = None
        if not style_anchor_path and scene_id > 1:
            first_scene = project.scenes[0]
            if first_scene.image_path and os.path.exists(first_scene.image_path):
                style_anchor_path = first_scene.image_path

        # Visual Bible dict
        vb_dict = project.visual_bible.model_dump() if project.visual_bible else None

        scene.status = MVSceneStatus.GENERATING

        # 캐릭터 앵커 이미지 조회
        char_anchor_paths = self._get_character_anchors_for_scene(project, scene)

        # 캐릭터 외형+의상 설명 주입 (Visual Bible 기준)
        final_prompt = self._inject_character_descriptions(project, scene, scene.image_prompt)

        # 인종 키워드 자동 주입
        _eth = getattr(project, 'character_ethnicity', None)
        _eth_v = _eth.value if hasattr(_eth, 'value') else str(_eth or 'auto')
        _ETH_KW = {
            "korean": "Korean", "japanese": "Japanese", "chinese": "Chinese",
            "southeast_asian": "Southeast Asian", "european": "European",
            "black": "Black", "hispanic": "Hispanic",
        }
        ethnicity_keyword = _ETH_KW.get(_eth_v, "")
        if ethnicity_keyword and ethnicity_keyword.lower() not in final_prompt.lower():
            final_prompt = f"{ethnicity_keyword} characters, {final_prompt}"

        # 캐릭터 미등장 씬: 정의 안 된 인물 등장 방지
        _regen_neg = scene.negative_prompt or ""
        if not scene.characters_in_scene:
            _no_people = "random person, unnamed person, elderly man, old man, bystander, stranger, human figure"
            _regen_neg = f"{_no_people}, {_regen_neg}" if _regen_neg else _no_people

        # 시대/배경 키워드 자동 주입 (concept에서 추출)
        era_prefix, era_negative = self._extract_era_setting(project.concept)
        if era_prefix and era_prefix.lower() not in final_prompt.lower():
            final_prompt = f"{era_prefix}, {final_prompt}"
        if era_negative:
            _regen_neg = f"{era_negative}, {_regen_neg}" if _regen_neg else era_negative

        try:
            image_path, _ = self.image_agent.generate_image(
                scene_id=scene.scene_id,
                prompt=final_prompt,
                style=project.style.value,
                output_dir=image_dir,
                style_anchor_path=style_anchor_path,
                genre=project.genre.value,
                mood=project.mood.value,
                negative_prompt=_regen_neg or scene.negative_prompt,
                visual_bible=vb_dict,
                color_mood=scene.color_mood,
                character_reference_paths=char_anchor_paths or None,
                camera_directive=scene.camera_directive,
            )
            scene.image_path = image_path
            scene.video_path = None  # I2V 결과 초기화 (이미지 변경됨)
            scene.status = MVSceneStatus.COMPLETED
            print(f"  Regenerated: {image_path}")
        except Exception as e:
            scene.status = MVSceneStatus.FAILED
            print(f"  [ERROR] Regeneration failed: {e}")
            raise

        self._save_manifest(project, project_dir)
        return scene

    def generate_scene_i2v(self, project: MVProject, scene_id: int) -> MVScene:
        """
        특정 씬의 이미지를 Veo I2V로 비디오 변환

        Args:
            project: MVProject 객체
            scene_id: 씬 ID (1-based)

        Returns:
            업데이트된 MVScene
        """
        if scene_id < 1 or scene_id > len(project.scenes):
            raise ValueError(f"Invalid scene_id: {scene_id}")

        scene = project.scenes[scene_id - 1]
        project_dir = f"{self.output_base_dir}/{project.project_id}"

        if not scene.image_path or not os.path.exists(scene.image_path):
            raise ValueError(f"Scene {scene_id} has no image")

        print(f"[MV I2V] Scene {scene_id} video generation...")

        from agents.video_agent import VideoAgent
        video_agent = VideoAgent()

        video_dir = f"{project_dir}/media/video"
        os.makedirs(video_dir, exist_ok=True)
        video_path = f"{video_dir}/scene_{scene_id:02d}_i2v.mp4"

        try:
            result_path = video_agent._call_veo_api(
                prompt=scene.image_prompt,
                style=project.style.value,
                mood=project.mood.value,
                duration_sec=min(int(scene.duration_sec), 8),  # Veo max ~8s
                output_path=video_path,
                first_frame_image=scene.image_path
            )
            scene.video_path = result_path or video_path
            print(f"  I2V complete: {scene.video_path}")
        except Exception as e:
            print(f"  [ERROR] I2V failed: {e}")
            raise

        self._save_manifest(project, project_dir)
        return scene

    # ================================================================
    # 유틸리티
    # ================================================================

    def _final_render(
        self,
        video_in: str,
        audio_path: str,
        out_path: str,
        srt_path: Optional[str] = None,
        watermark_text: Optional[str] = None,
        watermark_opacity: float = 0.3,
    ):
        """
        통합 최종 렌더링: 자막 burn-in + 음악 합성 + 워터마크를 단일 FFmpeg 패스로 처리.

        subtitle_fix.md 명세에 따라:
        - fps=30 고정
        - subtitles= 필터로 SRT burn-in
        - aresample=48000, asetpts=N/SR/TB
        - H.264 High Profile, AAC 192k
        """
        import platform
        import subprocess

        video_abs = os.path.abspath(video_in)
        audio_abs = os.path.abspath(audio_path)
        out_abs = os.path.abspath(out_path)

        # 플랫폼별 폰트
        system = platform.system()
        if system == "Windows":
            font_name = "Malgun Gothic"
        else:
            from agents.subtitle_utils import _detect_linux_font
            font_name = _detect_linux_font()

        # --- 비디오 필터 체인 구성 ---
        vf_parts = ["fps=30"]

        if srt_path and os.path.exists(srt_path):
            sub_abs = os.path.abspath(srt_path)
            sub_escaped = sub_abs.replace("\\", "/").replace(":", "\\:")
            if srt_path.endswith(".ass"):
                # ASS: 스타일이 파일에 내장되어 있으므로 ass= 필터 사용
                vf_parts.append(f"ass='{sub_escaped}'")
            else:
                # SRT: force_style로 폰트/크기 지정
                force_style = (
                    f"FontName={font_name},"
                    f"FontSize=42,"
                    f"Outline=2,"
                    f"Shadow=1,"
                    f"MarginV=60"
                )
                vf_parts.append(f"subtitles='{sub_escaped}':force_style='{force_style}'")

        if watermark_text:
            opacity_hex = f"{watermark_opacity:.2f}"
            vf_parts.append(
                f"drawtext=text='{watermark_text}':"
                f"fontsize=24:fontcolor=white@{opacity_hex}:"
                f"x=w-tw-20:y=20"
            )

        vf_filter = ",".join(vf_parts)

        # --- FFmpeg 커맨드 ---
        cmd = [
            "ffmpeg", "-y",
            "-i", video_abs,
            "-i", audio_abs,
            "-filter_complex",
            f"[0:v]{vf_filter}[v];[1:a]aresample=48000,asetpts=N/SR/TB[a]",
            "-map", "[v]", "-map", "[a]",
            "-c:v", "libx264", "-pix_fmt", "yuv420p",
            "-profile:v", "high", "-level", "4.1", "-r", "30",
            "-c:a", "aac", "-ar", "48000", "-b:a", "192k",
            "-shortest",
            out_abs
        ]

        # 동적 timeout
        try:
            duration = self.ffmpeg_composer.get_video_duration(video_abs)
            timeout = max(300, int(duration * 5))
        except Exception:
            timeout = 600

        print(f"  [Final Render] Combined pass: subtitle={'Yes' if srt_path else 'No'}, watermark={'Yes' if watermark_text else 'No'}")
        print(f"  [Final Render] VF: {vf_filter}")
        print(f"  [Final Render] Timeout: {timeout}s")

        result = subprocess.run(
            cmd, capture_output=True, text=True,
            encoding='utf-8', errors='replace', timeout=timeout
        )

        if result.returncode != 0:
            print(f"  [ERROR] Final render failed (rc={result.returncode})")
            print(f"  [ERROR] stderr: {result.stderr[-500:] if result.stderr else '(empty)'}")
            # 폴백: 자막 없이 기존 final_encode 사용
            print(f"  [FALLBACK] Retrying without subtitles via final_encode...")
            self.ffmpeg_composer.final_encode(
                video_in=video_in,
                audio_path=audio_path,
                out_path=out_path,
                watermark_text=watermark_text,
            )
        else:
            out_size = os.path.getsize(out_abs) if os.path.exists(out_abs) else 0
            if out_size < 1024:
                print(f"  [ERROR] Output too small ({out_size}B), falling back to final_encode")
                self.ffmpeg_composer.final_encode(
                    video_in=video_in,
                    audio_path=audio_path,
                    out_path=out_path,
                    watermark_text=watermark_text,
                )
            else:
                print(f"  [SUCCESS] Final render: {out_abs} ({out_size:,} bytes)")

    def _generate_lyrics_srt(
        self,
        scenes: List[MVScene],
        output_path: str,
        timed_lyrics: Optional[list] = None
    ):
        """
        가사 SRT 자막 파일 생성 (하이브리드 방식)

        - 수정 안 한 씬: timed_lyrics의 해당 시간대 엔트리 그대로 사용 (정밀 싱크)
        - 수정한 씬 (lyrics_modified=True): lyrics_text를 씬 구간 내 균등 분배
        - timed_lyrics 없으면: 전체 씬 기반 균등 분할 (기존 fallback)
        """
        srt_lines = []

        # 수정된 씬이 있는지 확인
        modified_scenes = {s.scene_id for s in scenes if getattr(s, 'lyrics_modified', False)}

        if timed_lyrics and len(timed_lyrics) > 0:
            # 전처리: 비단조 타임스탬프 감지 및 보간
            timed_lyrics = self._fix_broken_timestamps(timed_lyrics, scenes)

            if not modified_scenes:
                # 수정된 씬 없음 → 기존 timed_lyrics 그대로 사용
                print(f"    Using timed lyrics ({len(timed_lyrics)} entries, no modifications)")
                srt_lines = self._srt_from_timed_lyrics(timed_lyrics)
            else:
                # 하이브리드: 수정 안 한 씬은 timed_lyrics, 수정한 씬은 균등 분배
                print(f"    Hybrid SRT: {len(modified_scenes)} modified scene(s), {len(timed_lyrics)} timed entries")
                all_entries = []

                for scene in scenes:
                    if scene.scene_id in modified_scenes:
                        # 수정된 씬: lyrics_text를 씬 구간 내 균등 분배
                        scene_entries = self._lyrics_text_to_timed_entries(scene)
                        if scene_entries:
                            print(f"      Scene {scene.scene_id}: using edited lyrics ({len(scene_entries)} lines, {scene.start_sec:.1f}s~{scene.end_sec:.1f}s)")
                        all_entries.extend(scene_entries)
                    else:
                        # 수정 안 한 씬: timed_lyrics에서 해당 시간대 엔트리 사용
                        scene_entries = [
                            e for e in timed_lyrics
                            if scene.start_sec <= float(e.get("t", 0)) < scene.end_sec
                        ]
                        all_entries.extend(scene_entries)

                # 시간순 정렬 후 중복 제거 + SRT 생성
                all_entries.sort(key=lambda e: float(e.get("t", 0)))
                all_entries = self._deduplicate_merged_entries(all_entries)
                srt_lines = self._srt_from_timed_lyrics(all_entries)
        else:
            # fallback: 씬 기반 균등 분할 (timed_lyrics 없음)
            print(f"    Using scene-based lyrics (no timestamps)")
            idx = 0
            for scene in scenes:
                if not scene.lyrics_text:
                    continue

                # 씬 내 줄 단위 분배
                entries = self._lyrics_text_to_timed_entries(scene)
                for entry in entries:
                    idx += 1
                    start_tc = self._sec_to_srt_timecode(float(entry["t"]))
                    end_sec = float(entry.get("end", float(entry["t"]) + 4))
                    end_tc = self._sec_to_srt_timecode(end_sec)

                    srt_lines.append(str(idx))
                    srt_lines.append(f"{start_tc} --> {end_tc}")
                    srt_lines.append(entry["text"])
                    srt_lines.append("")

        # 파일 저장
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("\n".join(srt_lines))

        entry_count = len([l for l in srt_lines if l.startswith("00:") or l.startswith("01:") or l.startswith("02:")])
        print(f"    SRT generated: {output_path} ({entry_count} entries)")

    @staticmethod
    def _deduplicate_merged_entries(entries: list) -> list:
        """하이브리드 SRT 병합 후 최종 중복 제거 (시간순 정렬된 상태 전제)"""
        if len(entries) <= 1:
            return entries
        result = [entries[0]]
        for entry in entries[1:]:
            prev = result[-1]
            time_diff = abs(float(entry.get("t", 0)) - float(prev.get("t", 0)))
            t_a = prev.get("text", "").strip()
            t_b = entry.get("text", "").strip()
            # 3초 이내 + 텍스트 동일/포함 관계면 중복
            if time_diff < 3.0 and (t_a == t_b or t_a in t_b or t_b in t_a):
                continue
            # 0.5초 미만이면 무조건 중복
            if time_diff < 0.5:
                continue
            result.append(entry)
        return result

    def _srt_from_timed_lyrics(self, timed_lyrics: list) -> list:
        """timed_lyrics 배열을 SRT 라인으로 변환

        end_sec 계산 시 다음 '표시될' 엔트리의 타임스탬프를 사용.
        (빈 텍스트/섹션 마커는 건너뛰어 gap이 생기지 않도록)
        """
        # 1차: 표시할 엔트리만 필터링
        displayed = []
        for entry in timed_lyrics:
            text = entry.get("text", "").strip()
            if not text:
                continue
            if _SECTION_MARKER_RE.match(text):
                continue
            displayed.append(entry)

        # 2차: 필터링된 목록에서 SRT 생성 (end_sec = 다음 표시 엔트리 기준)
        srt_lines = []
        for i, entry in enumerate(displayed):
            text = entry["text"].strip()
            start_sec = float(entry.get("t", 0))

            if i + 1 < len(displayed):
                next_start = float(displayed[i + 1].get("t", start_sec + 4))
                end_sec = min(next_start - 0.05, start_sec + 5)
            else:
                end_sec = start_sec + 4
            end_sec = max(end_sec, start_sec + 1.0)

            # 따옴표 제거 (가사 대사 표현 "..." → 깔끔한 자막)
            text = text.strip('"').strip('\u201c').strip('\u201d')

            srt_lines.append(str(i + 1))
            srt_lines.append(f"{self._sec_to_srt_timecode(start_sec)} --> {self._sec_to_srt_timecode(end_sec)}")
            srt_lines.append(text)
            srt_lines.append("")
        return srt_lines

    def _lyrics_text_to_timed_entries(self, scene) -> list:
        """씬의 lyrics_text를 씬 구간 내 균등 분배하여 timed_lyrics 엔트리로 변환"""
        if not scene.lyrics_text:
            return []

        lines = [l.strip() for l in scene.lyrics_text.strip().split('\n')
                 if l.strip() and not _SECTION_MARKER_RE.match(l.strip())]
        if not lines:
            return []

        duration = scene.end_sec - scene.start_sec
        if len(lines) == 1:
            interval = duration
        else:
            interval = duration / len(lines)

        entries = []
        for i, line in enumerate(lines):
            t = scene.start_sec + (i * interval)
            end = t + min(interval - 0.3, 5.0)
            end = max(end, t + 1.0)
            entries.append({"t": round(t, 2), "text": line, "end": round(end, 2)})
        return entries

    def _fix_broken_timestamps(self, timed_lyrics: list, scenes: List[MVScene]) -> list:
        """
        Gemini가 반환한 타임스탬프 중 비단조(뒤로 가는) 값만 선별 보간.

        기존 문제: 첫 번째 비단조 지점 이후를 전부 보간 → 정상 타임스탬프까지 덮어씀.
        개선: 실제로 깨진 엔트리(연속 비단조 구간)만 찾아 앞뒤 유효값 사이에서 보간.
        정상 타임스탬프는 절대 변경하지 않음.
        """
        if not timed_lyrics or len(timed_lyrics) < 2:
            return timed_lyrics

        n = len(timed_lyrics)
        timestamps = [float(e.get("t", 0)) for e in timed_lyrics]
        total_duration = max(s.end_sec for s in scenes) if scenes else timestamps[-1] + 30

        # 비단조 엔트리 마킹 (이전 최댓값보다 작거나 같은 것)
        needs_fix = [False] * n
        running_max = timestamps[0]
        for i in range(1, n):
            if timestamps[i] <= running_max:
                needs_fix[i] = True
            else:
                running_max = timestamps[i]

        fix_count = sum(needs_fix)
        if fix_count == 0:
            return timed_lyrics

        print(f"    [WARNING] {fix_count} non-monotonic timestamp(s) detected out of {n}")

        # 연속된 깨진 구간(run)별로 앞뒤 유효값 사이에서 보간
        i = 0
        while i < n:
            if needs_fix[i]:
                run_start = i
                while i < n and needs_fix[i]:
                    i += 1
                run_end = i  # exclusive

                # 앞쪽 유효 타임스탬프
                prev_t = timestamps[run_start - 1] if run_start > 0 else 0
                # 뒤쪽 유효 타임스탬프
                next_t = timestamps[run_end] if run_end < n else total_duration

                # next_t가 prev_t보다 작으면 (뒤쪽도 깨졌을 때) 안전 마진 추가
                if next_t <= prev_t:
                    next_t = prev_t + (run_end - run_start + 1) * 2.0

                count = run_end - run_start
                interval = (next_t - prev_t) / (count + 1)

                for k in range(count):
                    new_t = round(prev_t + interval * (k + 1), 2)
                    timestamps[run_start + k] = new_t
                    timed_lyrics[run_start + k]["t"] = new_t

                print(f"    [FIX] Entries {run_start+1}-{run_end}: interpolated {prev_t:.1f}s ~ {next_t:.1f}s")
            else:
                i += 1

        return timed_lyrics

    def _sec_to_srt_timecode(self, seconds: float) -> str:
        """초를 SRT 타임코드 형식으로 변환 (HH:MM:SS,mmm)"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        millis = int((seconds % 1) * 1000)
        return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"

    def _find_cjk_font(self) -> str:
        """한국어/CJK 지원 폰트 경로 찾기 (drawtext용)"""
        candidates = [
            "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
            "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
            "/usr/share/fonts/opentype/nanum/NanumGothic.ttf",
            "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        ]
        for path in candidates:
            if os.path.exists(path):
                return path.replace("\\", "/").replace(":", "\\:")
        # fc-match로 한국어 폰트 자동 탐색
        try:
            import subprocess as _sp
            r = _sp.run(["fc-match", "--format=%{file}", ":lang=ko"], capture_output=True, text=True, timeout=5)
            if r.returncode == 0 and r.stdout.strip():
                return r.stdout.strip().replace("\\", "/").replace(":", "\\:")
        except Exception:
            pass
        return "Sans"

    def _save_manifest(self, project: MVProject, project_dir: str):
        """매니페스트 저장 (atomic write로 손상 방지)"""
        manifest_path = f"{project_dir}/manifest.json"
        temp_path = f"{manifest_path}.tmp"

        # Pydantic → dict
        data = project.model_dump(mode='json')

        with open(temp_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2, default=str)

        # atomic rename (기존 파일 덮어쓰기)
        os.replace(temp_path, manifest_path)

    def load_project(self, project_id: str) -> Optional[MVProject]:
        """프로젝트 로드 (로컬 우선, R2 fallback)"""
        manifest_path = f"{self.output_base_dir}/{project_id}/manifest.json"

        print(f"[MV Pipeline] Loading project: {project_id}")
        print(f"[MV Pipeline] manifest exists: {os.path.exists(manifest_path)}")

        if not os.path.exists(manifest_path):
            # R2 fallback: 매니페스트 다운로드하여 로컬 복원
            try:
                from utils.storage import StorageManager
                storage = StorageManager()
                raw = storage.get_object(f"videos/{project_id}/manifest.json")
                if raw:
                    project_dir = f"{self.output_base_dir}/{project_id}"
                    os.makedirs(project_dir, exist_ok=True)
                    with open(manifest_path, 'w', encoding='utf-8') as f:
                        f.write(raw.decode('utf-8'))
                    print(f"[MV Pipeline] Manifest restored from R2: {manifest_path}")
                else:
                    print(f"[MV Pipeline] Project not found locally or in R2: {project_id}")
                    return None
            except Exception as e:
                print(f"[MV Pipeline] R2 fallback failed: {e}")
                return None

        with open(manifest_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        return MVProject(**data)
