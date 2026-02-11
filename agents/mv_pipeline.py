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

            # 가사 처리: 사용자 가사 있으면 타이밍만 싱크, 없으면 Gemini 추출
            if user_lyrics and user_lyrics.strip():
                print(f"\n[Step 1.5] Syncing user lyrics with music timing...")
                synced_lyrics = self.music_analyzer.sync_user_lyrics_with_gemini(stored_music_path, user_lyrics.strip())
                if synced_lyrics:
                    analysis_result["extracted_lyrics"] = synced_lyrics
                    timed_lyrics = getattr(self.music_analyzer, '_last_timed_lyrics', None)
                    if timed_lyrics:
                        analysis_result["timed_lyrics"] = timed_lyrics
                # Raw STT 문장 보존 (타이밍 에디터용)
                stt_sentences = getattr(self.music_analyzer, '_last_stt_sentences', None)
                if stt_sentences:
                    analysis_result["stt_sentences"] = stt_sentences
                extracted_lyrics = synced_lyrics
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

        project.scenes = scenes
        project.progress = 20

        print(f"  Total scenes: {len(scenes)}")

        # Gemini LLM으로 씬별 프롬프트 일괄 생성 시도
        gemini_prompts = self._generate_prompts_with_gemini(project, request)

        for i, scene in enumerate(project.scenes):
            print(f"\n  [Scene {scene.scene_id}] {scene.start_sec:.1f}s - {scene.end_sec:.1f}s")

            if gemini_prompts and i < len(gemini_prompts):
                scene.image_prompt = gemini_prompts[i]
            else:
                # fallback: 템플릿 기반 프롬프트
                scene.image_prompt = self._generate_image_prompt(
                    scene=scene,
                    project=project,
                    request=request,
                    scene_index=i,
                    total_scenes=len(project.scenes)
                )

            print(f"    Prompt: {scene.image_prompt[:80]}...")

        self._save_manifest(project, project_dir)

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

            # 가사 요약 (너무 길면 앞부분만)
            lyrics_summary = ""
            if project.lyrics:
                lyrics_summary = project.lyrics[:500]
            elif project.music_analysis and project.music_analysis.extracted_lyrics:
                lyrics_summary = project.music_analysis.extracted_lyrics[:500]

            # 씬 타임라인 정보 (LLM이 appears_in/scene_blocking을 올바르게 생성하도록)
            scene_timeline = ""
            if project.scenes:
                timeline_parts = []
                for s in project.scenes:
                    lyrics_preview = (s.lyrics_text or "")[:40].replace("\n", " ")
                    timeline_parts.append(
                        f"  Scene {s.scene_id}: {s.start_sec:.1f}s-{s.end_sec:.1f}s"
                        + (f' lyrics: "{lyrics_preview}..."' if lyrics_preview else "")
                    )
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
                    "eye shape, face shape, skin tone, body type. Be CONCRETE (e.g. 'Korean woman in her mid-20s, "
                    "long straight black hair, almond eyes, fair skin, slender build')\n"
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
                    "- NEVER change a character's ethnicity, age, or core features between scenes.\n"
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
                    "eye shape, face shape, skin tone, body type. Be CONCRETE (e.g. 'Korean woman in her mid-20s, "
                    "long straight black hair, almond eyes, fair skin, slender build')\n"
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
                    "- NEVER change a character's ethnicity, age, or core features between scenes.\n"
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
            role_to_path = char_manager.cast_mv_characters(
                characters=characters,
                project=project,
                project_dir=project_dir,
                candidates_per_pose=2,
            )

            # 결과를 MVCharacter.anchor_image_path에 저장
            for character in characters:
                path = role_to_path.get(character.role)
                if path:
                    character.anchor_image_path = path
                    print(f"    [OK] {character.role} -> {path}")

        except Exception as e:
            print(f"  [WARNING] CharacterManager failed, falling back to simple method: {e}")
            self._generate_character_anchors_simple(project, project_dir)

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
            "realistic": "hyperrealistic photograph, DSLR quality, natural lighting, sharp focus, NOT anime, NOT cartoon",
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
                scene_descriptions.append(
                    f"Scene {i+1} [{scene.start_sec:.1f}s-{scene.end_sec:.1f}s] "
                    f"({scene.visual_description}){broll_tag}\n{lyrics_part}"
                )

            scenes_text = "\n".join(scene_descriptions)

            # 스타일별 구체적 비주얼 가이드
            style_guide = {
                "cinematic": "cinematic film still, dramatic chiaroscuro lighting, shallow depth of field, anamorphic lens flare, color graded like a Hollywood blockbuster",
                "anime": "Japanese anime cel-shaded illustration, bold outlines, vibrant saturated colors, anime eyes and proportions, manga-inspired composition",
                "webtoon": "Korean webtoon digital art style, clean sharp lines, flat color blocks, manhwa character design, vertical scroll composition",
                "realistic": "hyperrealistic photograph, DSLR quality, natural lighting, photojournalistic, sharp focus, real-world textures",
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
                "- 캐릭터 외형(인종, 나이, 헤어, 체형)을 프롬프트에 구체적으로 포함하세요.\n"
                "- *** ETHNICITY IS MANDATORY IN EVERY PROMPT ***: Every prompt MUST explicitly state "
                "the character's ethnicity/race (e.g. 'Korean man', 'Korean woman'). "
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
            MVStyle.REALISTIC: "hyperrealistic photograph, DSLR quality, natural lighting, NOT anime, NOT cartoon, NOT illustration",
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
        """씬에 등장하는 캐릭터의 앵커 이미지 경로 반환 (최대 3개)"""
        if not project.visual_bible or not project.visual_bible.characters:
            return []
        char_map = {
            c.role: c.anchor_image_path
            for c in project.visual_bible.characters
            if c.anchor_image_path and os.path.exists(c.anchor_image_path)
        }
        return [
            char_map[r]
            for r in (scene.characters_in_scene or [])[:3]
            if r in char_map
        ]

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

        # 프롬프트 조립: action_pose를 최우선에 배치 (앵커 이미지의 직립 자세 덮어쓰기)
        prefix_parts = []
        if action_pose:
            prefix_parts.append(f"POSE: {action_pose}")
        if expression:
            prefix_parts.append(f"EXPRESSION: {expression}")
        char_block = " | ".join(char_descs)
        prefix_parts.append(char_block)

        return f"{'. '.join(prefix_parts)}. {prompt}"

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

            # B-roll 시도: intro/outro/bridge + 캐릭터 미등장
            seg_type = self._extract_segment_type(scene)
            if (pexels and seg_type in BROLL_SEGMENTS
                    and not scene.characters_in_scene):
                queries = pexels.generate_stock_queries(
                    scene_prompt=scene.image_prompt,
                    lyrics_text=scene.lyrics_text,
                    segment_type=seg_type,
                    genre=project.genre.value,
                    mood=project.mood.value,
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

                # 시대/배경 키워드 자동 주입 (concept에서 추출)
                if era_prefix and era_prefix.lower() not in final_prompt.lower():
                    final_prompt = f"{era_prefix}, {final_prompt}"
                # 시대 부정 키워드 (negative_prompt에 추가)
                _scene_neg = scene.negative_prompt or ""
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

    def _generate_cut_plan(self, scenes: List[MVScene]) -> List[dict]:
        """
        코드 기반 컷 플래닝: 세그먼트 타입에 따라 컷 수/길이를 자동 분배.
        I2V(video_path) 씬은 분할하지 않고 단일 컷으로 통과.

        Returns:
            List of cut dicts: parent_scene_id, cut_index, duration_sec,
            reframe, crop_anchor, effect_type, zoom_range, image_path
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

        reframe_cycle = ["wide", "medium", "close", "detail"]
        effect_cycle = ["zoom_in", "pan_left", "zoom_out", "pan_right", "diagonal"]
        anchor_positions = [
            (0.33, 0.33), (0.5, 0.5), (0.67, 0.33),
            (0.33, 0.67), (0.67, 0.67), (0.5, 0.33),
        ]

        cut_plan = []
        global_cut_idx = 0

        for scene in scenes:
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
                    "is_broll": getattr(scene, 'is_broll', False),
                })
                global_cut_idx += 1
                continue

            if not scene.image_path or not os.path.exists(scene.image_path):
                continue

            # 세그먼트 타입 추출
            seg_type = self._extract_segment_type(scene)
            config = cut_config.get(seg_type, default_config)
            num_cuts = config["cuts"]

            # duration 분배: 균등 분할
            total_dur = scene.duration_sec
            cut_dur = total_dur / num_cuts

            for ci in range(num_cuts):
                reframe = reframe_cycle[global_cut_idx % len(reframe_cycle)]
                effect = effect_cycle[global_cut_idx % len(effect_cycle)]
                anchor = anchor_positions[global_cut_idx % len(anchor_positions)]

                # chorus 4+ 컷이면 detail 추가
                if seg_type in ("chorus", "hook") and ci >= 3:
                    reframe = "detail"

                cut_plan.append({
                    "parent_scene_id": scene.scene_id,
                    "cut_index": ci,
                    "duration_sec": round(cut_dur, 2),
                    "reframe": reframe,
                    "crop_anchor": anchor,
                    "effect_type": effect,
                    "zoom_range": (1.0, 1.08),
                    "image_path": scene.image_path,
                    "video_path": None,
                })
                global_cut_idx += 1

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

        return cut_plan

    def _extract_segment_type(self, scene: MVScene) -> str:
        """씬의 visual_description에서 segment_type 추출."""
        desc = (scene.visual_description or "").lower()
        # "verse section" → "verse", "chorus section" → "chorus"
        for seg_type in ["pre_chorus", "post_chorus", "chorus", "hook", "verse", "bridge", "intro", "outro"]:
            if seg_type.replace("_", " ") in desc or seg_type.replace("_", "-") in desc or seg_type in desc:
                return seg_type
        return "verse"  # default

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
            # 1. 파생 컷 플래닝 + 클립 생성
            cut_plan = self._generate_cut_plan(completed_scenes)
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

            # 2. 클립들 이어붙이기 (씬 간 크로스페이드 전환)
            concat_video = f"{project_dir}/media/video/concat.mp4"
            # 빈 그룹 제거
            scene_groups = [g for g in scene_groups if g]
            total_dur = sum(s.duration_sec for s in completed_scenes)
            print(f"  Concatenating {len(video_clips)} clips in {len(scene_groups)} scenes (total ~{total_dur:.0f}s)...")
            project.current_step = f"영상 {len(scene_groups)}개 씬 크로스페이드 합성 중..."
            self._save_manifest(project, project_dir)

            if len(scene_groups) >= 2:
                concat_result = self.ffmpeg_composer.concatenate_with_crossfade(
                    scene_groups=scene_groups,
                    output_path=concat_video,
                    fade_duration=0.3,
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

            # 3. 가사 자막 생성 및 burn-in (가사가 있고 자막 옵션이 켜진 경우)
            video_with_subtitles = concat_video
            has_lyrics = bool(project.lyrics) or any(s.lyrics_text for s in project.scenes)
            subtitle_on = getattr(project, 'subtitle_enabled', True)
            print(f"  [Lyrics Check] project.lyrics = '{(project.lyrics or '')[:50]}...' (truthy={bool(project.lyrics)}), scene lyrics={any(s.lyrics_text for s in project.scenes)}, subtitle_enabled={subtitle_on}")
            if has_lyrics and subtitle_on:
                print(f"  Adding lyrics subtitles...")
                srt_path = f"{project_dir}/media/subtitles/lyrics.srt"
                os.makedirs(os.path.dirname(srt_path), exist_ok=True)

                # SRT 파일 생성 (edited_timed_lyrics > timed_lyrics 우선순위)
                timed_lyrics = None
                if project.edited_timed_lyrics:
                    timed_lyrics = project.edited_timed_lyrics
                    print(f"    Using edited_timed_lyrics ({len(timed_lyrics)} entries)")
                elif project.music_analysis and project.music_analysis.timed_lyrics:
                    timed_lyrics = project.music_analysis.timed_lyrics
                self._generate_lyrics_srt(project.scenes, srt_path, timed_lyrics=timed_lyrics)

                # 자막 burn-in
                subtitled_video = f"{project_dir}/media/video/subtitled.mp4"
                try:
                    self.ffmpeg_composer.overlay_subtitles(
                        video_in=concat_video,
                        srt_path=srt_path,
                        out_path=subtitled_video
                    )
                    if os.path.exists(subtitled_video) and os.path.getsize(subtitled_video) > 1024:
                        video_with_subtitles = subtitled_video
                        print(f"    Subtitles added successfully")
                    else:
                        print(f"    [WARNING] Subtitle burn-in failed, using video without subtitles")
                except Exception as sub_err:
                    print(f"    [WARNING] Subtitle error: {sub_err}")

            project.progress = 95
            project.current_step = "음악 합성 중..."
            self._save_manifest(project, project_dir)

            # 4. 최종 인코딩 (H.264 High + AAC + 조건부 워터마크)
            final_video = f"{project_dir}/final_mv.mp4"
            watermark = "Made with Klippa" if getattr(project, 'watermark_enabled', True) else None
            self.ffmpeg_composer.final_encode(
                video_in=video_with_subtitles,
                audio_path=project.music_file_path,
                out_path=final_video,
                watermark_text=watermark,
            )

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

        # Step 2: 씬 생성
        project = self.generate_scenes(project, request)
        if on_progress:
            on_progress(project)

        # Visual Bible + Style Anchor + Character Anchors
        project = self.generate_visual_bible(project)
        project = self.generate_style_anchor(project)
        project = self.generate_character_anchors(project)

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

        # 시대/배경 키워드 자동 주입 (concept에서 추출)
        era_prefix, era_negative = self._extract_era_setting(project.concept)
        if era_prefix and era_prefix.lower() not in final_prompt.lower():
            final_prompt = f"{era_prefix}, {final_prompt}"
        _regen_neg = scene.negative_prompt or ""
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
