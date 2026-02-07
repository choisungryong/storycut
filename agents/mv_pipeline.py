"""
Music Video Pipeline - 뮤직비디오 생성 파이프라인

Phase 1: 기본 MV 생성
- 음악 분석 → 씬 분할 → 이미지 생성 → 영상 합성
"""

import os
import json
import uuid
import time
from typing import Dict, Any, List, Optional, Callable
from pathlib import Path
from datetime import datetime

from schemas.mv_models import (
    MVProject, MVScene, MVProjectStatus, MVSceneStatus,
    MVGenre, MVMood, MVStyle, MusicAnalysis, MVProjectRequest
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

    # ================================================================
    # Step 1: 음악 업로드 & 분석
    # ================================================================

    def upload_and_analyze(
        self,
        music_file_path: str,
        project_id: Optional[str] = None
    ) -> MVProject:
        """
        음악 파일 업로드 및 분석

        Args:
            music_file_path: 업로드된 음악 파일 경로
            project_id: 프로젝트 ID (없으면 자동 생성)

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

            # Gemini로 가사 자동 추출 (타임스탬프 포함)
            print(f"\n[Step 1.5] Extracting lyrics with Gemini...")
            extracted_lyrics = self.music_analyzer.extract_lyrics_with_gemini(stored_music_path)
            if extracted_lyrics:
                analysis_result["extracted_lyrics"] = extracted_lyrics
                # 타임스탬프 가사 저장
                timed_lyrics = getattr(self.music_analyzer, '_last_timed_lyrics', None)
                if timed_lyrics:
                    analysis_result["timed_lyrics"] = timed_lyrics

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
        project.genre = request.genre
        project.mood = request.mood
        project.style = request.style
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
        """자동 씬 분할 (음악 분석 기반)"""
        scenes = []

        if not project.music_analysis:
            raise ValueError("Music analysis required for auto scene creation")

        # 음악 세그먼트 기반으로 씬 생성
        for i, segment in enumerate(project.music_analysis.segments):
            # 가사에서 해당 구간 텍스트 추출 (간단한 분할)
            lyrics_text = self._extract_lyrics_for_segment(
                request.lyrics,
                i,
                len(project.music_analysis.segments)
            )

            scene = MVScene(
                scene_id=i + 1,
                start_sec=segment.start_sec,
                end_sec=segment.end_sec,
                duration_sec=segment.duration_sec,
                visual_description=f"{segment.segment_type} section",
                lyrics_text=lyrics_text,
                image_prompt=""
            )
            scenes.append(scene)

        return scenes

    def _extract_lyrics_for_segment(
        self,
        lyrics: Optional[str],
        segment_index: int,
        total_segments: int
    ) -> str:
        """가사에서 해당 구간 텍스트 추출 (간단한 균등 분할)"""
        if not lyrics:
            return ""

        lines = [l.strip() for l in lyrics.strip().split('\n') if l.strip()]
        if not lines:
            return ""

        # 균등 분할
        lines_per_segment = max(1, len(lines) // total_segments)
        start_idx = segment_index * lines_per_segment
        end_idx = min(start_idx + lines_per_segment, len(lines))

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
            scene_descriptions = []
            for i, scene in enumerate(project.scenes):
                lyrics_part = f'  가사: "{scene.lyrics_text}"' if scene.lyrics_text else "  가사: (없음)"
                scene_descriptions.append(
                    f"Scene {i+1} [{scene.start_sec:.1f}s-{scene.end_sec:.1f}s] "
                    f"({scene.visual_description})\n{lyrics_part}"
                )

            scenes_text = "\n".join(scene_descriptions)

            # 스타일별 구체적 비주얼 가이드
            style_guide = {
                "cinematic": "cinematic film still, dramatic chiaroscuro lighting, shallow depth of field, anamorphic lens flare, color graded like a Hollywood blockbuster",
                "anime": "Japanese anime cel-shaded illustration, bold outlines, vibrant saturated colors, anime eyes and proportions, manga-inspired composition",
                "webtoon": "Korean webtoon digital art style, clean sharp lines, flat color blocks, manhwa character design, vertical scroll composition",
                "realistic": "hyperrealistic photograph, DSLR quality, natural lighting, photojournalistic, sharp focus, real-world textures",
                "illustration": "digital painting illustration, painterly brushstrokes, concept art quality, rich color palette, artstation trending",
                "abstract": "abstract expressionist art, surreal dreamlike imagery, bold geometric shapes, color field painting, non-representational"
            }.get(request.style.value, "cinematic film still, dramatic lighting")

            genre_guide = {
                "fantasy": "magical fantasy world, enchanted forests, glowing runes, ethereal creatures, mythical landscapes",
                "romance": "intimate romantic scenes, warm golden hour lighting, soft bokeh, couples in tender moments",
                "action": "high-energy action scenes, dynamic motion blur, explosive effects, intense close-ups",
                "horror": "dark horror atmosphere, unsettling shadows, eerie fog, distorted perspectives, muted desaturated colors",
                "scifi": "futuristic sci-fi environment, neon holographics, cyberpunk city, advanced technology, chrome surfaces",
                "drama": "dramatic emotional scenes, theatrical lighting, expressive faces, strong contrast",
                "comedy": "bright cheerful scenes, exaggerated expressions, warm vivid colors, playful compositions",
                "abstract": "surreal abstract visuals, impossible geometry, color explosions, dreamscape"
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

            system_prompt = (
                "당신은 뮤직비디오 비주얼 디렉터입니다. "
                "각 씬에 대해 이미지 생성 AI가 사용할 영어 프롬프트를 만들어주세요.\n\n"
                "규칙:\n"
                "- 각 씬마다 완전히 다른 구체적인 장면을 묘사하세요\n"
                "- 가사의 감정과 의미를 시각적으로 표현하세요\n"
                "- 인물, 배경, 조명, 색감, 구도를 구체적으로 지정하세요\n"
                "- 노래의 흐름에 따라 시각적 스토리가 진행되도록 하세요\n"
                "- 각 프롬프트는 영어로 1-2문장, 쉼표로 구분된 키워드 형태\n"
                "- 절대 씬 번호나 설명 없이 프롬프트만 출력\n"
                "- 정확히 씬 개수만큼 줄을 출력하세요 (한 줄에 하나의 프롬프트)\n"
                "- 중요: 모든 프롬프트 끝에 반드시 'no text, no letters, no words, no writing, no watermark'를 포함하세요\n\n"
                f"[필수 비주얼 스타일]\n"
                f"모든 씬에 다음 스타일을 강하게 적용하세요:\n"
                f"- 아트 스타일: {style_guide}\n"
                f"- 장르 비주얼: {genre_guide}\n"
                f"- 분위기/톤: {mood_guide}\n"
                f"각 프롬프트의 첫 부분에 스타일 키워드를 반드시 포함하세요."
            )

            user_prompt = (
                f"컨셉: {request.concept or '자유'}\n"
                f"총 씬 수: {len(project.scenes)}\n\n"
                f"씬 정보:\n{scenes_text}\n\n"
                f"위 {len(project.scenes)}개 씬에 대해 각각 이미지 생성 프롬프트를 한 줄씩 출력하세요."
            )

            print(f"  [Gemini] Generating {len(project.scenes)} scene prompts...")

            response = client.models.generate_content(
                model="gemini-2.0-flash",
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

            print(f"  [Gemini] Generated {len(lines)} unique prompts")
            return lines

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
            MVStyle.ABSTRACT: "abstract expressionist art, surreal dreamlike, non-representational"
        }
        style_prefix = style_map.get(request.style, "cinematic")

        # 장르 키워드
        genre_map = {
            MVGenre.FANTASY: "fantasy world, magical, epic",
            MVGenre.ROMANCE: "romantic atmosphere, emotional",
            MVGenre.ACTION: "dynamic action, intense",
            MVGenre.HORROR: "dark, horror atmosphere, eerie",
            MVGenre.SCIFI: "futuristic, sci-fi, technology",
            MVGenre.DRAMA: "dramatic, emotional depth",
            MVGenre.COMEDY: "bright, cheerful, fun",
            MVGenre.ABSTRACT: "abstract, artistic, surreal"
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

        # 스타일 앵커: 첫 번째 씬 이미지를 앵커로 사용하여 비주얼 일관성 유지
        style_anchor_path = None

        for i, scene in enumerate(project.scenes):
            print(f"\n  [Scene {scene.scene_id}/{total_scenes}] Generating image...")
            if style_anchor_path:
                print(f"    [Anchor] Using style anchor: {os.path.basename(style_anchor_path)}")
            scene.status = MVSceneStatus.GENERATING

            try:
                # 이미지 생성 (스타일 앵커 참조)
                image_path, _ = self.image_agent.generate_image(
                    scene_id=scene.scene_id,
                    prompt=scene.image_prompt,
                    style=project.style.value,
                    output_dir=image_dir,
                    style_anchor_path=style_anchor_path,
                )

                scene.image_path = image_path
                scene.status = MVSceneStatus.COMPLETED

                # 첫 번째 성공 이미지를 스타일 앵커로 설정
                if style_anchor_path is None and image_path and os.path.exists(image_path):
                    style_anchor_path = image_path
                    print(f"    [Anchor] Style anchor set: {os.path.basename(image_path)}")

                print(f"    Image saved: {image_path}")

                # 진행률 업데이트
                progress_per_scene = 50 / total_scenes  # 20% → 70%
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

        # 이미지 생성 완료 → IMAGES_READY 상태로 전환 (리뷰 대기)
        project.status = MVProjectStatus.IMAGES_READY
        project.progress = 70
        project.current_step = "이미지 생성 완료 - 리뷰 대기"
        self._save_manifest(project, project_dir)

        return project

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
            # 1. 각 이미지를 비디오 클립으로 변환 (Ken Burns 효과)
            video_clips = []
            effect_types = ["zoom_in", "zoom_out", "pan_left", "pan_right"]  # 다양하게

            for i, scene in enumerate(completed_scenes):
                clip_path = f"{project_dir}/media/video/scene_{scene.scene_id:02d}.mp4"

                print(f"    [Scene {scene.scene_id}] image_path: {scene.image_path}")

                # I2V로 생성된 비디오가 이미 있으면 그대로 사용
                if scene.video_path and os.path.exists(scene.video_path):
                    video_clips.append(scene.video_path)
                    print(f"    Scene {scene.scene_id}: {scene.duration_sec:.1f}s clip (I2V pre-generated)")
                    continue

                # 이미지 파일 존재 확인
                if not scene.image_path or not os.path.exists(scene.image_path):
                    print(f"    [SKIP] Image not found: {scene.image_path}")
                    continue

                # Ken Burns 효과로 이미지 → 비디오
                effect = effect_types[i % len(effect_types)]
                try:
                    self.ffmpeg_composer.ken_burns_clip(
                        image_path=scene.image_path,
                        out_path=clip_path,
                        duration_sec=scene.duration_sec,
                        effect_type=effect
                    )
                    scene.video_path = clip_path
                    video_clips.append(clip_path)
                    print(f"    Scene {scene.scene_id}: {scene.duration_sec:.1f}s clip created (Ken Burns)")
                except Exception as kb_err:
                    print(f"    [WARNING] Ken Burns failed: {str(kb_err)[-200:]}")
                    # Fallback: 정적 이미지 → 비디오 (FFmpegComposer 사용으로 해상도 일관성 보장)
                    try:
                        self.ffmpeg_composer._image_to_static_video(
                            image_path=scene.image_path,
                            duration_sec=scene.duration_sec,
                            output_path=clip_path
                        )
                        scene.video_path = clip_path
                        video_clips.append(clip_path)
                        print(f"    Scene {scene.scene_id}: {scene.duration_sec:.1f}s clip created (static)")
                    except Exception as static_err:
                        print(f"    [SKIP] Static also failed: {str(static_err)[-200:]}")
                        continue

            project.progress = 85
            project.current_step = "영상 클립 생성 완료, 이어붙이는 중..."
            self._save_manifest(project, project_dir)

            # 비디오 클립이 없으면 실패
            if not video_clips:
                project.status = MVProjectStatus.FAILED
                project.error_message = "No video clips were created. Check image generation."
                return project

            print(f"  Video clips created: {len(video_clips)}")

            # 2. 클립들 이어붙이기
            concat_video = f"{project_dir}/media/video/concat.mp4"
            print(f"  Concatenating {len(video_clips)} clips (total ~{sum(s.duration_sec for s in completed_scenes):.0f}s)...")
            project.current_step = f"영상 {len(video_clips)}개 이어붙이는 중..."
            self._save_manifest(project, project_dir)

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

            # 3. 가사 자막 생성 및 burn-in (가사가 있는 경우)
            video_with_subtitles = concat_video
            print(f"  [Lyrics Check] project.lyrics = '{(project.lyrics or '')[:50]}...' (truthy={bool(project.lyrics)})")
            if project.lyrics:
                print(f"  Adding lyrics subtitles...")
                srt_path = f"{project_dir}/media/subtitles/lyrics.srt"
                os.makedirs(os.path.dirname(srt_path), exist_ok=True)

                # SRT 파일 생성 (타임스탬프 가사 우선 사용)
                timed_lyrics = None
                if project.music_analysis and project.music_analysis.timed_lyrics:
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

            # 4. 음악 합성
            final_video = f"{project_dir}/final_mv.mp4"
            self.ffmpeg_composer._add_audio_to_video(
                video_in=video_with_subtitles,
                audio_path=project.music_file_path,
                out_path=final_video
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

        print(f"[MV Regenerate] Scene {scene_id} image regeneration...")

        # 스타일 앵커: scene_01 이미지 사용 (첫 씬 재생성이 아닌 경우)
        style_anchor_path = None
        if scene_id > 1:
            first_scene = project.scenes[0]
            if first_scene.image_path and os.path.exists(first_scene.image_path):
                style_anchor_path = first_scene.image_path

        scene.status = MVSceneStatus.GENERATING

        try:
            image_path, _ = self.image_agent.generate_image(
                scene_id=scene.scene_id,
                prompt=scene.image_prompt,
                style=project.style.value,
                output_dir=image_dir,
                style_anchor_path=style_anchor_path,
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
        가사 SRT 자막 파일 생성

        타임스탬프 가사가 있으면 실제 타이밍 사용, 없으면 씬 기반 균등 분할
        """
        srt_lines = []

        if timed_lyrics and len(timed_lyrics) > 0:
            # 타임스탬프 기반 SRT 생성 (정확한 싱크)
            print(f"    Using timed lyrics ({len(timed_lyrics)} entries)")

            # 전처리: 비단조 타임스탬프 감지 및 보간
            timed_lyrics = self._fix_broken_timestamps(timed_lyrics, scenes)

            srt_idx = 0
            for i, entry in enumerate(timed_lyrics):
                text = entry.get("text", "").strip()
                if not text:
                    continue

                start_sec = float(entry.get("t", 0))
                # 다음 가사 시작 0.3초 전까지 또는 최대 5초 표시
                if i + 1 < len(timed_lyrics):
                    next_start = float(timed_lyrics[i + 1].get("t", start_sec + 4))
                    end_sec = min(next_start - 0.3, start_sec + 5)  # 겹침 방지 0.3초 gap
                else:
                    end_sec = start_sec + 4
                # 최소 1초는 표시
                end_sec = max(end_sec, start_sec + 1.0)

                srt_idx += 1
                start_tc = self._sec_to_srt_timecode(start_sec)
                end_tc = self._sec_to_srt_timecode(end_sec)

                srt_lines.append(str(srt_idx))
                srt_lines.append(f"{start_tc} --> {end_tc}")
                srt_lines.append(text)
                srt_lines.append("")
        else:
            # fallback: 씬 기반 균등 분할
            print(f"    Using scene-based lyrics (no timestamps)")
            idx = 0
            for scene in scenes:
                if not scene.lyrics_text:
                    continue

                lyrics = scene.lyrics_text.strip()
                idx += 1

                start_tc = self._sec_to_srt_timecode(scene.start_sec)
                end_tc = self._sec_to_srt_timecode(scene.end_sec)

                srt_lines.append(str(idx))
                srt_lines.append(f"{start_tc} --> {end_tc}")
                srt_lines.append(lyrics)
                srt_lines.append("")

        # 파일 저장
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("\n".join(srt_lines))

        entry_count = len([l for l in srt_lines if l.startswith("00:") or l.startswith("01:") or l.startswith("02:")])
        print(f"    SRT generated: {output_path}")

    def _fix_broken_timestamps(self, timed_lyrics: list, scenes: List[MVScene]) -> list:
        """
        Gemini가 반환한 타임스탬프 중 비단조(뒤로 가는) 값을 감지하여 보간.

        예: [1.2, 5.0, ..., 57.4, 1.0, 1.0, 1.0, ...]
        → 57.4 이후 1.0들은 깨진 것이므로 균등 보간으로 대체
        """
        if not timed_lyrics or len(timed_lyrics) < 2:
            return timed_lyrics

        # 마지막 유효 타임스탬프 인덱스 찾기
        last_valid_idx = 0
        last_valid_t = float(timed_lyrics[0].get("t", 0))

        for i in range(1, len(timed_lyrics)):
            t = float(timed_lyrics[i].get("t", 0))
            if t > last_valid_t:
                last_valid_idx = i
                last_valid_t = t
            else:
                # 이 지점부터 타임스탬프가 깨짐
                break
        else:
            # 모든 타임스탬프가 단조 증가 → 문제 없음
            return timed_lyrics

        broken_start = last_valid_idx + 1
        broken_count = len(timed_lyrics) - broken_start

        if broken_count <= 0:
            return timed_lyrics

        print(f"    [WARNING] Broken timestamps detected from entry {broken_start + 1}/{len(timed_lyrics)}")
        print(f"    [FIX] Interpolating {broken_count} entries after {last_valid_t:.1f}s")

        # 총 음악 길이 구하기
        total_duration = max(s.end_sec for s in scenes) if scenes else last_valid_t + broken_count * 3
        remaining_duration = total_duration - last_valid_t

        # 남은 시간을 깨진 엔트리들에 균등 분배
        interval = remaining_duration / (broken_count + 1)

        for i in range(broken_count):
            idx = broken_start + i
            interpolated_t = round(last_valid_t + interval * (i + 1), 1)
            timed_lyrics[idx]["t"] = interpolated_t

        print(f"    [FIX] Interpolated range: {timed_lyrics[broken_start]['t']}s ~ {timed_lyrics[-1]['t']}s")

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
        """프로젝트 로드"""
        manifest_path = f"{self.output_base_dir}/{project_id}/manifest.json"

        print(f"[MV Pipeline] Loading project: {project_id}")
        print(f"[MV Pipeline] Manifest path: {manifest_path}")
        print(f"[MV Pipeline] CWD: {os.getcwd()}")
        print(f"[MV Pipeline] outputs/ exists: {os.path.exists('outputs')}")
        print(f"[MV Pipeline] project_dir exists: {os.path.exists(f'{self.output_base_dir}/{project_id}')}")
        print(f"[MV Pipeline] manifest exists: {os.path.exists(manifest_path)}")

        if not os.path.exists(manifest_path):
            # 디렉토리 내용 출력
            if os.path.exists('outputs'):
                dirs = os.listdir('outputs')
                print(f"[MV Pipeline] outputs/ contents: {dirs[:10]}")  # 처음 10개만
            return None

        with open(manifest_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        return MVProject(**data)
