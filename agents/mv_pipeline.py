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

            project.music_analysis = MusicAnalysis(**analysis_result)
            project.status = MVProjectStatus.READY
            project.progress = 10

            print(f"  Duration: {project.music_analysis.duration_sec:.1f}s")
            print(f"  BPM: {project.music_analysis.bpm or 'N/A'}")
            print(f"  Segments: {len(project.music_analysis.segments)}")

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

        # 각 씬에 이미지 프롬프트 생성
        for i, scene in enumerate(project.scenes):
            print(f"\n  [Scene {scene.scene_id}] {scene.start_sec:.1f}s - {scene.end_sec:.1f}s")

            # 이미지 프롬프트 생성
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

    def _generate_image_prompt(
        self,
        scene: MVScene,
        project: MVProject,
        request: MVProjectRequest,
        scene_index: int,
        total_scenes: int
    ) -> str:
        """씬별 이미지 프롬프트 생성"""

        # 스타일 프리픽스
        style_map = {
            MVStyle.CINEMATIC: "cinematic film still, dramatic lighting, high contrast",
            MVStyle.ANIME: "anime style, vibrant colors, detailed illustration",
            MVStyle.WEBTOON: "webtoon style, manhwa aesthetics, clean lines",
            MVStyle.REALISTIC: "photorealistic, 4K, detailed",
            MVStyle.ILLUSTRATION: "digital illustration, artistic",
            MVStyle.ABSTRACT: "abstract art, surreal, artistic interpretation"
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
            "professional quality"
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

        for i, scene in enumerate(project.scenes):
            print(f"\n  [Scene {scene.scene_id}/{total_scenes}] Generating image...")
            scene.status = MVSceneStatus.GENERATING

            try:
                # 이미지 생성
                image_path, _ = self.image_agent.generate_image(
                    scene_id=scene.scene_id,
                    prompt=scene.image_prompt,
                    style=project.style.value,
                    output_dir=image_dir
                )

                scene.image_path = image_path
                scene.status = MVSceneStatus.COMPLETED

                print(f"    Image saved: {image_path}")

                # 진행률 업데이트
                progress_per_scene = 50 / total_scenes  # 20% → 70%
                project.progress = int(20 + (i + 1) * progress_per_scene)

            except Exception as e:
                scene.status = MVSceneStatus.FAILED
                print(f"    [ERROR] Image generation failed: {e}")

            # 콜백 호출
            if on_scene_complete:
                try:
                    on_scene_complete(scene, i + 1, total_scenes)
                except Exception as cb_e:
                    print(f"    [WARNING] Callback failed: {cb_e}")

            # 매니페스트 저장 (각 씬마다)
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

            # 비디오 클립이 없으면 실패
            if not video_clips:
                project.status = MVProjectStatus.FAILED
                project.error_message = "No video clips were created. Check image generation."
                return project

            print(f"  Video clips created: {len(video_clips)}")

            # 2. 클립들 이어붙이기
            concat_video = f"{project_dir}/media/video/concat.mp4"
            self.ffmpeg_composer.concatenate_videos(
                video_paths=video_clips,
                output_path=concat_video
            )

            project.progress = 90

            # 3. 가사 자막 생성 및 burn-in (가사가 있는 경우)
            video_with_subtitles = concat_video
            if project.lyrics:
                print(f"  Adding lyrics subtitles...")
                srt_path = f"{project_dir}/media/subtitles/lyrics.srt"
                os.makedirs(os.path.dirname(srt_path), exist_ok=True)

                # SRT 파일 생성
                self._generate_lyrics_srt(project.scenes, srt_path)

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

        # Step 4: 영상 합성
        project = self.compose_video(project)

        elapsed = time.time() - start_time
        print(f"\n{'='*60}")
        print(f"[MV Pipeline] Completed in {elapsed:.1f}s")
        print(f"  Status: {project.status}")
        if project.final_video_path:
            print(f"  Output: {project.final_video_path}")
        print(f"{'='*60}\n")

        return project

    # ================================================================
    # 유틸리티
    # ================================================================

    def _generate_lyrics_srt(
        self,
        scenes: List[MVScene],
        output_path: str
    ):
        """씬별 가사를 SRT 자막 파일로 생성"""
        srt_lines = []

        for i, scene in enumerate(scenes, start=1):
            if not scene.lyrics_text:
                continue

            # 시간 포맷 변환 (초 → SRT 타임코드)
            start_tc = self._sec_to_srt_timecode(scene.start_sec)
            end_tc = self._sec_to_srt_timecode(scene.end_sec)

            # 가사 텍스트 (줄바꿈 유지)
            lyrics = scene.lyrics_text.strip()

            srt_lines.append(str(i))
            srt_lines.append(f"{start_tc} --> {end_tc}")
            srt_lines.append(lyrics)
            srt_lines.append("")  # 빈 줄로 구분

        # 파일 저장
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("\n".join(srt_lines))

        print(f"    SRT generated: {output_path} ({len([s for s in scenes if s.lyrics_text])} entries)")

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
