"""
STORYCUT 통합 파이프라인

전체 실행 플로우를 관리하는 메인 파이프라인.
outputs/<project_id>/ 구조로 모든 산출물 관리.

실행 플로우:
1. (optional) TopicFindingAgent - 주제 후보 생성
2. StoryAgent - 스토리 생성
3. SceneOrchestrator - Scene 처리 (맥락 상속, 영상/음성 생성)
4. FFmpegComposer - 최종 합성
5. OptimizationAgent - 제목/썸네일/AB테스트 패키지
"""

import os
import json
import time
import uuid
from dotenv import load_dotenv

# .env 파일 로드
load_dotenv()

from datetime import datetime
from typing import Dict, Any, Optional, List
from pathlib import Path

from schemas import (
    FeatureFlags,
    ProjectRequest,
    Scene,
    Manifest,
    ManifestOutputs,
    CostEstimate,
)
from agents import (
    StoryAgent,
    SceneOrchestrator,
    OptimizationAgent,
    CharacterManager,
    StyleAnchorAgent,
    ConsistencyValidator,
)
from utils.ffmpeg_utils import FFmpegComposer


class StorycutPipeline:
    """
    STORYCUT 통합 파이프라인

    모든 에이전트를 조율하고 Manifest를 관리합니다.
    """

    def __init__(self, output_base_dir: str = "outputs"):
        """
        Initialize pipeline.

        Args:
            output_base_dir: 출력 기본 디렉토리
        """
        self.output_base_dir = output_base_dir
        self.story_agent = StoryAgent()
        self.optimization_agent = OptimizationAgent()

    def run(self, request: ProjectRequest) -> Manifest:
        """
        [Legacy] 전체 파이프라인 한 번에 실행.
        """
        # 1. 스토리 생성
        story_data = self.generate_story_only(request)
        
        # 2. 영상 생성 (스토리 기반)
        return self.generate_video_from_story(story_data, request)

    def generate_story_only(self, request: ProjectRequest) -> Dict[str, Any]:
        """Step 1: 스토리만 생성"""
        print(f"\n[STEP 1] Generating story for topic: {request.topic}")
        return self._generate_story(request)

    def generate_video_from_story(self, story_data: Dict[str, Any], request: ProjectRequest, project_id: str = None) -> Manifest:
        """Step 2~5: 확정된 스토리로 영상 생성"""
        start_time = time.time()
        
        if not project_id:
            project_id = str(uuid.uuid4())[:8]
            
        project_dir = self._create_project_structure(project_id)

        # Manifest 초기화
        from schemas import GlobalStyle, CharacterSheet

        manifest = Manifest(
            project_id=project_id,
            input=request,
            status="processing",
            title=story_data.get("title"),
            script=json.dumps(story_data, ensure_ascii=False)
        )

        # v2.0: character_sheet와 global_style 저장
        if "character_sheet" in story_data:
            manifest.character_sheet = {
                token: CharacterSheet(**data)
                for token, data in story_data["character_sheet"].items()
            }
            print(f"[v2.0] Loaded {len(manifest.character_sheet)} characters from story")

        if "global_style" in story_data:
            manifest.global_style = GlobalStyle(**story_data["global_style"])
            print(f"[v2.0] Global style: {manifest.global_style.art_style}")

        # [STEP 1.3] Style Anchor - 프로젝트 전체 룩 앵커 이미지 생성
        style_anchor_path = None
        env_anchors = {}
        style_anchor_agent = StyleAnchorAgent()

        if manifest.global_style:
            print(f"\n[STEP 1.3] Generating style anchor image...")
            style_anchor_path = style_anchor_agent.generate_style_anchor(
                global_style=manifest.global_style,
                project_dir=project_dir
            )

        # [STEP 1.4] Environment Anchors - 씬별 환경 앵커 이미지 생성
        if manifest.global_style and "scenes" in story_data:
            print(f"\n[STEP 1.4] Generating environment anchor images...")
            env_anchors = style_anchor_agent.generate_environment_anchors(
                scenes=story_data["scenes"],
                global_style=manifest.global_style,
                project_dir=project_dir
            )

        # [STEP 1.5] Character Casting - 마스터 앵커 이미지 생성
        if manifest.character_sheet:
            print(f"\n[STEP 1.5] Casting characters (generating master anchor images)...")
            character_manager = CharacterManager()
            character_images = character_manager.cast_characters(
                character_sheet=manifest.character_sheet,
                global_style=manifest.global_style,
                project_dir=project_dir
            )

            # story_data에도 master_image_path 반영 (SceneOrchestrator가 참조)
            if "character_sheet" in story_data:
                for token, image_path in character_images.items():
                    if token in story_data["character_sheet"]:
                        story_data["character_sheet"][token]["master_image_path"] = image_path

        try:
            print(f"\n{'='*60}")
            print(f"STORYCUT Pipeline - Video Generation - Project: {project_id}")
            print(f"{'='*60}")

            # Step 2: Scene 처리 (맥락 상속 포함)
            print("\n[STEP 2/6] Processing scenes with context carry-over...")
            orchestrator = SceneOrchestrator(feature_flags=request.feature_flags)
            final_video = orchestrator.process_story(
                story_data=story_data,
                output_path=f"{project_dir}/final_video.mp4",
                request=request,
                style_anchor_path=style_anchor_path,
                environment_anchors=env_anchors,
            )

            # Scene 정보 업데이트
            manifest.scenes = self._convert_scenes_to_schema(story_data["scenes"])
            manifest.outputs.final_video_path = final_video

            # Step 3: 자막 생성 및 영상에 적용 (옵션)
            if request.subtitles:
                print("\n[STEP 3/6] Generating subtitles and applying to video...")
                final_video = self._generate_and_apply_subtitles(
                    manifest.scenes,
                    project_dir,
                    final_video
                )
                manifest.outputs.final_video_path = final_video

            # Step 3.5: Film Look 후처리 (v2.0)
            if request.feature_flags.film_look:
                print("\n[STEP 3.5/6] Applying film look (grain + color grading)...")
                final_video = self._apply_film_look(final_video, project_dir)
                manifest.outputs.final_video_path = final_video

            # Step 4: Optimization 패키지 생성
            if request.feature_flags.optimization_pack:
                # v2.1: Check if StoryAgent already generated optimization data
                if "youtube_opt" in story_data:
                    print("\n[STEP 4/6] Using pre-generated optimization package from StoryAgent...")
                    opt = story_data["youtube_opt"]
                    manifest.outputs.title_candidates = opt.get("title_candidates", [])
                    manifest.outputs.thumbnail_texts = [opt.get("thumbnail_text")] if opt.get("thumbnail_text") else []
                    manifest.outputs.hashtags = opt.get("hashtags", [])
                    
                    # Save as separate JSON for frontend compatibility
                    opt_package = {
                        "title_candidates": manifest.outputs.title_candidates,
                        "thumbnail_texts": manifest.outputs.thumbnail_texts,
                        "hashtags": manifest.outputs.hashtags
                    }
                    opt_path = self.optimization_agent.save_optimization_package(opt_package, project_dir, project_id)
                    manifest.outputs.metadata_json_path = opt_path
                    
                else:
                    print("\n[STEP 4/6] Generating optimization package (Legacy)...")
                    opt_package = self.optimization_agent.run(
                        topic=request.topic or manifest.title,
                        script=manifest.script,
                        scenes=manifest.scenes,
                        request=request
                    )
                
                manifest.outputs.title_candidates = opt_package.get("title_candidates", [])
                manifest.outputs.thumbnail_texts = opt_package.get("thumbnail_texts", [])
                manifest.outputs.hashtags = opt_package.get("hashtags", [])
                
                opt_path = self.optimization_agent.save_optimization_package(opt_package, project_dir, project_id)
                manifest.outputs.metadata_json_path = opt_path

            # Step 5: Manifest 저장
            print("\n[STEP 5/6] Saving manifest...")
            manifest.status = "completed"
            manifest.execution_time_sec = time.time() - start_time
            manifest.cost_estimate = self._estimate_costs(manifest)

            manifest_path = self._save_manifest(manifest, project_dir)
            
            return manifest

        except Exception as e:
            manifest.status = "failed"
            manifest.error_message = str(e)
            self._save_manifest(manifest, project_dir)
            raise

    def _create_project_structure(self, project_id: str) -> str:
        """
        프로젝트 디렉토리 구조 생성.

        outputs/<project_id>/
        ├── manifest.json
        ├── final_video.mp4
        ├── scenes/
        │   ├── scene_01.json
        │   ├── scene_02.json
        │   └── ...
        ├── media/
        │   ├── video/
        │   ├── audio/
        │   ├── images/
        │   └── subtitles/
        └── optimization_<project_id>.json
        """
        project_dir = f"{self.output_base_dir}/{project_id}"

        dirs = [
            project_dir,
            f"{project_dir}/scenes",
            f"{project_dir}/media/video",
            f"{project_dir}/media/audio",
            f"{project_dir}/media/images",
            f"{project_dir}/media/subtitles",
            f"{project_dir}/media/characters",  # v2.0: 캐릭터 마스터 이미지
        ]

        for d in dirs:
            os.makedirs(d, exist_ok=True)

        return project_dir

    def _generate_story(self, request: ProjectRequest) -> Dict[str, Any]:
        """
        스토리 생성.

        Args:
            request: ProjectRequest

        Returns:
            Story JSON
        """
        story_data = self.story_agent.generate_story(
            genre=request.genre or "emotional",
            mood=request.mood or "dramatic",
            style=request.style_preset or "cinematic",
            total_duration_sec=request.duration_target_sec or 60,
            user_idea=request.topic or request.user_idea
        )

        return story_data

    def _convert_scenes_to_schema(
        self,
        scene_dicts: List[Dict[str, Any]]
    ) -> List[Scene]:
        """
        Scene 딕셔너리를 Schema 객체로 변환 (v2.0 호환).

        Args:
            scene_dicts: Scene 딕셔너리 목록

        Returns:
            Scene 객체 목록
        """
        scenes = []
        for idx, sd in enumerate(scene_dicts, start=1):
            scene = Scene(
                index=idx,
                scene_id=sd.get("scene_id", idx),
                sentence=sd.get("narration", ""),
                narration=sd.get("narration"),
                visual_description=sd.get("visual_description"),
                mood=sd.get("mood"),
                duration_sec=sd.get("duration_sec", 5),
                # v2.0 필드
                narrative=sd.get("narrative"),
                image_prompt=sd.get("image_prompt"),
                characters_in_scene=sd.get("characters_in_scene", []),
            )
            scenes.append(scene)
        return scenes

    def _generate_subtitles(
        self,
        scenes: List[Scene],
        project_dir: str
    ) -> str:
        """
        자막 파일 생성.

        Args:
            scenes: Scene 목록
            project_dir: 프로젝트 디렉토리

        Returns:
            SRT 파일 경로
        """
        composer = FFmpegComposer()

        scene_dicts = [
            {
                "narration": s.narration or s.sentence,
                "duration_sec": s.duration_sec
            }
            for s in scenes
        ]

        srt_path = f"{project_dir}/media/subtitles/full.srt"
        return composer.generate_srt_from_scenes(scene_dicts, srt_path)

    def _generate_and_apply_subtitles(
        self,
        scenes: List[Scene],
        project_dir: str,
        input_video: str
    ) -> str:
        """
        자막 파일 생성 및 영상에 적용 (Burn-in).

        Args:
            scenes: Scene 목록
            project_dir: 프로젝트 디렉토리
            input_video: 입력 영상 경로

        Returns:
            자막이 적용된 최종 영상 경로
        """
        composer = FFmpegComposer()

        # 1. 자막 파일 생성
        scene_dicts = [
            {
                "narration": s.narration or s.sentence,
                "duration_sec": s.duration_sec
            }
            for s in scenes
        ]

        srt_path = f"{project_dir}/media/subtitles/full.srt"
        composer.generate_srt_from_scenes(scene_dicts, srt_path)
        print(f"  Generated subtitle file: {srt_path}")

        # 2. 자막을 영상에 burn-in
        output_with_subtitles = f"{project_dir}/final_video_with_subtitles.mp4"
        try:
            subtitled_video, subtitle_success = composer.overlay_subtitles(
                input_video,
                srt_path,
                output_with_subtitles
            )
            if subtitle_success:
                print(f"  Applied subtitles to video: {subtitled_video}")
                return subtitled_video
            else:
                print(f"  [Warning] Subtitle burn-in failed (likely OOM). Using original video.")
                return input_video
        except Exception as e:
            print(f"  [Warning] Subtitle burn-in failed: {e}. Using original video.")
            return input_video

    def _estimate_costs(self, manifest: Manifest) -> CostEstimate:
        """
        비용 추정.

        Args:
            manifest: Manifest 객체

        Returns:
            CostEstimate 객체
        """
        # 대략적인 추정치
        llm_tokens = len(manifest.script or "") * 2  # 입력 + 출력
        video_seconds = sum(s.duration_sec for s in manifest.scenes)
        image_count = len([s for s in manifest.scenes if s.generation_method == "image+kenburns"])
        tts_characters = sum(len(s.narration or "") for s in manifest.scenes)

        # 비용 계산 (대략적)
        estimated_usd = (
            (llm_tokens / 1000) * 0.03 +  # GPT-4 토큰
            image_count * 0.02 +           # DALL-E 이미지
            (tts_characters / 1000) * 0.015  # TTS
        )

        # Hook 비디오 사용 시 추가 비용
        if manifest.input.feature_flags.hook_scene1_video:
            estimated_usd += 0.5  # Runway 등 비디오 API

        return CostEstimate(
            llm_tokens=llm_tokens,
            video_seconds=video_seconds,
            image_count=image_count,
            tts_characters=tts_characters,
            estimated_usd=round(estimated_usd, 2)
        )

    def _save_manifest(self, manifest: Manifest, project_dir: str) -> str:
        """
        Manifest를 JSON으로 저장.

        Args:
            manifest: Manifest 객체
            project_dir: 프로젝트 디렉토리

        Returns:
            저장된 파일 경로
        """
        manifest_path = f"{project_dir}/manifest.json"

        # Pydantic 모델을 JSON으로 직렬화
        manifest_dict = manifest.model_dump(mode="json")

        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest_dict, f, ensure_ascii=False, indent=2, default=str)

        # Scene별 JSON 저장
        for scene in manifest.scenes:
            scene_path = f"{project_dir}/scenes/scene_{scene.scene_id:02d}.json"
            scene_dict = scene.model_dump(mode="json")
            with open(scene_path, "w", encoding="utf-8") as f:
                json.dump(scene_dict, f, ensure_ascii=False, indent=2)

        return manifest_path

    def _save_scene_json(self, scene: Scene, project_dir: str) -> str:
        """
        개별 Scene JSON 저장.

        Args:
            scene: Scene 객체
            project_dir: 프로젝트 디렉토리

        Returns:
            저장된 파일 경로
        """
        scene_path = f"{project_dir}/scenes/scene_{scene.scene_id:02d}.json"
        scene_dict = scene.model_dump(mode="json")

        with open(scene_path, "w", encoding="utf-8") as f:
            json.dump(scene_dict, f, ensure_ascii=False, indent=2)

        return scene_path

    def _apply_film_look(
        self,
        input_video: str,
        project_dir: str,
        grain_intensity: int = 10,
        saturation: float = 1.1,
        contrast: float = 1.05
    ) -> str:
        """
        필름 룩 후처리 적용 (v2.0).

        Args:
            input_video: 입력 영상 경로
            project_dir: 프로젝트 디렉토리
            grain_intensity: 그레인 강도 (0-30)
            saturation: 채도 (1.0 = 원본)
            contrast: 대비 (1.0 = 원본)

        Returns:
            필름 룩이 적용된 영상 경로
        """
        composer = FFmpegComposer()

        output_path = f"{project_dir}/final_video_film_look.mp4"

        try:
            result = composer.apply_film_look(
                video_in=input_video,
                out_path=output_path,
                grain_intensity=grain_intensity,
                saturation=saturation,
                contrast=contrast
            )
            print(f"  Film look applied: {result}")
            return result
        except Exception as e:
            print(f"  [Warning] Film look failed: {e}. Using original video.")
            return input_video


def run_pipeline(
    topic: str = None,
    genre: str = "emotional",
    mood: str = "dramatic",
    style: str = "cinematic, high contrast",
    duration: int = 60,
    feature_flags: Dict[str, bool] = None
) -> Manifest:
    """
    파이프라인 간편 실행 함수.

    Args:
        topic: 영상 주제
        genre: 장르
        mood: 분위기
        style: 영상 스타일
        duration: 목표 영상 길이 (초)
        feature_flags: Feature flags 딕셔너리

    Returns:
        Manifest 객체
    """
    # Feature flags 설정
    ff = FeatureFlags()
    if feature_flags:
        for key, value in feature_flags.items():
            if hasattr(ff, key):
                setattr(ff, key, value)

    # ProjectRequest 생성
    request = ProjectRequest(
        topic=topic,
        genre=genre,
        mood=mood,
        style_preset=style,
        duration_target_sec=duration,
        feature_flags=ff,
    )

    # 파이프라인 실행
    pipeline = StorycutPipeline()
    return pipeline.run(request)


if __name__ == "__main__":
    # 테스트 실행
    manifest = run_pipeline(
        topic="오래된 폐병원에서 발견된 미스터리한 일기장",
        genre="mystery",
        mood="suspenseful",
        duration=60,
        feature_flags={
            "hook_scene1_video": False,  # 비용 절감을 위해 OFF
            "ffmpeg_kenburns": True,
            "context_carry_over": True,
            "optimization_pack": True,
        }
    )

    print(f"\nFinal video: {manifest.outputs.final_video_path}")
    print(f"Title candidates: {manifest.outputs.title_candidates}")
