"""
Scene Orchestrator: Manages scene-by-scene processing with context carry-over.

P1 핵심 기능:
- 이전 장면의 핵심 키워드(인물/장소/감정/행동)를 다음 장면 프롬프트에 상속
- Scene 간 일관성 유지
"""

import os
import json
from typing import Dict, Any, List, Optional
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))

from agents.video_agent import VideoAgent
from agents.tts_agent import TTSAgent
from agents.music_agent import MusicAgent
from agents.composer_agent import ComposerAgent
from schemas import FeatureFlags, Scene, SceneEntities, ProjectRequest


class SceneOrchestrator:
    """
    Scene 단위 처리 오케스트레이터

    P1 핵심: Context Carry-over (맥락 상속)
    - 이전 장면의 핵심 키워드를 다음 장면 프롬프트에 강제 포함
    - 인물/장소/감정/행동 일관성 유지
    """

    def __init__(self, feature_flags: FeatureFlags = None):
        """
        Initialize Scene Orchestrator with all sub-agents.

        Args:
            feature_flags: Feature flags configuration
        """
        self.feature_flags = feature_flags or FeatureFlags()
        self.video_agent = VideoAgent(feature_flags=self.feature_flags)
        self.tts_agent = TTSAgent()
        self.music_agent = MusicAgent()
        self.composer_agent = ComposerAgent()

        # LLM 클라이언트 (맥락 추출용)
        self._llm_client = None

    @property
    def llm_client(self):
        """Lazy initialization of LLM client."""
        if self._llm_client is None:
            try:
                from openai import OpenAI
                self._llm_client = OpenAI()
            except Exception:
                self._llm_client = None
        return self._llm_client

    # =========================================================================
    # P1: Context Carry-over (맥락 상속)
    # =========================================================================

    def extract_entities(
        self,
        sentence: str,
        inherited_keywords: List[str] = None
    ) -> SceneEntities:
        """
        문장에서 엔티티(인물/장소/감정/행동) 추출.

        P1: 맥락 상속을 위한 엔티티 추출

        Args:
            sentence: 장면 문장
            inherited_keywords: 이전 장면에서 상속받은 키워드

        Returns:
            SceneEntities 객체
        """
        if not self.llm_client:
            # LLM 없으면 기본 엔티티 반환
            return SceneEntities(
                characters=[],
                location=None,
                props=[],
                mood=None,
                action=None
            )

        inherited_context = ", ".join(inherited_keywords) if inherited_keywords else "없음"

        prompt = f"""
다음 문장에서 핵심 엔티티를 추출하세요:

문장: {sentence}
이전 장면 맥락: {inherited_context}

JSON 형식으로 출력:
{{
    "characters": ["인물1", "인물2"],
    "location": "장소",
    "props": ["소품1", "소품2"],
    "mood": "분위기/감정",
    "action": "주요 행동"
}}

주의:
- 이전 장면 맥락과 일관성을 유지하세요
- 뜬금없는 인물/장소 변경 감지 시 이전 맥락 우선
- 명시적 언급이 없으면 null 사용
"""

        try:
            response = self.llm_client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "JSON만 출력하세요. 다른 설명 없이."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=300
            )

            content = response.choices[0].message.content.strip()
            # JSON 파싱
            if content.startswith("```"):
                content = content.split("```")[1]
                if content.startswith("json"):
                    content = content[4:]

            data = json.loads(content)
            return SceneEntities(**data)

        except Exception as e:
            print(f"  Entity extraction failed: {e}")
            return SceneEntities()

    def summarize_prev_scene(self, scene: Scene) -> str:
        """
        이전 장면 요약 생성.

        Args:
            scene: 이전 Scene 객체

        Returns:
            요약 문자열
        """
        parts = []

        if scene.entities.characters:
            parts.append(f"인물: {', '.join(scene.entities.characters)}")
        if scene.entities.location:
            parts.append(f"장소: {scene.entities.location}")
        if scene.entities.mood:
            parts.append(f"분위기: {scene.entities.mood}")
        if scene.entities.action:
            parts.append(f"행동: {scene.entities.action}")

        return " / ".join(parts) if parts else scene.sentence[:50]

    def extract_key_terms(self, scene: Scene) -> List[str]:
        """
        이전 장면에서 핵심 키워드 추출.

        P1: 다음 장면 프롬프트에 상속할 키워드

        Args:
            scene: 이전 Scene 객체

        Returns:
            키워드 목록
        """
        keywords = []

        # 엔티티에서 키워드 추출
        if scene.entities.characters:
            keywords.extend(scene.entities.characters[:2])  # 최대 2명
        if scene.entities.location:
            keywords.append(scene.entities.location)
        if scene.entities.mood:
            keywords.append(scene.entities.mood)
        if scene.entities.action:
            keywords.append(scene.entities.action)

        return keywords[:5]  # 최대 5개 키워드

    def build_prompt(
        self,
        sentence: str,
        inherited: List[str],
        entities: SceneEntities,
        style: str = None
    ) -> str:
        """
        영상 생성 프롬프트 구성.

        P1: inherited 키워드는 반드시 포함

        Args:
            sentence: 장면 문장
            inherited: 이전 장면에서 상속받은 키워드
            entities: 장면 엔티티
            style: 영상 스타일

        Returns:
            영상 생성 프롬프트
        """
        style = style or "cinematic, high contrast, webtoon-like"
        inherited_str = ", ".join(inherited) if inherited else "none"

        # 엔티티를 문자열로 변환
        entities_parts = []
        if entities.characters:
            entities_parts.append(f"Characters: {', '.join(entities.characters)}")
        if entities.location:
            entities_parts.append(f"Location: {entities.location}")
        if entities.props:
            entities_parts.append(f"Props: {', '.join(entities.props)}")
        if entities.mood:
            entities_parts.append(f"Mood: {entities.mood}")
        if entities.action:
            entities_parts.append(f"Action: {entities.action}")

        entities_str = " | ".join(entities_parts) if entities_parts else "N/A"

        prompt = f"""[STYLE] {style}
[INHERITED CONTEXT] {inherited_str}
[SCENE SENTENCE] {sentence}
[ENTITIES] {entities_str}
[RULES]
- 이전 장면과 동일 인물/공간/톤을 유지한다.
- 뜬금없는 배경/소품 변경 금지.
- 감정은 과장하되 개연성 유지."""

        return prompt

    def build_negative_prompt(self, style: str = None) -> str:
        """
        네거티브 프롬프트 생성.

        Args:
            style: 영상 스타일

        Returns:
            네거티브 프롬프트
        """
        return (
            "blurry, low quality, distorted, disfigured, "
            "watermark, text, logo, bad anatomy, extra limbs, "
            "inconsistent characters, different faces"
        )

    # =========================================================================
    # 메인 처리 로직
    # =========================================================================

    def process_story(
        self,
        story_data: Dict[str, Any],
        output_path: str = "output/youtube_ready.mp4",
        request: ProjectRequest = None,
        progress_callback: Any = None
    ) -> str:
        """
        Scene JSON에서 최종 영상까지 전체 처리.

        P1: 맥락 상속 적용

        Args:
            story_data: Story JSON (scenes 포함)
            output_path: 최종 영상 출력 경로
            request: ProjectRequest (feature flags 포함)

        Returns:
            최종 영상 파일 경로
        """
        print(f"\n{'='*60}")
        print(f"STORYCUT - Processing Story: {story_data['title']}")
        print(f"{'='*60}\n")

        # Feature flags 업데이트
        if request:
            self.feature_flags = request.feature_flags
            self.video_agent.feature_flags = request.feature_flags

        scenes = story_data["scenes"]
        total_scenes = len(scenes)
        style = story_data.get("style", request.style_preset if request else "cinematic")

        print(f"Total scenes: {total_scenes}")
        print(f"Target duration: {story_data['total_duration_sec']} seconds")
        print(f"Context carry-over: {'ON' if self.feature_flags.context_carry_over else 'OFF'}\n")

        # Scene 처리
        video_clips = []
        narration_clips = []
        processed_scenes = []
        prev_scene = None

        for i, scene_data in enumerate(scenes, 1):
            print(f"{'─'*60}")
            print(f"Processing Scene {i}/{total_scenes} (ID: {scene_data['scene_id']})")
            print(f"{'─'*60}")

            # Scene 객체 생성
            scene = Scene(
                index=i,
                scene_id=scene_data["scene_id"],
                sentence=scene_data.get("narration", ""),
                narration=scene_data.get("narration"),
                visual_description=scene_data.get("visual_description"),
                mood=scene_data.get("mood"),
                duration_sec=scene_data.get("duration_sec", 5),
            )

            # P1: Context Carry-over
            if self.feature_flags.context_carry_over and prev_scene:
                scene.context_summary = self.summarize_prev_scene(prev_scene)
                scene.inherited_keywords = self.extract_key_terms(prev_scene)
                print(f"  [CONTEXT] Inherited: {scene.inherited_keywords}")
            else:
                scene.inherited_keywords = []

            # 엔티티 추출
            scene.entities = self.extract_entities(
                scene.sentence,
                scene.inherited_keywords
            )

            # 프롬프트 생성
            scene.prompt = self.build_prompt(
                sentence=scene.sentence,
                inherited=scene.inherited_keywords,
                entities=scene.entities,
                style=style
            )
            scene.negative_prompt = self.build_negative_prompt(style)

            # 영상 생성
            video_path = self.video_agent.generate_video(
                scene_id=scene.scene_id,
                visual_description=scene.visual_description or scene.prompt,
                style=style,
                mood=scene.mood,
                duration_sec=scene.duration_sec,
                scene=scene
            )
            video_clips.append(video_path)
            scene.assets.video_path = video_path

            # 내레이션 생성
            audio_path = self.tts_agent.generate_speech(
                scene_id=scene.scene_id,
                narration=scene.narration,
                emotion=scene.mood
            )
            narration_clips.append(audio_path)
            scene.assets.narration_path = audio_path

            processed_scenes.append(scene)
            prev_scene = scene

            print(f"Scene {i} complete\n")

            if progress_callback:
                import asyncio
                if asyncio.iscoroutinefunction(progress_callback):
                    # 만약 비동기 콜백이라면 (run_in_executor 등으로 처리 필요할 수 있음)
                    # 현재 구조는 동기 함수이므로, 비동기 호출이 어렵다면 
                    # 호출 측에서 래핑하거나 여기서 event_loop를 써야 함.
                    # 간단히는, 호출 측에서 동기 래퍼를 주거나 여기서 await하지 않고 fire-and-forget
                    pass
                else:
                    # Sync callback
                    progress_callback(scene, i)

        # 배경 음악 선택
        print(f"{'─'*60}")
        music_path = self.music_agent.select_music(
            genre=story_data["genre"],
            mood=story_data.get("mood", "neutral"),
            duration_sec=story_data["total_duration_sec"]
        )
        print(f"{'─'*60}\n")

        # 최종 영상 합성
        final_video = self.composer_agent.compose_video(
            video_clips=video_clips,
            narration_clips=narration_clips,
            music_path=music_path,
            output_path=output_path
        )

        print(f"\n{'='*60}")
        print(f"SUCCESS! Video ready for YouTube upload")
        print(f"{'='*60}")
        print(f"File: {os.path.abspath(final_video)}\n")

        return final_video

    def process_scenes_from_script(
        self,
        script_text: str,
        request: ProjectRequest
    ) -> List[Scene]:
        """
        스크립트 텍스트에서 Scene 목록 생성.

        P1: 맥락 상속 적용

        Args:
            script_text: 전체 스크립트 텍스트
            request: ProjectRequest (feature flags 포함)

        Returns:
            Scene 객체 목록
        """
        # 문장 단위 분할
        sentences = self._split_into_sentences(script_text)
        scenes = []
        prev_scene = None

        for idx, sentence in enumerate(sentences, start=1):
            scene = Scene(
                index=idx,
                scene_id=idx,
                sentence=sentence,
                narration=sentence,
            )

            # P1: Context Carry-over
            if request.feature_flags.context_carry_over and prev_scene:
                scene.context_summary = self.summarize_prev_scene(prev_scene)
                scene.inherited_keywords = self.extract_key_terms(prev_scene)
            else:
                scene.inherited_keywords = []

            # 엔티티 추출
            scene.entities = self.extract_entities(
                sentence,
                scene.inherited_keywords
            )

            # 프롬프트 생성
            scene.prompt = self.build_prompt(
                sentence=sentence,
                inherited=scene.inherited_keywords,
                entities=scene.entities,
                style=request.style_preset
            )
            scene.negative_prompt = self.build_negative_prompt(request.style_preset)

            scenes.append(scene)
            prev_scene = scene

        return scenes

    def _split_into_sentences(self, text: str) -> List[str]:
        """
        텍스트를 문장 단위로 분할.

        Args:
            text: 전체 텍스트

        Returns:
            문장 목록
        """
        import re

        # 한국어 및 영어 문장 분할
        # 마침표, 물음표, 느낌표 기준
        sentences = re.split(r'(?<=[.!?])\s+', text.strip())

        # 빈 문장 제거 및 정리
        sentences = [s.strip() for s in sentences if s.strip()]

        # 너무 긴 문장은 분할
        result = []
        for s in sentences:
            if len(s) > 100:
                # 쉼표나 연결어 기준으로 추가 분할
                parts = re.split(r'(?<=,)\s+|(?<=그리고)\s+|(?<=하지만)\s+', s)
                result.extend([p.strip() for p in parts if p.strip()])
            else:
                result.append(s)

        return result

    def retry_scene(
        self,
        scene: Dict[str, Any],
        story_style: str = "cinematic"
    ) -> tuple[str, str]:
        """
        단일 Scene 재처리.

        Args:
            scene: Scene 데이터
            story_style: 영상 스타일

        Returns:
            (video_path, audio_path) 튜플
        """
        print(f"Retrying scene {scene['scene_id']}...")

        video_path = self.video_agent.generate_video(
            scene_id=scene["scene_id"],
            visual_description=scene["visual_description"],
            style=story_style,
            mood=scene["mood"],
            duration_sec=scene["duration_sec"]
        )

        audio_path = self.tts_agent.generate_speech(
            scene_id=scene["scene_id"],
            narration=scene["narration"],
            emotion=scene["mood"]
        )

        return video_path, audio_path

    def generate_subtitle_files(
        self,
        scenes: List[Scene],
        output_dir: str = "media/subtitles"
    ) -> List[str]:
        """
        각 Scene에 대한 SRT 자막 파일 생성.

        Args:
            scenes: Scene 목록
            output_dir: 출력 디렉토리

        Returns:
            SRT 파일 경로 목록
        """
        from utils.ffmpeg_utils import FFmpegComposer

        os.makedirs(output_dir, exist_ok=True)
        composer = FFmpegComposer()

        srt_paths = []

        for scene in scenes:
            srt_path = f"{output_dir}/scene_{scene.scene_id:02d}.srt"

            # 단일 Scene용 SRT 생성
            scene_data = [{
                "narration": scene.narration or scene.sentence,
                "duration_sec": scene.duration_sec
            }]

            composer.generate_srt_from_scenes(scene_data, srt_path)
            scene.assets.subtitle_srt_path = srt_path
            srt_paths.append(srt_path)

        return srt_paths

    def get_processing_stats(
        self,
        scenes: List[Scene]
    ) -> Dict[str, Any]:
        """
        처리 통계 반환.

        Args:
            scenes: 처리된 Scene 목록

        Returns:
            통계 딕셔너리
        """
        video_methods = {}
        for scene in scenes:
            method = scene.generation_method or "unknown"
            video_methods[method] = video_methods.get(method, 0) + 1

        return {
            "total_scenes": len(scenes),
            "video_generation_methods": video_methods,
            "context_carry_over_enabled": self.feature_flags.context_carry_over,
            "hook_scene_video_enabled": self.feature_flags.hook_scene1_video,
        }
