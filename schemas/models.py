"""
STORYCUT Data Models

공통 데이터 모델 정의 (Pydantic 기반)
- FeatureFlags: 기능 플래그 (P0)
- ProjectRequest: 프로젝트 요청 정보
- Scene: 장면 데이터
- Manifest: 프로젝트 메타데이터
"""

from datetime import datetime
from enum import Enum
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field
import uuid


class TargetPlatform(str, Enum):
    """대상 플랫폼"""
    YOUTUBE_LONG = "youtube_long"
    YOUTUBE_SHORTS = "youtube_shorts"


class FeatureFlags(BaseModel):
    """
    기능 플래그 설정

    기본값 OFF → 안정화 후 ON 전환 가능
    """
    # P0: Scene 1 Hook 전용 고품질 비디오 생성
    hook_scene1_video: bool = Field(
        default=False,
        description="Scene 1에서 고품질 비디오 생성 강제 (비용 주의)"
    )

    # P0: FFmpeg Ken Burns Effect
    ffmpeg_kenburns: bool = Field(
        default=True,
        description="이미지 기반 씬에 Ken Burns(줌/팬) 효과 적용"
    )

    # P0: Audio Ducking
    ffmpeg_audio_ducking: bool = Field(
        default=False,
        description="내레이션 존재 시 BGM 자동 감쇠"
    )

    # P0: Subtitle Burn-in
    subtitle_burn_in: bool = Field(
        default=True,
        description="자막을 영상에 burn-in 처리"
    )

    # P1: Context Carry-over
    context_carry_over: bool = Field(
        default=True,
        description="이전 장면 키워드를 다음 장면 프롬프트에 상속"
    )

    # P2: Optimization Pack
    optimization_pack: bool = Field(
        default=True,
        description="제목/썸네일/AB 테스트 메타데이터 생성"
    )

    # Optional: Topic Finding
    topic_finding: bool = Field(
        default=False,
        description="트렌드 기반 주제 후보 생성 (외부 API 필요)"
    )


class ProjectRequest(BaseModel):
    """프로젝트 요청 정보"""
    topic: Optional[str] = Field(default=None, description="영상 주제")
    style_preset: Optional[str] = Field(
        default="cinematic, high contrast",
        description="영상 스타일 프리셋"
    )
    target_platform: TargetPlatform = Field(
        default=TargetPlatform.YOUTUBE_LONG,
        description="대상 플랫폼"
    )
    language: str = Field(default="ko", description="언어 코드")
    duration_target_sec: Optional[int] = Field(
        default=60,
        description="목표 영상 길이 (초)"
    )

    # 콘텐츠 옵션
    voice_over: bool = Field(default=True, description="내레이션 포함 여부")
    bgm: bool = Field(default=True, description="배경 음악 포함 여부")
    subtitles: bool = Field(default=True, description="자막 포함 여부")

    # 장르/무드
    genre: Optional[str] = Field(default="emotional", description="장르")
    mood: Optional[str] = Field(default="dramatic", description="분위기")
    user_idea: Optional[str] = Field(default=None, description="사용자 아이디어")

    # Feature Flags
    feature_flags: FeatureFlags = Field(
        default_factory=FeatureFlags,
        description="기능 플래그"
    )


class SceneAssets(BaseModel):
    """장면 에셋 경로"""
    image_path: Optional[str] = Field(default=None, description="생성된 이미지 경로")
    video_path: Optional[str] = Field(default=None, description="생성된 비디오 경로")
    narration_path: Optional[str] = Field(default=None, description="내레이션 오디오 경로")
    subtitle_srt_path: Optional[str] = Field(default=None, description="자막 SRT 파일 경로")
    bgm_path: Optional[str] = Field(default=None, description="배경 음악 경로")


class SceneTiming(BaseModel):
    """장면 타이밍 정보"""
    start_ms: int = Field(default=0, description="시작 시간 (밀리초)")
    end_ms: int = Field(default=0, description="종료 시간 (밀리초)")
    duration_ms: int = Field(default=0, description="지속 시간 (밀리초)")


class SceneEntities(BaseModel):
    """장면 엔티티 정보 (맥락 상속용)"""
    characters: List[str] = Field(default_factory=list, description="등장 인물")
    location: Optional[str] = Field(default=None, description="장소")
    props: List[str] = Field(default_factory=list, description="소품")
    mood: Optional[str] = Field(default=None, description="분위기/감정")
    action: Optional[str] = Field(default=None, description="행동/동작")


class Scene(BaseModel):
    """
    장면 데이터

    각 Scene은 독립적으로 처리 가능 (장애 격리)
    """
    index: int = Field(..., description="장면 인덱스 (1부터 시작)")
    scene_id: int = Field(..., description="장면 ID (레거시 호환)")
    sentence: str = Field(..., description="장면 대사/내레이션")

    # 맥락 상속 (P1)
    context_summary: Optional[str] = Field(
        default=None,
        description="이전 씬 요약 (상속용)"
    )
    inherited_keywords: List[str] = Field(
        default_factory=list,
        description="이전 씬에서 상속받은 키워드"
    )
    entities: SceneEntities = Field(
        default_factory=SceneEntities,
        description="장면 엔티티 정보"
    )

    # 프롬프트
    prompt: str = Field(default="", description="영상/이미지 생성 프롬프트")
    negative_prompt: Optional[str] = Field(
        default=None,
        description="네거티브 프롬프트"
    )

    # 레거시 호환 필드
    narration: Optional[str] = Field(default=None, description="내레이션 텍스트")
    visual_description: Optional[str] = Field(default=None, description="시각적 설명")
    mood: Optional[str] = Field(default=None, description="분위기")
    duration_sec: int = Field(default=5, description="지속 시간 (초)")

    # 에셋 및 타이밍
    assets: SceneAssets = Field(
        default_factory=SceneAssets,
        description="장면 에셋"
    )
    timing: SceneTiming = Field(
        default_factory=SceneTiming,
        description="타이밍 정보"
    )

    # 메타데이터
    is_hook: bool = Field(
        default=False,
        description="Hook 씬 여부 (Scene 1)"
    )
    generation_method: Optional[str] = Field(
        default=None,
        description="생성 방식 (video/image+kenburns)"
    )

    def model_post_init(self, __context):
        """초기화 후 처리"""
        # scene_id와 index 동기화
        if self.scene_id is None:
            self.scene_id = self.index
        # Hook 씬 설정
        if self.index == 1:
            self.is_hook = True
        # sentence와 narration 동기화
        if self.narration and not self.sentence:
            self.sentence = self.narration
        elif self.sentence and not self.narration:
            self.narration = self.sentence


class ManifestOutputs(BaseModel):
    """Manifest 출력 정보"""
    final_video_path: Optional[str] = Field(default=None, description="최종 영상 경로")
    shorts_video_path: Optional[str] = Field(default=None, description="쇼츠 영상 경로")

    # Optimization Agent 출력 (P2)
    title_candidates: List[str] = Field(
        default_factory=list,
        description="제목 후보 (3종)"
    )
    thumbnail_prompts: List[str] = Field(
        default_factory=list,
        description="썸네일 이미지 프롬프트 (2종)"
    )
    thumbnail_texts: List[str] = Field(
        default_factory=list,
        description="썸네일 문구 (3종)"
    )
    hashtags: List[str] = Field(
        default_factory=list,
        description="해시태그 (10개)"
    )
    description: Optional[str] = Field(default=None, description="영상 설명문")
    metadata_json_path: Optional[str] = Field(
        default=None,
        description="메타데이터 JSON 경로"
    )
    ab_test_meta: Optional[Dict[str, Any]] = Field(
        default=None,
        description="AB 테스트 메타데이터"
    )

    # Topic Finding 출력 (Optional)
    topic_candidates: List[str] = Field(
        default_factory=list,
        description="주제 후보 (topic_finding 활성화 시)"
    )


class CostEstimate(BaseModel):
    """비용 추정 정보"""
    llm_tokens: int = Field(default=0, description="LLM 토큰 사용량")
    video_seconds: int = Field(default=0, description="비디오 생성 시간 (초)")
    image_count: int = Field(default=0, description="이미지 생성 개수")
    tts_characters: int = Field(default=0, description="TTS 문자 수")
    estimated_usd: float = Field(default=0.0, description="예상 비용 (USD)")


class Manifest(BaseModel):
    """
    프로젝트 Manifest

    입력/모델/비용추정/씬별 산출물 경로/실행 시간/에러 로그 경로
    """
    project_id: str = Field(
        default_factory=lambda: str(uuid.uuid4())[:8],
        description="프로젝트 ID"
    )
    created_at: datetime = Field(
        default_factory=datetime.now,
        description="생성 시간"
    )

    # 입력 정보
    input: ProjectRequest = Field(
        default_factory=ProjectRequest,
        description="프로젝트 요청 정보"
    )

    # 스토리 정보
    title: Optional[str] = Field(default=None, description="영상 제목")
    script: Optional[str] = Field(default=None, description="전체 스크립트")

    # 장면 목록
    scenes: List[Scene] = Field(
        default_factory=list,
        description="장면 목록"
    )

    # 출력 정보
    outputs: ManifestOutputs = Field(
        default_factory=ManifestOutputs,
        description="출력 정보"
    )

    # 비용 추정
    cost_estimate: CostEstimate = Field(
        default_factory=CostEstimate,
        description="비용 추정"
    )

    # 로그 정보
    logs: Dict[str, str] = Field(
        default_factory=dict,
        description="로그 파일 경로"
    )

    # 실행 정보
    execution_time_sec: Optional[float] = Field(
        default=None,
        description="실행 시간 (초)"
    )
    status: str = Field(
        default="pending",
        description="상태 (pending/processing/completed/failed)"
    )
    error_message: Optional[str] = Field(
        default=None,
        description="에러 메시지"
    )

    def to_legacy_story_data(self) -> Dict[str, Any]:
        """기존 story_data 형식으로 변환 (하위 호환성)"""
        return {
            "title": self.title or self.input.topic or "Untitled",
            "genre": self.input.genre,
            "mood": self.input.mood,
            "style": self.input.style_preset,
            "total_duration_sec": self.input.duration_target_sec,
            "scenes": [
                {
                    "scene_id": scene.scene_id,
                    "narration": scene.narration or scene.sentence,
                    "visual_description": scene.visual_description or scene.prompt,
                    "mood": scene.mood or scene.entities.mood,
                    "duration_sec": scene.duration_sec,
                }
                for scene in self.scenes
            ]
        }

    @classmethod
    def from_legacy_story_data(
        cls,
        story_data: Dict[str, Any],
        request: Optional[ProjectRequest] = None
    ) -> "Manifest":
        """기존 story_data에서 Manifest 생성"""
        if request is None:
            request = ProjectRequest(
                topic=story_data.get("title"),
                genre=story_data.get("genre"),
                mood=story_data.get("mood"),
                style_preset=story_data.get("style"),
                duration_target_sec=story_data.get("total_duration_sec", 60),
            )

        scenes = []
        for idx, scene_data in enumerate(story_data.get("scenes", []), start=1):
            scene = Scene(
                index=idx,
                scene_id=scene_data.get("scene_id", idx),
                sentence=scene_data.get("narration", ""),
                narration=scene_data.get("narration"),
                visual_description=scene_data.get("visual_description"),
                prompt=scene_data.get("visual_description", ""),
                mood=scene_data.get("mood"),
                duration_sec=scene_data.get("duration_sec", 5),
            )
            scenes.append(scene)

        return cls(
            title=story_data.get("title"),
            input=request,
            scenes=scenes,
        )
