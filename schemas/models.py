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
from typing import Optional, List, Dict, Any, Tuple
from pydantic import BaseModel, Field
import uuid


class TargetPlatform(str, Enum):
    """대상 플랫폼"""
    YOUTUBE_LONG = "youtube_long"
    YOUTUBE_SHORTS = "youtube_shorts"


class SceneStatus(str, Enum):
    """씬 처리 상태"""
    PENDING = "pending"
    GENERATING_IMAGE = "generating_image"
    GENERATING_TTS = "generating_tts"
    GENERATING_VIDEO = "generating_video"
    COMPOSING = "composing"
    COMPLETED = "completed"
    FAILED = "failed"


class CameraWork(str, Enum):
    """카메라 워크 (Ken Burns 효과 다양화)"""
    ZOOM_IN = "zoom_in"
    ZOOM_OUT = "zoom_out"
    PAN_LEFT = "pan_left"
    PAN_RIGHT = "pan_right"
    PAN_UP = "pan_up"
    PAN_DOWN = "pan_down"
    STATIC = "static"


class PoseType(str, Enum):
    """캐릭터 포즈 유형"""
    FRONT = "front"
    THREE_QUARTER = "three_quarter"
    SIDE = "side"
    FULL_BODY = "full_body"
    EMOTION_NEUTRAL = "emotion_neutral"
    EMOTION_INTENSE = "emotion_intense"


class PoseAnchor(BaseModel):
    """포즈별 앵커 이미지"""
    pose: PoseType
    image_path: str
    score: float = Field(default=0.0, description="품질 점수 (0~1)")


class AnchorSet(BaseModel):
    """캐릭터의 멀티포즈 앵커 세트"""
    character_token: str
    poses: Dict[str, PoseAnchor] = Field(default_factory=dict)
    best_pose: str = "three_quarter"

    def get_pose_image(self, pose: str) -> Optional[str]:
        """pose에 해당하는 이미지 경로 반환, 없으면 best_pose 폴백."""
        anchor = self.poses.get(pose)
        if anchor:
            return anchor.image_path
        fallback = self.poses.get(self.best_pose)
        if fallback:
            return fallback.image_path
        # 아무 포즈라도 반환
        if self.poses:
            return next(iter(self.poses.values())).image_path
        return None


class ValidationResult(BaseModel):
    """일관성 검증 결과"""
    scene_id: int
    passed: bool
    overall_score: float = 0.0
    dimension_scores: Dict[str, float] = Field(default_factory=dict)
    issues: List[str] = Field(default_factory=list)
    attempt_number: int = 1


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

    # v2.0: Film Look
    film_look: bool = Field(
        default=False,
        description="필름 그레인 + 색보정 후처리 (시네마틱 룩)"
    )

    # v2.0: Consistency Validation
    consistency_validation: bool = Field(
        default=False,
        description="Gemini Vision 기반 일관성 검증 활성화"
    )
    consistency_max_retries: int = Field(
        default=3,
        description="일관성 검증 실패 시 최대 재시도 횟수"
    )


class ProjectRequest(BaseModel):
    """프로젝트 요청 정보"""
    model_config = {"populate_by_name": True}  # voice -> voice_id alias 지원

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
    voice_id: str = Field(
        default="uyVNoMrnUku1dZyVEXwD",
        alias="voice",
        description="ElevenLabs voice ID (default: Adam)"
    )
    duration_target_sec: Optional[int] = Field(
        default=60,
        description="목표 영상 길이 (초)"
    )
    image_model: str = Field(
        default="standard",
        description="이미지 생성 모델 (standard/premium)"
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


class CharacterSheet(BaseModel):
    """캐릭터 시트 (시각적 일관성 유지용)"""
    name: str = Field(..., description="캐릭터 이름")
    gender: str = Field(default="unknown", description="성별")
    age: str = Field(default="unknown", description="나이대")
    appearance: str = Field(..., description="외형 상세 (머리, 눈, 피부, 특징)")
    clothing_default: str = Field(default="", description="기본 의상")
    emotion_range: List[str] = Field(default_factory=list, description="감정 범위")
    visual_seed: int = Field(default=42, description="이미지 생성 시드 (일관성용)")

    # 마스터 캐릭터 이미지 (참조용)
    master_image_path: Optional[str] = Field(
        default=None,
        description="마스터 캐릭터 이미지 경로 (로컬)"
    )
    master_image_id: Optional[str] = Field(
        default=None,
        description="마스터 캐릭터 이미지 ID (NanoBanana 등 API 참조용)"
    )
    master_image_url: Optional[str] = Field(
        default=None,
        description="마스터 캐릭터 이미지 URL (외부 접근용)"
    )

    # v2.0: 멀티포즈 앵커 세트
    anchor_set: Optional[AnchorSet] = Field(
        default=None,
        description="멀티포즈 앵커 이미지 세트"
    )


class GlobalStyle(BaseModel):
    """글로벌 스타일 설정"""
    art_style: str = Field(
        default="cinematic animation, high contrast, dramatic lighting",
        description="아트 스타일"
    )
    color_palette: str = Field(
        default="desaturated blues and warm amber highlights",
        description="색상 팔레트"
    )
    aspect_ratio: str = Field(default="16:9", description="화면 비율")
    visual_seed: int = Field(default=12345, description="전체 프로젝트 시드")

    # v2.0: 스타일/환경 앵커
    style_anchor_path: Optional[str] = Field(
        default=None,
        description="프로젝트 전체 룩 앵커 이미지 경로"
    )
    environment_anchors: Dict[int, str] = Field(
        default_factory=dict,
        description="씬별 환경 앵커 이미지 경로 (scene_id -> path)"
    )


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

    # v2.0 새 필드 (캐릭터 참조 시스템)
    narrative: Optional[str] = Field(default=None, description="내부 참조용 장면 설명")
    image_prompt: Optional[str] = Field(default=None, description="이미지 생성 전용 프롬프트")
    characters_in_scene: List[str] = Field(
        default_factory=list,
        description="이 장면에 등장하는 캐릭터 토큰 목록"
    )

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

    # 씬 상태 관리 (Phase 1)
    status: SceneStatus = Field(
        default=SceneStatus.PENDING,
        description="씬 처리 상태"
    )
    error_message: Optional[str] = Field(
        default=None,
        description="에러 메시지 (실패 시)"
    )
    retry_count: int = Field(
        default=0,
        description="재시도 횟수"
    )

    # TTS 기반 duration (Phase 1)
    tts_duration_sec: Optional[float] = Field(
        default=None,
        description="TTS 실제 오디오 길이 (초)"
    )

    # 카메라 워크 (Phase 1)
    camera_work: CameraWork = Field(
        default=CameraWork.ZOOM_IN,
        description="Ken Burns 효과 방향"
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

    # v2.0 캐릭터 참조 시스템
    global_style: Optional[GlobalStyle] = Field(
        default=None,
        description="글로벌 스타일 설정"
    )
    character_sheet: Dict[str, CharacterSheet] = Field(
        default_factory=dict,
        description="캐릭터 시트 (토큰별)"
    )

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




class GenerateVideoRequest(BaseModel):
    """영상 생성 요청 (확정된 스토리 포함)"""
    project_id: Optional[str] = Field(default=None, description="프로젝트 ID (옵션)")
    request_params: ProjectRequest = Field(..., description="초기 요청 파라미터")
    story_data: Dict[str, Any] = Field(..., description="확정된 스토리 데이터")
