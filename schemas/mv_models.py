"""
Music Video Mode - Data Models

뮤직비디오 모드용 Pydantic 스키마 정의
"""

from enum import Enum
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from datetime import datetime


class MVGenre(str, Enum):
    """뮤직비디오 장르"""
    FANTASY = "fantasy"
    ROMANCE = "romance"
    ACTION = "action"
    HORROR = "horror"
    SCIFI = "scifi"
    DRAMA = "drama"
    COMEDY = "comedy"
    ABSTRACT = "abstract"


class MVMood(str, Enum):
    """뮤직비디오 분위기"""
    EPIC = "epic"
    DREAMY = "dreamy"
    ENERGETIC = "energetic"
    CALM = "calm"
    DARK = "dark"
    ROMANTIC = "romantic"
    MELANCHOLIC = "melancholic"
    UPLIFTING = "uplifting"


class MVStyle(str, Enum):
    """비주얼 스타일"""
    CINEMATIC = "cinematic"
    ANIME = "anime"
    WEBTOON = "webtoon"
    REALISTIC = "realistic"
    ILLUSTRATION = "illustration"
    ABSTRACT = "abstract"
    HOYOVERSE = "hoyoverse"


class GenreProfile(BaseModel):
    """장르별 비주얼 프로필 (genre_profiles.yaml에서 로드)"""
    palette_base: List[str] = Field(default_factory=list)
    palette_mode: str = Field(default="guide")
    lighting: str = Field(default="")
    motif_library: List[str] = Field(default_factory=list)
    avoid_keywords: List[str] = Field(default_factory=list)
    atmosphere: str = Field(default="")
    composition_guide: str = Field(default="")
    shot_bias: Dict[str, float] = Field(default_factory=dict)
    arc_tone_sequence: List[str] = Field(default_factory=list)
    prompt_lexicon: Dict[str, str] = Field(default_factory=lambda: {"positive": "", "negative": ""})
    reference_artists: List[str] = Field(default_factory=list)


class MVCharacter(BaseModel):
    """MV 캐릭터 정의"""
    role: str = Field(..., description="역할명 (예: 주인공, 상대역)")
    description: str = Field(..., description="구체적 외형 -- 인종, 나이, 헤어, 얼굴, 체형")
    outfit: str = Field(default="", description="의상")
    appears_in: List[int] = Field(default_factory=list, description="등장 씬 ID 목록")
    anchor_image_path: Optional[str] = Field(None, description="앵커 이미지 경로")


class MVSceneBlocking(BaseModel):
    """씬별 연출 지시"""
    scene_id: int
    shot_type: str = Field(default="medium", description="wide/medium/close-up/extreme-close-up")
    narrative_beat: str = Field(default="", description="서사적 의미")
    characters: List[str] = Field(default_factory=list, description="등장 캐릭터 역할명")
    expression: Optional[str] = Field(None, description="표정/감정")
    lighting: Optional[str] = Field(None, description="씬별 조명")
    action_pose: Optional[str] = Field(None, description="캐릭터 동작/포즈 (은유적 가사→현실적 동작으로 해석)")
    visual_continuity: Optional[str] = Field(None, description="이전 씬과의 시각적 연결 방식 (same_location_time_change/same_character_state_change/cause_and_effect/motif_callback/contrast_cut/establishing)")


class MVNarrativeArc(BaseModel):
    """전체 서사 구조"""
    acts: List[Dict[str, str]] = Field(default_factory=list,
        description="[{'scenes': '1-4', 'description': '일상', 'tone': 'warm'}]")


class VisualBible(BaseModel):
    """뮤직비디오 비주얼 바이블 - LLM이 생성하는 전체 비주얼 가이드"""
    color_palette: List[str] = Field(default_factory=list, description="주요 색상 5개 (hex)")
    lighting_style: str = Field(default="", description="조명 스타일 (예: 'warm golden hour')")
    recurring_motifs: List[str] = Field(default_factory=list, description="반복 모티프 (예: 'falling petals')")
    character_archetypes: List[str] = Field(default_factory=list, description="캐릭터 아키타입")
    atmosphere: str = Field(default="", description="전체 분위기 서술")
    avoid_keywords: List[str] = Field(default_factory=list, description="회피할 비주얼 키워드")
    composition_notes: str = Field(default="", description="구도 가이드")
    reference_artists: List[str] = Field(default_factory=list, description="참고 아티스트/작품")

    # Director's Brief 확장
    characters: List[MVCharacter] = Field(default_factory=list)
    scene_blocking: List[MVSceneBlocking] = Field(default_factory=list)
    narrative_arc: Optional[MVNarrativeArc] = Field(None)


class MVSceneStatus(str, Enum):
    """씬 상태"""
    PENDING = "pending"
    GENERATING = "generating"
    COMPLETED = "completed"
    FAILED = "failed"


class MVProjectStatus(str, Enum):
    """프로젝트 상태"""
    UPLOADED = "uploaded"
    ANALYZING = "analyzing"
    READY = "ready"
    GENERATING = "generating"
    IMAGES_READY = "images_ready"  # 이미지 생성 완료, 리뷰 대기
    COMPOSING = "composing"
    COMPLETED = "completed"
    FAILED = "failed"


# ============================================================
# 음악 분석 관련
# ============================================================

class MusicSegment(BaseModel):
    """음악 구간 정보"""
    segment_type: str = Field(..., description="구간 타입 (intro, verse, chorus, bridge, outro)")
    start_sec: float = Field(..., description="시작 시간 (초)")
    end_sec: float = Field(..., description="종료 시간 (초)")
    duration_sec: float = Field(..., description="구간 길이 (초)")
    energy_level: Optional[float] = Field(None, description="에너지 레벨 (0-1)")


class MusicAnalysis(BaseModel):
    """음악 분석 결과"""
    duration_sec: float = Field(..., description="전체 길이 (초)")
    bpm: Optional[float] = Field(None, description="BPM (분당 비트 수)")
    mood: Optional[str] = Field(None, description="감지된 분위기")
    energy: Optional[float] = Field(None, description="전체 에너지 레벨 (0-1)")
    segments: List[MusicSegment] = Field(default_factory=list, description="구간 분할")
    key_timestamps: List[float] = Field(default_factory=list, description="주요 전환 포인트")
    extracted_lyrics: Optional[str] = Field(None, description="Gemini로 자동 추출된 가사")
    timed_lyrics: Optional[List[Dict[str, Any]]] = Field(None, description="타임스탬프 포함 가사 [{t: 초, text: 가사}]")


# ============================================================
# 가사 관련
# ============================================================

class LyricSegment(BaseModel):
    """가사 구간"""
    segment_id: int = Field(..., description="구간 ID")
    start_sec: float = Field(..., description="시작 시간")
    end_sec: float = Field(..., description="종료 시간")
    text: str = Field(..., description="가사 텍스트")
    segment_type: Optional[str] = Field(None, description="구간 타입 (verse, chorus 등)")


# ============================================================
# 씬 관련
# ============================================================

class MVScene(BaseModel):
    """뮤직비디오 씬"""
    scene_id: int = Field(..., description="씬 ID")
    start_sec: float = Field(..., description="시작 시간")
    end_sec: float = Field(..., description="종료 시간")
    duration_sec: float = Field(..., description="씬 길이")

    # 비주얼 정보
    image_prompt: str = Field(..., description="이미지 생성 프롬프트")
    visual_description: Optional[str] = Field(None, description="비주얼 설명 (한국어)")

    # 가사 연동
    lyrics_text: Optional[str] = Field(None, description="해당 구간 가사")
    lyrics_modified: bool = Field(default=False, description="사용자가 가사를 수정했는지 여부")

    # 이미지 파생 (비용/일관성 최적화)
    is_derived: bool = Field(default=False, description="파생 씬 여부 (기존 이미지 재사용)")
    source_scene_id: Optional[int] = Field(None, description="파생 원본 씬 ID")
    derived_effect: Optional[str] = Field(None, description="파생 효과 (hflip, crop_left, crop_right)")

    # 상태
    status: MVSceneStatus = Field(default=MVSceneStatus.PENDING)

    # 결과물
    image_path: Optional[str] = Field(None, description="생성된 이미지 경로")
    video_path: Optional[str] = Field(None, description="생성된 비디오 경로")

    # 연출
    transition: str = Field(default="crossfade", description="전환 효과 (cut, crossfade, fade)")
    camera_work: Optional[str] = Field(None, description="카메라 워크")

    # Visual Bible 강화 프롬프트 (Pass 3에서 생성)
    negative_prompt: Optional[str] = Field(None, description="네거티브 프롬프트")
    color_mood: Optional[str] = Field(None, description="이 씬의 색감/무드 키워드")
    camera_directive: Optional[str] = Field(None, description="카메라 연출 지시")

    # Director's Brief
    characters_in_scene: List[str] = Field(default_factory=list, description="등장 캐릭터 역할명")


# ============================================================
# 프로젝트 관련
# ============================================================

class MVCharacterSetup(str, Enum):
    """캐릭터 구성"""
    MALE_FEMALE = "male_female"        # 남녀 커플
    FEMALE_FEMALE = "female_female"    # 여여 커플
    MALE_MALE = "male_male"           # 남남 커플
    SOLO_MALE = "solo_male"           # 솔로 남성
    SOLO_FEMALE = "solo_female"       # 솔로 여성
    GROUP = "group"                   # 그룹 (3인 이상)
    AUTO = "auto"                     # AI 자동 결정


class MVCharacterEthnicity(str, Enum):
    """캐릭터 인종/외형"""
    AUTO = "auto"              # AI 자동 결정
    KOREAN = "korean"          # 한국인
    JAPANESE = "japanese"      # 일본인
    CHINESE = "chinese"        # 중국인
    SOUTHEAST_ASIAN = "southeast_asian"  # 동남아시아인
    EUROPEAN = "european"      # 유럽/백인
    BLACK = "black"            # 흑인
    HISPANIC = "hispanic"      # 히스패닉/라틴
    MIXED = "mixed"            # 다인종 조합


class MVProjectRequest(BaseModel):
    """MV 생성 요청"""
    project_id: Optional[str] = Field(None, description="프로젝트 ID (없으면 자동 생성)")

    # 가사 & 컨셉
    lyrics: Optional[str] = Field(None, description="가사 텍스트")
    concept: Optional[str] = Field(None, description="비주얼 컨셉 설명")

    # 캐릭터 설정
    character_setup: MVCharacterSetup = Field(default=MVCharacterSetup.AUTO, description="캐릭터 구성")
    character_ethnicity: MVCharacterEthnicity = Field(default=MVCharacterEthnicity.AUTO, description="캐릭터 인종/외형")

    # 스타일 설정
    genre: MVGenre = Field(default=MVGenre.FANTASY)
    mood: MVMood = Field(default=MVMood.EPIC)
    style: MVStyle = Field(default=MVStyle.CINEMATIC)

    # 수동 씬 분할 (Phase 1)
    manual_scenes: Optional[List[Dict[str, Any]]] = Field(
        None,
        description="수동 씬 분할 [{'start_sec': 0, 'end_sec': 30, 'description': '...'}, ...]"
    )


class MVProject(BaseModel):
    """MV 프로젝트 전체 정보"""
    project_id: str = Field(..., description="프로젝트 ID")
    status: MVProjectStatus = Field(default=MVProjectStatus.UPLOADED)

    # 음악 정보
    music_file_path: str = Field(..., description="업로드된 음악 파일 경로")
    music_analysis: Optional[MusicAnalysis] = Field(None, description="음악 분석 결과")

    # 입력 정보
    lyrics: Optional[str] = Field(None)
    concept: Optional[str] = Field(None)
    character_setup: MVCharacterSetup = Field(default=MVCharacterSetup.AUTO)
    character_ethnicity: MVCharacterEthnicity = Field(default=MVCharacterEthnicity.AUTO)
    genre: MVGenre = Field(default=MVGenre.FANTASY)
    mood: MVMood = Field(default=MVMood.EPIC)
    style: MVStyle = Field(default=MVStyle.CINEMATIC)

    # 씬 목록
    scenes: List[MVScene] = Field(default_factory=list)

    # 결과물
    final_video_path: Optional[str] = Field(None)
    thumbnail_path: Optional[str] = Field(None)

    # 메타데이터
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: Optional[datetime] = Field(None)
    error_message: Optional[str] = Field(None)

    # 진행률
    progress: int = Field(default=0, description="진행률 (0-100)")
    current_step: Optional[str] = Field(None, description="현재 진행 단계")

    # Visual Bible (Pass 1에서 생성)
    visual_bible: Optional[VisualBible] = Field(None, description="비주얼 바이블")
    style_anchor_path: Optional[str] = Field(None, description="전용 스타일 앵커 이미지 경로")


class MVUploadResponse(BaseModel):
    """음악 업로드 응답"""
    project_id: str
    status: str
    music_analysis: MusicAnalysis
    extracted_lyrics: Optional[str] = Field(None, description="자동 추출된 가사")
    message: str = "음악 업로드 및 분석 완료"


class MVGenerateResponse(BaseModel):
    """MV 생성 요청 응답"""
    project_id: str
    status: str
    total_scenes: int
    estimated_time_sec: int
    message: str = "뮤직비디오 생성이 시작되었습니다"


class MVStatusResponse(BaseModel):
    """MV 상태 조회 응답"""
    project_id: str
    status: MVProjectStatus
    progress: int
    current_step: Optional[str]
    scenes: List[MVScene]
    error_message: Optional[str] = None


class MVResultResponse(BaseModel):
    """MV 결과 조회 응답"""
    project_id: str
    status: MVProjectStatus
    video_url: Optional[str]
    thumbnail_url: Optional[str]
    duration_sec: float
    scenes: List[MVScene]
    download_url: Optional[str]
