# StoryCut Project Memory

## 응답 규칙
- **모든 응답은 반드시 한국어로 작성** (코드, 변수명, 커밋 메시지 제외)

## 핵심 컨셉
StoryCut은 **텍스트 스크립트로 영상을 만드는 AI 에이전트 서비스**입니다.

## 주요 기능
- 텍스트/주제 입력 → 스토리 자동 생성 → 영상 자동 제작
- 유튜브 롱폼/쇼츠 영상 자동 생성
- AI 이미지 생성 + Ken Burns 효과로 영상화
- TTS 내레이션 + BGM 자동 믹싱
- 자막 자동 생성 및 burn-in

## 프로젝트 구조

```
storycut/
├── api_server.py          # FastAPI 메인 서버
├── pipeline.py            # 통합 파이프라인 (메인 실행 흐름)
├── agents/                # AI 에이전트 모듈
│   ├── story_agent.py     # 스토리 생성 (Gemini LLM)
│   ├── character_manager.py # 캐릭터 멀티포즈 앵커 (v2.0 리팩토링)
│   ├── scene_orchestrator.py # 씬 조율 + 맥락 상속 + ConsistencyValidator 연동
│   ├── image_agent.py     # 이미지 생성 (Gemini/Replicate)
│   ├── video_agent.py     # 비디오 생성 (Veo 3.1 I2V 전용 + 정책)
│   ├── tts_agent.py       # TTS 음성 생성
│   ├── music_agent.py     # BGM 선택
│   ├── composer_agent.py  # 최종 합성
│   ├── optimization_agent.py # 유튜브 최적화
│   ├── style_anchor.py    # [v2.0] 스타일/환경 앵커 이미지 생성
│   └── consistency_validator.py # [v2.0] Gemini Vision 일관성 검증
├── schemas/
│   └── models.py          # Pydantic 데이터 모델 (PoseType, AnchorSet, ValidationResult 추가)
├── utils/
│   ├── ffmpeg_utils.py    # FFmpeg 유틸리티 (Ken Burns, 자막, 덕킹)
│   ├── prompt_builder.py  # 멀티모달 프롬프트 빌더 (7단계 LOCK 시스템)
│   ├── error_manager.py   # 에러 관리
│   └── storage.py         # 스토리지 유틸
├── config/
│   ├── __init__.py        # 설정 로더 (load_veo_policy, load_style_tokens 추가)
│   ├── veo_policy.yaml    # [v2.0] Veo I2V 모션 화이트리스트/블랙리스트
│   └── style_tokens.yaml  # [v2.0] 스타일 토큰 화이트리스트
├── cli/                   # CLI 도구
└── outputs/               # 생성된 영상 출력 디렉토리
```

## 핵심 파일 분석

### `pipeline.py` - 메인 파이프라인
- `StorycutPipeline` 클래스: 전체 실행 흐름 관리
- `run()`: 스토리 생성 → 영상 생성 전체 실행
- `generate_story_only()`: 스토리만 생성 (Step 1)
- `generate_video_from_story()`: 확정된 스토리로 영상 생성 (Step 2~6)

### `schemas/models.py` - 데이터 모델
- `ProjectRequest`: 프로젝트 요청 정보
- `Scene`: 씬 데이터 (narration, prompt, assets 등)
- `Manifest`: 프로젝트 메타데이터 전체
- `CharacterSheet`: 캐릭터 시트 (외형, 시드, 마스터 이미지)
- `GlobalStyle`: 글로벌 스타일 설정
- `FeatureFlags`: 기능 플래그 (hook_scene1_video, ffmpeg_kenburns, film_look 등)

### `api_server.py` - FastAPI 서버
- `POST /generate/story`: 스토리만 생성
- `POST /generate/video`: 확정된 스토리로 영상 생성
- `GET /status/{project_id}`: 프로젝트 상태 조회

## 파이프라인 흐름 (v2.0)

```
1. StoryAgent.generate_story()
   └─ Gemini LLM으로 스토리 JSON 생성
   └─ character_sheet, global_style 포함

2. CharacterManager.cast_characters() [v2.0 신규]
   └─ 각 캐릭터 마스터 앵커 이미지 생성
   └─ outputs/{project_id}/media/characters/{token}.png

3. SceneOrchestrator.process_story()
   └─ 씬별 반복 처리
   ├─ TTSAgent: 내레이션 음성 생성
   ├─ VideoAgent: 이미지+Ken Burns 또는 Veo 비디오
   └─ 자막 burn-in

4. FFmpegComposer.apply_film_look() [v2.0 신규]
   └─ 필름 그레인 + 색보정

5. ComposerAgent.compose_video()
   └─ 씬 연결 + 오디오 믹싱

6. OptimizationAgent.run()
   └─ 제목/썸네일/해시태그 생성
```

## 주요 API 키 (환경변수)
- `GOOGLE_API_KEY`: Gemini LLM, 이미지, Veo 비디오
- `REPLICATE_API_TOKEN`: Replicate 이미지 생성 (백업)
- `OPENAI_API_KEY`: DALL-E 이미지 (선택)

## Feature Flags
```python
FeatureFlags(
    hook_scene1_video=False,    # Scene 1 고품질 비디오 (Veo)
    ffmpeg_kenburns=True,       # Ken Burns 효과
    ffmpeg_audio_ducking=False, # 오디오 덕킹
    subtitle_burn_in=True,      # 자막 burn-in
    context_carry_over=True,    # 맥락 상속
    optimization_pack=True,     # 유튜브 최적화
    film_look=False,            # 필름 그레인/색보정 [v2.0]
)
```

## 출력 구조
```
outputs/{project_id}/
├── manifest.json           # 프로젝트 메타데이터
├── final_video.mp4         # 최종 영상
├── scenes/                 # 씬별 JSON
├── media/
│   ├── video/              # 씬 비디오
│   ├── images/             # 씬 이미지
│   ├── audio/              # 내레이션
│   ├── subtitles/          # SRT 자막
│   └── characters/         # 캐릭터 마스터 이미지 [v2.0]
└── optimization_{id}.json  # 유튜브 최적화 데이터
```

## 기술 스택
- Python, FastAPI
- Google Gemini API (LLM, 이미지, 비디오 Veo 3.1)
- FFmpeg (영상 합성, Ken Burns, 자막, 오디오 덕킹)
- Replicate (이미지 생성 백업)
- Pydantic (데이터 검증)

## 배포
- Railway (백엔드)
- Vercel (프론트엔드)
