# STORYCUT v2.0

**STORYCUT**은 텍스트로 된 이야기를 입력하면, 자동으로 **유튜브 업로드가 가능한 완성형 영상(MP4)**을 생성해주는 AI 기반 스토리 영상 제작 에이전트입니다.

이 프로젝트의 목적은 **조회수/수익형 유튜브 제작 머신**으로,
**여러 사용자가 STORYCUT 에이전트를 활용해 각자 자신의 이야기 영상을 만들고 유튜브에 업로드하도록 돕는 제작 도구**를 제공하는 것입니다.

---

## v2.0 주요 기능

### P0 (필수/최우선)
- **Scene 1 Hook 전용 고품질 비디오 생성**: 첫 장면은 고품질 비디오로 강제 생성 (비용 최적화)
- **Ken Burns Effect**: 이미지 기반 장면에 줌/팬 효과 적용하여 영상처럼 표현
- **Audio Ducking**: 내레이션 시 BGM 자동 감쇠
- **Subtitle Burn-in**: 자막을 영상에 직접 렌더링

### P1 (우선순위 높음)
- **Context Carry-over**: 이전 장면의 핵심 키워드(인물/장소/감정/행동)를 다음 장면에 상속

### P2 (추가 기능)
- **Optimization Agent**: 유튜브 제목 3종, 썸네일 프롬프트 2종, 해시태그, AB 테스트 메타데이터 생성

---

## 시스템 아키텍처

```
User Input (CLI / API)
  ↓
[Feature Flags Configuration]
  ↓
Story Agent (Scene JSON 생성)
  ↓
Scene Orchestrator (맥락 상속 + 장면 단위 분해)
  ↓
┌─────────────────────────────────────────┐
│  Scene 1 (Hook)        │  Scene 2~N    │
│  → High-quality Video  │  → Image      │
│     (Runway/API)       │  → Ken Burns  │
│  → Fallback: Image+KB  │               │
└─────────────────────────────────────────┘
  ↓
Video Agent / Image Agent / TTS Agent / Music Agent
  ↓
FFmpeg Composer (Ducking + Subtitle + Concat)
  ↓
Optimization Agent (제목/썸네일/AB테스트)
  ↓
outputs/<project_id>/
  ├─ manifest.json
  ├─ final_video.mp4
  ├─ scenes/scene_*.json
  └─ optimization_*.json
```

---

## 배포 아키텍처

STORYCUT은 두 가지 배포 방식을 지원합니다:

### 🏠 로컬 배포 (개발/테스트)
- FastAPI + WebSocket
- 로컬 파일 시스템

### ☁️ 클라우드 배포 (프로덕션)
- Cloudflare Workers (API)
- Cloudflare Pages (웹 UI)
- D1 Database (메타데이터)
- R2 Storage (영상 파일)
- Queue (비동기 작업)
- 외부 서버 (Python 백엔드 - Railway/Render)

자세한 배포 가이드: [cloudflare/DEPLOYMENT_GUIDE.md](cloudflare/DEPLOYMENT_GUIDE.md)

---

## 설치 및 실행

### 1. 필수 요구사항

- Python 3.10 이상
- FFmpeg (시스템에 설치 필요)

#### FFmpeg 설치

**Windows:**
```bash
# Chocolatey 사용
choco install ffmpeg

# 또는 Scoop 사용
scoop install ffmpeg
```

**macOS:**
```bash
brew install ffmpeg
```

**Linux:**
```bash
sudo apt-get install ffmpeg
```

### 2. Python 패키지 설치

```bash
pip install -r requirements.txt
```

### 3. 환경 변수 설정

`.env.example` 파일을 `.env`로 복사하고 API 키를 설정합니다:

```bash
cp .env.example .env
```

`.env` 파일 편집:
```
OPENAI_API_KEY=your_openai_api_key_here
RUNWAY_API_KEY=your_runway_api_key_here  # 선택사항 (Hook 비디오용)
```

### 4. 실행

#### 🌐 웹 UI (권장)
```bash
# Windows
start_server.bat

# Mac/Linux
python api_server.py
```

브라우저에서 접속:
```
http://localhost:8000
```

**주요 기능:**
- ✨ 직관적인 UI로 장르/분위기 선택
- 📊 실시간 진행상황 확인 (WebSocket)
- 🎬 영상 미리보기 및 다운로드
- 🚀 최적화 패키지 (제목/썸네일/해시태그) 즉시 복사

자세한 내용: [WEBUI_GUIDE.md](WEBUI_GUIDE.md)

---

#### 💻 CLI 모드
```bash
python cli/storycut_cli.py
```

#### 📦 프로그래매틱 모드
```python
from pipeline import run_pipeline

manifest = run_pipeline(
    topic="오래된 폐병원에서 발견된 미스터리한 일기장",
    genre="mystery",
    mood="suspenseful",
    duration=60,
    feature_flags={
        "hook_scene1_video": False,  # 비용 절감
        "ffmpeg_kenburns": True,
        "context_carry_over": True,
        "optimization_pack": True,
    }
)

print(f"Video: {manifest.outputs.final_video_path}")
print(f"Titles: {manifest.outputs.title_candidates}")
```

---

## Feature Flags

`config/feature_flags.yaml`에서 기능을 ON/OFF 할 수 있습니다:

| Flag | 기본값 | 설명 |
|------|--------|------|
| `hook_scene1_video` | OFF | Scene 1에서 고품질 비디오 생성 (비용 주의) |
| `ffmpeg_kenburns` | ON | 이미지에 Ken Burns 효과 적용 |
| `ffmpeg_audio_ducking` | OFF | 내레이션 시 BGM 자동 감쇠 |
| `subtitle_burn_in` | ON | 자막을 영상에 burn-in |
| `context_carry_over` | ON | 장면 간 맥락 상속 |
| `optimization_pack` | ON | 제목/썸네일/AB테스트 패키지 생성 |
| `topic_finding` | OFF | 트렌드 기반 주제 추천 (향후 지원) |

---

## 프로젝트 구조

```
storycut/
├── cli/
│   └── storycut_cli.py          # CLI 진입점
│
├── agents/                       # 역할 기반 에이전트
│   ├── story_agent.py           # 스토리 생성
│   ├── scene_orchestrator.py    # Scene 오케스트레이션 (맥락 상속)
│   ├── video_agent.py           # 영상 생성 (Hook + KenBurns)
│   ├── image_agent.py           # 이미지 생성
│   ├── tts_agent.py             # 내레이션 생성
│   ├── music_agent.py           # 배경 음악 선택
│   ├── composer_agent.py        # 영상 합성
│   └── optimization_agent.py    # 유튜브 최적화
│
├── schemas/                      # Pydantic 데이터 모델
│   ├── __init__.py
│   └── models.py                # FeatureFlags, Scene, Manifest 등
│
├── config/                       # 설정 파일
│   ├── __init__.py
│   └── feature_flags.yaml       # Feature flags 설정
│
├── utils/                        # 유틸리티
│   └── ffmpeg_utils.py          # FFmpeg 래퍼 (KenBurns, Ducking, Subtitle)
│
├── pipeline.py                   # 통합 파이프라인
├── prompts/                      # 에이전트 프롬프트
└── outputs/                      # 프로젝트별 출력 디렉토리
    └── <project_id>/
        ├── manifest.json
        ├── final_video.mp4
        ├── scenes/
        └── optimization_*.json
```

---

## 출력 구조 (Manifest)

```json
{
  "project_id": "a1b2c3d4",
  "title": "영상 제목",
  "status": "completed",
  "scenes": [...],
  "outputs": {
    "final_video_path": "outputs/a1b2c3d4/final_video.mp4",
    "title_candidates": [
      "충격! 폐병원의 비밀이 밝혀졌다",
      "폐병원에서 발견된 일기장, 그 안에는?",
      "미스터리 일기장의 진실"
    ],
    "thumbnail_prompts": [
      "Dramatic close-up portrait, shocked expression...",
      "Split image composition, before and after..."
    ],
    "hashtags": ["#미스터리", "#폐병원", "#쇼츠", ...],
    "ab_test_meta": {...}
  },
  "cost_estimate": {
    "llm_tokens": 5000,
    "video_seconds": 60,
    "estimated_usd": 0.35
  }
}
```

---

## 주요 기능 상세

### 1. Ken Burns Effect
이미지 기반 장면에 줌/팬 효과를 적용하여 정적 이미지를 영상처럼 표현합니다.

- **zoom_in**: 중앙에서 확대
- **zoom_out**: 확대에서 축소
- **pan_left/right**: 좌우 이동
- **diagonal**: 대각선 이동 + 줌

### 2. Audio Ducking
내레이션이 있는 구간에서 BGM 볼륨을 자동으로 감쇠시킵니다.

- sidechaincompress 필터 사용
- 설정 가능: threshold, ratio, attack, release

### 3. Context Carry-over
이전 장면의 핵심 엔티티(인물, 장소, 감정, 행동)를 다음 장면 프롬프트에 상속합니다.

- 일관된 캐릭터 표현
- 장소/배경 연속성
- 감정 흐름 유지

### 4. Optimization Agent
유튜브 최적화 패키지를 자동 생성합니다:

- **제목 후보 3종**: 충격형, 질문형, 요약형
- **썸네일 문구 3종**: 2-5단어 임팩트 문구
- **썸네일 프롬프트 2종**: 이미지 생성용 (과장된 감정, 높은 대비)
- **해시태그 10개**: 관련성 높은 순서
- **AB 테스트 메타데이터**: titleA/B, thumbnailA/B

---

## API 키 안내

### OpenAI API (필수)
- 스토리 생성: GPT-4
- 음성 생성: OpenAI TTS
- 이미지 생성: DALL-E 3
- 가입: https://platform.openai.com/

### Runway ML (선택 - Hook 비디오용)
- Scene 1 고품질 영상 생성에 사용
- 가입: https://runwayml.com/
- 없으면 이미지+Ken Burns로 대체

---

## 문제 해결

### FFmpeg 관련 오류
```bash
# FFmpeg 설치 확인
ffmpeg -version

# sidechaincompress 필터 지원 확인
ffmpeg -filters | grep sidechain
```

### Ken Burns 오류
- 입력 이미지 해상도가 너무 낮으면 실패할 수 있음
- 최소 1280x720 이상 권장

### 자막 burn-in 오류
- Windows에서 경로에 한글이 있으면 실패할 수 있음
- 영문 경로 사용 권장

---

## 비용 추정

| 항목 | 대략적 비용 |
|------|-------------|
| GPT-4 토큰 (1K) | $0.03 |
| DALL-E 3 이미지 | $0.02/장 |
| OpenAI TTS (1K 문자) | $0.015 |
| Runway 비디오 (초) | $0.05/초 |

60초 영상 기준 예상 비용: **$0.30 ~ $0.80** (Hook 비디오 제외)

---

## 완료 기준 (Definition of Done)

- [x] (D1) Scene 1은 비디오 생성 경로, Scene 2~N은 이미지+KenBurns 경로
- [x] (D2) 내레이션 있으면 BGM 자동 감쇠, 없으면 원복
- [x] (D3) 자막이 영상 위에 burn-in (옵션으로 on/off)
- [x] (D4) 씬 프롬프트에 이전 씬 핵심 키워드 상속
- [x] (D5) Optimization Agent가 제목/썸네일/메타 JSON 출력
- [x] (D6) 기존 실행 방식(API/CLI)이 깨지지 않음

---

## 프로젝트 철학

- STORYCUT은 **창작을 대체하지 않는다**
- STORYCUT은 **창작을 가속한다**
- AI는 도구이며, 크리에이터가 주인이다
- **비용 최적화**: 고비용 모델은 Hook에만 사용

---

## 라이센스

MIT License

---

## 기여

이슈와 PR은 환영합니다!

---

## 한 줄 설명

> **STORYCUT v2.0 — 이야기를 입력하면, 조회수 터지는 유튜브 영상이 완성된다.**
