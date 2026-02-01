# Storycut 프로젝트 아키텍처 분석 보고서

## 1. 프로젝트 개요

**Storycut**은 텍스트 주제를 입력하면 스토리, 이미지, 음성, 영상을 포함한 완성된 숏폼/롱폼 비디오를 자동으로 생성하는 플랫폼입니다. **Cloudflare Workers**(Edge)를 통한 가벼운 요청 처리와 **Python FastAPI Server**(Backend)를 통한 무거운 AI 연산을 결합한 하이브리드 아키텍처를 채택하고 있습니다.

### 주요 기능
- **AI 스토리텔링**: 주제에 따른 자동 대본 생성 (Gemini/OpenAI).
- **멀티모달 생성**: 이미지 (DALL-E/Replicate), 음성 (ElevenLabs), 비디오 (Runway/Luma), 배경음악.
- **워크플로우 오케스트레이션**: 스토리 기획부터 최종 영상 합성까지 단계별 파이프라인 자동화.
- **하이브리드 인프라**: 확장성 있는 서버리스(Worker)와 고성능 연산 서버(Python)의 결합.

## 2. 설계 아키텍처

이 시스템은 **마이크로서비스형 하이브리드 패턴**을 따릅니다:

*   **프론트엔드**: Cloudflare를 통해 서빙되는 정적 웹 UI (HTML/Templates).
*   **게이트웨이 / 인증 계층**: Cloudflare Worker. 사용자 인증, DB 상호작용, 에셋 서빙 담당.
*   **코어 엔진**: Python FastAPI 서버. 무거운 `StorycutPipeline` 및 AI 에이전트 실행.
*   **스토리지**:
    *   **데이터베이스**: Cloudflare D1 (사용자, 프로젝트, 크레딧 관리).
    *   **오브젝트 스토리지**: Cloudflare R2 (이미지, 오디오, 최종 영상 파일).

### 아키텍처 다이어그램

```mermaid
graph TD
    User[사용자 / 웹 클라이언트]
    
    subgraph "Cloudflare Edge (Worker)"
        Worker[Worker.js]
        Auth[인증 로직]
        Router[라우터]
        D1[(D1 데이터베이스)]
        R2[(R2 스토리지)]
    end
    
    subgraph "연산 백엔드 (Python)"
        API[FastAPI 서버]
        Pipeline[Storycut 파이프라인]
        
        subgraph "AI 에이전트"
            StoryAgent[스토리 에이전트]
            SceneAgent[장면 오케스트레이터]
            ImageAgent[이미지 에이전트]
            TTSAgent[TTS 에이전트]
            OptAgent[최적화 에이전트]
        end
    end
    
    subgraph "외부 API"
        LLM[OpenAI / Gemini]
        Voice[ElevenLabs]
        ImgGen[Replicate / DALL-E]
    end

    User -->|HTTPS| Worker
    Worker -->|인증/데이터| D1
    Worker -->|미디어 서빙| R2
    
    Worker -->|생성 요청 (Webhook/Fetch)| API
    API -->|비동기 처리| Pipeline
    
    Pipeline --> StoryAgent
    Pipeline --> SceneAgent
    SceneAgent --> ImageAgent & TTSAgent
    
    ImageAgent --> ImgGen
    TTSAgent --> Voice
    StoryAgent --> LLM
    
    Pipeline -->|결과 업로드| R2
    Pipeline -->|상태 업데이트 (Webhook)| Worker
    
    %% WebSocket 연결 (잠재적 문제)
    User -.->|WebSocket (진행상황)| API
```

## 3. 주요 모듈 및 흐름

### 3.1. Cloudflare Worker (`worker.js`)
시스템의 메인 진입점 역할을 합니다.
*   **/api/auth/**: `bcrypt`와 `jose`(JWT)를 사용한 로그인/회원가입 처리. D1에 사용자 정보 저장.
*   **/api/generate**: 사용자 크레딧 확인, D1에 프로젝트 ID 생성 후 Python 백엔드로 요청 전달.
*   **/api/asset/**: R2 버킷의 미디어 파일을 보안 URL로 프록시하여 서빙.
*   **데이터베이스**: `users` 및 `projects` 테이블 관리.

### 3.2. Python 백엔드 (`api_server.py`)
핵심 로직을 수행합니다.
*   **FastAPI**: API 엔드포인트 제공 및 실시간 진행률 전송을 위한 WebSocket 관리.
*   **Pipeline Wrapper**: 메인 스레드 차단을 방지하기 위해 별도 스레드에서 파이프라인 실행.
*   **Webhook**: 작업 완료 시 Worker에게 상태 업데이트 전송 (또는 내부 처리).

### 3.3. Storycut 파이프라인 (`pipeline.py`)
`agents` 프레임워크를 활용한 작업 오케스트레이터입니다.
1.  **스토리 생성**: 장면, 내레이션, 시각적 묘사가 포함된 대본 작성.
2.  **장면 오케스트레이션**:
    *   장면별 순차 처리.
    *   캐릭터 일관성을 유지하며 이미지 생성.
    *   TTS 오디오 생성.
3.  **합성 (`ffmpeg_utils`)**: 이미지, 오디오, 자막을 결합하여 MP4 생성.
4.  **최적화**: 유튜브용 제목, 썸네일 텍스트, 해시태그 생성.

## 4. 기술 요건

### 인프라
- **Cloudflare 계정**: Workers, R2 Bucket, D1 Database.
- **Python 호스팅**: Railway, Render, EC2 등 (Python 3.10+ 구동 가능한 서버).
- **환경 변수**:
    *   `R2_ACCOUNT_ID`, `R2_ACCESS_KEY_ID`, `R2_SECRET_ACCESS_KEY`
    *   `OPENAI_API_KEY`, `GOOGLE_API_KEY` (Gemini)
    *   `ELEVENLABS_API_KEY`
    *   `DATABASE_URL` 등.

### 소프트웨어 스택
- **런타임**: Python 3.10+, Node.js (Worker 개발용).
- **주요 라이브러리**:
    *   Python: `fastapi`, `ffmpeg-python`, `pydantic`.
    *   Node: `wrangler`, `jose`, `bcryptjs`.
- **시스템 바이너리**: Python 서버에 `ffmpeg` 설치 필수.

## 5. 개선 포인트 및 제안사항

### 🔴 중요 (Critical)
1.  **WebSocket 프록시 부재**:
    *   **현황**: 클라이언트가 Python 백엔드와 직접 WebSocket 연결을 시도할 수 있지만, Worker를 거칠 경우 `worker.js`에 WebSocket 프록시 로직이 없어 연결이 실패할 수 있습니다.
    *   **해결**: 클라이언트가 Python 서버 주소로 직접 연결하도록 CORS/URL을 설정하거나, Worker에서 `WebSocketPair`를 사용해 프록시 기능을 구현해야 합니다.
2.  **하드코딩된 Secret**:
    *   `worker.js`에 JWT Secret 키(`my-secret-salt-key-change-this`)가 그대로 노출되어 있습니다.
    *   **해결**: `wrangler secret put`을 사용하여 환경 변수로 분리해야 합니다.
3.  **Fire-and-forget 신뢰성**:
    *   `ctx.waitUntil` 내에서 백엔드로 `fetch`를 보내고 응답을 확인하지 않는 구조입니다. 백엔드 장애 시 사용자는 크레딧만 잃고 결과를 받지 못할 수 있습니다.
    *   **해결**: 최소한 백엔드로부터 '접수 완료(202 Accepted)' 응답을 확인한 후 크레딧을 차감하거나, Cloudflare Queues를 도입하여 재시도 로직을 구축해야 합니다.

### 🟡 최적화 (Optimization)
1.  **Queue 시스템 도입**: 코드 내 주석 처리된 Queue 로직을 활성화하여, HTTP 요청 대신 비동기 메시지 큐로 작업을 분산 처리하면 안정성이 크게 향상됩니다.
2.  **정적 파일 호스팅**: `web/templates`를 Cloudflare Worker에서 직접 서빙하는 대신, **Cloudflare Pages** 기능을 활용하면 더 빠른 속도와 배포 편의성을 확보할 수 있습니다.

### 🟢 확장성 (Scalability)
1.  **데이터베이스 인덱싱**: D1의 `projects` 테이블에서 상태 조회(Polling)가 빈번할 수 있으므로 인덱싱 최적화가 필요합니다.
2.  **스토리지 관리**: R2에 쌓이는 임시 파일이나 오래된 프로젝트 파일을 정리하기 위한 Lifecycle 규칙 설정이 권장됩니다.
