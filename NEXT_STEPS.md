# STORYCUT v2.0 - 개발 로드맵

> 마지막 업데이트: 2026-02-01

---

## 완료된 작업

- [x] P0 기능: Hook 비디오, Ken Burns, Audio Ducking, Subtitle
- [x] P1 기능: Context Carry-over
- [x] P2 기능: Optimization Agent
- [x] 웹 UI 구현 (FastAPI + WebSocket)
- [x] Cloudflare 아키텍처 설계
- [x] Worker, Queue Consumer, D1 스키마 작성
- [x] 배포 가이드 작성
- [x] TTS 다각화 (Google Neural2 A/C, Gemini Flash/Pro)
- [x] Cloudflare Pages 자동 배포 (Git 연결)
- [x] R2 스토리지 연동 (생성 영상 자동 업로드)
- [x] Phase 2: Cloudflare 배포 완료
- [x] Phase 3.1: 인증 API 구현 (`/api/auth/register`, `/api/auth/login`, `/api/auth/me`)

---

## 긴급 보안 이슈 (즉시 조치 필요)

### 1. API 키 노출 (CRITICAL)
**파일:** `.env`
- Git 히스토리에 실제 API 키 노출됨 (OpenAI, Google, Replicate, ElevenLabs, R2)
- **조치:** BFG로 히스토리 정리 + 모든 키 재발급

### 2. 하드코딩된 JWT Secret (CRITICAL)
**파일:** `worker.js:17`
- `JWT_SECRET = 'my-secret-salt-key-change-this'`
- **조치:** wrangler.toml 환경변수로 이동

### 3. Mock 인증 시스템 (HIGH)
**파일:** `api_server.py:1120-1149`
- 아무 이메일/비밀번호로 로그인 가능
- **조치:** 실제 JWT + bcrypt 해싱 적용

### 4. Path Traversal 취약점 (HIGH)
**파일:** `api_server.py:1237-1258`
- `/api/asset/../../etc/passwd` 공격 가능
- **조치:** 경로 검증 및 sanitization 추가

### 5. 디버그 로그에 API 키 노출 (HIGH)
**파일:** `api_server.py:48, 51`
- **조치:** 프로덕션에서 제거 또는 마스킹

### 6. 에러 응답에 트레이스백 노출 (MEDIUM)
**파일:** `api_server.py:98-110`
- **조치:** 프로덕션에서 숨김 처리

---

## 보안 체크리스트

### 즉시 (오늘)
- [ ] `.env` Git 히스토리에서 제거
- [ ] 모든 API 키 재발급
- [ ] JWT Secret 환경변수로 이동
- [ ] 디버그 로그 제거

### 단기 (이번 주)
- [ ] Path Traversal 취약점 수정
- [ ] Mock 인증 → 실제 JWT 인증
- [ ] 에러 트레이스백 숨김
- [ ] CORS 설정 강화

### 중기 (이번 달)
- [ ] Rate Limiting 추가
- [ ] 보안 헤더 미들웨어 추가
- [ ] 입력값 검증 강화
- [ ] 로깅 시스템 도입

---

## 개발 로드맵

### Phase 1: 로컬 테스트 및 검증

#### 1.1 환경 설정
```bash
cp .env.example .env
# OPENAI_API_KEY 등 설정
```

#### 1.2 검증 항목
- [ ] 스토리 생성 정상 동작
- [ ] Scene 이미지 + Ken Burns 정상 생성
- [ ] TTS 내레이션 생성
- [ ] FFmpeg 합성 성공
- [ ] 최적화 패키지 생성 (제목/썸네일/해시태그)
- [ ] WebSocket 실시간 진행상황 업데이트
- [ ] 영상 다운로드

---

### Phase 2: Cloudflare 배포 (완료)

- [x] Wrangler 설치 및 로그인
- [x] D1 Database 생성 및 스키마 적용
- [x] R2 Storage 생성
- [x] Queue 생성
- [x] Worker 배포
- [x] Python 백엔드 Railway 배포
- [x] Cloudflare Pages 배포

---

### Phase 3: 인증 및 크레딧 시스템 (진행 중)

#### 3.1 인증 API (완료)
- [x] `/api/auth/register` - 회원가입
- [x] `/api/auth/login` - 로그인 (JWT 발급)
- [x] `/api/auth/me` - 현재 사용자 정보

#### 3.2 크레딧 시스템
- [ ] 영상 생성 시 크레딧 차감
- [ ] 크레딧 부족 시 에러 처리
- [ ] 크레딧 충전 페이지 UI

#### 3.3 결제 연동 (Stripe)
- [ ] Stripe Checkout 설정
- [ ] 결제 완료 시 크레딧 충전
- [ ] 결제 내역 조회 페이지

---

### Phase 4: UI/UX 개선

#### 4.1 로그인/회원가입 UI
- [ ] 로그인 모달
- [ ] 회원가입 폼
- [ ] 로그아웃 버튼
- [ ] 크레딧 잔액 표시

#### 4.2 UI/UX 리디자인
- [ ] 반응형 레이아웃 강화 (모바일 최적화)
- [ ] 다크 모드 고도화 (글래스모피즘)
- [ ] React / Next.js 도입 고려 (장기)

#### 4.3 기능 분리
- [ ] 대시보드 분리: 영상 생성 창과 히스토리/결과 창 분리
- [ ] 회원가입/로그인 페이지 별도 구축
- [ ] 설정(Settings) 페이지 독립

#### 4.4 에러 처리 개선
- [ ] API 키 없을 때 안내 메시지
- [ ] 크레딧 부족 시 충전 유도
- [ ] FFmpeg 오류 상세 메시지
- [ ] 재시도 버튼

---

### Phase 5: 성능 최적화

#### 5.1 캐싱
- [ ] R2 + CDN으로 영상 서빙
- [ ] D1 쿼리 결과 캐싱
- [ ] OpenAI API 응답 캐싱 (동일 프롬프트)

#### 5.2 비용 최적화
- [ ] Hook 비디오 사용률 모니터링
- [ ] DALL-E 3 → Stable Diffusion 옵션
- [ ] TTS 캐싱 (동일 텍스트)

#### 5.3 스케일링
- [ ] Queue 병렬 처리
- [ ] 여러 Python 백엔드 인스턴스
- [ ] 로드 밸런싱

---

### Phase 6: 분석 및 비즈니스

#### 6.1 Analytics
- [ ] 영상 생성 수 추적
- [ ] 장르별 인기도
- [ ] 평균 처리 시간
- [ ] 비용 분석

#### 6.2 AB 테스트 자동화
- [ ] 생성된 제목 AB 테스트 결과 수집
- [ ] 썸네일 클릭률 추적
- [ ] 최적화 알고리즘 개선

#### 6.3 소셜 기능
- [ ] 영상 공유 링크
- [ ] 커뮤니티 갤러리
- [ ] 인기 영상 랭킹

---

## 아키텍처 개선사항

| 항목 | 파일 | 문제 | 해결방안 |
|------|------|------|----------|
| 중복 라우트 | `api_server.py:682, 1206` | `/api/manifest/{project_id}` 2번 정의 | 하나로 통합 |
| 하드코딩 URL | `api_server.py:904`, `app.js:109` | 백엔드 URL 하드코딩 | 환경변수 사용 |
| CORS 설정 | `api_server.py:77-89` | `allow_methods=["*"]` | 필요한 것만 명시 |
| Rate Limiting | - | 무제한 API 호출 | `slowapi` 적용 |
| 로깅 | - | `print()` 사용 | `logging` 모듈 |
| 재시도 로직 | `agents/*.py` | 외부 API 재시도 없음 | `tenacity` 적용 |

---

## 선택적 개선 사항 (장기)

- [ ] 다국어 지원 (영어, 일본어 등)
- [ ] 음성 클로닝 (사용자 목소리)
- [ ] 커스텀 BGM 업로드
- [ ] 자막 편집 UI
- [ ] 영상 템플릿 갤러리
- [ ] API 문서 자동 생성 (Swagger UI)
- [ ] Webhook 지원 (영상 생성 완료 알림)
- [ ] 원클릭 유튜브 업로드
- [ ] 조회수 분석 및 개선 제안

---

## 문제 발생 시

### 로컬 테스트 실패
1. `.env` 파일 API 키 확인
2. `ffmpeg -version` 확인
3. `python api_server.py` 로그 확인

### Cloudflare 배포 실패
1. `wrangler.toml` 설정 확인
2. `wrangler deploy --dry-run` 실행
3. [DEPLOYMENT_GUIDE.md](cloudflare/DEPLOYMENT_GUIDE.md) 참고

---

## 배포 환경변수 참고

### Railway
```
ENVIRONMENT=production
BACKEND_URL=https://web-production-bb6bf.up.railway.app
OPENAI_API_KEY=sk-...
GOOGLE_API_KEY=AIza...
REPLICATE_API_TOKEN=r8_...
ELEVENLABS_API_KEY=sk_...
R2_ACCOUNT_ID=...
R2_ACCESS_KEY_ID=...
R2_SECRET_ACCESS_KEY=...
R2_BUCKET_NAME=storycut-videos
```

### Cloudflare Worker (wrangler.toml)
```toml
[vars]
ENVIRONMENT = "production"
BACKEND_URL = "https://web-production-bb6bf.up.railway.app"

# Secrets (wrangler secret put으로 설정)
# JWT_SECRET
# BACKEND_API_SECRET
```

---

## 보안 강화 코드 예시

<details>
<summary>Path Traversal 방어</summary>

```python
import re
from pathlib import Path

def sanitize_path_component(component: str) -> str:
    if not re.match(r'^[a-zA-Z0-9_-]+$', component):
        raise HTTPException(status_code=400, detail="잘못된 경로 형식")
    return component

@app.get("/api/asset/{project_id}/{asset_type}/{filename}")
async def get_asset(project_id: str, asset_type: str, filename: str):
    project_id = sanitize_path_component(project_id)
    filename = sanitize_path_component(Path(filename).stem) + Path(filename).suffix

    base_path = Path("outputs").resolve()
    final_path = (base_path / project_id / local_dir / filename).resolve()

    if not str(final_path).startswith(str(base_path)):
        raise HTTPException(status_code=403, detail="접근 거부")
```
</details>

<details>
<summary>Rate Limiting</summary>

```python
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

@app.post("/api/generate/story")
@limiter.limit("5/minute")
async def generate_story(request: Request, ...):
    pass
```
</details>

<details>
<summary>보안 헤더 미들웨어</summary>

```python
from starlette.middleware.base import BaseHTTPMiddleware

class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        return response

app.add_middleware(SecurityHeadersMiddleware)
```
</details>

<details>
<summary>로깅 시스템</summary>

```python
import logging
import sys

def setup_logger(name: str, level: str = "INFO") -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level))

    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    return logger

# 사용
logger = setup_logger(__name__)
logger.info("서버 시작")
```
</details>
