# STORYCUT Cloudflare 배포 가이드

## 🏗️ 아키텍처 개요

```
User Browser
    ↓
Cloudflare Pages (웹 UI)
    ↓
Cloudflare Workers (API)
    ↓
    ├─→ D1 Database (메타데이터)
    ├─→ R2 Storage (영상 파일)
    ├─→ Queue (비동기 작업)
    └─→ Python Backend (실제 영상 생성)
```

---

## 📋 사전 준비

### 1. Cloudflare 계정 및 도메인
- Cloudflare 계정 생성
- 도메인 등록 및 Cloudflare DNS 연결

### 2. Wrangler CLI 설치
```bash
npm install -g wrangler

# 로그인
wrangler login
```

---

## 🚀 배포 단계

### Step 1: D1 Database 생성

```bash
# 프로덕션 DB 생성
wrangler d1 create storycut-db

# 개발 DB 생성
wrangler d1 create storycut-db-dev

# 출력된 database_id를 wrangler.toml에 기록
```

### Step 2: 스키마 적용

```bash
# 프로덕션
wrangler d1 execute storycut-db --file=cloudflare/schema.sql

# 개발
wrangler d1 execute storycut-db-dev --file=cloudflare/schema.sql --env=dev
```

### Step 3: R2 Bucket 생성

```bash
# 프로덕션
wrangler r2 bucket create storycut-videos

# 개발
wrangler r2 bucket create storycut-videos-dev
```

### Step 4: Queue 생성

```bash
# 프로덕션
wrangler queues create storycut-video-queue
wrangler queues create storycut-dlq

# 개발
wrangler queues create storycut-video-queue-dev
```

### Step 5: Secret 설정

```bash
# Backend API Secret
wrangler secret put BACKEND_API_SECRET

# OpenAI API Key (백엔드에서 사용)
wrangler secret put OPENAI_API_KEY

# Stripe Secret Key
wrangler secret put STRIPE_SECRET_KEY
```

### Step 6: Worker 배포

```bash
# 프로덕션
wrangler deploy

# 개발
wrangler deploy --env=dev
```

### Step 7: Queue Consumer 배포

```bash
# queue-consumer.js를 별도 Worker로 배포
wrangler deploy cloudflare/queue-consumer.js --name storycut-queue-consumer
```

---

## 🌐 Cloudflare Pages 배포 (웹 UI)

### 방법 1: GitHub 연동 (권장)

1. GitHub에 코드 푸시
2. Cloudflare Dashboard → Pages → Create Project
3. GitHub 저장소 선택
4. 빌드 설정:
   - Build command: (없음 - 정적 파일)
   - Build output directory: `/web`

### 방법 2: Wrangler 직접 배포

```bash
wrangler pages deploy web --project-name=storycut-ui
```

---

## 🔧 환경 변수 설정 (Pages)

Pages 프로젝트 → Settings → Environment Variables:

```
API_URL=https://api.storycut.com
```

---

## 🐍 Python 백엔드 배포

### 옵션 A: Cloudflare Workers (Python Workers)

**주의**: Python Workers는 제한적이므로 권장하지 않음

### 옵션 B: 외부 서버 (권장)

**Railway / Render / Fly.io / AWS Lambda**

#### Railway 예시:

1. `railway.toml` 생성:
```toml
[build]
builder = "nixpacks"

[deploy]
startCommand = "uvicorn api_server:app --host 0.0.0.0 --port $PORT"

[env]
OPENAI_API_KEY = "${{OPENAI_API_KEY}}"
RUNWAY_API_KEY = "${{RUNWAY_API_KEY}}"
```

2. 배포:
```bash
railway login
railway init
railway up
```

3. 배포된 URL을 Cloudflare Worker의 `BACKEND_URL`에 설정:
```bash
wrangler secret put BACKEND_URL
# 입력: https://your-app.railway.app
```

---

## 📊 D1 초기 데이터 삽입

### 테스트 사용자 생성

```bash
wrangler d1 execute storycut-db --command "
INSERT INTO users (id, email, api_token, credits)
VALUES ('test-user-001', 'test@example.com', 'test-token-123', 100);
"
```

---

## 🧪 테스트

### 1. API 헬스 체크
```bash
curl https://api.storycut.com/api/health
```

### 2. 영상 생성 테스트
```bash
curl -X POST https://api.storycut.com/api/generate \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer test-token-123" \
  -d '{
    "topic": "테스트 주제",
    "genre": "mystery",
    "duration": 30
  }'
```

### 3. 상태 확인
```bash
curl https://api.storycut.com/api/status/{project_id}
```

---

## 📈 모니터링

### Cloudflare Dashboard

- **Workers**: 요청 수, 에러율, CPU 시간
- **D1**: 쿼리 수, 읽기/쓰기
- **R2**: 저장소 사용량, 요청 수
- **Queue**: 큐 깊이, 처리 속도

### Wrangler CLI로 로그 확인

```bash
# Worker 로그
wrangler tail

# Queue Consumer 로그
wrangler tail --name storycut-queue-consumer
```

---

## 💰 비용 예측

### Cloudflare 무료 플랜 제한

- **Workers**: 100,000 요청/일
- **D1**: 5GB 저장소, 500만 행 읽기/일
- **R2**: 10GB 저장소, 1백만 Class A 작업/월
- **Queue**: 100만 메시지/월

### 유료 플랜 (Workers Paid)

- **Workers**: $5/월 + $0.30/백만 요청
- **D1**: $5/월 + $1/GB 저장소 + $0.001/백만 행 읽기
- **R2**: $0.015/GB 저장소 + $4.50/백만 Class A 작업

---

## 🔐 보안 체크리스트

- [ ] API 토큰을 Secret으로 저장 (`wrangler secret put`)
- [ ] CORS 설정 확인
- [ ] Rate Limiting 구현 (Cloudflare Rate Limiting Rule)
- [ ] SQL Injection 방지 (Prepared Statements 사용)
- [ ] 사용자 입력 검증

---

## 🐛 문제 해결

### Worker 배포 실패
```bash
# wrangler.toml 문법 확인
wrangler deploy --dry-run
```

### D1 연결 오류
```bash
# D1 리스트 확인
wrangler d1 list

# 테이블 확인
wrangler d1 execute storycut-db --command "SELECT name FROM sqlite_master WHERE type='table';"
```

### Queue 처리 안 됨
```bash
# Queue 상태 확인
wrangler queues list

# Consumer 로그 확인
wrangler tail --name storycut-queue-consumer
```

---

## 📚 참고 자료

- [Cloudflare Workers 문서](https://developers.cloudflare.com/workers/)
- [D1 데이터베이스 가이드](https://developers.cloudflare.com/d1/)
- [R2 스토리지 가이드](https://developers.cloudflare.com/r2/)
- [Queues 가이드](https://developers.cloudflare.com/queues/)
- [Cloudflare Pages 문서](https://developers.cloudflare.com/pages/)
