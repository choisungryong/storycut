# STORYCUT v2.0 - 다음 단계 로드맵

## ✅ 완료된 작업

- [x] P0 기능: Hook 비디오, Ken Burns, Audio Ducking, Subtitle
- [x] P1 기능: Context Carry-over
- [x] P2 기능: Optimization Agent
- [x] 웹 UI 구현 (FastAPI + WebSocket)
- [x] Cloudflare 아키텍처 설계
- [x] Worker, Queue Consumer, D1 스키마 작성
- [x] 배포 가이드 작성
- [x] **TTS 다각화 (Google Neural2 A/C, Gemini Flash/Pro)**
- [x] **Cloudflare Pages 자동 배포 (Git 연결)**
- [x] **R2 스토리지 연동 (생성 영상 자동 업로드)**

---

## 🎯 다음 우선순위 작업 (New)

### **Phase 4: 프론트엔드 대개편 (Next)** 🎨
사용자 경험 개선 및 모던한 디자인 적용이 시급합니다.

#### 4.1 UI/UX 리디자인 (Design System)
- [ ] 현재의 단순 HTML/CSS 구조 탈피
- [ ] **React / Next.js 도입 고려** (장기적 관점) 또는 현 HTML 구조 내 **모던 컴포넌트 라이브러리** 적용
- [ ] 반응형 레이아웃 강화 (모바일 최적화)
- [ ] 다크 모드 고도화 (글래스모피즘 등 트렌디한 스타일)

#### 4.2 기능 분리 (Feature Separation)
- [ ] **대시보드 분리**: 영상 생성 창과 히스토리/결과 창 분리
- [ ] **회원가입/로그인 페이지 별도 구축**
- [ ] 설정(Settings) 페이지 독립

---

### **Phase 1: 로컬 테스트 및 검증** ⚡

## 🎯 다음 우선순위 작업

### **Phase 1: 로컬 테스트 및 검증** ⚡

#### 1.1 환경 설정
```bash
# .env 파일 생성
cp .env.example .env

# API 키 설정 (.env 파일 편집)
OPENAI_API_KEY=sk-...
RUNWAY_API_KEY=...  # 선택사항
```

#### 1.2 기본 테스트
```bash
# 서버 시작 (이미 실행 중)
python api_server.py

# 브라우저에서 테스트
http://localhost:8000

# 짧은 영상 (30초)으로 테스트
- 주제: "간단한 이야기"
- 장르: Mystery
- Feature Flags: Ken Burns만 ON
```

#### 1.3 검증 항목
- [ ] 스토리 생성 정상 동작
- [ ] Scene 1: 이미지 + Ken Burns 정상 생성 (Hook 비디오 OFF)
- [ ] Scene 2-N: 이미지 + Ken Burns 정상 생성
- [ ] TTS 내레이션 생성
- [ ] FFmpeg 합성 성공
- [ ] 최적화 패키지 생성 (제목/썸네일/해시태그)
- [ ] WebSocket 실시간 진행상황 업데이트
- [ ] 영상 다운로드

---

### **Phase 2: Cloudflare 배포** ☁️ (완료)

#### 2.1 Wrangler 설치 및 로그인
```bash
npm install -g wrangler
wrangler login
```

#### 2.2 D1 Database 생성 및 스키마 적용
```bash
# DB 생성
wrangler d1 create storycut-db

# 스키마 적용
wrangler d1 execute storycut-db --file=cloudflare/schema.sql

# 테스트 사용자 추가
wrangler d1 execute storycut-db --command "
INSERT INTO users (id, email, api_token, credits)
VALUES ('user-001', 'test@example.com', 'test-token-123', 100);
"
```

#### 2.3 R2 Storage 생성
```bash
wrangler r2 bucket create storycut-videos
```

#### 2.4 Queue 생성
```bash
wrangler queues create storycut-video-queue
wrangler queues create storycut-dlq
```

#### 2.5 Worker 배포
```bash
# wrangler.toml 수정 (database_id, bucket_name 등)
# Secret 설정
wrangler secret put BACKEND_API_SECRET
wrangler secret put OPENAI_API_KEY

# 배포
wrangler deploy
```

#### 2.6 Python 백엔드 배포 (Railway)
```bash
# Railway 설치
npm install -g @railway/cli

# 로그인 및 배포
railway login
railway init
railway up

# 환경 변수 설정
railway variables set OPENAI_API_KEY=sk-...
railway variables set RUNWAY_API_KEY=...
```

#### 2.7 Cloudflare Pages 배포
```bash
# GitHub에 푸시
git add .
git commit -m "Add Cloudflare deployment"
git push origin main

# Pages 프로젝트 생성 (Dashboard)
# Build output directory: /web
```

---

### **Phase 3: 인증 및 크레딧 시스템** 🔐 (진행 중)

#### 3.1 인증 API 구현 (완료)
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

### **Phase 4: UI/UX 개선** 🎨

#### 4.1 로그인/회원가입 UI
- [ ] 로그인 모달
- [ ] 회원가입 폼
- [ ] 로그아웃 버튼
- [ ] 크레딧 잔액 표시

#### 4.2 대시보드
- [ ] 프로젝트 히스토리
- [ ] 크레딧 사용 내역
- [ ] 알림 센터

#### 4.3 에러 처리 개선
- [ ] API 키 없을 때 안내 메시지
- [ ] 크레딧 부족 시 충전 유도
- [ ] FFmpeg 오류 상세 메시지
- [ ] 재시도 버튼

---

### **Phase 5: 성능 최적화** 🚀

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

### **Phase 6: 분석 및 비즈니스** 📊

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

## 📝 체크리스트

### 지금 당장 해야 할 일 (우선순위 높음)

1. **API 키 설정**
   ```bash
   # .env 파일 생성 및 편집
   cp .env.example .env
   # OPENAI_API_KEY 설정
   ```

2. **로컬 테스트 실행**
   ```bash
   # 브라우저에서 http://localhost:8000
   # 30초 짧은 영상 생성 테스트
   ```

3. **Wrangler 설치 및 D1 생성**
   ```bash
   npm install -g wrangler
   wrangler login
   wrangler d1 create storycut-db
   ```

4. **Python 백엔드 Railway 배포**
   ```bash
   railway login
   railway init
   railway up
   ```

5. **Worker 배포**
   ```bash
   wrangler deploy
   ```

---

## 🛠️ 선택적 개선 사항

### 나중에 추가할 수 있는 기능

- [ ] 다국어 지원 (영어, 일본어 등)
- [ ] 음성 클로닝 (사용자 목소리)
- [ ] 커스텀 BGM 업로드
- [ ] 자막 편집 UI
- [ ] 영상 템플릿 갤러리
- [ ] API 문서 자동 생성 (Swagger UI)
- [ ] Webhook 지원 (영상 생성 완료 알림)

---

## 📞 문제 발생 시

### 로컬 테스트 실패
1. `.env` 파일 API 키 확인
2. `ffmpeg -version` 확인
3. `python api_server.py` 로그 확인

### Cloudflare 배포 실패
1. `wrangler.toml` 설정 확인
2. `wrangler deploy --dry-run` 실행
3. [DEPLOYMENT_GUIDE.md](cloudflare/DEPLOYMENT_GUIDE.md) 참고

---

## 🎉 최종 목표

**완전히 자동화된 조회수/수익형 유튜브 제작 머신!**

1. 사용자가 주제 입력
2. AI가 자동으로 영상 생성
3. 최적화된 제목/썸네일/해시태그 제공
4. 원클릭 유튜브 업로드 (향후)
5. 조회수 분석 및 개선 제안 (향후)
