# STORYCUT - Quick Start Guide

빠르게 시작하는 가이드입니다.

---

## 1. 설치 (5분)

### 1.1 FFmpeg 설치

**Windows (PowerShell 관리자 권한):**
```powershell
choco install ffmpeg
```

**macOS:**
```bash
brew install ffmpeg
```

**Linux:**
```bash
sudo apt-get install ffmpeg
```

### 1.2 Python 패키지 설치

```bash
pip install -r requirements.txt
```

### 1.3 API 키 설정

`.env` 파일 생성:
```bash
cp .env.example .env
```

`.env` 편집:
```
OPENAI_API_KEY=sk-your-key-here
```

OpenAI API 키 발급: https://platform.openai.com/api-keys

---

## 2. 첫 번째 영상 생성

### 명령 실행

```bash
python cli/storycut_cli.py
```

### 입력 예시

```
Genre: emotional
Mood: melancholic
Style: cinematic animation
Duration: 90
Story idea: (Enter를 눌러 AI가 자동 생성하도록 함)
```

### 실행 과정

```
1. 스토리 생성 (10-30초)
2. Scene별 영상 생성 (Scene당 30-60초)
3. TTS 내레이션 생성 (Scene당 5-10초)
4. 배경 음악 선택 (즉시)
5. FFmpeg 합성 (10-20초)
```

### 결과

`output/youtube_ready.mp4` 파일이 생성됩니다.

---

## 3. 생성된 영상 확인

```bash
# Windows
start output/youtube_ready.mp4

# macOS
open output/youtube_ready.mp4

# Linux
xdg-open output/youtube_ready.mp4
```

---

## 4. 유튜브 업로드

1. YouTube Studio 접속
2. "만들기" > "동영상 업로드" 클릭
3. `youtube_ready.mp4` 선택
4. 제목, 설명, 태그 입력
5. 썸네일 업로드 (권장: 영상의 한 장면 사용)
6. 게시

---

## 5. MVP 제한사항

현재 MVP 버전은:

✅ 자동 스토리 생성
✅ Scene별 영상 생성 (placeholder 또는 API)
✅ TTS 내레이션
✅ 배경 음악
✅ 영상 합성

⚠️ 다음은 아직 미구현:
- 고품질 영상 생성 (API 키 필요)
- 자동 썸네일 생성
- 자막 생성
- 쇼츠 변환

---

## 6. 자주 묻는 질문

**Q: API 비용이 얼마나 드나요?**
A: OpenAI 기준, 90초 영상 1개당:
- GPT-4 스토리 생성: $0.05-0.10
- TTS (10 scenes): $0.15-0.30
- 합계: 약 $0.20-0.40

**Q: 영상 품질이 낮아요.**
A: MVP는 placeholder 영상을 사용합니다. Runway API 키를 설정하면 고품질 영상이 생성됩니다.

**Q: 한글 내레이션이 가능한가요?**
A: 네, OpenAI TTS는 다국어를 지원합니다. 한글 스토리를 입력하면 한글 내레이션이 생성됩니다.

**Q: 실패한 Scene만 다시 생성할 수 있나요?**
A: 현재 MVP에서는 전체 재생성만 가능합니다. Scene 단위 재시도는 향후 버전에서 지원 예정입니다.

---

## 7. 다음 단계

생성에 성공했다면:

1. ✅ 다양한 장르로 실험
2. ✅ 다른 mood/style 조합 시도
3. ✅ 영상 길이 최적화 (60-120초 권장)
4. ✅ 유튜브 업로드 및 피드백 수집

---

## 문제가 있나요?

GitHub Issues에 리포트해주세요: https://github.com/your-repo/storycut/issues

---

**Happy Creating! 🎬**
