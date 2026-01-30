# StoryCut v2.0 - 진행 중인 과제

## ✅ 완료된 과제

### [완료] TTS 우선순위 수정 및 테스트
- ElevenLabs를 Primary TTS로 변경
- OpenAI TTS → pyttsx3 폴백 구조 구현
- ElevenLabs v2.x API 연동 완료
- 테스트 성공: 한국어 및 영어 TTS 정상 작동

### [완료] 캐릭터 레퍼런스 시스템 구현
- image_agent.py: generate_image() 반환 타입을 tuple로 변경
- character_reference_id 파라미터 추가
- NanoBanana API에 character_reference 전달
- master_image_id 저장 및 활용 구조 완성

### [완료] GOOGLE_API_KEY 설정
- .env 파일에 GOOGLE_API_KEY 추가
- Gemini 3 Pro, NanoBanana, Veo 3.1 통합 준비 완료

---

## 🔄 진행 중인 과제

### [진행 중] Task #1: Phase 1 로컬 테스트 및 검증
**목표**: 전체 시스템 통합 테스트

**테스트 항목**:
- [ ] 스토리 생성 (Gemini 3 Pro)
  - character_sheet 포함 JSON 생성
  - global_style 적용

- [ ] 이미지 생성 (NanoBanana)
  - 캐릭터 레퍼런스 ID 전달
  - visual_seed 적용

- [ ] TTS 생성 (ElevenLabs)
  - 감정별 voice 적용
  - duration 계산

- [ ] Ken Burns 효과
  - FFmpeg 합성
  - 카메라 워크 적용

- [ ] 최종 영상 합성
  - 비디오 + 오디오 + 배경음악
  - WebSocket 진행상황 업데이트

**예상 소요 시간**: 30-60초 영상 기준

**시작일**: 2026-01-29

---

## 📅 대기 중인 과제

### [완료] Task #2: 캐릭터 레퍼런스 시스템 통합 테스트
**설명**: 마스터 캐릭터 생성 및 씬 간 일관성 테스트
- POST /api/projects/{id}/characters/generate 테스트
- master_image_id 저장 확인
- NanoBanana character_reference 전달 확인
- 여러 씬에서 캐릭터 외형 일관성 검증

**우선순위**: 높음

---

### [완료] Task #3: Veo 3.1 비디오 생성 API 통합
**설명**: Google Veo 3.1 API를 video_agent.py에 통합
- _call_veo_api() 메서드 구현
- GOOGLE_API_KEY 사용
- image_prompt를 비디오 프롬프트로 활용
- 이미지 기반 Ken Burns 대신 실제 비디오 생성

**우선순위**: 중간

---

### Task #4: 카메라 워크 다양화 개선
**설명**: Ken Burns 효과의 카메라 워크를 더 다양하게
- CameraWork enum 활용 (ZOOM_IN, ZOOM_OUT, PAN_LEFT, PAN_RIGHT, STATIC)
- ffmpeg_utils.py에 각 카메라 워크별 필터 적용
- scene.camera_work에 따라 동적 FFmpeg 필터 생성

**우선순위**: 낮음

---

## 📝 참고 사항

- 전체 로드맵은 `NEXT_STEPS.md` 참고
- API 키 설정은 `.env` 파일에서 관리
- 테스트 스크립트: `test_tts.py` (TTS 테스트)

**마지막 업데이트**: 2026-01-29 21:50
