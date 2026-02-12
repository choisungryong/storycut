# StoryCut/Klippa MV 품질 개선 – 최종 구현 지시문 (바로 전달용)

## 0) 목표 / 범위
### 목표
1) **Derived 컷의 줌/크롭이 이상해서 몰입이 깨지는 문제**(머리/턱 잘림, 랜덤 센터, 과도 줌) 제거  
2) **씬 간 연결이 “뚝뚝 끊김”으로 느껴지는 문제** 해결 (전환 연출화 + match-cut)  
3) **캐릭터 일관성/완성도**를 “상업용 기본기” 수준으로 강화 (QA + 프롬프트 락)

### 범위(이번 스프린트)
- **I2V는 제외**하고, 편집/합성 레이어에서 해결한다.
- 이미지 재생성 호출 수를 증가시키지 않거나(권장 0) 최소화한다.
- 브릿지샷 오버레이 에셋(mp4/webm)은 **현재 없음** → FFmpeg 내장 전환으로 대체하고, 나중에 에셋이 생기면 자동 적용되도록 플러그형 구조로 만든다.

---

## 1) 산출물(Deliverables)
1) `DerivedCutController` 모듈/클래스
2) `TransitionPlanner` 모듈/클래스 (Easing + BridgeShot + Match-cut 포함)
3) `CharacterQA` 모듈/클래스 (얼굴 임베딩 기반 앵커 검증)
4) `cuts.json` (derived cuts plan), `render.log` (QA/전환/캐릭터검증 로그)
5) 데모: 동일 MV로 **개선 전/후** 렌더 1개씩

---

## 2) 우선순위(필수 구현 순서)
1) DerivedCutController v2 (프레이밍 안정화)
2) TransitionPlanner v1 (끊김 완화: easing + 전환 + match-cut)
3) CharacterQA v1 (캐릭터 앵커 검증 + 렌즈/손 프롬프트 락)

---

# A. DerivedCutController v2 (필수)

## A1. 목적
Keyframe(원본 이미지) 1장을 여러 Derived Cut으로 분할하여 컷 수를 늘리되,
**랜덤 크롭/랜덤 센터/과도 줌**으로 인한 몰입 붕괴를 제거한다.

## A2. 입력/출력
### 입력
- `keyframe_image_path`
- `shot_role`: `hero|broll|transition`
- `shot_type`: `WIDE|MEDIUM|CLOSEUP|DETAIL`
- `cut_duration_sec`
- `subtitle_safe_area_ratio` (예: 0.20)
- (옵션) `subject_bbox` (없으면 내부에서 추출)

### 출력
- `DerivedCut[]` 각 cut은 최소 포함:
  - `crop_rect` (x,y,w,h)
  - `kenburns`: scale_from, scale_to, pan_from, pan_to
  - `shake_px`
  - `easing`
  - `qa_passed`, `qa_fail_reason`

## A3. 핵심 규칙(반드시)
### Rule 1) Subject-aware crop (필수)
Derived 생성 전 **피사체 bbox 확보**:
- 우선순위: `face_bbox` > `person_bbox` > `saliency_center`
- 구현: Gemini Vision 또는 OpenCV 얼굴/인물 검출
- Derived 중심점은 이미지 중앙이 아니라 **bbox 중심** 사용

### Rule 2) 랜덤 크롭 금지 – 템플릿 기반 크롭만 허용
- `WIDE`   scale 1.00~1.06  
- `MEDIUM` scale 1.06~1.15  
- `CLOSEUP` scale 1.12~1.22 (상한 1.25 절대 초과 금지)  
- `DETAIL` scale 1.10~1.30 (인물/얼굴 없는 컷에서만)

### Rule 3) Face-safe padding 강제(머리/턱 잘림 방지)
`CLOSEUP`/`MEDIUM`:
- 상단 padding ≥ 0.12H
- 하단 padding ≥ 0.10H
- 좌우 padding ≥ 0.08W
- bbox+padding이 **프레임 내 100% 포함**되어야 함

### Rule 4) 패닝 제한
- `max_dx = 0.03W`
- `max_dy = 0.02H`
- hero/closeup은 dy 더 낮게(예: 0.01H)

### Rule 5) 컷 내 줌 변화량 제한
- `Δscale <= 0.06` (6% 이내)

### Rule 6) Subtitle Safe Area 침범 금지
하단 `subtitle_safe_area_ratio` 영역에 얼굴 bbox가 겹치면 **폐기**

## A4. QA Gate(자동 폐기 조건)
아래 중 1개라도 위반 시 폐기 + 파라미터 재시도(재생성 없음):
- 얼굴 bbox 프레임 밖 1px라도 나감
- 턱/입 하단 padding 부족
- subject center 좌우 치우침 과다(프레임 중앙 기준 15% 초과)
- subtitle safe area 침범
- CLOSEUP scale > 1.25

## A5. 재시도 정책(비용 0)
- padding 증가 → scale 상한 감소 → center 재정렬 순으로 3~5회 재시도
- 실패 지속 시 `CLOSEUP -> MEDIUM` 강등(안전)

## A6. 로깅(필수)
- bbox_source(gemini/opencv/none), bbox_values
- shot_type, scale_from/to, dx/dy, easing
- qa_fail_reason(폐기된 컷 포함)

---

# B. TransitionPlanner v1 (필수) – Easing + BridgeShot + Match-cut

## B1. 목적
- 컷 내부 모션은 **Easing**으로 자연화  
- 컷 사이 연결은 **브릿지샷/전환 규칙**으로 뚝뚝 끊김 완화  
- 인물샷 연속은 **match-cut 규칙**으로 시선 점프 방지

## B2. Easing 규칙(컷 내부)
- linear 금지: `easeInOutSine` 또는 `easeInOutCubic` 기본
- 시작/끝 10% 구간 완만
- `Δscale <= 0.06` 유지

## B3. BridgeShot(전환 클립) 규칙
### 오버레이 에셋(현 상태)
- filmburn/lightleak/glitch mp4/webm **없음**
- 따라서 **FFmpeg 내장 전환으로 대체**:
  - `fadeblack` (2~4 frames)
  - `whiteflash` (1~2 frames, 과노출)
  - (옵션) `pixelize`/간단 글리치

### 구조(필수)
- `assets/overlays/` 폴더가 존재하고 파일이 있으면 사용
- 없으면 자동으로 내장 전환 fallback

## B4. 전환 선택 규칙(추천)
- default: `cut`
- 룩/소스 충돌(스톡↔AI, 팔레트 점프 큼) → `fadeblack(2~4f)`
- broll↔broll 무드 유사 → 짧은 `xfade(4~6f)` 또는 오버레이 있으면 filmburn
- 강박자/후렴 포인트 → `whiteflash(1~2f)`

## B5. xfade 제한(남발 금지)
- hero↔hero: xfade 금지(ghosting 유발)
- broll↔broll: 짧게만(4~6 frames)
- hero↔broll: xfade보다 fadeblack/whiteflash 우선

## B6. Match-cut(연속성) 규칙
### 1) 프레이밍 연속성
- 인물샷 연속 시 `subject_center_x` 유지:
  - 다음 컷 center는 이전 center에서 ±5% 이내로 제한
- 좌↔우 반전은 후렴 강박자에만 허용

### 2) 모션 방향 연속성
- 4~8컷 단위로 줌 방향 유지(줌인/줌아웃)
- 전환 구간에서만 방향 전환 허용

### 3) 밝기/채도 점프 완화(저비용)
- 컷별 대표 프레임 avgY/avgS 측정
- 점프가 크면 다음 컷 시작 3~6프레임 동안 3~8% 범위에서만 완화

## B7. TransitionPlanner I/O
### 입력
- `cuts[]` (각 cut에 role, source, palette_stats(avgY/avgS), subject_center_x/y, shot_type 포함)

### 출력
- `transition_to_next`: cut|xfade|fadeblack|whiteflash|pixelize|overlay
- `transition_frames`
- `overlay_asset_path`(있을 때만)

## B8. 로깅(필수)
- transition 선택 값/프레임/근거(룩충돌, role, beat 등)
- xfade 제한 적용 여부

---

# C. CharacterQA v1 (추가 – 캐릭터 완성도 강화)

## C1. 목적
- 이번 영상은 캐릭터 일관성이 전반적으로 괜찮지만,
  **클로즈업 왜곡/조명 변화/손 디테일**로 “고급감”이 떨어질 수 있다.
- 자동 QA로 **앵커 불일치(다른 배우)**를 조기에 차단한다.

## C2. 얼굴 임베딩 기반 앵커 검증(권장 필수)
### 입력
- `anchor_face_embedding` (Hero/Villain 등 캐릭터별)
- `generated_frame/image`

### 처리
1) 얼굴 탐지(없으면 `face_not_found` 로그)
2) 얼굴 임베딩 추출
3) 코사인 유사도 계산 → 임계치 미달이면 `sim_below_threshold`로 실패

### 결과
- 실패 시: (권장) 해당 컷은 derived 후보에서 제외하거나, 더 안전한 shot_type로 강등
- 임계치는 모델별로 초기엔 로그 기반 튜닝

## C3. 프롬프트 락(Style/Negative) – 캐릭터 샷에 강제
### 렌즈 고정(왜곡 방지)
- Positive: `portrait 85mm lens` 또는 `cinematic 50mm lens`, `natural facial proportions`
- Negative: `wide-angle distortion`, `fisheye`, `exaggerated facial features`

### 손/접촉 씬(커플/포옹) 토큰
- Positive: `natural hands`, `correct fingers`, `anatomically correct hands`
- Negative: `extra fingers`, `deformed hands`, `fused fingers`

## C4. 보조 검증(옵션)
- 의상/팔레트가 명확한 캐릭터는 torso 영역 dominant color로 이탈 감지(저비용)

## C5. 로깅(필수)
- face_detected 여부, similarity score
- fail_reason(face_not_found, sim_below_threshold, palette_out_of_range)

---

## 3) 수용 기준(Acceptance Criteria)
- (프레이밍) 얼굴/머리/턱 잘림이 거의 사라지고, 과도 줌/랜덤 센터가 제거됨
- (연결) 씬 전환이 부드러워지고 “툭툭 끊김” 체감이 크게 감소
- (전환) hero↔hero에서 ghosting 없이 자연스러운 컷 연결
- (캐릭터) 앵커 불일치(다른 배우) 컷이 자동으로 걸러지고 로그로 확인 가능
- (운영) 재생성 호출 수 증가 없이(권장 0) 편집/합성 레이어로 해결

---

## 4) 진행 메모
- 브릿지 오버레이 에셋이 없으므로 이번 스프린트는 **내장 전환으로 MVP 구현**
- 추후 `assets/overlays/`에 filmburn/lightleak/glitch 클립을 넣으면 자동 적용되게 확장 가능하도록 설계

