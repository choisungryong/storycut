1) DerivedCutController 구현 지시문 (Markdown)
목적

Keyframe(원본 이미지) 1장을 여러 개의 Derived Cut으로 분할해 컷 수를 늘리되, **줌/크롭이 이상해서 몰입이 깨지는 문제(머리/턱 잘림, 랜덤 센터, 과도 줌)**를 제거한다.
랜덤 크롭/랜덤 센터 금지, Subject-aware crop(피사체 인식) 기반으로만 프레이밍한다.

입력/출력
입력

keyframe_image_path

shot_role: hero | broll | transition

shot_type: WIDE | MEDIUM | CLOSEUP | DETAIL

cut_duration_sec

subtitle_safe_area: 하단 안전영역 비율(예: 0.20 = 하단 20%)

(옵션) subject_bbox: 외부에서 주입 가능(없으면 내부에서 추출)

출력

DerivedCut[] (컷 리스트)

각 컷은 crop_rect, kenburns_params, shake_px, easing, qa_passed 등을 포함

핵심 규칙 (반드시)
Rule 1) Subject-aware 프레이밍(필수)

Derived Cut 생성 전, 이미지에서 주 피사체 bbox를 확보한다.

우선순위:

face_bbox (hero/closeup/medium일 때 최우선)

person_bbox (wide/medium에서 사용)

없으면 saliency_center(fallback)

구현 옵션:

(권장) Gemini Vision 호출로 face_bbox, person_bbox 추출(JSON normalized 좌표)

(대체) OpenCV/로컬 얼굴 검출

Derived 컷의 center는 이미지 중앙이 아니라 bbox 중심을 사용한다.

Rule 2) 랜덤 크롭 금지 → 템플릿 크롭만 허용

크롭/줌은 아래 shot_type 템플릿 범위에서만 결정한다.

WIDE : scale 1.00 ~ 1.06 (거의 고정)

MEDIUM: scale 1.06 ~ 1.15

CLOSEUP: scale 1.12 ~ 1.22 (상한 1.25 절대 초과 금지)

DETAIL: scale 1.10 ~ 1.30 (인물/얼굴 없는 컷에서만)

Rule 3) Face-safe padding 강제(머리/턱 잘림 방지)

CLOSEUP / MEDIUM에서 반드시 적용:

bbox에 padding 적용:

상단 padding: 최소 0.12 * frame_h

하단 padding: 최소 0.10 * frame_h

좌우 padding: 최소 0.08 * frame_w

crop_rect는 bbox+padding이 100% 포함되도록 계산

머리 잘림 방지: face_bbox.top이 프레임 상단에 닿으면 실패로 처리(QA에서 폐기)

Rule 4) 패닝(센터 이동) 제한

랜덤 패닝/오프셋이 있어도 아래 제한을 강제:

max_dx = 0.03 * frame_w (3%)

max_dy = 0.02 * frame_h (2%)

hero/closeup은 dy를 더 낮게(예: 1%)

Rule 5) 컷 내 줌 변화량 제한(“줌이 이상함” 방지)

한 DerivedCut의 kenburns 줌 변화량:

Δscale <= 0.06 (최대 6% 이내)

예: 1.08 → 1.13 같은 급격 변화 금지

Rule 6) Subtitle Safe Area 침범 금지

하단 subtitle_safe_area(예: 20%) 영역에 얼굴 bbox가 겹치면 해당 derived 컷 폐기.

hero/medium/closeup에서 특히 강제

QA Gate(자동 폐기 조건)

DerivedCut 생성 후 아래 중 하나라도 위반하면 폐기하고 재시도(최대 N회):

얼굴 bbox가 프레임 밖으로 1px이라도 나감

턱/입이 프레임 아래쪽에 너무 붙음(하단 padding 부족)

인물 중심이 좌우로 과도하게 치우침(예: 프레임 중앙 기준 15% 이상)

subtitle_safe_area에 얼굴이 침범

closeup scale이 1.25 초과

재시도 정책 (비용 0)

재시도는 “재생성”이 아니라 템플릿 파라미터 변경으로만 수행:

shot_type 변경 금지(의도 유지)

padding 증가 → scale 상한 감소 → center 재정렬 순으로 시도

N=3~5회 내에 해결 못하면 MEDIUM으로 강등(가장 안전)

로깅(필수)

bbox_source: gemini|opencv|none

bbox_values

selected_shot_type, scale_from, scale_to, dx/dy

qa_fail_reason (폐기된 컷에 대해)

2) Easing + BridgeShot + TransitionPlanner 구현 지시문 (Markdown)
목적

컷 간 연결이 “뚝뚝 끊김”으로 느껴지는 문제를 해결한다.

컷 내부: Easing으로 기계적인 줌/패닝 제거

컷 사이: 크로스페이드 남발 대신 **브릿지 샷(0.1~0.3s)**로 전환을 연출화

전체: “연속성 규칙(match-cut)”로 시선 점프를 줄인다

A. Easing(컷 내부 모션) 규칙
적용 대상

모든 derived 컷의 zoompan / pan / scale 모션

규칙

linear 금지: easeInOutSine 또는 easeInOutCubic 기본

컷 시작/끝 10% 구간은 속도 완만(가감속)

Δscale <= 0.06 유지 (DerivedCutController 규칙과 동일)

기대 효과

“AI가 억지로 줌 걸어놓은 느낌” 제거

몰입감 상승

B. Bridge Shot(전환 클립) 규칙
브릿지 종류(assets 기반)

fadeblack: 2~4 frames

whiteflash: 1~2 frames(과노출)

filmburn/lightleak: 4~8 frames(0.13~0.27s)

glitch: 3~6 frames(0.10~0.20s)

브릿지는 오버레이 에셋(mp4/webm)을 로컬에 두고 재사용(비용 0)

선택 규칙(추천)

default: cut

다음 조건 중 하나면 브릿지 삽입:

룩 충돌(스톡↔AI, 팔레트 차이 큼) → fadeblack(2~4f)

둘 다 broll/transition이고 무드 유사 → filmburn(4~8f)

후렴/강박자 포인트 → whiteflash(1~2f)

템포가 빨라지는 구간(컷이 1초 이내) → glitch(3~6f)

크로스페이드(xfade) 사용 룰(남발 금지)

hero↔hero: xfade 금지(유령 겹침/얼굴 이상해 보임)

broll↔broll: 짧은 xfade 4~6 frames만 허용

hero↔broll: xfade보다 fadeblack 또는 filmburn 우선

C. Match-cut(연속성) 규칙
1) 프레이밍 연속성

인물샷 연속일 때:

이전 컷의 subject_center_x를 다음 컷에 전달

다음 컷의 center는 이전 center에서 ±5% 이내로 제한

좌↔우 “반전”은 후렴 강박자에만 허용

2) 모션 방향 연속성

기본: 4~8 컷 단위로 줌 방향 유지(줌인 유지/줌아웃 유지)

전환 구간에서만 방향 전환 허용

3) 밝기/채도 점프 완화(저비용)

컷별 대표 프레임의 평균 밝기/채도를 계산

다음 컷 시작 3~6프레임 동안 약한 정규화로 점프 완화

과한 보정 금지(룩 통일은 별도 단계에서)

D. TransitionPlanner I/O
입력

cuts[] (각 cut에 다음 필드 포함)

role, source(ai/stock), palette_stats(avgY/avgS), subject_center_x, shot_type

is_strong_beat(optional, 비트/후렴 지점)

출력

transition_to_next: cut|xfade|fadeblack|whiteflash|filmburn|glitch

transition_frames

overlay_asset_path(필요 시)

Acceptance Criteria

hero 구간에서 얼굴 겹침/ghosting 없이 자연스러운 컷 연결

룩 충돌 구간에서 “툭” 튀는 느낌이 fadeblack/filmburn로 완화

후렴 구간에서 전환이 리듬을 타며 “뮤비 편집감”이 생김