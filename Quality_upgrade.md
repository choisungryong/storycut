StoryCut / Klippa MV 고품질 마감 적용: 구현 지시서 (Final Sprint)
0) 목표(What we ship)

현재 MV 파이프라인(가사/씬 → 이미지(나노바나나) → (옵션)I2V → FFmpeg 합성)을 유지하면서, “상업용 느낌”을 좌우하는 핵심 기술 요소를 짧은 기간에 안정적으로 반영한다.

핵심 KPI(체감 품질)

컷 리듬(60~120 컷 체감): 이미지 생성 수를 늘리지 않고(기본 24장 유지), 편집에서 컷 수를 만든다.

캐릭터/룩 일관성: 앵커 검증 + 스타일 락 토큰 강제

중복샷 제거: pHash 기반 자동 디듀프 + 대체 정책

스톡 B-roll 혼합: 비용 절감 + 세트 반복감 감소

전환/마감 품질: cut/짧은 xfade/짧은 암전 자동 정책 + 인코딩 규격 고정 + 워터마크

1) 변경 범위(Non-goals 포함)
1.1 반드시 한다 (In scope)

Shot Planner 확장: 샷 역할(role) / 스톡 후보 / 전환 메타 / 파생컷 계획 출력

Derived Cuts(파생 컷) 생성: 24 Keyframe → 72~120 컷으로 분할(이미지 재생성 없이)

Style Lock 주입: 모든 프롬프트에 스타일 락/금지 토큰 강제

Dedup(pHash): 이미지 중복/유사 샷 자동 감지 → 재생성 대신 구도/스톡 대체 우선

Stock Fetcher(Pexels 우선): B-roll/Transition에만 사용, 실패 시 fallback

Transition Planner: cut/xfade/fadeblack 자동 선택

FFmpeg Final Spec: 30fps, 목표 비트레이트, 오디오 48kHz, 워터마크

1.2 이번 스프린트에서는 하지 않는다 (Out of scope)

고비용 CV 기반 “정체성 임베딩 검증(ArcFace)”의 완전 자동화(시간 부족)

대신 LLM+휴리스틱 기반 검증 + “금지 토큰” 강화 + (선택)간단 얼굴 검출 정도로 MVP

3D/2.5D depth parallax 고급 합성(추후)

2) 데이터 계약(Schemas)
2.1 입력: MV Storyboard (기존 구조에 필드 추가)

샷(=Keyframe) 단위 JSON에 아래를 추가한다.

{
  "shot_id": 1,
  "lyric": "…",
  "scene_prompt": "…",                  // 기존
  "duration_sec_base": 12.0,            // 기존 고정값
  "role": "hero|broll|transition",       // 신규
  "active_characters": ["Hero"],         // 기존/신규
  "style_lock": {                        // 신규
    "palette": "...",
    "lens": "...",
    "lighting": "...",
    "texture": "..."
  },
  "negative_lock": ["..."],              // 신규(금지 토큰)
  "stock_candidate": true,               // 신규
  "stock_query": ["...","..."],          // 신규(영문)
  "i2v_candidate": false,                // 기존/신규
  "keyframe_asset": {                    // 신규
    "source": "nanobanana|pexels|pixabay",
    "uri": "path_or_url",
    "meta": {}
  }
}

2.2 출력: Derived Cuts Plan (최종 편집용 컷 리스트)

Keyframe 24장을 편집 컷 72~120개로 확장한 리스트.

{
  "cut_id": "k01_c02",
  "parent_shot_id": 1,
  "start_sec": 0.0,
  "duration_sec": 0.9,
  "transform": {
    "reframe": "wide|medium|close|detail",
    "crop_rect": [x,y,w,h],
    "kenburns": {"zoom_from":1.0,"zoom_to":1.08,"pan":[0.02, -0.01]},
    "shake_px": 1,
    "overlay": ["film_grain","light_leak","rain_particles"],
    "freeze_frames": 0
  },
  "transition_to_next": "cut|xfade|fadeblack",
  "transition_frames": 6,
  "asset_uri": "keyframe_or_stock_asset_path",
  "role": "hero|broll|transition",
  "log": ["..."]
}

3) 핵심 설계: 비용 폭발 없이 60~120컷 만들기
3.1 Keyframes(이미지 생성)는 기본 24 유지

이미지 생성 호출 수는 최대한 유지

컷 수는 “편집 파생”으로 만든다.

3.2 Derived Cuts 생성 규칙 (MVP)

기본: 각 keyframe에서 3 cuts 파생 → 24×3=72컷

후렴/클라이맥스 구간: 4~5 cuts 파생 → 90~120컷 체감 가능

cut 길이 분포(권장):

Verse: 1.5~3.0s

Chorus/Hook: 0.6~1.2s

Bridge: 1.0~2.0s

샷당 12초 고정은 “내부 분할(서브컷)”로 해결 (합성 단계에서 분할)

3.3 파생 컷 변형(재생성 없이)

각 cut은 동일 이미지를 사용하되:

reframe/crop: wide→medium→close (삼분할 위치 변화 포함)

kenburns: zoom/pan

micro-shake: 1~2px

overlay: film grain / light leak / particles

freeze: 드랍 직전 3~6프레임만

4) Style Lock / Negative Lock (반드시)
4.1 프롬프트 빌더 변경

모든 Nanobanana 호출 프롬프트 앞/뒤에 아래를 강제 삽입:

STYLE LOCK(항상 동일): 팔레트/렌즈/조명/텍스처 문장

NEGATIVE LOCK(항상 동일 + MV별 추가):

“no cartoon, no anime, no illustration”

“no different ethnicity, no different skin tone”

“no different outfit colors, keep signature outfit”

“no extra characters, no random faces”

broll/transition에서는 “no people, no face” 강제

4.2 Shot Type 스케줄러(구도 다양성)

LLM이 shot마다 “camera instruction”을 반드시 포함:

wide / medium / close-up / detail

low angle / high angle / side / back view / silhouette

(중복 방지에 필수)

5) Dedup(pHash) + 대체 정책 (재생성 최소화)
5.1 pHash 계산

keyframe 이미지마다 pHash 계산

이전 keyframes와 해밍거리 비교

임계치(권장):

dist ≤ 5: 거의 동일 → FAIL

6 ≤ dist ≤ 10: 유사 → 상황에 따라 FAIL(우선 대체)

dist > 10: OK

5.2 처리 정책 (비용 보호)

FAIL 시 우선순위:

구도 변경(프롬프트 camera instruction 변경) 후 1회 재생성

그래도 FAIL이면 role 강등(broll/transition) + no people 환경샷으로 재생성(1회)

그래도 FAIL이면 스톡으로 대체(가능하면)

재생성 예산:

keyframe당 최대 1~2회(하드 제한)

6) Stock B-roll 혼합 (Pexels 우선, 선택적)
6.1 사용 규칙

role == hero: 절대 스톡 금지

role in {broll, transition}: 스톡 허용

스톡 비중 목표: 전체 cuts 기준 10~20% (과하면 이질감)

6.2 자동 분류(LLM)

lyric/scene이 환경 위주면 stock_candidate=true

stock_query(영문) 3~6개 생성

6.3 Pexels API (MVP)

env: PEXELS_API_KEY

영상 검색 → top N(10) → 해상도≥720p, 길이≥필요 duration → 다운로드 캐시

실패 시: Nanobanana 환경샷(no people) fallback

7) Transition Planner (Cut / Xfade / Fade-to-black)
7.1 기본 정책

Default: cut

xfade: 4~8 frames (30fps 기준 0.13~0.27s)

fadeblack: 2~4 frames (0.07~0.13s)

7.2 선택 규칙

HERO 인접(A.role==hero or B.role==hero) → 무조건 cut

둘 다 broll/transition + look 유사 → short xfade

source/룩 충돌(스톡↔AI, 팔레트 차이 큼) → fadeblack

7.3 look 유사도(MVP)

시간 부족 시 간단 휴리스틱:

같은 source면 유사로 간주

또는 대표 프레임 평균 밝기/채도만 비교(간단 통계)

8) FFmpeg 합성: 품질 규격 고정 + 워터마크
8.1 최종 출력 규격(필수)

Video: 1280×720 또는 1920×1080(현 상태 유지 가능), 30fps

Target bitrate: 3~6 Mbps(720p) / 6~10 Mbps(1080p)

Codec: H.264 High Profile 권장(가능하면)

Audio: AAC 48kHz, 160~192 kbps

Pixel format: yuv420p (호환성)

8.2 워터마크

텍스트: “Made with Klippa”

위치: 우상단(자막과 안 겹치게)

불투명도: 0.25~0.35

무료 플랜: 항상 ON / 유료: OFF 옵션(플래그로)

8.3 마감 룩(공통 적용)

subtle grain

slight saturation/contrast normalization

(과한 LUT 금지: 이질감 증가)

9) 로깅/설명 가능성(필수)

각 shot/cut마다 아래 로그 저장:

role 결정 근거

style lock 주입 여부

pHash 값/유사도/디듀프 결과

스톡 사용 여부(pexels_id 등)

전환 선택(cut/xfade/fadeblack) 이유

재생성 횟수/예산 소비

사용자 UI에는 간단히:

“일관성 체크 ✅ / 중복 제거 ✅ / 스톡 B-roll 적용 ✅” 정도만 노출 가능

10) 개발 태스크(우선순위)
P0 (오늘 반드시)

Derived Cuts Plan + Composer 적용(24→72컷 이상)

Style Lock + Negative Lock 강제

pHash Dedup + 재생성 제한/대체 정책

Final FFmpeg 인코딩 규격(특히 오디오 160~192kbps)

워터마크 삽입(플래그)

P1 (되면 매우 좋음)

Pexels Stock Fetcher + 캐싱 + fallback

Transition Planner 자동화(xfade/fadeblack)

11) 수용 기준(Acceptance Criteria)

이미지 생성 수를 24(±소폭)로 유지하면서도, 최종 MV는 72컷 이상 “컷 체감”이 난다.

HERO 샷에서 캐릭터가 튀는(인종/피부톤/의상 급변) 사례가 눈에 띄게 감소한다.

유사 샷 반복이 줄고(디듀프 작동 로그), 재생성 예산이 통제된다.

스톡은 hero에 섞이지 않고 broll/transition에만 제한적으로 들어간다.

최종 출력이 30fps, 오디오 48kHz 160~192kbps로 고정된다.

“Made with Klippa” 워터마크가 정상 삽입된다.

12) 구현 팁(코딩에이전트 참고)

“컷 수 확장”이 가장 큰 체감 레버이므로, Pexels/전환 자동화보다 먼저 끝낸다.

파생 컷은 이미지 재생성 없이 FFmpeg crop/scale/pan/zoom으로 처리한다.

스톡/오버레이 루프는 로컬 assets 폴더로 두고 반복 사용(비용 0).