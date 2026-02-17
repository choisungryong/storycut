# [지시서] MV 출력 품질 + 카메라 문법(4필드) + DerivedCut/Easing/Transition 반영

> 기준: 최신 생성 영상 분석 결과 반영  
> 참고: OpenDirection Prompt Builder의 카메라 문법(Geometry/Execution/Temporal/Narrative) 및 “카테고리당 1~2개 태그 제한” 운영 팁을 설계에 반영함.  
> - https://opendirection.studio/prompt-builder

---

## 0) 관찰 요약(이번 영상에서 “더 올릴 수 있는” 핵심)

### A. 인코딩(가장 확실한 체감 포인트)
현재 출력은 **1080p/30fps 형태는 좋지만**, 비디오 비트레이트가 낮아(대략 **~1.4Mbps 수준**) 디테일이 뭉개지고 “스틸 합성” 느낌이 강화됩니다.  
➡️ **출력 인코딩 목표를 3~6Mbps(720p 기준) / 6~10Mbps(1080p 기준)**로 올리면 체감이 큽니다.

### B. “줌이 이상함/끊김”은 생성 이미지 문제가 아니라 **DerivedCut(리프레임) + Transition 설계** 문제
- 씬 자체(이미지)는 준수하나, 리프레임이 **Δscale·패닝·세이프패딩** 규칙 없이 돌아가면 “기계적인 줌”이 됩니다.
- 씬 간 연결이 hard cut만이면 MV에서 “뚝뚝 끊김” 체감이 커집니다.

---

## 1) OpenDirection Prompt Builder 노하우를 반영한 설계 변경(필수)

OpenDirection 방식처럼 씬 JSON에 카메라 문법을 **4개 필드**로 분리하고, **각 카테고리당 1~2개 태그로 제한**하여 일관성을 확보합니다.

- Geometry: movement / vector / angle / shot_size  
- Execution: rig / relation  
- Temporal: timing / transition  
- Narrative: intent

> 운영 원칙: 프롬프트를 길게 쓰는 것이 아니라 **필드를 구조화**하고 **태그 수를 제한**하는 것이 목표입니다.

---

# 2) 구현 요구사항(우선순위 순)

## [P0] 출력 인코딩 정책 업그레이드 (즉시 반영)

### 목표
- (권장) **720p / 30fps / 3~6Mbps / AAC 160~192k**
- (대안) 1080p 유지 시 **6~10Mbps** 이상으로

### FFmpeg 템플릿(ABR 방식)

**720p 권장**
```bash
ffmpeg -y -i input.mp4 \
  -vf "scale=1280:720:flags=lanczos" \
  -c:v libx264 -profile:v high -level 4.1 -pix_fmt yuv420p \
  -b:v 5M -maxrate 6M -bufsize 12M \
  -g 60 -keyint_min 60 -sc_threshold 0 \
  -c:a aac -ar 48000 -b:a 192k \
  output_720p.mp4
```

**1080p 유지**
```bash
ffmpeg -y -i input.mp4 \
  -c:v libx264 -profile:v high -level 4.1 -pix_fmt yuv420p \
  -b:v 8M -maxrate 10M -bufsize 20M \
  -g 60 -keyint_min 60 -sc_threshold 0 \
  -c:a aac -ar 48000 -b:a 192k \
  output_1080p.mp4
```

---

## [P0] 씬 JSON에 “카메라 4필드” 추가 + 태그 수 제한(카테고리당 1~2개)

### 2.1 스키마 변경
`MVScene` 또는 scene 모델에 아래 필드를 추가:

```json
"camera": {
  "geometry": {
    "movement": ["Static" | "Dolly In" | "Pan Left" ...], 
    "vector":   ["Orbit" | "Arc CW" | "Pedestal Up" ...],
    "angle":    ["Eye Level" | "Low Angle" | "High Angle" ...],
    "shot_size": ["Close-Up" | "Medium Shot" | "Wide Shot" ...]
  },
  "execution": {
    "rig": ["Handheld" | "Tripod" | "Crane" ...],
    "relation": ["POV" | "Over-the-Shoulder" ...]
  },
  "temporal": {
    "timing": ["Normal Speed" | "Slow" ...],
    "transition": ["cut" | "xfade" | "fadeblack" | "fadewhite"]
  },
  "narrative": {
    "intent": ["reveal" | "intimate" | "tension" | "closure" ...]
  }
}
```

### 2.2 태그 제한 규칙(필수)
- movement/vector/angle/shot_size **각각 최대 1개**
- execution( rig / relation ) 각각 최대 1개
- temporal: transition 1개
- narrative: intent 1개  
- 미지정이면 기본값:
  - movement=Static, shot_size=Medium Shot, transition=cut

### 2.3 LLM 프롬프트 지시(Story → Scene 생성 단계)
- “카메라 4필드를 반드시 출력”
- “각 카테고리에서 1개만 선택”
- “전환은 훅/구간 전환에서만 xfade/fade, 기본은 cut”

---

## [P0] DerivedCutController(Easing 포함) — “이상한 줌” 근절용 하드 제약

### 3.1 CutPlan 파라미터화(씬마다 고정)
각 컷에 아래 파라미터를 산출하여 render 단계에 전달:

```json
"reframe": {
  "scale_start": 1.00,
  "scale_end": 1.06,
  "pan_x": 0.02,
  "pan_y": -0.01,
  "ease": "inOutQuad",
  "face_safe_padding": {"top": 0.18, "bottom": 0.10, "left": 0.08, "right": 0.08}
}
```

### 3.2 하드 제약(필수)
- `scale_end` 범위:
  - Close: 1.00~1.08
  - Medium: 1.00~1.06
  - Wide: 1.00~1.03
- `Δscale_per_sec` 제한: **0.02 이하**
- pan 제한: 프레임 너비/높이의 **±3% 이내**
- 얼굴 검출 실패 시:
  - **줌 금지(scale_end=scale_start)** + 미세 패닝만 허용(또는 정지)

### 3.3 Easing(필수)
지원 easing(최소):
- linear
- inOutQuad
- inOutCubic

---

## [P0] Transition Planner — 씬 연결 “뚝뚝 끊김” 최소 세트 적용

에셋(필름번/lightleak) 없이도 FFmpeg 내장 전환만으로 충분히 개선됩니다.

### 4.1 룰 기반 전환 선택(필수)
- 기본: `cut`
- 같은 인물/비슷한 구도 지속: `xfade(0.12~0.20s)`
- 감정 전환/행(Act) 전환: `fadeblack(0.10~0.18s)`
- 훅/강조(후렴 진입 등): `fadewhite(0.06~0.10s)`

### 4.2 xfade 과다 방지(필수)
- xfade 연속 사용 금지(최소 2컷 간격)
- xfade는 **전체 컷의 20~30% 이내**로 제한

---

## [P1] “Match-cut(연속성)” 최소 규칙(씬 간 어색함 추가 감소)

### 5.1 연속성 제약
- 동일 인물 연속이면:
  - 얼굴 크기(또는 bbox 스케일) 변화 **±5~8% 이내**
  - 중심점 이동량 제한(±3% 이내)
  - 줌 방향 연속성 유지(zoom-in 다음도 zoom-in)

### 5.2 디듀프(pHash)와 연동
- pHash 유사하면:
  1) 동일 컷의 reframe 파라미터만 변경(줌/팬 방향 반대로)
  2) 그래도 비슷하면 재생성(예산 1회)
  3) 최종 폴백: establishing/detail 브릿지컷(환경 중심)

---

## [P1] 로깅/재현성(필수: 운영 효율 급상승)

`out/<project_id>/render.log`에 씬별로 기록:
- scene_id / shot_size / transition / reframe(scale_start/end, pan, ease)
- face_detected 여부 + padding 적용 여부
- xfade 사용 여부/시간
- 최종 인코딩 스펙(해상도/bitrate)

---

# 3) 파일별 작업 지시(코딩에이전트 착수용)

## 3.1 `schemas/mv_models.py` (또는 scene schema)
- `camera` 4필드 구조 추가
- `reframe` 파라미터 구조 추가
- `transition` 필드(temporal.transition) 확정

## 3.2 `scene_orchestrator.py` 또는 MV scene 생성 모듈
- LLM 출력에 `camera` 필드 포함시키기
- 각 카테고리 1개 태그 제한 validator 추가(초과 시 첫 1개만 채택)

## 3.3 `prompt_builder.py`
- `camera.geometry.shot_size` → shot_type 프레이밍 토큰 매핑
- camera movement/vector는 “derived cut”과 충돌 가능하므로:
  - **I2I 프롬프트에는 movement를 최소로**, 대신 derived cut에서 반영(권장)

## 3.4 `derived_cut_controller.py` (또는 리프레임 모듈)
- scale/pan/ease/face-safe padding 구현
- 얼굴 미검출 시 안전 모드(줌 금지)

## 3.5 `transition_planner.py` (없으면 mv_pipeline 내 함수로)
- 룰 기반 transition 선택
- xfade 제한 로직
- FFmpeg filtergraph 생성

## 3.6 `mv_pipeline.py`
- render 단계에서:
  - reframe 적용 → transition 적용 → 최종 인코딩(목표 bitrate) 순으로 파이프 구성
- `Made with Klippa` 워터마크는 corner 고정(이미 적용되어 있으면 유지)

---

# 4) “OpenDirection 스타일”을 우리 MV에 맞게 운영하는 원칙(팀 룰)

1) 카메라 태그는 4필드로 관리(Geometry/Execution/Temporal/Narrative)  
2) **카테고리당 1~2개만 선택**(과다 태그 금지)  
3) “카메라 움직임”은 생성 모델 프롬프트에서 과욕 부리지 말고, **DerivedCut에서 물리적으로 구현**한다.  
4) 전환은 “항상 넣기”가 아니라, 룰 기반으로 제한적으로(20~30%) 넣는다.

---

# 5) 완료 체크(납품 확인용)

- [ ] 최종 출력이 720p라면 3~6Mbps(또는 1080p 6~10Mbps) 달성
- [ ] scene JSON에 camera 4필드가 저장되고, 카테고리별 1개 제한이 동작
- [ ] DerivedCut easing 적용(Linear 대비 ‘툭툭’ 감소)
- [ ] Transition 룰 적용 + xfade 제한
- [ ] render.log에 씬별 파라미터가 남음
