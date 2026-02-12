[긴급: 가사 자막 싱크 안정화 최종안 - Whisper 없이]

전제: 기존 'STT 타임스탬프 기반 timed_lyrics' + drawtext 다건 때문에 자막이 누락/깜빡임/밀림이 발생함.
이번 수정은 "절대 엉키지 않도록" 안정성 최우선으로 구현한다.

1) 자막 텍스트
- 자막 내용은 user_lyrics_text 100%만 사용한다(STT 텍스트는 합성에 사용 금지).
- STT는 텍스트가 아니라 '보컬 시작/끝 시간 추정' 용도로만 선택적으로 사용.

2) 타이밍(균등분배의 치명적 단점 방지)
- 절대 "전체 T 기준 완전 균등분배"를 하지 않는다.
- 반드시 '가사 구간 anchors'를 잡고, 그 구간 안에서만 분배한다.
  - anchor_start = (자동추정) 첫 보컬 시작 시점
  - anchor_end = (자동추정) 마지막 보컬 끝 시점 또는 T
- 분배 범위: [anchor_start, anchor_end]
- intro/outro(범위 밖)는 자막을 출력하지 않는다.

3) anchor 자동추정(필수)
- VAD 기반으로 anchor_start/anchor_end를 추정한다.
  - 보컬/음성 탐지 실패 시 fallback: anchor_start=0, anchor_end=T
- 추정 결과를 render.log에 저장한다.

4) 사용자 보정(필수, UI 최소)
- 사용자가 anchor_start를 1번만 보정할 수 있게 한다:
  - 방법 A: "가사 시작이 이 시점부터" 슬라이더(±5초)
  - 방법 B: "현재 재생 위치를 가사 시작으로 설정" 버튼 1개
- 사용자가 보정하면 그 값을 최우선으로 사용한다.
- (선택) anchor_end도 같은 방식으로 보정 가능하게 하면 더 좋다(필수 아님).

5) 자막 합성(누락/깜빡임 방지)
- drawtext 다건을 완전히 중단하고, lyrics.srt(또는 lyrics.ass) 파일을 생성하여 subtitles= 로 burn-in 한다.
- FFmpeg는 fps=30 고정, 오디오는 aresample=48000 + asetpts 적용해서 타임베이스 흔들림을 제거한다.

6) 구현 산출물
- tmp/lyrics.srt 생성
- render.log: lyrics_source=user_input_only, anchor_start/end(auto+user), timeline_mode=anchored_uniform, N_lines, T, srt_path

Acceptance:
- 같은 음원/가사로 2회 렌더 시 자막이 누락/깜빡임 없이 전 구간 표시
- intro에서 자막이 먼저 나오지 않음(앵커 범위 밖은 출력 금지)
- 후렴에서 자막이 뒤로 크게 밀리지 않음(앵커 범위 내 분배)
