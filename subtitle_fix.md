"""
[FIX SPEC] MV 자막 타이밍(텍스트 유사도/매칭 0) 기반 정렬 개선

목표:
- 텍스트 유사도/가사-음성 매칭은 전혀 사용하지 않는다.
- WhisperX/VAD 타이밍만으로 "가사가 보이는 시간축"을 만든다.
- 앞 가사가 누락되지 않게 한다(현재 t0_lyric=44s로 잡혀 DROP 발생).
- 자막이 '후루룩' 지나가지 않게 최소 표시 시간과 캡션 정책을 적용한다.

현상(로그 근거):
- LYRIC_T0가 word_density로 44.18s로 잡히고,
  그 이전 VAD phrase들이 "<=t0_lyric" 이유로 DROP 되어 Intro/초반 가사가 사라짐.
- 라인당 1.8s 같은 구간이 발생해 빠르게 지나감.

핵심 변경:
1) t0_lyric 산정 로직 제거/대체
   - "word_density 기반 t0_lyric" 폐기 (또는 안전장치로만 사용)
   - DROP 로직에서 t0_lyric 기준 컷을 절대 사용하지 않는다.
   - 자막 타임라인 시작(t_start)은 아래 중 "가장 빠른" 유효 vocal 기준으로 잡는다:
        t_start = first_vad_start + pre_roll
     where:
        first_vad_start = min(phrase.start for phrase in vad_phrases if phrase.dur >= MIN_PHRASE_DUR)
        pre_roll = max(0.0, PRE_ROLL_SEC)  # 보통 0.2~0.5s
   - WhisperX anchor(t0=7.3s) 는 "음성 존재" 참고값일 뿐, 가사 시작 컷오프로 쓰지 않는다.

2) VAD phrase → 라인 배분은 "시간만" 사용
   - 각 VAD phrase는 (start,end) 구간을 갖는다.
   - 전체 가사 라인 리스트(lines)를 순서대로 이 phrase들에 채워 넣는다.
   - 배분 기준은 텍스트 유사도 금지이므로:
        - phrase의 길이에 비례하여 "할당 라인 수"를 정한다.
        - 라인 가중치는 글자수(또는 음절수) 기반으로만 사용 가능 (의미/유사도 X).
        - 너무 짧은 phrase에는 라인을 넣지 말고 다음 phrase로 carry-over.

3) '후루룩' 방지 정책(필수)
   - 최소 표시 시간: MIN_LINE_SEC = 2.2 (권장 2.0~2.5)
   - 최대 표시 시간: MAX_LINE_SEC = 6.0 (너무 늘어지는 것 방지)
   - phrase 구간이 부족하면:
        - 라인을 "다음 phrase로 넘김(carry)" 하거나
        - 캡션을 2줄 묶기(merge) 옵션 적용
   - phrase 구간이 남으면:
        - 라인별 시간을 MAX_LINE_SEC까지 늘리고 남는 시간은 "뒤쪽 여백"으로 둔다.

4) Intro/메타 라인 처리
   - [Intro], [Verse] 같은 섹션 헤더는 기본적으로 자막 이벤트로 넣지 않는다(현재 [Final Chorus] 같은 라인이 섞일 수 있음).
   - 괄호 무대지시 "(정적)" 같은 라인은 옵션 처리:
        - show_stage_directions=False면 제거
        - True면 일반 라인보다 짧게(예: clamp 1.5~3.0s)

5) 구현 포인트(코드 수정 위치 가이드)
   - mv_pipeline.py 또는 자막 정렬 모듈에서:
        - LYRIC_T0 계산하는 함수/블록 제거
        - DROP 조건: reason "<=t0_lyric" 관련 로직 완전 제거
        - WINDOW span 계산은:
              span = (last_vad_end) - (first_vad_start)
          으로 산정 (anchor/t0_lyric 기반 금지)
        - ALLOC 로직을 아래 allocate_lines_to_phrases()로 교체

6) 참고로 현재 환경 경고/에러
   - torchcodec/FFmpeg 경고는 VAD 디코딩 최적화 문제일 뿐, 자막 누락의 직접 원인은 아님.
   - WinError 10054는 브라우저 range 요청 끊김(206 Partial Content)로 흔함. 기능 이슈 핵심 아님.

-----------------------------------------
권장 의사코드
-----------------------------------------
"""

from dataclasses import dataclass
from typing import List, Tuple

@dataclass
class Phrase:
    start: float
    end: float
    words: int = 0

def sanitize_lyrics_lines(raw_lines: List[str]) -> List[str]:
    # 섹션 헤더 제거: [Intro], [Verse 1] 등
    out = []
    for s in raw_lines:
        t = s.strip()
        if not t:
            continue
        if t.startswith("[") and t.endswith("]"):
            continue
        # 필요 시 (정적) 같은 무대지시 제거 옵션
        out.append(t)
    return out

def allocate_lines_to_phrases(
    phrases: List[Phrase],
    lines: List[str],
    pre_roll: float = 0.3,
    min_phrase_dur: float = 1.2,
    min_line_sec: float = 2.2,
    max_line_sec: float = 6.0,
    merge_short: bool = True,
) -> List[Tuple[float, float, str]]:
    """
    텍스트 유사도 없이, VAD phrase 시간만으로 라인 타임라인 생성.
    """
    valid = [p for p in phrases if (p.end - p.start) >= min_phrase_dur]
    if not valid or not lines:
        return []

    t0 = valid[0].start + pre_roll
    # 첫 phrase start보다 pre_roll이 커서 역전되면 보정
    if t0 > valid[0].end - 0.1:
        t0 = valid[0].start

    # 전체 사용 가능 시간은 "valid phrase들의 union"이지만 단순화를 위해 합산/순차 배치
    # (phrase 사이 gap은 그대로 gap으로 둔다)
    i_line = 0
    events = []

    def push_event(a: float, b: float, text: str):
        if b <= a:
            return
        events.append((a, b, text))

    for pi, p in enumerate(valid):
        if i_line >= len(lines):
            break

        seg_start = p.start if pi > 0 else t0
        seg_end = p.end
        seg_dur = max(0.0, seg_end - seg_start)
        if seg_dur < min_line_sec:
            continue  # 너무 짧은 phrase는 스킵(다음 phrase로 carry)

        # phrase에 몇 라인을 넣을지: 최소시간 기반으로 가능한 개수로 제한
        max_lines_here = int(seg_dur // min_line_sec)
        if max_lines_here <= 0:
            continue

        # 일단 가능한 만큼 채우되, 남은 라인 수 고려
        remaining = len(lines) - i_line
        n = min(max_lines_here, remaining)

        # 라인별 시간 분배(글자수 가중치 가능. 의미매칭 X)
        weights = [max(1, len(lines[i_line + k])) for k in range(n)]
        wsum = sum(weights)
        cursor = seg_start

        for k in range(n):
            share = seg_dur * (weights[k] / wsum)
            dur = max(min_line_sec, min(max_line_sec, share))
            # 남은 시간과 남은 라인 고려해 마지막 라인까지 커버되게 보정
            if k == n - 1:
                end = seg_end
            else:
                end = min(seg_end, cursor + dur)

            push_event(cursor, end, lines[i_line + k])
            cursor = end
        i_line += n

    # 라인이 남으면: 뒤쪽 phrase에 못 넣은 상태 -> 정책적으로
    # 1) 마지막 이벤트 뒤에 이어붙이기(권장 X) or
    # 2) 마지막 phrase end를 넘기지 않는 선에서 merge_short 켜서 2줄로 묶어 재배치
    # 여기서는 안전하게: 남은 라인은 마지막 phrase 구간 내에서 merge 처리하도록 상위 로직에서 재시도.
    if i_line < len(lines) and merge_short:
        # 간단 처리: 남은 라인을 2줄씩 묶어서 라인 수를 줄인 뒤 재호출하도록 호출측에서 구현
        pass

    return events

"""
테스트/로그 기대값:
- 더 이상 LYRIC_T0 / DROP(reason "<=t0_lyric") 로그가 나오면 안 됨.
- CAP0의 시작 시간이 44초대가 아니라, 첫 VAD phrase(예: 2.44~9.65 또는 11.35~...) 근처로 이동해야 함.
- 라인별 sec_per_line이 2.2s 아래로 내려가지 않아야 함(후루룩 방지).
"""
