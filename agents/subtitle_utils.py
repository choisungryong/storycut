import re
import platform
import subprocess
from dataclasses import dataclass
from typing import List, Tuple, Optional
from utils.logger import get_logger
from utils.ffmpeg_utils import get_media_duration
logger = get_logger("subtitle_utils")


# 괄호 없는 섹션 마커 패턴 (대소문자 무시)
# "Chorus", "Scene 2", "Chorus — 꿈을 세우다", "Scene 1 — 만남" 등 모두 매칭
_SECTION_BARE_RE = re.compile(
    r'^(?:verse|chorus|pre[-\s]?chorus|post[-\s]?chorus|hook|bridge|'
    r'outro|intro|interlude|instrumental|refrain|final\s+chorus|rap|scene)'
    r'(?:\s*\d*)'            # 선택적 번호 (Scene 1, Verse 2)
    r'(?:\s*[-—:].+)?'       # 선택적 부제 (— 만남, : 시련)
    r'\s*$',
    re.IGNORECASE
)


@dataclass
class SubtitleLine:
    start: float
    end: float
    text: str


@dataclass
class AnchorResult:
    """보컬 앵커 추정 결과"""
    anchor_start: float
    anchor_end: float
    method: str  # "segments", "vad", "fallback", "user_override"


def _detect_linux_font() -> str:
    """Linux에서 사용 가능한 CJK 폰트 탐색"""
    try:
        result = subprocess.run(
            ["fc-list", ":lang=ko", "family"],
            capture_output=True, text=True, timeout=5
        )
        families = result.stdout.strip()
        for preferred in ["Noto Sans CJK KR", "Noto Sans CJK", "NanumGothic", "NanumBarunGothic"]:
            if preferred in families:
                return preferred
    except Exception:
        pass
    return "Noto Sans CJK KR"


# ── 기본 유틸 ──────────────────────────────────────────────

def ffprobe_duration_sec(media_path: str) -> float:
    """ffprobe로 미디어 길이(초) 반환. get_media_duration 래퍼."""
    return get_media_duration(media_path)


def split_lyrics_lines(user_lyrics_text: str) -> List[str]:
    """가사 텍스트를 줄 단위로 분리하고, 섹션 마커/빈 줄 제거."""
    # 줄 시작의 [...]  마커 제거 (예: [Intro], [Verse 1], [Final Chorus])
    strip_bracket = re.compile(r'^\[.*?\]\s*')
    # 줄 시작/끝의 (...) 마커 제거 (예: (낮게, 속삭이듯), (정적), (간주))
    strip_paren = re.compile(r'^\(.*?\)\s*|\s*\(.*?\)$')
    raw = user_lyrics_text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    # 모든 따옴표 문자 완전 제거 (strip은 가장자리만 제거하므로 replace 사용)
    quote_chars = '"\'\u201c\u201d\u2018\u2019\u201e\u201f\u00ab\u00bb\uff02\uff07'
    cleaned = []
    for ln in raw:
        txt = ln.strip()
        for ch in quote_chars:
            txt = txt.replace(ch, '')
        # 마커 제거 (줄 전체가 마커면 빈 문자열이 됨)
        txt = strip_bracket.sub('', txt).strip()
        txt = strip_paren.sub('', txt).strip()
        # 괄호 없는 섹션 마커 제거 (Verse 1, Pre-Chorus 등)
        if _SECTION_BARE_RE.match(txt):
            txt = ''
        cleaned.append(txt)
    return [ln for ln in cleaned if ln]


# ── 앵커 추정 ─────────────────────────────────────────────

def anchors_from_segments(segments: list) -> Optional[AnchorResult]:
    """
    MusicAnalysis.segments 에서 보컬 구간 앵커 추정.
    intro/outro를 제외한 첫/마지막 세그먼트의 시작/끝을 사용.
    """
    if not segments:
        return None

    vocal_types = {"verse", "chorus", "pre_chorus", "pre-chorus", "hook", "rap", "bridge"}
    vocal_segs = []
    for seg in segments:
        seg_type = getattr(seg, "segment_type", None) or seg.get("segment_type", "") if isinstance(seg, dict) else getattr(seg, "segment_type", "")
        if seg_type.lower().replace(" ", "_") in vocal_types:
            start = getattr(seg, "start_sec", None) or (seg.get("start_sec", 0) if isinstance(seg, dict) else 0)
            end = getattr(seg, "end_sec", None) or (seg.get("end_sec", 0) if isinstance(seg, dict) else 0)
            vocal_segs.append((float(start), float(end)))

    if not vocal_segs:
        return None

    anchor_start = min(s[0] for s in vocal_segs)
    anchor_end = max(s[1] for s in vocal_segs)

    if anchor_start >= anchor_end:
        return None

    return AnchorResult(anchor_start, anchor_end, "segments")


def anchors_from_vad(media_path: str, silence_db: float = -30,
                     min_silence_dur: float = 1.5) -> Optional[AnchorResult]:
    """
    FFmpeg silencedetect 기반 간이 VAD.
    첫 silence_end → anchor_start, 마지막 silence_start → anchor_end.
    """
    try:
        total = ffprobe_duration_sec(media_path)
    except Exception:
        return None

    cmd = [
        "ffmpeg", "-i", media_path,
        "-af", f"silencedetect=n={silence_db}dB:d={min_silence_dur}",
        "-f", "null", "-"
    ]
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True,
            encoding="utf-8", errors="replace", timeout=60
        )
    except Exception:
        return None

    silence_ends = []
    silence_starts = []
    for line in result.stderr.split("\n"):
        if "silence_end:" in line:
            m = re.search(r"silence_end:\s*([\d.]+)", line)
            if m:
                silence_ends.append(float(m.group(1)))
        elif "silence_start:" in line:
            m = re.search(r"silence_start:\s*([\d.]+)", line)
            if m:
                silence_starts.append(float(m.group(1)))

    anchor_start = silence_ends[0] if silence_ends else 0.0
    anchor_end = silence_starts[-1] if silence_starts else total

    # 앵커가 너무 좁으면 (3초 미만) 실패로 간주
    if anchor_end - anchor_start < 3.0:
        return None

    return AnchorResult(anchor_start, anchor_end, "vad")


def detect_anchors(
    media_path: str,
    segments: Optional[list] = None,
    user_anchor_start: Optional[float] = None,
    user_anchor_end: Optional[float] = None,
) -> AnchorResult:
    """
    앵커 추정 우선순위:
    1) 사용자 보정값
    2) MusicAnalysis segments 기반
    3) FFmpeg VAD 기반
    4) fallback: (0, T)
    """
    total = ffprobe_duration_sec(media_path)

    # 1) 사용자 보정
    if user_anchor_start is not None:
        a_start = user_anchor_start
        a_end = user_anchor_end if user_anchor_end is not None else total
        return AnchorResult(a_start, a_end, "user_override")

    # 2) 세그먼트 기반
    if segments:
        result = anchors_from_segments(segments)
        if result:
            return result

    # 3) VAD 기반
    vad_result = anchors_from_vad(media_path)
    if vad_result:
        return vad_result

    # 4) fallback
    return AnchorResult(0.0, total, "fallback")


# ── 타임라인 생성 ──────────────────────────────────────────

def clamp_timeline_uniform(lines: List[str], total_duration: float,
                           min_dur: float = 0.8, max_dur: float = 4.0) -> List[SubtitleLine]:
    """[0, total_duration] 범위 내 균등 분배 (offset 없음)"""
    n = len(lines)
    if n == 0:
        return []
    total_duration = max(total_duration, 0.1)
    uniform = total_duration / n

    # if in range -> simple uniform
    if min_dur <= uniform <= max_dur:
        t = 0.0
        out = []
        for txt in lines:
            start = t
            end = min(total_duration, t + uniform)
            out.append(SubtitleLine(start, end, txt))
            t = end
        out[-1].end = total_duration
        return out

    # clamp dur
    dur = min(max(uniform, min_dur), max_dur)
    base_total = dur * n

    if base_total <= total_duration:
        extra = total_duration - base_total
        per = extra / n
        t = 0.0
        out = []
        for txt in lines:
            start = t
            end = min(total_duration, t + dur + per)
            out.append(SubtitleLine(start, end, txt))
            t = end
        out[-1].end = total_duration
        return out

    # fallback strict uniform if cannot fit
    t = 0.0
    out = []
    uniform = total_duration / n
    for txt in lines:
        start = t
        end = min(total_duration, t + uniform)
        out.append(SubtitleLine(start, end, txt))
        t = end
    out[-1].end = total_duration
    return out


def clamp_timeline_anchored(
    lines: List[str],
    anchor_start: float,
    anchor_end: float,
    min_dur: float = 0.8,
    max_dur: float = 4.0,
) -> List[SubtitleLine]:
    """
    [anchor_start, anchor_end] 범위 안에서만 가사 분배.
    intro/outro(범위 밖)에는 자막이 출력되지 않음.
    """
    duration = anchor_end - anchor_start
    raw = clamp_timeline_uniform(lines, duration, min_dur, max_dur)
    # offset by anchor_start
    for entry in raw:
        entry.start += anchor_start
        entry.end += anchor_start
    return raw


def _extract_vocal_ranges(
    segments: list,
    anchor_start: float = 0.0,
    anchor_end: float = 9999.0,
) -> List[Tuple[float, float]]:
    """세그먼트 목록에서 보컬 구간만 추출 (시간순 정렬)."""
    vocal_types = {"verse", "chorus", "pre_chorus", "pre-chorus", "hook", "rap", "bridge"}
    ranges = []
    for seg in segments:
        if isinstance(seg, dict):
            seg_type = seg.get("segment_type", "")
            start = float(seg.get("start_sec", 0))
            end = float(seg.get("end_sec", 0))
        else:
            seg_type = getattr(seg, "segment_type", "")
            start = float(getattr(seg, "start_sec", 0))
            end = float(getattr(seg, "end_sec", 0))
        if seg_type.lower().replace(" ", "_") in vocal_types and start < end:
            ranges.append((max(start, anchor_start), min(end, anchor_end)))
    ranges.sort(key=lambda r: r[0])
    return ranges


def clamp_timeline_vocal_segments(
    lines: List[str],
    segments: list,
    anchor_start: float,
    anchor_end: float,
    min_dur: float = 0.8,
    max_dur: float = 4.0,
) -> List[SubtitleLine]:
    """
    보컬 세그먼트 내에서만 가사를 분배.
    간주/인트로/아웃트로 구간에는 자막을 배치하지 않음.
    """
    vocal_ranges = _extract_vocal_ranges(segments, anchor_start, anchor_end)

    if not vocal_ranges:
        return clamp_timeline_anchored(lines, anchor_start, anchor_end, min_dur, max_dur)

    n = len(lines)
    if n == 0:
        return []

    total_vocal_dur = sum(e - s for s, e in vocal_ranges)
    if total_vocal_dur <= 0:
        return clamp_timeline_anchored(lines, anchor_start, anchor_end, min_dur, max_dur)

    result: List[SubtitleLine] = []
    line_idx = 0

    for vi, (vstart, vend) in enumerate(vocal_ranges):
        if line_idx >= n:
            break
        seg_dur = vend - vstart
        # 이 보컬 구간에 비례 배분할 줄 수
        seg_lines = max(1, round(n * seg_dur / total_vocal_dur))
        # 마지막 보컬 구간이면 남은 줄 전부 배정
        if vi == len(vocal_ranges) - 1:
            seg_lines = n - line_idx
        else:
            seg_lines = min(seg_lines, n - line_idx)

        sub_lines = lines[line_idx:line_idx + seg_lines]
        sub_timeline = clamp_timeline_uniform(sub_lines, seg_dur, min_dur, max_dur)
        for entry in sub_timeline:
            entry.start += vstart
            entry.end += vstart
        result.extend(sub_timeline)
        line_idx += seg_lines

    return result


def snap_away_from_instrumental(
    aligned: list,
    segments: list,
    anchor_start: float = 0.0,
    anchor_end: float = 9999.0,
) -> list:
    """
    STT 정렬 결과에서 간주 구간에 배치된 자막을 다음 보컬 구간으로 밀어냄.
    aligned: List[AlignedSubtitle]
    """
    vocal_ranges = _extract_vocal_ranges(segments, anchor_start, anchor_end)
    if not vocal_ranges:
        return aligned

    def _in_vocal(t: float) -> bool:
        return any(s <= t < e for s, e in vocal_ranges)

    def _next_vocal_start(t: float) -> Optional[float]:
        for s, e in vocal_ranges:
            if s > t:
                return s
        return None

    for a in aligned:
        if not _in_vocal(a.start):
            nv = _next_vocal_start(a.start)
            if nv is not None:
                dur = a.end - a.start
                a.start = nv
                a.end = nv + dur

    return aligned


# ── SRT 출력 ───────────────────────────────────────────────

def format_srt_timestamp(seconds: float) -> str:
    seconds = max(seconds, 0.0)
    ms = int(round(seconds * 1000))
    hh = ms // 3600000
    ms -= hh * 3600000
    mm = ms // 60000
    ms -= mm * 60000
    ss = ms // 1000
    ms -= ss * 1000
    return f"{hh:02d}:{mm:02d}:{ss:02d},{ms:03d}"


def write_srt(timeline: List[SubtitleLine], out_path: str) -> None:
    buf = []
    for i, s in enumerate(timeline, start=1):
        start = format_srt_timestamp(s.start)
        end = format_srt_timestamp(max(s.end, s.start + 0.01))
        text = s.text.replace("\n", " ")
        buf.append(str(i))
        buf.append(f"{start} --> {end}")
        buf.append(text)
        buf.append("")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(buf))


# ── STT 정렬 + ASS 출력 ──────────────────────────────────

@dataclass
class AlignedSubtitle:
    start: float
    end: float
    text: str
    confidence: float  # 0-100


def align_lyrics_with_stt(
    user_lyrics_lines: List[str],
    stt_segments: List[dict],
    anchor_start: float,
    anchor_end: float,
    min_confidence: float = 55.0,
    min_dur: float = 0.8,
    max_dur: float = 5.0,
) -> List[AlignedSubtitle]:
    """
    사용자 가사와 STT 세그먼트를 fuzzy matching으로 정렬.

    Args:
        user_lyrics_lines: 사용자 가사 줄 리스트 (빈줄/섹션 마커 제거 후)
        stt_segments: [{start, end, text}, ...] Gemini STT 결과
        anchor_start: 보컬 시작 시간
        anchor_end: 보컬 종료 시간
        min_confidence: 매칭 최소 confidence (0-100)
        min_dur: 최소 자막 표시 시간
        max_dur: 최대 자막 표시 시간

    Returns:
        정렬된 AlignedSubtitle 리스트
    """
    from rapidfuzz import fuzz

    if not user_lyrics_lines:
        return []

    # STT 없으면 균등 분배 fallback
    if not stt_segments:
        timeline = clamp_timeline_anchored(user_lyrics_lines, anchor_start, anchor_end)
        return [AlignedSubtitle(s.start, s.end, s.text, 0.0) for s in timeline]

    # 시간순 정렬
    sorted_stt = sorted(stt_segments, key=lambda s: float(s.get("start", 0)))
    n_lines = len(user_lyrics_lines)
    total_dur = anchor_end - anchor_start
    # 가사 한 줄당 예상 시간 (시간 윈도우 계산용)
    expected_line_dur = total_dur / max(1, n_lines)
    # 탐색 윈도우: 예상 위치 ± 3배 (반복 가사/간주 구간 커버)
    time_window = max(10.0, expected_line_dur * 3)

    result: List[AlignedSubtitle] = []
    search_start = 0  # 시간순 전진만 허용 (뒤로 돌아가지 않음)

    for line_idx, line in enumerate(user_lyrics_lines):
        clean_line = re.sub(r'[\s\-.,!?~]+', '', line).lower()
        if not clean_line:
            continue

        # 이 줄의 예상 시간 위치
        expected_time = anchor_start + (line_idx / max(1, n_lines)) * total_dur
        max_search_time = expected_time + time_window

        best_score = 0.0
        best_idx = -1

        for si in range(search_start, len(sorted_stt)):
            seg = sorted_stt[si]
            seg_start = float(seg.get("start", 0))

            # 시간 윈도우 초과 시 탐색 중단
            if seg_start > max_search_time:
                break

            seg_text = re.sub(r'[\s\-.,!?~]+', '', seg.get("text", "")).lower()
            if not seg_text:
                continue
            score = fuzz.partial_ratio(clean_line, seg_text)
            if score > best_score:
                best_score = score
                best_idx = si

        if best_score >= min_confidence and best_idx >= 0:
            seg = sorted_stt[best_idx]
            # 전진: 다음 줄은 이 매칭 이후부터 탐색
            search_start = best_idx + 1
            result.append(AlignedSubtitle(
                start=float(seg["start"]),
                end=float(seg["end"]),
                text=line,
                confidence=best_score,
            ))
        else:
            # 매칭 실패: 예상 위치 기반 보간 (이전 줄 end가 있으면 그 뒤부터)
            if result:
                interp_start = max(result[-1].end + 0.1, expected_time)
            else:
                interp_start = expected_time
            interp_end = interp_start + min(expected_line_dur, max_dur)
            result.append(AlignedSubtitle(
                start=interp_start,
                end=interp_end,
                text=line,
                confidence=0.0,
            ))

    # 후처리: 단조 증가 강제 + dur 클램핑 + anchor 범위 클램핑
    for i in range(len(result)):
        # 단조 증가 강제
        if i > 0 and result[i].start <= result[i - 1].start:
            result[i].start = result[i - 1].end + 0.1

        # dur 클램핑
        dur = result[i].end - result[i].start
        if dur < min_dur:
            result[i].end = result[i].start + min_dur
        elif dur > max_dur:
            result[i].end = result[i].start + max_dur

        # anchor 범위 클램핑
        result[i].start = max(anchor_start, min(anchor_end - 0.5, result[i].start))
        result[i].end = max(result[i].start + 0.3, min(anchor_end, result[i].end))

    return result


def write_ass(
    timeline: List[AlignedSubtitle],
    out_path: str,
    font_name: Optional[str] = None,
) -> None:
    """
    ASS 자막 파일 생성 (pysubs2 사용).

    Args:
        timeline: AlignedSubtitle 리스트
        out_path: 출력 .ass 파일 경로
        font_name: 폰트명 (None이면 플랫폼별 자동 선택)
    """
    import pysubs2

    if font_name is None:
        system = platform.system()
        if system == "Windows":
            font_name = "Malgun Gothic"
        else:
            # Docker/Linux: Noto Sans CJK 우선, 없으면 NanumGothic fallback
            font_name = _detect_linux_font()

    subs = pysubs2.SSAFile()

    # 기본 스타일 설정
    style = pysubs2.SSAStyle()
    style.fontname = font_name
    style.fontsize = 42
    style.outline = 2.5
    style.shadow = 1.5
    style.marginv = 60
    style.alignment = pysubs2.Alignment.BOTTOM_CENTER
    style.primarycolor = pysubs2.Color(255, 255, 255, 0)  # white
    style.outlinecolor = pysubs2.Color(0, 0, 0, 0)  # black outline
    style.backcolor = pysubs2.Color(0, 0, 0, 128)  # semi-transparent shadow
    subs.styles["Default"] = style

    for entry in timeline:
        event = pysubs2.SSAEvent()
        event.start = int(entry.start * 1000)  # ms
        event.end = int(entry.end * 1000)
        event.text = entry.text.replace("\n", "\\N")
        event.style = "Default"
        subs.events.append(event)

    subs.save(out_path, encoding="utf-8-sig")
