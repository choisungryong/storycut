import re
import subprocess
from dataclasses import dataclass
from typing import List, Tuple, Optional


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


# ── 기본 유틸 ──────────────────────────────────────────────

def ffprobe_duration_sec(media_path: str) -> float:
    cmd = ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of",
           "default=noprint_wrappers=1:nokey=1", media_path]
    out = subprocess.check_output(cmd, text=True).strip()
    return float(out)


def split_lyrics_lines(user_lyrics_text: str) -> List[str]:
    raw = user_lyrics_text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    lines = [ln.strip() for ln in raw]
    return [ln for ln in lines if ln]


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
