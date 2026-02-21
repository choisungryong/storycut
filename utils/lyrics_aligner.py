"""
Timing-only lyric subtitle aligner v6.

NO text similarity.  NO DP on strings.  NO word-density detection.
Uses ALL VAD phrases with capacity-based allocation (dur // min_line_sec).
Carry-over for short phrases.  Line merging when too dense.
"""

import json
import os
import re
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np


# ── Constants ──

PRE_ROLL      = 0.3    # seconds before first VAD phrase start
MIN_PHRASE_DUR = 1.2   # minimum phrase duration to accept lines
MIN_LINE_SEC  = 2.2    # minimum display time per caption (prevents rushing)
MAX_LINE_SEC  = 6.0    # maximum display time per caption
MIN_GAP       = 0.02   # minimum gap between consecutive captions
MAX_LINE_CHARS = 18

# VAD
VAD_FRAME_MS   = 30
VAD_HOP_MS     = 10
VAD_MIN_PHRASE  = 0.80
VAD_MERGE_GAP  = 0.25
VAD_ENERGY_PCT = 15


# ── Logger ──

def _log(tag: str, d: dict):
    print(f"{tag} {json.dumps(d, ensure_ascii=False)}")


# ── Data ──

@dataclass
class Caption:
    start_sec: float
    end_sec: float
    text: str
    confidence: float = 0.0

@dataclass
class Phrase:
    start: float
    end: float
    @property
    def dur(self): return self.end - self.start


# ================================================================
# 1) VAD on vocals.wav  (energy-based)
# ================================================================

def _vad_from_audio(wav_path: str, audio_duration: float) -> List[Phrase]:
    try:
        from scipy.io import wavfile
        sr, data = wavfile.read(wav_path)
    except Exception as e:
        _log("VAD_ERROR", {"error": str(e)})
        return []

    if data.ndim > 1:
        data = data.mean(axis=1)
    data = data.astype(np.float32)
    peak = np.max(np.abs(data))
    if peak > 0:
        data /= peak

    frame_len = int(sr * VAD_FRAME_MS / 1000)
    hop_len = int(sr * VAD_HOP_MS / 1000)
    n_frames = max(1, (len(data) - frame_len) // hop_len + 1)

    energy = np.zeros(n_frames, dtype=np.float32)
    for i in range(n_frames):
        s = i * hop_len
        energy[i] = np.sqrt(np.mean(data[s:s + frame_len] ** 2))

    thr = max(np.percentile(energy, VAD_ENERGY_PCT), 0.01)
    voiced = energy > thr

    regions = []
    in_r, rs = False, 0.0
    for i in range(n_frames):
        t = i * VAD_HOP_MS / 1000.0
        if voiced[i] and not in_r:
            rs = t; in_r = True
        elif not voiced[i] and in_r:
            regions.append(Phrase(rs, t)); in_r = False
    if in_r:
        regions.append(Phrase(rs, n_frames * VAD_HOP_MS / 1000.0))

    # merge close
    merged = []
    for r in regions:
        if merged and (r.start - merged[-1].end) < VAD_MERGE_GAP:
            merged[-1] = Phrase(merged[-1].start, r.end)
        else:
            merged.append(r)

    # drop short, clamp
    result = []
    for r in merged:
        if r.dur < VAD_MIN_PHRASE:
            continue
        s = max(0.0, r.start)
        e = min(audio_duration, r.end) if audio_duration > 0 else r.end
        if e > s:
            result.append(Phrase(round(s, 3), round(e, 3)))
    return result


# ================================================================
# 2) Capacity-based allocation: lines → phrases
# ================================================================

def _allocate_to_phrases(
    phrases: List[Phrase],
    lines: List[str],
) -> Tuple[List[str], List[Tuple[float, float, str]]]:
    """
    Allocate lyric lines to VAD phrases based on duration capacity.
    Two-pass: (1) proportional allocation by duration, (2) distribute within each phrase.
    Short phrases (< MIN_PHRASE_DUR) are skipped; lines carry over.

    Returns (merged_lines, list of (start_sec, end_sec, text) events).
    """
    valid = [p for p in phrases if p.dur >= MIN_PHRASE_DUR]
    if not valid or not lines:
        return lines, []

    # Effective durations (first phrase gets PRE_ROLL trimmed)
    eff_durs = []
    for pi, p in enumerate(valid):
        if pi == 0:
            seg_start = p.start + PRE_ROLL
            if seg_start > p.end - 0.1:
                seg_start = p.start
            eff_durs.append(max(0.0, p.end - seg_start))
        else:
            eff_durs.append(p.dur)

    # Total capacity (hard max per phrase)
    capacities = [max(1, int(d // MIN_LINE_SEC)) for d in eff_durs]
    total_capacity = sum(capacities)

    # Merge lines if more than total capacity
    merged_lines = list(lines)
    while len(merged_lines) > total_capacity and len(merged_lines) > 1:
        new_merged = []
        i = 0
        merged_one = False
        while i < len(merged_lines):
            if not merged_one and i + 1 < len(merged_lines):
                new_merged.append(merged_lines[i] + "\n" + merged_lines[i + 1])
                i += 2
                merged_one = True
            else:
                new_merged.append(merged_lines[i])
                i += 1
        if len(new_merged) == len(merged_lines):
            break
        merged_lines = new_merged

    if len(merged_lines) != len(lines):
        _log("MERGE", {"before": len(lines), "after": len(merged_lines),
                        "capacity": total_capacity})

    n_lines = len(merged_lines)
    total_dur = sum(eff_durs)

    # ── Pass 1: Proportional allocation (duration-based) ──
    # Each phrase gets lines proportional to its duration, capped by capacity
    raw_alloc = [(d / total_dur) * n_lines if total_dur > 0 else 0 for d in eff_durs]
    alloc = [min(capacities[i], max(0, round(raw_alloc[i]))) for i in range(len(valid))]

    # Fix rounding: distribute difference
    diff = n_lines - sum(alloc)
    if diff != 0:
        # Sort by fractional remainder
        frac = [(raw_alloc[i] - alloc[i], i) for i in range(len(valid))]
        if diff > 0:
            frac.sort(key=lambda x: -x[0])
            for _, idx in frac:
                if diff <= 0:
                    break
                if alloc[idx] < capacities[idx]:
                    alloc[idx] += 1
                    diff -= 1
        else:
            frac.sort(key=lambda x: x[0])
            for _, idx in frac:
                if diff >= 0:
                    break
                if alloc[idx] > 0:
                    alloc[idx] -= 1
                    diff += 1

    # Log allocations
    for i in range(len(valid)):
        if alloc[i] > 0 or i < 5:
            spl = round(eff_durs[i] / alloc[i], 2) if alloc[i] > 0 else 0
            _log("ALLOC", {"phrase": i, "n_lines": alloc[i],
                           "t": [round(valid[i].start, 2), round(valid[i].end, 2)],
                           "sec_per_line": spl, "capacity": capacities[i]})

    # ── Pass 2: Distribute lines within each phrase ──
    events = []
    i_line = 0

    for pi, p in enumerate(valid):
        n = alloc[pi]
        if n <= 0 or i_line >= len(merged_lines):
            continue

        seg_start = p.start + PRE_ROLL if pi == 0 else p.start
        if seg_start > p.end - 0.1:
            seg_start = p.start
        seg_end = p.end
        seg_dur = max(0.0, seg_end - seg_start)

        chunk = merged_lines[i_line:i_line + n]
        i_line += n

        # Distribute by char-count weight
        weights = [max(1, len(re.sub(r'\s', '', txt))) for txt in chunk]
        wsum = sum(weights)
        cursor = seg_start

        for k, txt in enumerate(chunk):
            share = seg_dur * (weights[k] / wsum)
            dur = max(MIN_LINE_SEC, min(MAX_LINE_SEC, share))

            if k == len(chunk) - 1:
                end = seg_end
            else:
                end = min(seg_end, cursor + dur)

            if end <= cursor:
                end = cursor + 0.5

            events.append((round(cursor, 3), round(end, 3), txt))
            cursor = end

    # If lines remain (rounding edge case), append after last event
    if i_line < len(merged_lines) and events:
        last_end = events[-1][1]
        remaining_lines = merged_lines[i_line:]
        per = max(MIN_LINE_SEC, 3.0)
        cursor = last_end + MIN_GAP
        for txt in remaining_lines:
            end = cursor + per
            events.append((round(cursor, 3), round(end, 3), txt))
            cursor = end + MIN_GAP
        _log("OVERFLOW", {"extra_lines": len(remaining_lines),
                           "appended_after": last_end})

    return merged_lines, events


# ================================================================
# 3) Convert events → Captions with smoothing
# ================================================================

def _events_to_captions(events: List[Tuple[float, float, str]]) -> List[Caption]:
    """Convert (start, end, text) events to Caption objects with overlap fix."""
    caps = []
    for s, e, txt in events:
        # Fix overlap with previous
        if caps and s < caps[-1].end_sec + MIN_GAP:
            s = round(caps[-1].end_sec + MIN_GAP, 3)
        if s >= e:
            e = round(s + 0.5, 3)
        caps.append(Caption(round(s, 3), round(e, 3), _wrap(txt)))
    return caps


# ================================================================
# Helpers
# ================================================================

def _wrap(text: str, mx: int = MAX_LINE_CHARS) -> str:
    text = text.strip()
    if len(re.sub(r'\s', '', text)) <= mx:
        return text
    words = text.split()
    if len(words) <= 1:
        return text
    mid = len(text) // 2
    bp, bd = -1, len(text)
    pos = 0
    for i, w in enumerate(words[:-1]):
        pos += len(w)
        d = abs(pos - mid)
        if d < bd:
            bd = d; bp = i
        pos += 1
    if bp >= 0:
        return " ".join(words[:bp + 1]) + "\n" + " ".join(words[bp + 1:])
    return text


_SECTION_BARE_RE = re.compile(
    r'^(?:verse|chorus|pre[-\s]?chorus|post[-\s]?chorus|hook|bridge|'
    r'outro|intro|interlude|instrumental|refrain|final\s+chorus|rap)'
    r'\s*\d*\s*$',
    re.IGNORECASE
)


def preprocess_lyrics(lyrics_text: str) -> List[str]:
    raw = lyrics_text.replace('\r\n', '\n').replace('\r', '\n').strip()
    raw = re.sub(r'\[.*?\]', '', raw)
    result = []
    for ln in raw.split('\n'):
        ln = ln.strip()
        if ln and re.sub(r'[\s.,;:!?\-~(){}]', '', ln) and not _SECTION_BARE_RE.match(ln):
            result.append(ln)
    return result


def _sec_to_ass(sec: float) -> str:
    h = int(sec // 3600)
    m = int((sec % 3600) // 60)
    s = sec % 60
    cs = int(round((s - int(s)) * 100))
    return f"{h}:{m:02d}:{int(s):02d}.{cs:02d}"


def _sec_to_srt(sec: float) -> str:
    h = int(sec // 3600)
    m = int((sec % 3600) // 60)
    s = sec % 60
    ms = int(round((s - int(s)) * 1000))
    return f"{h:02d}:{m:02d}:{int(s):02d},{ms:03d}"


def render_ass(caps: List[Caption], path: str) -> str:
    hdr = """[Script Info]
ScriptType: v4.00+
PlayResX: 1920
PlayResY: 1080

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,Noto Sans CJK KR,52,&H00FFFFFF,&H000000FF,&H00111111,&H64000000,0,0,0,0,100,100,0,0,1,3,0,2,80,80,90,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    lines = [hdr.strip()]
    for c in caps:
        txt = c.text.replace('\n', '\\N')
        lines.append(f"Dialogue: 0,{_sec_to_ass(c.start_sec)},{_sec_to_ass(c.end_sec)},Default,,0,0,0,,{txt}")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        f.write("\n".join(lines) + "\n")
    return path


def render_srt(caps: List[Caption], path: str) -> str:
    lines = []
    for i, c in enumerate(caps, 1):
        lines += [str(i), f"{_sec_to_srt(c.start_sec)} --> {_sec_to_srt(c.end_sec)}", c.text, ""]
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        f.write("\n".join(lines))
    return path


# ================================================================
# Linear fallback
# ================================================================

def _captions_linear(lines: List[str], t0: float, t1: float) -> List[Caption]:
    n = len(lines)
    dt = (t1 - t0) / n
    caps = []
    for i, txt in enumerate(lines):
        s = t0 + i * dt
        e = t0 + (i + 1) * dt
        e = max(e, s + 0.5)
        if caps and s < caps[-1].end_sec + MIN_GAP:
            s = caps[-1].end_sec + MIN_GAP
        if s >= e:
            e = s + 0.5
        caps.append(Caption(round(s, 3), round(e, 3), _wrap(txt)))
    return caps


# ================================================================
# Gemini Alignment Entry (primary path)
# ================================================================

def generate_from_gemini_alignment(
    gemini_aligned: List[dict],
    lyrics_lines: List[str],
    ass_path: str,
    srt_path: str,
    min_dur: float = 1.5,
    max_dur: float = 8.0,
) -> Tuple[List[Caption], str, str]:
    """
    Gemini align_lyrics_to_audio() 결과를 직접 ASS/SRT로 변환.
    gemini_aligned: [{"index": 0, "start": 12.5, "end": 16.2, "text": "..."}, ...]

    VAD/WhisperX 불필요. Gemini가 오디오를 직접 들어 타이밍을 잡았으므로
    start/end를 그대로 사용하되, 최소/최대 표시시간만 보정.
    """
    if not gemini_aligned:
        _log("GEMINI_ALIGN", {"status": "empty_input"})
        return [], ass_path, srt_path

    n_input = len(gemini_aligned)
    n_lyrics = len(lyrics_lines)

    # Sort by start time (Gemini should already be sorted, but safety)
    entries = sorted(gemini_aligned, key=lambda x: x.get("start", 0))

    captions: List[Caption] = []
    for entry in entries:
        s = float(entry.get("start", 0))
        e = float(entry.get("end", 0))
        idx = int(entry.get("index", -1))
        text = str(entry.get("text", "")).strip()

        # Use original lyrics if text is empty or index is valid
        if not text and 0 <= idx < n_lyrics:
            text = lyrics_lines[idx]
        if not text:
            continue
        # 섹션 마커 필터 (Gemini가 text에 마커를 포함할 수 있음)
        if _SECTION_BARE_RE.match(text):
            continue

        # Enforce minimum duration
        dur = e - s
        if dur < min_dur:
            e = s + min_dur
        # Enforce maximum duration
        if e - s > max_dur:
            e = s + max_dur

        # Fix overlap with previous caption
        if captions and s < captions[-1].end_sec + MIN_GAP:
            s = captions[-1].end_sec + MIN_GAP
        if s >= e:
            e = s + min_dur

        captions.append(Caption(round(s, 3), round(e, 3), _wrap(text)))

    if not captions:
        _log("GEMINI_ALIGN", {"status": "no_captions"})
        return [], ass_path, srt_path

    # Log summary
    _log("GEMINI_ALIGN", {
        "status": "ok",
        "n_input": n_input,
        "n_captions": len(captions),
        "coverage": f"{len(captions)}/{n_lyrics}",
        "timeline": [captions[0].start_sec, captions[-1].end_sec],
    })

    # Check for rushing
    for i, c in enumerate(captions):
        dur = c.end_sec - c.start_sec
        if dur < 1.0:
            _log("WARN_RUSH", {"cap": i, "dur": round(dur, 2), "text": c.text[:30]})

    # Render
    render_ass(captions, ass_path)
    render_srt(captions, srt_path)

    return captions, ass_path, srt_path


# ================================================================
# Main Entry (VAD fallback - legacy)
# ================================================================

def generate_synced_subtitles(
    vocals_wav_path: str,
    lyrics_lines: List[str],
    audio_duration_sec: float,
    ass_path: str,
    srt_path: str,
    whisperx_segments: Optional[List[dict]] = None,
    gemini_segments: Optional[List[dict]] = None,
    whisper_words: Optional[List[dict]] = None,
) -> Tuple[List[Caption], str, str]:
    """
    Timing-only lyric aligner v6.
    No text similarity. No word-density detection. No LYRIC_T0 / DROP.
    All VAD phrases used. Capacity-based allocation.
    """
    if not lyrics_lines:
        _log("ALIGN_END", {"status": "no_lyrics"})
        return [], ass_path, srt_path

    n = len(lyrics_lines)

    # ── 1) VAD ──
    vad_phrases = None
    if vocals_wav_path and os.path.exists(vocals_wav_path):
        vad_phrases = _vad_from_audio(vocals_wav_path, audio_duration_sec)
        if vad_phrases:
            _log("VAD", {"n_phrases": len(vad_phrases),
                         "total_vocal": round(sum(p.dur for p in vad_phrases), 1)})

    # ── 2) Phrase-based allocation ──
    if vad_phrases and len(vad_phrases) >= 2:
        t_start = vad_phrases[0].start
        t_end = vad_phrases[-1].end

        # Log WINDOW
        span = t_end - t_start
        avg_line = span / max(1, n)
        _log("WINDOW", {"N_lines": n, "t_start": round(t_start, 2),
                         "t_end": round(t_end, 2), "span": round(span, 1),
                         "avg_line": round(avg_line, 2)})

        # Allocate lines to phrases
        merged_lines, events = _allocate_to_phrases(vad_phrases, lyrics_lines)
        n_effective = len(merged_lines)

        if events:
            # Log phrase allocations
            valid = [p for p in vad_phrases if p.dur >= MIN_PHRASE_DUR]
            for i, p in enumerate(valid[:10]):
                cap = int(p.dur // MIN_LINE_SEC)
                _log("PHRASE", {"i": i, "t": [round(p.start, 2), round(p.end, 2)],
                                "dur": round(p.dur, 2), "capacity": cap})

            captions = _events_to_captions(events)
            method = "capacity"
        else:
            captions = _captions_linear(lyrics_lines, t_start, t_end)
            method = "linear"
            n_effective = n
    else:
        # No VAD: linear fallback over full audio
        t_start = 0.0
        t_end = audio_duration_sec if audio_duration_sec > 0 else 300.0
        captions = _captions_linear(lyrics_lines, t_start, t_end)
        method = "linear"
        n_effective = n

    _log("METHOD", {"mode": method, "n_lines": n_effective, "n_captions": len(captions)})

    if not captions:
        _log("ALIGN_END", {"status": "no_captions"})
        return [], ass_path, srt_path

    # ── 3) Acceptance checks ──
    _log("CAP0", {"t": [captions[0].start_sec, captions[0].end_sec],
                   "text": captions[0].text[:40]})
    _log("CAP_LAST", {"t": [captions[-1].start_sec, captions[-1].end_sec],
                       "text": captions[-1].text[:40]})

    # Check per-caption rushing
    for i, c in enumerate(captions):
        dur = c.end_sec - c.start_sec
        if dur < MIN_LINE_SEC * 0.5:
            _log("WARN_RUSH", {"cap": i, "dur": round(dur, 2),
                                "text": c.text[:30]})

    # Coverage
    coverage = captions[-1].end_sec / audio_duration_sec if audio_duration_sec > 0 else 1.0
    if coverage < 0.5:
        _log("WARN", {"type": "LOW_COVERAGE", "ratio": round(coverage, 2),
                       "last_end": round(captions[-1].end_sec, 1),
                       "audio_dur": round(audio_duration_sec, 1)})

    # ── 4) Render ──
    render_ass(captions, ass_path)
    render_srt(captions, srt_path)

    _log("ALIGN_END", {
        "status": "ok",
        "method": method,
        "n_captions": len(captions),
        "timeline": [captions[0].start_sec, captions[-1].end_sec],
    })

    return captions, ass_path, srt_path
