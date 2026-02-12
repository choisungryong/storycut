[자막 싱크/누락 문제 해결 - 이번 스프린트 지시]

목표:
- drawtext 다건 방식은 중단하고, SRT(또는 ASS) 자막 파일을 생성해서 FFmpeg subtitles= 로 burn-in 하세요.
- 자막 텍스트는 100% 사용자 입력 가사(user_lyrics_text)를 정답으로 사용합니다.
- Google STT 텍스트는 자막 내용에 쓰지 마세요(노래는 정렬 깨짐). 있으면 타이밍 힌트(스냅) 정도만 선택적으로 사용합니다.
- Whisper/forced-alignment는 이번에 하지 않습니다.

구현 범위:
1) user_lyrics_text -> 라인 분해 (빈 줄 제거)
2) ffprobe로 오디오 길이 T(초) 구하기
3) 라인 수 N에 대해 타이밍을 "균등 분배"로 생성
   - 기본: start=i*T/N, end=(i+1)*T/N
   - min_dur=0.8s, max_dur=4.0s clamp(너무 짧/길면 조정)
4) lyrics.srt 파일 생성 (UTF-8)
5) FFmpeg에서 subtitles=lyrics.srt 로 burn-in
   - fps=30 고정
   - 오디오 aresample=48000, asetpts=N/SR/TB 적용
6) render.log에 아래 기록:
   - lyrics_source=user_input_only
   - timeline_mode=uniform(또는 uniform+snap)
   - N_lines, audio_duration, srt_path

코드 제공(그대로 사용 가능):
- 새 파일: agents/subtitle_utils.py 로 추가

--- agents/subtitle_utils.py ---
(아래 코드 블록을 파일로 생성)

[CODE_START]
import subprocess
from dataclasses import dataclass
from typing import List, Tuple, Optional

@dataclass
class SubtitleLine:
    start: float
    end: float
    text: str

def ffprobe_duration_sec(media_path: str) -> float:
    cmd = ["ffprobe","-v","error","-show_entries","format=duration","-of",
           "default=noprint_wrappers=1:nokey=1", media_path]
    out = subprocess.check_output(cmd, text=True).strip()
    return float(out)

def split_lyrics_lines(user_lyrics_text: str) -> List[str]:
    raw = user_lyrics_text.replace("\r\n","\n").replace("\r","\n").split("\n")
    lines = [ln.strip() for ln in raw]
    return [ln for ln in lines if ln]

def clamp_timeline_uniform(lines: List[str], total_duration: float,
                           min_dur: float = 0.8, max_dur: float = 4.0) -> List[SubtitleLine]:
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
        text = s.text.replace("\n"," ")
        buf.append(str(i))
        buf.append(f"{start} --> {end}")
        buf.append(text)
        buf.append("")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(buf))
[CODE_END]

mv_pipeline.py 적용 지점:
- 기존 drawtext 기반 자막 합성 로직을 제거/우회하고,
- 아래 흐름으로 교체:
  1) T = ffprobe_duration_sec(audio_path)
  2) lines = split_lyrics_lines(user_lyrics_text)
  3) timeline = clamp_timeline_uniform(lines, T)
  4) write_srt(timeline, tmp/lyrics.srt)
  5) ffmpeg 렌더 시 -vf "fps=30,subtitles=tmp/lyrics.srt:force_style='FontName=Noto Sans CJK KR,FontSize=42,Outline=2,Shadow=1,MarginV=60'"
     오디오: -af "aresample=48000,asetpts=N/SR/TB"
- render.log에 N_lines, audio_duration, timeline_mode 등을 기록

FFmpeg 커맨드 예시(참고):
ffmpeg -y -i video_input.mp4 \
 -vf "fps=30,subtitles=tmp/lyrics.srt:force_style='FontName=Noto Sans CJK KR,FontSize=42,Outline=2,Shadow=1,MarginV=60'" \
 -af "aresample=48000,asetpts=N/SR/TB" \
 -c:v libx264 -pix_fmt yuv420p -profile:v high -level 4.1 -r 30 \
 -c:a aac -ar 48000 -b:a 192k \
 out.mp4

완료 기준:
- 같은 음원/가사로 2회 렌더해도 자막이 깜빡이거나 누락되지 않아야 함
- 자막 내용은 항상 사용자 입력 가사와 동일해야 함
