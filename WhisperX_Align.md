You are implementing a forced-alignment subtitle generator for SONG LYRICS.
Inputs:
- vocals.wav (Demucs vocals-only output; not the full mix)
- lyrics_align.txt (ground-truth lyrics, cleaned: no section tags, no quotes/ellipses, repeats expanded)
- whisperx output JSON containing word-level timestamps (list of words with start/end + token text)

Outputs:
- Primary: lyrics.ass (ASS subtitles)
- Fallback: lyrics.srt (SRT subtitles)

Goal:
Produce near-perfect sync for lyric lines, including repeated choruses, intro-with-no-lyrics, and mixed Korean/English.

Hard requirements:
1) Do NOT start subtitles at 0 if the intro has no vocals.
2) Do NOT globally align duplicated chorus blocks; handle repeats sequentially with time-window constraints.
3) Provide robust resynchronization to prevent drift mid-song.
4) Always generate both ASS and SRT from the same internal Caption[] timeline.

========================================================
A. Data Model
========================================================
Define the internal representation:

type Caption = {
  startSec: number;   // float seconds
  endSec: number;     // float seconds
  text: string;       // may contain '\n' line breaks (later rendered as '\N' for ASS)
  confidence?: number;
};

Define constants:
MIN_DUR = 0.80
MAX_DUR = 2.20
MAX_LINES = 2
MAX_LINE_CHARS = 18  // Korean readability target
ONSET_RUN_WORDS = 3
ONSET_MAX_GAP = 0.35 // seconds
ANCHOR_MIN_LEN = 6   // tokens

========================================================
B. Preprocess Lyrics
========================================================
Parse lyrics_align.txt into "blocks" and "lines":

- A block is a group of non-empty lines separated by blank lines.
- Preserve line breaks within blocks (each line becomes a caption candidate).
- Normalize each line for matching:
  - strip
  - remove punctuation, quotes, ellipses
  - collapse spaces
  - for English: lowercase
  - keep Korean spacing as-is but normalize whitespace
Return:
- blocks: Array<Block>
  where Block = { rawLines: string[], normTokens: string[] }

Also build a "chorus signature" per block:
- signature = hash(join(normTokens, ' '))
- blocks with identical signatures are considered repeats (e.g., Chorus).

========================================================
C. Load WhisperX Words
========================================================
Load word-level list from whisperx JSON:
words = [{startSec, endSec, tokenNorm, tokenRaw}]
Normalize tokens similarly to lyrics normalization.

Filter out junk tokens:
- remove empty tokens
- optionally remove tokens that are purely non-letter and non-Korean
- keep timing as authoritative

========================================================
D. Detect First Vocal Onset (Intro Handling)
========================================================
Find the earliest point where vocals likely begin:
- scan words in time order
- find the first index i such that:
  - words[i..i+ONSET_RUN_WORDS-1] exist
  - each consecutive gap (start of next - end of prev) <= ONSET_MAX_GAP
  - tokens are not empty/junk
Set vocalOnsetSec = words[i].startSec

Rule:
- No caption may start before vocalOnsetSec - 0.2
- If your alignment yields earlier times, clamp them forward.

========================================================
E. Alignment Strategy (Core)
========================================================
We align sequentially, block by block, line by line, using:
1) windowed search (time constraints)
2) dynamic programming token alignment (Needleman–Wunsch / Levenshtein)
3) anchors for resynchronization

E1) Maintain a moving cursor in the word list:
- cursorWordIdx initially = index of vocalOnsetSec
- cursorTimeSec = vocalOnsetSec

E2) For each block in order:
- Determine a search window in words:
  - windowStartIdx = cursorWordIdx
  - windowEndIdx = windowStartIdx + W where W covers ~30-60 seconds of audio (or N words), adjustable
  - IMPORTANT: if this block is a repeated chorus, enforce that its windowStartIdx is AFTER the previous chorus end (do not let it match earlier chorus occurrences).
- Token alignment:
  - Align block.normTokens to words[windowStartIdx:windowEndIdx].tokenNorm
  - Use DP alignment scoring:
    match +2, substitution -1, insertion -1, deletion -1
  - Choose the best alignment position (optionally via sliding windows):
    - evaluate multiple candidate sub-windows (e.g., shift by 3-5 seconds / M words)
    - pick the sub-window with highest normalized score

E3) Anchor-based resync:
- Within the block, find anchor phrases (unique sequences) from lyrics tokens:
  - anchors are sequences length >= ANCHOR_MIN_LEN
  - prefer sequences that occur only once in the whole lyrics (global uniqueness)
- During alignment, force at least one anchor to match within the chosen window.
- After confirming the anchor match, update cursorWordIdx to the end of the matched anchor region.
This prevents drift and mid-song mismatch.

========================================================
F. Convert Alignment to Line-Level Captions
========================================================
For each lyric line in the block:
- Obtain its tokens (normTokensLine)
- Find matched word indices for these tokens within the block alignment mapping
- Derive line start/end times:
  - startSec = start time of first matched word
  - endSec = end time of last matched word
If a line has partial or no matches:
- Interpolate using neighboring matched lines:
  - start = prev.end (or cursorTimeSec if first)
  - end = next.start (or start + MIN_DUR if last)
- Ensure duration:
  - if end - start < MIN_DUR => extend end to start + MIN_DUR
  - if end - start > MAX_DUR => allow but consider splitting by punctuation/spacing

Clamp intro:
- if start < vocalOnsetSec - 0.2 => start = vocalOnsetSec
- ensure start < end always

Readability wrap:
- Wrap to MAX_LINES using whitespace boundaries
- For Korean, measure chars excluding spaces; for mixed text, measure visible chars.
- Insert '\n' line breaks (later converted to '\N' in ASS).

Create Caption[] in order.

Update cursorWordIdx/cursorTimeSec:
- cursorTimeSec = endSec of last caption in block
- cursorWordIdx = nearest word index after cursorTimeSec

========================================================
G. Render ASS (Primary)
========================================================
Render a minimal ASS file:

[Script Info]
ScriptType: v4.00+
PlayResX: 1920
PlayResY: 1080

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default, Noto Sans CJK KR, 52, &H00FFFFFF, &H000000FF, &H00111111, &H64000000, 0,0,0,0, 100,100, 0,0, 1, 3, 0, 2, 80,80,90,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
Dialogue: 0, H:MM:SS.cc, H:MM:SS.cc, Default,,0,0,0,, <Text with '\N' for line breaks>

Time formatting:
- ASS uses centiseconds (cc)
- Convert seconds accordingly.

========================================================
H. Render SRT (Fallback)
========================================================
From the same Caption[]:
- SRT time format: HH:MM:SS,mmm
- Use the same wrapped text but with '\n' line breaks.

========================================================
I. Validation & Debugging
========================================================
Implement these validations:
- captions sorted by time, non-overlapping (or minimal overlaps)
- no caption starts before vocalOnsetSec - 0.2
- durations >= MIN_DUR
- detect chorus blocks: ensure each repeat is mapped to increasing time windows

Emit debug artifacts:
- alignment score per block
- chosen window start/end times per block
- anchor matches used
- any interpolated lines count

If alignment score is below threshold for a block:
- expand search window
- relax anchor requirement
- fallback to simpler DP without anchors
- still produce captions but mark low confidence (optional)
