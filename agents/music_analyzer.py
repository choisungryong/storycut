"""
Music Analyzer - 음악 파일 분석 모듈

Phase 1: 기본 분석 (길이, BPM)
Phase 2: 고급 분석 (구간 감지, 분위기 추정)
"""

import os
from typing import Optional, List
from pathlib import Path

# Phase 1: pydub만 사용 (librosa는 Phase 2에서)
try:
    from pydub import AudioSegment
    PYDUB_AVAILABLE = True
except ImportError:
    PYDUB_AVAILABLE = False
    print("[MusicAnalyzer] pydub not available. Install with: pip install pydub")

# Phase 2: librosa (선택적)
try:
    import librosa
    import numpy as np
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False


class MusicAnalyzer:
    """
    음악 파일 분석기

    Phase 1 기능:
    - 파일 길이 (duration)
    - BPM 추정 (librosa 있으면)
    - 기본 구간 분할 (균등 분할)

    Phase 2 기능 (추후):
    - 자동 구간 감지 (intro, verse, chorus)
    - 분위기 추정
    - 비트 타임스탬프
    """

    SUPPORTED_FORMATS = ['.mp3', '.wav', '.m4a', '.ogg', '.flac']
    MAX_DURATION_SEC = 600  # 10분 제한

    def __init__(self):
        self.pydub_available = PYDUB_AVAILABLE
        self.librosa_available = LIBROSA_AVAILABLE

        if not self.pydub_available:
            print("[MusicAnalyzer] WARNING: pydub not installed. Limited functionality.")

    def analyze(self, audio_path: str) -> dict:
        """
        음악 파일 분석

        Args:
            audio_path: 음악 파일 경로

        Returns:
            MusicAnalysis 딕셔너리:
            {
                "duration_sec": float,
                "bpm": float or None,
                "mood": str or None,
                "energy": float or None,
                "segments": [...],
                "key_timestamps": [...]
            }
        """
        print(f"[MusicAnalyzer] Analyzing: {audio_path}")

        # 파일 존재 확인
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        # 포맷 확인
        ext = Path(audio_path).suffix.lower()
        if ext not in self.SUPPORTED_FORMATS:
            raise ValueError(f"Unsupported format: {ext}. Supported: {self.SUPPORTED_FORMATS}")

        result = {
            "duration_sec": 0,
            "bpm": None,
            "mood": None,
            "energy": None,
            "segments": [],
            "key_timestamps": []
        }

        # Phase 1: pydub로 기본 분석
        if self.pydub_available:
            result["duration_sec"] = self._get_duration_pydub(audio_path)
        else:
            # pydub 없으면 librosa로 시도
            if self.librosa_available:
                result["duration_sec"] = self._get_duration_librosa(audio_path)
            else:
                raise RuntimeError("No audio library available. Install pydub or librosa.")

        # 길이 제한 확인
        if result["duration_sec"] > self.MAX_DURATION_SEC:
            raise ValueError(
                f"Audio too long: {result['duration_sec']:.1f}s. "
                f"Maximum allowed: {self.MAX_DURATION_SEC}s"
            )

        print(f"  Duration: {result['duration_sec']:.2f}s")

        # Phase 1+: librosa로 BPM 분석 (가능하면)
        if self.librosa_available:
            try:
                bpm = self._get_bpm_librosa(audio_path)
                result["bpm"] = bpm
                print(f"  BPM: {bpm:.1f}")
            except Exception as e:
                print(f"  BPM analysis failed: {e}")

        # Phase 1: 기본 구간 분할 (균등 분할)
        result["segments"] = self._create_basic_segments(result["duration_sec"])

        print(f"  Segments: {len(result['segments'])}")
        print(f"[MusicAnalyzer] Analysis complete")

        return result

    def _get_duration_pydub(self, audio_path: str) -> float:
        """pydub으로 길이 구하기"""
        audio = AudioSegment.from_file(audio_path)
        return len(audio) / 1000.0  # ms -> sec

    def _get_duration_librosa(self, audio_path: str) -> float:
        """librosa로 길이 구하기"""
        duration = librosa.get_duration(path=audio_path)
        return duration

    def _get_bpm_librosa(self, audio_path: str) -> float:
        """librosa로 BPM 추정"""
        y, sr = librosa.load(audio_path, sr=22050, duration=60)  # 처음 60초만
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)

        # tempo가 배열일 수 있음
        if hasattr(tempo, '__len__'):
            return float(tempo[0]) if len(tempo) > 0 else 120.0
        return float(tempo)

    def _create_basic_segments(
        self,
        duration_sec: float,
        target_segment_duration: float = 12.0
    ) -> List[dict]:
        """
        기본 균등 구간 분할

        Args:
            duration_sec: 전체 길이
            target_segment_duration: 목표 구간 길이 (기본 12초)

        Returns:
            구간 목록
        """
        segments = []
        num_segments = max(1, int(duration_sec / target_segment_duration))

        # 최소 6개, 최대 24개 구간
        num_segments = max(6, min(24, num_segments))

        segment_duration = duration_sec / num_segments

        # 구간 타입 패턴 (일반적인 노래 구조)
        if num_segments <= 4:
            types = ["intro", "verse", "chorus", "outro"]
        elif num_segments <= 6:
            types = ["intro", "verse", "chorus", "verse", "chorus", "outro"]
        elif num_segments <= 9:
            types = ["intro", "verse", "pre-chorus", "chorus", "verse", "chorus", "bridge", "chorus", "outro"]
        else:
            types = ["intro", "verse", "verse", "pre-chorus", "chorus", "chorus",
                     "verse", "verse", "pre-chorus", "chorus", "bridge", "chorus", "outro"]

        for i in range(num_segments):
            start = i * segment_duration
            end = min((i + 1) * segment_duration, duration_sec)

            segment_type = types[i % len(types)]

            segments.append({
                "segment_type": segment_type,
                "start_sec": round(start, 2),
                "end_sec": round(end, 2),
                "duration_sec": round(end - start, 2),
                "energy_level": None  # Phase 2에서 구현
            })

        return segments

    # ================================================================
    # Google Cloud Speech-to-Text (Chirp 3) - word-level timestamps
    # ================================================================

    def transcribe_with_gemini_audio(self, audio_path: str) -> Optional[list]:
        """
        Gemini 2.5 Flash Audio API로 보컬 구간 문장 단위 타임스탬프 추출.

        Returns:
            [{"start": float, "end": float, "text": str}, ...] 또는 None
        """
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            print("  [Gemini-STT] GOOGLE_API_KEY not set")
            return None

        try:
            from google import genai
            from google.genai import types
            import shutil
            import tempfile
            import json

            client = genai.Client(api_key=api_key)

            # non-ASCII 파일명 대응: temp ASCII 파일로 복사
            ext = Path(audio_path).suffix
            temp_dir = tempfile.gettempdir()
            temp_path = os.path.join(temp_dir, f"gemini_stt_upload{ext}")
            shutil.copy2(audio_path, temp_path)

            try:
                audio_file = client.files.upload(file=temp_path)
                print(f"  [Gemini-STT] File uploaded: {audio_file.name}")
            finally:
                try:
                    os.remove(temp_path)
                except OSError:
                    pass

            prompt_text = (
                "Listen to this music carefully and detect all vocal (singing) segments.\n"
                "For each sung sentence/phrase, output the start time, end time, and the sung text.\n\n"
                "Rules:\n"
                "1. 'start' = time in seconds when the vocal phrase begins\n"
                "2. 'end' = time in seconds when the vocal phrase ends\n"
                "3. start < end, and entries must be in chronological order\n"
                "4. Skip instrumental/non-vocal sections entirely\n"
                "5. Transcribe the actual sung words (Korean, English, etc.)\n"
                "6. Group words into natural sentence-level phrases (not single words)\n\n"
                "Output: JSON array only\n"
                '[{"start": 12.5, "end": 16.2, "text": "sung phrase"}, ...]'
            )

            print(f"  [Gemini-STT] Extracting vocal segments...")
            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=[audio_file, prompt_text],
                config=types.GenerateContentConfig(
                    temperature=0.0,
                    response_mime_type="application/json",
                )
            )

            result = json.loads(response.text.strip())
            if not isinstance(result, list):
                print(f"  [Gemini-STT] Invalid response format")
                return None

            # Validate entries
            valid = []
            for entry in result:
                start = float(entry.get("start", 0))
                end = float(entry.get("end", 0))
                text = str(entry.get("text", "")).strip()
                if start < end and text:
                    valid.append({"start": round(start, 2), "end": round(end, 2), "text": text})

            if valid:
                print(f"  [Gemini-STT] Got {len(valid)} segments: {valid[0]['start']}s ~ {valid[-1]['end']}s")
            else:
                print(f"  [Gemini-STT] No valid segments")
                return None

            return valid

        except Exception as e:
            print(f"  [Gemini-STT] Failed: {e}")
            import traceback
            traceback.print_exc()
            return None

    def transcribe_with_google_stt(self, audio_path: str, language: str = "ko-KR") -> Optional[list]:
        """
        Google Cloud Speech-to-Text V1 API로 단어 단위 타임스탬프 추출

        Returns:
            [{"t": float, "text": str}, ...] 단어별 타임스탬프, 실패 시 None
        """
        import requests as _requests
        import base64
        import subprocess
        import time as _time

        api_key = os.getenv("GOOGLE_STT_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not api_key:
            print("  [STT] No API key (GOOGLE_STT_API_KEY or GOOGLE_API_KEY)")
            return None

        print(f"  [STT] Preparing audio for Google Cloud Speech-to-Text...")

        # FLAC mono 16kHz로 변환 (STT 최적 포맷)
        import tempfile
        flac_path = os.path.join(tempfile.gettempdir(), "stt_input.flac")
        try:
            proc = subprocess.run(
                ["ffmpeg", "-y", "-i", audio_path, "-ar", "16000", "-ac", "1", flac_path],
                capture_output=True, timeout=60, encoding="utf-8", errors="replace"
            )
            if proc.returncode != 0:
                print(f"  [STT] FFmpeg conversion failed: {proc.stderr[:200]}")
                return None
        except Exception as e:
            print(f"  [STT] FFmpeg error: {e}")
            return None

        # 파일 크기 체크 + 필요 시 OGG_OPUS로 재인코딩
        flac_size = os.path.getsize(flac_path)
        file_mb = flac_size / (1024 * 1024)
        print(f"  [STT] FLAC size: {file_mb:.1f}MB")

        encoding = "FLAC"
        audio_path_final = flac_path

        if file_mb > 9:
            # FLAC이 너무 크면 OGG_OPUS로 재인코딩 (10~20x 압축)
            ogg_path = os.path.join(tempfile.gettempdir(), "stt_input.ogg")
            try:
                proc2 = subprocess.run(
                    ["ffmpeg", "-y", "-i", flac_path, "-ar", "16000", "-ac", "1",
                     "-c:a", "libopus", "-b:a", "32k", ogg_path],
                    capture_output=True, timeout=60, encoding="utf-8", errors="replace"
                )
                if proc2.returncode == 0:
                    ogg_mb = os.path.getsize(ogg_path) / (1024 * 1024)
                    print(f"  [STT] Re-encoded as OGG_OPUS: {ogg_mb:.1f}MB (from {file_mb:.1f}MB FLAC)")
                    encoding = "OGG_OPUS"
                    audio_path_final = ogg_path
                else:
                    print(f"  [STT] OGG_OPUS re-encode failed, trying with FLAC anyway")
            except Exception as e:
                print(f"  [STT] OGG re-encode error: {e}")

        # 파일 읽기 + base64 인코딩
        with open(audio_path_final, "rb") as f:
            audio_content = f.read()
        # 임시 파일 정리
        for tmp in [flac_path, os.path.join(tempfile.gettempdir(), "stt_input.ogg")]:
            try:
                os.remove(tmp)
            except OSError:
                pass

        file_mb = len(audio_content) / (1024 * 1024)
        if file_mb > 10:
            print(f"  [STT] File still too large ({file_mb:.1f}MB > 10MB)")
            return None

        audio_b64 = base64.b64encode(audio_content).decode()

        # longrunningrecognize 비동기 호출
        sample_rate = 16000 if encoding == "FLAC" else 48000
        url = f"https://speech.googleapis.com/v1/speech:longrunningrecognize?key={api_key}"
        payload = {
            "config": {
                "encoding": encoding,
                "sampleRateHertz": sample_rate,
                "languageCode": language,
                "enableWordTimeOffsets": True,
                "model": "latest_long",
                "useEnhanced": True,
            },
            "audio": {"content": audio_b64}
        }

        print(f"  [STT] Calling Google Cloud Speech-to-Text (longrunningrecognize)...")
        try:
            resp = _requests.post(url, json=payload, timeout=30)
        except Exception as e:
            print(f"  [STT] Request failed: {e}")
            return None

        if resp.status_code != 200:
            error_text = resp.text[:300]
            print(f"  [STT] API error {resp.status_code}: {error_text}")
            if "PERMISSION_DENIED" in error_text or "API key" in error_text:
                print(f"  [STT] Hint: API key may not have Speech-to-Text API enabled.")
                print(f"  [STT] Enable it at: https://console.cloud.google.com/apis/library/speech.googleapis.com")
            return None

        operation = resp.json()
        operation_name = operation.get("name")
        if not operation_name:
            print(f"  [STT] No operation name in response")
            return None

        # 결과 폴링 (최대 3분)
        poll_url = f"https://speech.googleapis.com/v1/operations/{operation_name}?key={api_key}"
        print(f"  [STT] Waiting for transcription result...")
        for attempt in range(90):
            _time.sleep(2)
            try:
                poll_resp = _requests.get(poll_url, timeout=15)
                if poll_resp.status_code != 200:
                    continue
                result = poll_resp.json()
                if result.get("done"):
                    break
            except Exception:
                continue
        else:
            print("  [STT] Timeout (3min) waiting for transcription")
            return None

        if "error" in result:
            print(f"  [STT] Error: {result['error'].get('message', result['error'])}")
            return None

        # 단어 단위 타임스탬프 파싱
        response_data = result.get("response", {})
        results = response_data.get("results", [])

        words = []
        for r in results:
            alt = r.get("alternatives", [{}])[0]
            for w in alt.get("words", []):
                start_time = w.get("startTime", "0s")
                t = float(start_time.rstrip("s"))
                words.append({"t": round(t, 2), "text": w.get("word", "")})

        if words:
            print(f"  [STT] Transcribed: {len(words)} words, {words[0]['t']}s ~ {words[-1]['t']}s")
        else:
            print(f"  [STT] No words detected")
            return None

        return words

    def _align_stt_with_lyrics(self, stt_words: list, user_lyrics: str) -> list:
        """
        STT 단어 타임스탬프와 사용자 가사를 정렬하여 줄 단위 timed_lyrics 생성

        방식: STT 텍스트를 연결 → 사용자 가사 각 줄의 위치를 찾아 타임스탬프 매핑
        """
        import re as _re

        # 1. STT 단어들을 연결하면서 각 문자의 타임스탬프 기록
        char_timestamps = []  # [(char, timestamp), ...]
        for word in stt_words:
            for ch in word["text"]:
                char_timestamps.append((ch, word["t"]))

        # 공백 제거한 STT 전체 텍스트
        stt_full = "".join(ch for ch, _ in char_timestamps)
        stt_full_lower = stt_full.lower()

        # 2. 사용자 가사 줄별 처리
        lines = [l.strip() for l in user_lyrics.strip().split('\n') if l.strip()]
        lines = [l for l in lines if not _re.match(r'^\[.*?\]$', l)]  # 섹션 마커 제거

        result = []
        search_start = 0  # 순차 검색 위치

        for line in lines:
            # 공백/특수문자 제거하여 검색용 텍스트 생성
            clean_line = _re.sub(r'[\s\-.,!?~]+', '', line).lower()
            if not clean_line:
                continue

            # STT 텍스트에서 위치 찾기 (순차 전진)
            pos = stt_full_lower.find(clean_line, search_start)

            if pos == -1 and search_start > 0:
                # 못 찾으면 처음부터 다시 (반복 가사 등)
                pos = stt_full_lower.find(clean_line, 0)

            if pos == -1:
                # 부분 매칭 시도 (앞 절반만)
                half = clean_line[:len(clean_line) // 2]
                if len(half) >= 3:
                    pos = stt_full_lower.find(half, search_start)
                    if pos == -1:
                        pos = stt_full_lower.find(half, 0)

            if pos >= 0 and pos < len(char_timestamps):
                t = char_timestamps[pos][1]
                result.append({"t": t, "text": line})
                search_start = pos + len(clean_line)
            else:
                # 매칭 실패 → 이전 타임스탬프에서 보간
                if result:
                    result.append({"t": round(result[-1]["t"] + 3.0, 2), "text": line})
                else:
                    result.append({"t": 0.0, "text": line})

        if result:
            print(f"  [STT Align] Matched {len(result)} lyrics lines, {result[0]['t']}s ~ {result[-1]['t']}s")

        return result

    def _stt_words_to_timed_lyrics(self, stt_words: list) -> list:
        """
        STT 단어를 문장 단위 timed_lyrics로 그룹화 (사용자 가사 없을 때)
        1초 이상 gap이 있으면 새 문장으로 분리
        """
        if not stt_words:
            return []

        sentences = []
        current_words = [stt_words[0]]

        for i in range(1, len(stt_words)):
            gap = stt_words[i]["t"] - stt_words[i - 1]["t"]
            if gap > 1.0:
                text = " ".join(w["text"] for w in current_words)
                sentences.append({"t": current_words[0]["t"], "text": text})
                current_words = [stt_words[i]]
            else:
                current_words.append(stt_words[i])

        if current_words:
            text = " ".join(w["text"] for w in current_words)
            sentences.append({"t": current_words[0]["t"], "text": text})

        return sentences

    # ================================================================
    # Main lyrics sync: STT first, Gemini fallback
    # ================================================================

    def sync_user_lyrics_with_gemini(self, audio_path: str, user_lyrics: str) -> Optional[str]:
        """
        사용자 가사를 음악과 타이밍 싱크 (v3 - Google STT + Gemini fallback)

        1차: Google Cloud STT로 단어 단위 타임스탬프 → 사용자 가사와 정렬
        2차 (fallback): Gemini로 참조 기반 추출

        Args:
            audio_path: 음악 파일 경로
            user_lyrics: 사용자가 입력한 가사 텍스트

        Returns:
            가사 텍스트 (timed_lyrics는 self._last_timed_lyrics에 저장)
        """
        print(f"[MusicAnalyzer] Syncing lyrics with music timing (v3)...")

        # === 1차: Google Cloud STT ===
        stt_words = self.transcribe_with_google_stt(audio_path)
        if stt_words and len(stt_words) >= 5:
            # Raw STT 문장 보존 (타이밍 에디터용)
            self._last_stt_sentences = self._stt_words_to_timed_lyrics(stt_words)
            print(f"  [STT] Success! Aligning {len(stt_words)} words with user lyrics...")
            result = self._align_stt_with_lyrics(stt_words, user_lyrics)
            if result and len(result) >= 3:
                self._last_timed_lyrics = result
                plain = "\n".join(e["text"] for e in result if e.get("text"))
                print(f"  [STT] Alignment complete: {len(result)} entries")
                return plain
            else:
                print(f"  [STT] Alignment produced too few results, falling back to Gemini")
        else:
            self._last_stt_sentences = None

        # === 2차: Gemini fallback ===
        print(f"  [Fallback] Using Gemini for lyrics sync...")

        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            print("  [Gemini] GOOGLE_API_KEY not set, skipping")
            self._last_timed_lyrics = None
            return user_lyrics

        try:
            from google import genai
            from google.genai import types
            import shutil
            import tempfile
            import re as _re_local

            client = genai.Client(api_key=api_key)

            # 섹션 마커 제거 ([Chorus], [Verse 1] 등)
            cleaned_lyrics = _re_local.sub(r'^\[.*?\]\s*$', '', user_lyrics, flags=_re_local.MULTILINE).strip()
            cleaned_lyrics = _re_local.sub(r'\n{3,}', '\n\n', cleaned_lyrics)

            # 오디오 길이 확인
            duration = self._get_audio_duration(audio_path)
            print(f"  [Gemini] Audio duration: {duration:.1f}s")

            if duration <= 90:
                result = self._sync_single_pass(client, audio_path, cleaned_lyrics)
            else:
                result = self._sync_chunked(client, audio_path, cleaned_lyrics, duration)

            if result and self._validate_timestamps(result):
                self._last_timed_lyrics = result
                # STT 실패 시 Gemini 결과를 stt_sentences 대체로 저장
                if not self._last_stt_sentences:
                    self._last_stt_sentences = [{"t": e["t"], "text": e["text"]} for e in result if e.get("text")]
                plain = "\n".join(e["text"] for e in result if e.get("text"))
                print(f"  [Gemini] Sync complete: {len(result)} entries, {len(plain)} chars")
                print(f"  [Gemini] Time range: {result[0]['t']}s ~ {result[-1]['t']}s")
                return plain
            elif result:
                print(f"  [Gemini] Timestamps imperfect, using with interpolation fix")
                self._last_timed_lyrics = result
                if not self._last_stt_sentences:
                    self._last_stt_sentences = [{"t": e["t"], "text": e["text"]} for e in result if e.get("text")]
                plain = "\n".join(e["text"] for e in result if e.get("text"))
                return plain
            else:
                print(f"  [Gemini] Extraction failed, using lyrics without timing")
                self._last_timed_lyrics = None
                return user_lyrics

        except Exception as e:
            print(f"  [Gemini] Lyrics sync failed: {e}")
            import traceback
            traceback.print_exc()
            self._last_timed_lyrics = None
            return user_lyrics

    def _sync_single_pass(self, client, audio_path: str, ref_lyrics: str) -> Optional[list]:
        """참조 가사 기반 단일 패스 추출 (90초 이하)"""
        from google.genai import types
        import shutil
        import tempfile

        ext = Path(audio_path).suffix
        temp_dir = tempfile.gettempdir()
        temp_path = os.path.join(temp_dir, f"gemini_refsync_upload{ext}")
        shutil.copy2(audio_path, temp_path)

        try:
            audio_file = client.files.upload(file=temp_path)
            print(f"  [Gemini] File uploaded: {audio_file.name}")
        finally:
            try:
                os.remove(temp_path)
            except OSError:
                pass

        prompt_text = (
            "이 음악을 처음부터 끝까지 주의 깊게 들으면서 가사를 타임스탬프와 함께 추출하세요.\n\n"
            "아래는 참조용 가사입니다. 텍스트는 이것을 기준으로 매칭하세요:\n"
            "== 참조 가사 ==\n"
            f"{ref_lyrics}\n"
            "== 끝 ==\n\n"
            "작업 방법:\n"
            "1. 음악을 듣고 보컬이 시작되는 정확한 시간을 감지하세요\n"
            "2. 들리는 가사를 위 참조 가사에서 찾아 매칭하세요\n"
            "3. 참조 가사에 있는 텍스트를 그대로 사용하세요 (임의 수정 금지)\n"
            "4. 보컬이 없는 구간(간주/인트로/아웃트로)은 건너뛰세요\n\n"
            "출력 규칙:\n"
            "- t = 해당 가사가 보컬로 불리기 시작하는 정확한 시간(초)\n"
            "- t는 반드시 단조 증가 (이전 값보다 항상 커야 함)\n"
            "- 후렴구가 반복되면 매번 새로운 t를 부여하세요\n"
            "- [Chorus], [Verse] 같은 섹션 마커는 포함하지 마세요\n"
            "- 한 줄이 30자를 초과하면 적절히 분할하세요\n\n"
            "출력: JSON 배열만\n"
            '[{"t": 5.2, "text": "가사 첫줄"}, {"t": 8.7, "text": "가사 둘째줄"}, ...]'
        )

        print(f"  [Gemini] Extracting with reference lyrics (single pass)...")
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[audio_file, prompt_text],
            config=types.GenerateContentConfig(
                temperature=0.0,
                response_mime_type="application/json",
            )
        )

        return self._parse_gemini_response(response.text, "ref-sync")

    def _sync_chunked(self, client, audio_path: str, ref_lyrics: str, total_duration: float) -> Optional[list]:
        """참조 가사 기반 청크 분할 추출 (90초 초과)"""
        from google.genai import types
        import subprocess
        import shutil
        import tempfile

        # 가사를 대략적으로 분할 (청크에 해당하는 가사 범위 힌트)
        lyrics_lines = [l.strip() for l in ref_lyrics.split('\n') if l.strip()]
        total_lines = len(lyrics_lines)

        chunk_duration = 60
        overlap = 5
        chunks = []
        start = 0
        while start < total_duration:
            end = min(start + chunk_duration, total_duration)
            chunks.append((start, end))
            start = end - overlap
            if end >= total_duration:
                break

        print(f"  [Gemini] Chunked ref-sync: {len(chunks)} chunks for {total_duration:.0f}s audio")

        all_entries = []
        ext = Path(audio_path).suffix
        temp_dir = tempfile.gettempdir()

        for i, (chunk_start, chunk_end) in enumerate(chunks):
            chunk_path = os.path.join(temp_dir, f"gemini_refsync_chunk_{i}{ext}")

            try:
                # ffmpeg로 청크 추출
                cmd = [
                    "ffmpeg", "-y",
                    "-ss", str(chunk_start),
                    "-t", str(chunk_end - chunk_start),
                    "-i", audio_path,
                    "-acodec", "copy",
                    chunk_path
                ]
                subprocess.run(
                    cmd, capture_output=True, timeout=30,
                    encoding='utf-8', errors='replace'
                )

                if not os.path.exists(chunk_path):
                    print(f"  [Gemini] Chunk {i+1} extraction failed")
                    continue

                upload_path = os.path.join(temp_dir, f"gemini_refsync_upload_{i}{ext}")
                shutil.copy2(chunk_path, upload_path)

                try:
                    audio_file = client.files.upload(file=upload_path)
                except Exception as upload_err:
                    print(f"  [Gemini] Chunk {i+1} upload failed: {upload_err}")
                    continue
                finally:
                    try:
                        os.remove(upload_path)
                    except OSError:
                        pass

                # 이 청크에 해당하는 참조 가사 범위 추정
                ratio_start = chunk_start / total_duration
                ratio_end = chunk_end / total_duration
                line_start = int(ratio_start * total_lines)
                line_end = min(int(ratio_end * total_lines) + 3, total_lines)  # 여유분
                chunk_ref = "\n".join(lyrics_lines[line_start:line_end])

                prompt_text = (
                    f"이 오디오 클립은 전체 노래의 {chunk_start:.0f}초~{chunk_end:.0f}초 구간입니다.\n"
                    "이 클립을 듣고 가사를 타임스탬프와 함께 추출하세요.\n\n"
                    "참조용 가사 (이 구간에 해당하는 부분):\n"
                    f"== 참조 ==\n{chunk_ref}\n== 끝 ==\n\n"
                    "규칙:\n"
                    "1. t = 이 클립 내에서 해당 가사가 시작되는 시간(초) (0부터 시작)\n"
                    "2. t는 반드시 단조 증가\n"
                    "3. 참조 가사의 텍스트를 그대로 사용하세요\n"
                    "4. 보컬이 있는 구간만 기록\n"
                    "5. 한 줄 최대 30자\n\n"
                    "출력: JSON 배열만\n"
                    '[{"t": 0.0, "text": "가사"}, {"t": 3.5, "text": "가사"}, ...]'
                )

                print(f"  [Gemini] Chunk {i+1}/{len(chunks)} ({chunk_start:.0f}s~{chunk_end:.0f}s)...")
                response = client.models.generate_content(
                    model="gemini-2.5-flash",
                    contents=[audio_file, prompt_text],
                    config=types.GenerateContentConfig(
                        temperature=0.0,
                        response_mime_type="application/json",
                    )
                )

                chunk_entries = self._parse_gemini_response(response.text, f"ref-chunk{i+1}")
                if chunk_entries:
                    for entry in chunk_entries:
                        entry["t"] = round(entry["t"] + chunk_start, 1)
                    print(f"    -> {len(chunk_entries)} entries ({chunk_entries[0]['t']}s ~ {chunk_entries[-1]['t']}s)")
                    all_entries.extend(chunk_entries)

            finally:
                try:
                    os.remove(chunk_path)
                except OSError:
                    pass

        if not all_entries:
            print(f"  [Gemini] No lyrics from any chunk")
            return None

        all_entries.sort(key=lambda e: e["t"])
        merged = self._deduplicate_entries(all_entries)
        print(f"  [Gemini] Merged: {len(merged)} entries, range: {merged[0]['t']}s ~ {merged[-1]['t']}s")
        return merged

    def extract_lyrics_with_gemini(self, audio_path: str) -> Optional[str]:
        """
        가사 추출 (타임스탬프 포함)

        1순위: Google Cloud STT (word-level 타임스탬프)
        2순위: Gemini 2.5 Flash
        3순위: Whisper 하이브리드

        Args:
            audio_path: 음악 파일 경로

        Returns:
            추출된 가사 텍스트 (실패 시 None)
        """
        print(f"[MusicAnalyzer] Extracting lyrics...")

        # Step 0: Google Cloud STT 우선 시도
        stt_words = self.transcribe_with_google_stt(audio_path)
        if stt_words and len(stt_words) >= 5:
            result = self._stt_words_to_timed_lyrics(stt_words)
            # Raw STT 문장 보존 (타이밍 에디터용)
            self._last_stt_sentences = list(result) if result else None
            if result and len(result) >= 3:
                self._last_timed_lyrics = result
                plain = "\n".join(e["text"] for e in result if e.get("text"))
                print(f"  [STT] Extracted: {len(result)} lines, {len(plain)} chars")
                return plain
        else:
            self._last_stt_sentences = None

        # Step 1: Gemini 2.5 Flash로 시도
        gemini_result = self._extract_with_gemini_25(audio_path)

        if gemini_result:
            # 타임스탬프 품질 검증
            if self._validate_timestamps(gemini_result):
                self._last_timed_lyrics = gemini_result
                # STT 실패 시 Gemini 결과를 stt_sentences 대체로 저장
                if not self._last_stt_sentences:
                    self._last_stt_sentences = [{"t": e["t"], "text": e["text"]} for e in gemini_result if e.get("text")]
                plain = "\n".join(e["text"] for e in gemini_result if e.get("text"))
                print(f"  [Gemini 2.5] Final: {len(gemini_result)} entries, {len(plain)} chars")
                print(f"  [Gemini 2.5] Time range: {gemini_result[0]['t']}s ~ {gemini_result[-1]['t']}s")
                return plain
            else:
                print(f"  [Gemini 2.5] Timestamps broken, falling back to Whisper...")

        # Step 2: Fallback - Whisper 하이브리드
        print(f"  [Fallback] Trying Whisper...")
        whisper_result = self._extract_with_whisper(audio_path)

        if whisper_result:
            # Gemini 텍스트가 있으면 하이브리드
            gemini_text = self._extract_gemini_text_only(audio_path)
            if gemini_text:
                merged = self._merge_whisper_gemini(whisper_result, gemini_text)
                if merged:
                    self._last_timed_lyrics = merged
                    if not self._last_stt_sentences:
                        self._last_stt_sentences = [{"t": e["t"], "text": e["text"]} for e in merged if e.get("text")]
                    plain = "\n".join(e["text"] for e in merged if e.get("text"))
                    print(f"  [Hybrid] Final: {len(merged)} entries")
                    return plain

            self._last_timed_lyrics = whisper_result
            if not self._last_stt_sentences:
                self._last_stt_sentences = [{"t": e["t"], "text": e["text"]} for e in whisper_result if e.get("text")]
            plain = "\n".join(e["text"] for e in whisper_result if e.get("text"))
            print(f"  [Whisper-only] Final: {len(whisper_result)} entries")
            return plain

        # Step 3: 최후 fallback - Gemini 결과를 보간으로 수정해서 사용
        if gemini_result:
            print(f"  [Last resort] Using Gemini with interpolated timestamps")
            self._last_timed_lyrics = gemini_result
            plain = "\n".join(e["text"] for e in gemini_result if e.get("text"))
            return plain

        return None

    def _extract_with_gemini_25(self, audio_path: str) -> Optional[list]:
        """
        Gemini 2.5 Flash로 가사+타임스탬프 추출
        긴 노래(>90초)는 청크 단위로 분할하여 누락 방지

        Returns:
            [{"t": float, "text": str}, ...] 또는 None
        """
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            print("  [Gemini 2.5] GOOGLE_API_KEY not set")
            return None

        try:
            # 오디오 길이 확인
            duration = self._get_audio_duration(audio_path)
            print(f"  [Gemini 2.5] Audio duration: {duration:.1f}s")

            if duration <= 90:
                # 짧은 노래: 단일 패스
                return self._gemini25_single_pass(audio_path)
            else:
                # 긴 노래: 청크 분할 추출
                return self._gemini25_chunked(audio_path, duration)

        except Exception as e:
            print(f"  [Gemini 2.5] Failed: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _get_audio_duration(self, audio_path: str) -> float:
        """오디오 파일 길이(초) 반환"""
        if PYDUB_AVAILABLE:
            try:
                audio = AudioSegment.from_file(audio_path)
                return len(audio) / 1000.0
            except Exception:
                pass

        # fallback: ffprobe
        import subprocess
        try:
            result = subprocess.run(
                ["ffprobe", "-v", "quiet", "-show_entries", "format=duration",
                 "-of", "default=noprint_wrappers=1:nokey=1", audio_path],
                capture_output=True, text=True, timeout=30,
                encoding='utf-8', errors='replace'
            )
            return float(result.stdout.strip())
        except Exception:
            return 0.0

    def _gemini25_single_pass(self, audio_path: str) -> Optional[list]:
        """Gemini 2.5 단일 패스 추출 (짧은 노래용)"""
        from google import genai
        from google.genai import types
        import shutil
        import tempfile
        import json

        client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

        # non-ASCII 파일명 대응
        ext = Path(audio_path).suffix
        temp_dir = tempfile.gettempdir()
        temp_path = os.path.join(temp_dir, f"gemini25_upload{ext}")
        shutil.copy2(audio_path, temp_path)

        try:
            audio_file = client.files.upload(file=temp_path)
            print(f"  [Gemini 2.5] File uploaded: {audio_file.name}")
        finally:
            try:
                os.remove(temp_path)
            except OSError:
                pass

        prompt_text = (
            "이 음악 파일을 정밀하게 들으면서 가사를 타임스탬프와 함께 추출하세요.\n\n"
            "핵심 규칙:\n"
            "1. t = 해당 가사가 보컬로 불리기 시작하는 정확한 시간(초)\n"
            "2. t는 반드시 단조 증가 (이전 값보다 항상 커야 함)\n"
            "3. 후렴구가 반복될 때마다 t가 달라져야 합니다\n"
            "4. 노래 전체를 처음부터 끝까지 빠짐없이 기록\n"
            "5. 간주/인스트루멘탈 구간은 건너뛰세요\n"
            "6. 한 줄 최대 30자\n\n"
            "출력: JSON 배열만\n"
            '[{"t": 5.2, "text": "가사"}, {"t": 8.7, "text": "가사"}, ...]'
        )

        print(f"  [Gemini 2.5] Extracting lyrics (single pass)...")
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[audio_file, prompt_text],
            config=types.GenerateContentConfig(
                temperature=0.0,
                response_mime_type="application/json",
            )
        )

        return self._parse_gemini_response(response.text, "single")

    def _gemini25_chunked(self, audio_path: str, total_duration: float) -> Optional[list]:
        """
        긴 노래를 청크로 분할하여 Gemini 2.5로 각각 추출 후 병합
        - 청크 크기: 60초
        - 오버랩: 5초 (경계 누락 방지)
        """
        from google import genai
        from google.genai import types
        import subprocess
        import shutil
        import tempfile
        import json

        client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

        # 청크 구간 계산
        chunk_duration = 60  # 60초씩
        overlap = 5  # 5초 오버랩
        chunks = []
        start = 0
        while start < total_duration:
            end = min(start + chunk_duration, total_duration)
            chunks.append((start, end))
            start = end - overlap  # 오버랩 포함
            if end >= total_duration:
                break

        print(f"  [Gemini 2.5] Chunked extraction: {len(chunks)} chunks for {total_duration:.0f}s audio")
        for i, (s, e) in enumerate(chunks):
            print(f"    Chunk {i+1}: {s:.0f}s ~ {e:.0f}s")

        all_entries = []
        ext = Path(audio_path).suffix
        temp_dir = tempfile.gettempdir()

        for i, (chunk_start, chunk_end) in enumerate(chunks):
            chunk_path = os.path.join(temp_dir, f"gemini25_chunk_{i}{ext}")

            try:
                # ffmpeg로 청크 추출
                chunk_duration_sec = chunk_end - chunk_start
                cmd = [
                    "ffmpeg", "-y",
                    "-ss", str(chunk_start),
                    "-t", str(chunk_duration_sec),
                    "-i", audio_path,
                    "-acodec", "copy",
                    chunk_path
                ]
                subprocess.run(
                    cmd, capture_output=True, timeout=30,
                    encoding='utf-8', errors='replace'
                )

                if not os.path.exists(chunk_path):
                    print(f"  [Gemini 2.5] Chunk {i+1} extraction failed")
                    continue

                # Gemini에 업로드
                upload_path = os.path.join(temp_dir, f"gemini25_chunk_upload_{i}{ext}")
                shutil.copy2(chunk_path, upload_path)

                try:
                    audio_file = client.files.upload(file=upload_path)
                except Exception as upload_err:
                    print(f"  [Gemini 2.5] Chunk {i+1} upload failed: {upload_err}")
                    continue
                finally:
                    try:
                        os.remove(upload_path)
                    except OSError:
                        pass

                # 청크별 프롬프트 (구간 정보 포함)
                prompt_text = (
                    f"이 오디오 클립은 전체 노래의 {chunk_start:.0f}초~{chunk_end:.0f}초 구간입니다.\n"
                    "이 클립에서 들리는 가사를 타임스탬프와 함께 추출하세요.\n\n"
                    "핵심 규칙:\n"
                    "1. t = 이 클립 내에서 해당 가사가 시작되는 시간(초) (0부터 시작)\n"
                    "2. t는 반드시 단조 증가\n"
                    "3. 보컬이 있는 구간만 기록, 간주/인스트루멘탈 건너뛰기\n"
                    "4. 한 줄 최대 30자\n\n"
                    "출력: JSON 배열만\n"
                    '[{"t": 0.0, "text": "가사"}, {"t": 3.5, "text": "가사"}, ...]'
                )

                print(f"  [Gemini 2.5] Chunk {i+1}/{len(chunks)} ({chunk_start:.0f}s~{chunk_end:.0f}s)...")
                response = client.models.generate_content(
                    model="gemini-2.5-flash",
                    contents=[audio_file, prompt_text],
                    config=types.GenerateContentConfig(
                        temperature=0.0,
                        response_mime_type="application/json",
                    )
                )

                chunk_entries = self._parse_gemini_response(response.text, f"chunk{i+1}")
                if chunk_entries:
                    # 타임스탬프에 청크 시작 시간 오프셋 추가
                    for entry in chunk_entries:
                        entry["t"] = round(entry["t"] + chunk_start, 1)
                    print(f"    -> {len(chunk_entries)} entries ({chunk_entries[0]['t']}s ~ {chunk_entries[-1]['t']}s)")
                    all_entries.extend(chunk_entries)

            finally:
                try:
                    os.remove(chunk_path)
                except OSError:
                    pass

        if not all_entries:
            print(f"  [Gemini 2.5] No lyrics from any chunk")
            return None

        # 시간순 정렬
        all_entries.sort(key=lambda e: e["t"])

        # 오버랩 구간 중복 제거
        merged = self._deduplicate_entries(all_entries)

        print(f"  [Gemini 2.5] Merged: {len(merged)} entries, range: {merged[0]['t']}s ~ {merged[-1]['t']}s")
        return merged

    def _parse_gemini_response(self, response_text: str, label: str) -> Optional[list]:
        """Gemini 응답 JSON 파싱 및 유효 엔트리 필터"""
        import json

        try:
            result = json.loads(response_text.strip())
        except json.JSONDecodeError:
            print(f"  [Gemini 2.5/{label}] JSON parse failed")
            return None

        if not isinstance(result, list) or not result:
            print(f"  [Gemini 2.5/{label}] No lyrics detected")
            return None

        entries = []
        for entry in result:
            text = entry.get("text", "").strip()
            if text:
                entries.append({"t": round(float(entry.get("t", 0)), 1), "text": text})

        if not entries:
            return None

        return entries

    def _deduplicate_entries(self, entries: list) -> list:
        """
        오버랩 구간에서 발생하는 중복 가사 제거
        - 시간차 3초 이내 + 텍스트 유사도 높은 엔트리 병합
        """
        if len(entries) <= 1:
            return entries

        merged = [entries[0]]
        for entry in entries[1:]:
            prev = merged[-1]
            time_diff = abs(entry["t"] - prev["t"])

            # 시간차 3초 이내이고 텍스트가 비슷하면 중복으로 간주
            if time_diff < 3.0 and self._text_similar(prev["text"], entry["text"]):
                # 더 나중 시간의 것을 유지 (오버랩 뒤쪽이 더 정확)
                continue
            # 시간차 0.5초 미만이면 동일 엔트리로 간주
            elif time_diff < 0.5:
                continue
            else:
                merged.append(entry)

        return merged

    @staticmethod
    def _text_similar(a: str, b: str) -> bool:
        """두 텍스트가 유사한지 비교 (순서 기반)"""
        if a == b:
            return True
        # 한쪽이 다른쪽에 포함
        if a in b or b in a:
            return True
        # 순서 기반 공통 접두사 길이 비교 (set 비교는 한글에서 오판 많음)
        shorter = min(len(a), len(b))
        if shorter == 0:
            return False
        match_count = sum(1 for ca, cb in zip(a, b) if ca == cb)
        return match_count / shorter > 0.6

    def _validate_timestamps(self, timed_lyrics: list) -> bool:
        """
        타임스탬프 품질 검증

        조건:
        - 80% 이상이 단조 증가해야 함
        - 마지막 타임스탬프가 전체의 50% 이상 지점이어야 함
        """
        if not timed_lyrics or len(timed_lyrics) < 3:
            return False

        # 단조 증가 비율 체크
        monotonic_count = 0
        for i in range(1, len(timed_lyrics)):
            if timed_lyrics[i]["t"] > timed_lyrics[i - 1]["t"]:
                monotonic_count += 1

        monotonic_ratio = monotonic_count / (len(timed_lyrics) - 1)

        # 마지막 타임스탬프 vs 첫 타임스탬프 범위 체크
        first_t = timed_lyrics[0]["t"]
        last_t = timed_lyrics[-1]["t"]
        time_span = last_t - first_t

        print(f"  [Validate] Monotonic ratio: {monotonic_ratio:.1%}, time span: {first_t:.1f}s ~ {last_t:.1f}s ({time_span:.1f}s)")

        # 80% 이상 단조 증가 + 시간 범위가 30초 이상
        if monotonic_ratio >= 0.8 and time_span >= 30:
            print(f"  [Validate] PASS")
            return True

        print(f"  [Validate] FAIL (monotonic={monotonic_ratio:.1%}, span={time_span:.1f}s)")
        return False

    def _extract_with_whisper(self, audio_path: str) -> Optional[list]:
        """
        OpenAI Whisper로 정확한 타이밍의 가사 추출

        Returns:
            [{"t": float, "text": str}, ...] 또는 None
        """
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            print("  [Whisper] OPENAI_API_KEY not set, skipping")
            return None

        try:
            from openai import OpenAI
            import json

            client = OpenAI(api_key=api_key)

            # 파일 크기 확인 (Whisper 제한: 25MB)
            file_size = os.path.getsize(audio_path)
            if file_size > 25 * 1024 * 1024:
                print(f"  [Whisper] File too large ({file_size / 1024 / 1024:.1f}MB > 25MB)")
                return None

            print(f"  [Whisper] Transcribing audio...")

            with open(audio_path, "rb") as f:
                transcript = client.audio.transcriptions.create(
                    model="whisper-1",
                    file=f,
                    response_format="verbose_json",
                    timestamp_granularities=["segment"],
                )

            # 응답에서 세그먼트 추출
            segments = None
            if hasattr(transcript, 'segments'):
                segments = transcript.segments
            elif isinstance(transcript, dict):
                segments = transcript.get('segments', [])

            if not segments:
                print(f"  [Whisper] No segments detected")
                return None

            timed_lyrics = []
            for seg in segments:
                # 객체/딕셔너리 양쪽 지원
                if isinstance(seg, dict):
                    text = seg.get("text", "").strip()
                    start = float(seg.get("start", 0))
                    end = float(seg.get("end", 0))
                else:
                    text = getattr(seg, "text", "").strip()
                    start = float(getattr(seg, "start", 0))
                    end = float(getattr(seg, "end", 0))

                if not text:
                    continue
                # 너무 짧은 세그먼트 필터 (노이즈)
                if end - start < 0.3:
                    continue

                timed_lyrics.append({
                    "t": round(start, 1),
                    "text": text
                })

            if not timed_lyrics:
                print(f"  [Whisper] No lyrics found")
                return None

            print(f"  [Whisper] Got {len(timed_lyrics)} segments")
            print(f"  [Whisper] Time range: {timed_lyrics[0]['t']}s ~ {timed_lyrics[-1]['t']}s")
            return timed_lyrics

        except Exception as e:
            print(f"  [Whisper] Failed: {e}")
            return None

    def _extract_gemini_text_only(self, audio_path: str) -> Optional[List[str]]:
        """
        Gemini로 가사 텍스트만 추출 (타임스탬프 없이)

        Returns:
            가사 줄 리스트 또는 None
        """
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            return None

        try:
            from google import genai
            from google.genai import types
            import shutil
            import tempfile

            client = genai.Client(api_key=api_key)

            # non-ASCII 파일명 대응
            ext = Path(audio_path).suffix
            temp_dir = tempfile.gettempdir()
            temp_path = os.path.join(temp_dir, f"gemini_text_upload{ext}")
            shutil.copy2(audio_path, temp_path)

            try:
                audio_file = client.files.upload(file=temp_path)
            finally:
                try:
                    os.remove(temp_path)
                except OSError:
                    pass

            print(f"  [Gemini] Extracting lyrics text...")

            prompt_text = (
                "이 음악의 가사를 한 줄씩 추출하세요.\n"
                "- 가사만 출력 (타임스탬프 불필요)\n"
                "- 한 줄당 최대 30자\n"
                "- 가사가 없으면 빈 문자열\n"
                "- 순서대로 줄바꿈으로 구분"
            )

            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=[audio_file, prompt_text],
                config=types.GenerateContentConfig(temperature=0.0)
            )

            text = response.text.strip()
            if not text:
                return None

            lines = [l.strip() for l in text.split('\n') if l.strip()]
            print(f"  [Gemini] Got {len(lines)} lyrics lines")
            return lines

        except Exception as e:
            print(f"  [Gemini] Text extraction failed: {e}")
            return None

    def _merge_whisper_gemini(
        self,
        whisper_timed: list,
        gemini_lines: List[str]
    ) -> Optional[list]:
        """
        Whisper 타이밍 + Gemini 텍스트 병합

        Whisper 세그먼트 수와 Gemini 줄 수가 비슷하면 1:1 대응,
        다르면 Whisper 타이밍을 유지하고 Gemini 텍스트를 순차 매핑
        """
        if not whisper_timed or not gemini_lines:
            return None

        w_count = len(whisper_timed)
        g_count = len(gemini_lines)

        print(f"  [Merge] Whisper: {w_count} segments, Gemini: {g_count} lines")

        # Whisper와 Gemini 줄 수가 비슷하면 (±30%) 1:1 매핑
        ratio = g_count / w_count if w_count > 0 else 0
        if 0.7 <= ratio <= 1.3:
            # 1:1 매핑 (Gemini 텍스트로 교체)
            merged = []
            for i, entry in enumerate(whisper_timed):
                g_idx = min(int(i * g_count / w_count), g_count - 1)
                merged.append({
                    "t": entry["t"],
                    "text": gemini_lines[g_idx]
                })
            print(f"  [Merge] 1:1 mapping applied ({w_count} entries)")
            return merged

        # Gemini 줄이 더 많으면: Whisper 타이밍 구간에 Gemini 줄을 균등 분배
        if g_count > w_count:
            merged = []
            lines_per_seg = g_count / w_count
            for i, entry in enumerate(whisper_timed):
                g_start = int(i * lines_per_seg)
                g_end = int((i + 1) * lines_per_seg)
                # 해당 구간의 Gemini 줄들을 합침
                combined_text = ' '.join(gemini_lines[g_start:g_end])
                if len(combined_text) > 40:
                    # 너무 길면 첫 줄만
                    combined_text = gemini_lines[g_start]
                merged.append({
                    "t": entry["t"],
                    "text": combined_text
                })
            print(f"  [Merge] Gemini text mapped to Whisper timing ({len(merged)} entries)")
            return merged

        # Whisper가 더 많으면: Whisper 그대로 사용 (Gemini 텍스트 무시)
        print(f"  [Merge] Keeping Whisper as-is (more segments)")
        return whisper_timed

    def _extract_with_gemini_only(self, audio_path: str) -> Optional[str]:
        """
        Gemini만으로 가사+타임스탬프 추출 (Whisper 실패 시 fallback)
        """
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            print("[MusicAnalyzer] GOOGLE_API_KEY not set.")
            return None

        try:
            from google import genai
            from google.genai import types
            import shutil
            import tempfile
            import json

            client = genai.Client(api_key=api_key)

            ext = Path(audio_path).suffix
            temp_dir = tempfile.gettempdir()
            temp_path = os.path.join(temp_dir, f"gemini_lyrics_upload{ext}")
            shutil.copy2(audio_path, temp_path)

            try:
                audio_file = client.files.upload(file=temp_path)
            finally:
                try:
                    os.remove(temp_path)
                except OSError:
                    pass

            prompt_text = (
                "이 음악의 가사를 타임스탬프와 함께 추출하세요.\n\n"
                "규칙:\n"
                "- t: 보컬 시작 시간(초), 반드시 단조 증가\n"
                "- 한 줄 최대 30자\n"
                "- 가사 없으면 빈 배열 []\n\n"
                "출력: JSON 배열만\n"
                '[{"t": 0.0, "text": "가사"}, ...]'
            )

            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=[audio_file, prompt_text],
                config=types.GenerateContentConfig(
                    temperature=0.0,
                    response_mime_type="application/json",
                )
            )

            timed_lyrics = json.loads(response.text.strip())
            if not timed_lyrics or not isinstance(timed_lyrics, list):
                return None

            # 유효 엔트리 필터
            valid = [e for e in timed_lyrics if e.get("text", "").strip()]
            if not valid:
                return None

            self._last_timed_lyrics = valid
            plain = "\n".join(e["text"] for e in valid)
            print(f"  [Gemini-only] {len(valid)} entries, {len(plain)} chars")
            return plain

        except Exception as e:
            print(f"[MusicAnalyzer] Gemini extraction failed: {e}")
            self._last_timed_lyrics = None
            return None

    def get_suggested_scene_count(self, duration_sec: float) -> int:
        """
        추천 씬 개수 계산

        - 20초당 1씬 기준
        - 최소 4씬, 최대 15씬
        """
        suggested = int(duration_sec / 20)
        return max(4, min(15, suggested))


# ============================================================
# Phase 2 기능 (추후 구현)
# ============================================================

class AdvancedMusicAnalyzer(MusicAnalyzer):
    """
    Phase 2: 고급 음악 분석

    - 자동 구간 감지 (librosa segmentation)
    - 분위기 추정 (spectral features)
    - 비트 타임스탬프 (beat tracking)
    """

    def analyze_advanced(self, audio_path: str) -> dict:
        """고급 분석 (Phase 2)"""
        # 기본 분석 먼저
        result = self.analyze(audio_path)

        if not self.librosa_available:
            print("[AdvancedMusicAnalyzer] librosa not available. Skipping advanced analysis.")
            return result

        try:
            y, sr = librosa.load(audio_path, sr=22050)

            # 비트 타임스탬프
            result["key_timestamps"] = self._get_beat_timestamps(y, sr)

            # 분위기 추정
            result["mood"] = self._estimate_mood(y, sr)

            # 에너지 레벨
            result["energy"] = self._estimate_energy(y)

        except Exception as e:
            print(f"[AdvancedMusicAnalyzer] Advanced analysis failed: {e}")

        return result

    def _get_beat_timestamps(self, y, sr) -> List[float]:
        """비트 타임스탬프 추출"""
        tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
        beat_times = librosa.frames_to_time(beat_frames, sr=sr)
        # 주요 비트만 (4비트마다)
        return [round(t, 3) for t in beat_times[::4]][:50]  # 최대 50개

    def _estimate_mood(self, y, sr) -> str:
        """분위기 추정 (간단한 휴리스틱)"""
        # Spectral centroid (밝기)
        spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
        avg_centroid = float(np.mean(spectral_centroids))

        # RMS Energy
        rms = librosa.feature.rms(y=y)[0]
        avg_rms = float(np.mean(rms))

        # 간단한 분류
        if avg_rms > 0.1 and avg_centroid > 2000:
            return "energetic"
        elif avg_rms < 0.05:
            return "calm"
        elif avg_centroid < 1500:
            return "dark"
        else:
            return "neutral"

    def _estimate_energy(self, y) -> float:
        """에너지 레벨 (0-1)"""
        rms = librosa.feature.rms(y=y)[0]
        avg_rms = float(np.mean(rms))
        # 0-1 범위로 정규화
        return min(1.0, avg_rms * 5)
