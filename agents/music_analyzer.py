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
        target_segment_duration: float = 8.0
    ) -> List[dict]:
        """
        기본 균등 구간 분할

        Args:
            duration_sec: 전체 길이
            target_segment_duration: 목표 구간 길이 (기본 8초)

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

    def extract_lyrics_with_gemini(self, audio_path: str) -> Optional[str]:
        """
        Gemini API로 음악에서 가사 자동 추출 (타임스탬프 포함)

        Args:
            audio_path: 음악 파일 경로

        Returns:
            추출된 가사 텍스트 (실패 시 None)
        """
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            print("[MusicAnalyzer] GOOGLE_API_KEY not set. Skipping lyrics extraction.")
            return None

        try:
            from google import genai
            from google.genai import types
            import shutil
            import tempfile

            client = genai.Client(api_key=api_key)

            print(f"[MusicAnalyzer] Extracting lyrics with Gemini...")

            # 한국어 파일명 등 non-ASCII 경로 대응: 임시 ASCII 파일명으로 복사
            ext = Path(audio_path).suffix
            temp_dir = tempfile.gettempdir()
            temp_path = os.path.join(temp_dir, f"gemini_lyrics_upload{ext}")
            shutil.copy2(audio_path, temp_path)

            # 오디오 파일 업로드
            try:
                audio_file = client.files.upload(file=temp_path)
                print(f"  File uploaded: {audio_file.name}")
            finally:
                # 임시 파일 정리
                try:
                    os.remove(temp_path)
                except OSError:
                    pass

            # Gemini에 타임스탬프 포함 가사 추출 요청
            prompt_text = (
                "이 음악을 듣고 가사를 타임스탬프와 함께 추출해주세요.\n"
                "각 줄마다 가사가 시작되는 시간(초)을 포함해주세요.\n\n"
                "출력 형식 (JSON 배열):\n"
                '[{"t": 0.0, "text": "첫 번째 가사 줄"}, {"t": 5.2, "text": "두 번째 가사 줄"}, ...]\n\n'
                "규칙:\n"
                "- t는 해당 가사가 불리기 시작하는 시간(초, 소수점 1자리)\n"
                "- 인스트루멘탈 구간은 건너뛰세요\n"
                "- 가사가 없는 음악이면 빈 배열 []을 출력하세요\n"
                "- JSON 형식만 출력하고 다른 텍스트는 포함하지 마세요"
            )
            response = client.models.generate_content(
                model="gemini-2.0-flash",
                contents=[audio_file, prompt_text],
                config=types.GenerateContentConfig(
                    temperature=0.1,
                    response_mime_type="application/json",
                )
            )

            import json
            result_text = response.text.strip()

            # JSON 파싱
            timed_lyrics = json.loads(result_text)

            if not timed_lyrics or not isinstance(timed_lyrics, list):
                print(f"  No lyrics detected (instrumental)")
                return None

            # 플레인 텍스트 가사 (호환용)
            plain_lyrics = "\n".join([item["text"] for item in timed_lyrics if item.get("text")])

            if not plain_lyrics.strip():
                print(f"  No lyrics detected (empty)")
                return None

            # timed_lyrics를 별도 속성에 저장
            self._last_timed_lyrics = timed_lyrics

            print(f"  Lyrics extracted: {len(plain_lyrics)} chars, {len(timed_lyrics)} timed entries")
            return plain_lyrics

        except Exception as e:
            print(f"[MusicAnalyzer] Lyrics extraction failed: {e}")
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
