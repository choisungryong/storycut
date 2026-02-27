"""
TTS Agent: Generates narration audio for each scene.
ElevenLabs 전용 — 실패 시 silent placeholder fallback.
멀티 화자 대화 지원 (v3.0).
"""

import os
import subprocess
import time
import tempfile
from typing import Optional, List, Dict, Any
from dataclasses import dataclass
from dotenv import load_dotenv
from utils.logger import get_logger
logger = get_logger("tts_agent")


load_dotenv()

# Voice list cache
_voice_cache: Dict[str, Any] = {"data": None, "expires": 0}


@dataclass
class TTSResult:
    """TTS 생성 결과"""
    audio_path: str
    duration_sec: float  # 실제 오디오 길이
    sentence_timings: Optional[List[Dict[str, Any]]] = None  # 문장별 실제 발화 타이밍
    char_alignment: Optional[Dict[str, Any]] = None  # ElevenLabs 캐릭터별 정밀 타이밍


class TTSAgent:
    """
    Generates narration audio using ElevenLabs TTS.
    Fallback: silent placeholder audio.
    """

    def __init__(self, voice: str = "uyVNoMrnUku1dZyVEXwD"):
        """
        Initialize TTS Agent.

        Args:
            voice: ElevenLabs voice ID (default: Adam)
        """
        self.voice = voice
        self.elevenlabs_key = os.getenv("ELEVENLABS_API_KEY")
        if self.elevenlabs_key:
            logger.info("[TTS Agent] Provider: ElevenLabs")
        else:
            logger.warning("[TTS Agent] Warning: ELEVENLABS_API_KEY not set — will use silent placeholder")

    def generate_speech(
        self,
        scene_id: int,
        narration: str,
        emotion: str = "neutral",
        output_path: str = None
    ) -> TTSResult:
        """
        Generate narration audio for a scene.

        Args:
            scene_id: Scene identifier
            narration: Text to speak
            emotion: Emotional tone (for compatible TTS services)
            output_path: 프로젝트별 출력 경로 (미지정 시 공유 경로 fallback)

        Returns:
            TTSResult with audio_path and duration_sec
        """
        logger.info(f"  [TTS Agent] Generating narration for scene {scene_id}...")
        logger.info(f"     Text: {narration[:60]}...")

        # Build output path — 프로젝트별 경로 우선, 미지정 시 공유 경로 fallback
        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
        else:
            output_dir = "media/audio"
            os.makedirs(output_dir, exist_ok=True)
            output_path = f"{output_dir}/narration_{scene_id:02d}.mp3"

        audio_path = None
        char_alignment = None

        # 1. Try ElevenLabs with alignment (자막 정밀 싱크용)
        if self.elevenlabs_key:
            try:
                result = self._call_elevenlabs_with_alignment(narration, self.voice, output_path)
                audio_path = result["audio_path"]
                char_alignment = result.get("alignment")
            except Exception as e:
                logger.error(f"     [Warning] ElevenLabs TTS failed: {e}")
                audio_path = None

        # 2. Fallback: silent placeholder
        if audio_path is None:
            logger.info(f"     [Fallback] Using silent placeholder audio")
            audio_path = self._generate_placeholder_audio(scene_id, narration, output_path)

        # 측정: 실제 오디오 길이 (FFprobe 사용, 실패 시 텍스트 기반 추정)
        duration_sec = self._get_audio_duration(audio_path, narration_text=narration)

        logger.info(f"     [TTS Agent] Audio saved: {audio_path} (duration: {duration_sec:.2f}s)")
        return TTSResult(audio_path=audio_path, duration_sec=duration_sec, char_alignment=char_alignment)

    def get_available_voices(self) -> List[Dict[str, Any]]:
        """
        ElevenLabs 음성 목록 반환 (5분 캐시).

        Returns:
            [{"voice_id": "...", "name": "...", "category": "...",
              "preview_url": "...", "labels": {"gender": "male", "age": "young"}}]
        """
        global _voice_cache

        # Check cache
        if _voice_cache["data"] and time.time() < _voice_cache["expires"]:
            return _voice_cache["data"]

        if not self.elevenlabs_key:
            logger.info("[TTS Agent] No ELEVENLABS_API_KEY - returning empty voice list")
            return []

        try:
            from elevenlabs.client import ElevenLabs
            client = ElevenLabs(api_key=self.elevenlabs_key)
            response = client.voices.get_all()

            voices = []
            for v in response.voices:
                labels = {}
                if hasattr(v, 'labels') and v.labels:
                    labels = dict(v.labels) if isinstance(v.labels, dict) else {}

                voices.append({
                    "voice_id": v.voice_id,
                    "name": v.name,
                    "category": getattr(v, 'category', 'custom'),
                    "preview_url": getattr(v, 'preview_url', ''),
                    "labels": labels,
                })

            # Cache for 5 minutes
            _voice_cache["data"] = voices
            _voice_cache["expires"] = time.time() + 300

            logger.info(f"[TTS Agent] Loaded {len(voices)} voices from ElevenLabs")
            return voices

        except Exception as e:
            logger.error(f"[TTS Agent] Failed to load voices: {e}")
            return []

    def generate_dialogue_audio(
        self,
        dialogue_lines: List[Dict[str, str]],
        character_voices: List[Dict[str, str]],
        output_path: str,
        silence_gap: float = 0.3,
    ) -> 'TTSResult':
        """
        멀티 화자 TTS: 각 대사를 해당 음성으로 생성 후 하나로 스티칭.

        Args:
            dialogue_lines: [{"speaker": "narrator", "text": "...", "emotion": ""}]
            character_voices: [{"speaker": "narrator", "voice_id": "..."}]
            output_path: 최종 오디오 출력 경로
            silence_gap: 화자 전환 시 무음 간격 (초)

        Returns:
            TTSResult with audio_path and duration_sec
        """
        if not dialogue_lines:
            return TTSResult(audio_path=output_path, duration_sec=0.0)

        # Build speaker → voice_id map
        voice_map = {}
        for cv in character_voices:
            voice_map[cv["speaker"]] = cv["voice_id"]

        # Default voice for unmapped speakers
        default_voice = self.voice

        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # Generate individual audio clips
        temp_dir = tempfile.mkdtemp(prefix="tts_dialogue_")
        clip_paths = []
        line_timings = []
        current_time = 0.0

        for i, line in enumerate(dialogue_lines):
            speaker = line.get("speaker", "narrator")
            text = line.get("text", "").strip()
            if not text:
                continue

            voice_id = voice_map.get(speaker, default_voice)
            clip_path = os.path.join(temp_dir, f"line_{i:03d}.mp3")

            # Generate individual line
            try:
                if self.elevenlabs_key:
                    self._call_elevenlabs_api(text, voice_id, clip_path)
                else:
                    self._generate_placeholder_audio(i, text, clip_path)
            except Exception as e:
                logger.error(f"  [TTS] Line {i} ({speaker}) failed: {e}. Using placeholder.")
                self._generate_placeholder_audio(i, text, clip_path)

            # Get duration
            duration = self._get_audio_duration(clip_path, narration_text=text)

            line_timings.append({
                "speaker": speaker,
                "text": text,
                "start": current_time,
                "end": current_time + duration,
                "duration": duration,
            })

            clip_paths.append(clip_path)
            current_time += duration + silence_gap

        if not clip_paths:
            return TTSResult(audio_path=output_path, duration_sec=0.0)

        # Stitch clips with silence gaps using FFmpeg concat
        self._stitch_audio_clips(clip_paths, output_path, silence_gap)

        total_duration = self._get_audio_duration(output_path)

        # Clean up temp files
        for p in clip_paths:
            try:
                os.remove(p)
            except OSError:
                pass
        try:
            os.rmdir(temp_dir)
        except OSError:
            pass

        # 타이밍 검증: sentence_timings 합산 vs 실제 스티칭 오디오 길이
        if line_timings:
            timings_end = line_timings[-1]["end"]
            timings_sum = sum(t["duration"] for t in line_timings)
            gap_total = (len(line_timings) - 1) * silence_gap
            logger.info(f"  [TTS SYNC DEBUG] clips={len(clip_paths)}, silence_gap={silence_gap}")
            logger.info(f"  [TTS SYNC DEBUG] individual durations: {[round(t['duration'],2) for t in line_timings]}")
            logger.info(f"  [TTS SYNC DEBUG] timings_last_end={timings_end:.2f}s, clips_sum={timings_sum:.2f}s + gaps={gap_total:.2f}s = {timings_sum+gap_total:.2f}s")
            logger.info(f"  [TTS SYNC DEBUG] actual_stitched={total_duration:.2f}s, DRIFT={timings_end - total_duration:+.2f}s")

        logger.info(f"  [TTS] Dialogue audio: {len(clip_paths)} clips, {total_duration:.1f}s total")
        return TTSResult(audio_path=output_path, duration_sec=total_duration, sentence_timings=line_timings)

    def generate_speech_with_timing(
        self,
        scene_id: int,
        narration: str,
        emotion: str = "neutral",
        output_path: str = None
    ) -> TTSResult:
        """
        나레이션을 문장 단위로 TTS 생성하여 실제 발화 타이밍을 측정.
        자막과 TTS 싱크를 정확하게 맞추기 위한 메서드.

        Returns:
            TTSResult with sentence_timings populated
        """
        import re

        if not narration or not narration.strip():
            return self.generate_speech(scene_id, narration, emotion, output_path)

        # 문장 분리 (. ? ! 기준)
        sentences = re.split(r'(?<=[.?!。])\s+', narration.strip())
        # 빈 문장 제거 + 너무 짧은 문장 병합
        merged = []
        for s in sentences:
            s = s.strip()
            if not s:
                continue
            if merged and len(merged[-1]) < 10:
                merged[-1] = merged[-1] + " " + s
            else:
                merged.append(s)
        sentences = merged if merged else [narration.strip()]

        # 문장이 1개면 일반 TTS로 처리
        if len(sentences) <= 1:
            result = self.generate_speech(scene_id, narration, emotion, output_path)
            result.sentence_timings = [{
                "text": narration.strip(),
                "start": 0.0,
                "end": result.duration_sec,
                "duration": result.duration_sec,
            }]
            return result

        logger.info(f"  [TTS] 문장별 TTS 생성: {len(sentences)}개 문장")

        # 문장별 TTS 생성
        temp_dir = tempfile.mkdtemp(prefix="tts_sentence_")
        clip_paths = []
        sentence_timings = []
        current_time = 0.0

        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
        else:
            output_dir = "media/audio"
            os.makedirs(output_dir, exist_ok=True)
            output_path = f"{output_dir}/narration_{scene_id:02d}.mp3"

        for i, sentence in enumerate(sentences):
            clip_path = os.path.join(temp_dir, f"sent_{i:03d}.mp3")

            try:
                if self.elevenlabs_key:
                    self._call_elevenlabs_api(sentence, self.voice, clip_path)
                else:
                    self._generate_placeholder_audio(i, sentence, clip_path)
            except Exception as e:
                logger.error(f"  [TTS] Sentence {i} failed: {e}. Using placeholder.")
                self._generate_placeholder_audio(i, sentence, clip_path)

            duration = self._get_audio_duration(clip_path, narration_text=sentence)

            sentence_timings.append({
                "text": sentence,
                "start": current_time,
                "end": current_time + duration,
                "duration": duration,
            })
            clip_paths.append(clip_path)
            current_time += duration
            logger.info(f"    문장 {i+1}: {sentence[:30]}... → {duration:.1f}s")

        if not clip_paths:
            return self.generate_speech(scene_id, narration, emotion, output_path)

        # 오디오 스티칭 (무음 없이 — 단일 화자 연속 발화)
        self._stitch_audio_clips(clip_paths, output_path, silence_gap=0.0)
        total_duration = self._get_audio_duration(output_path)

        # 정리
        for p in clip_paths:
            try:
                os.remove(p)
            except OSError:
                pass
        try:
            os.rmdir(temp_dir)
        except OSError:
            pass

        # 타이밍 검증
        if sentence_timings:
            timings_end = sentence_timings[-1]["end"]
            timings_sum = sum(t["duration"] for t in sentence_timings)
            logger.info(f"  [TTS SYNC DEBUG] timings_last_end={timings_end:.2f}s, clips_sum={timings_sum:.2f}s")
            logger.info(f"  [TTS SYNC DEBUG] actual_stitched={total_duration:.2f}s, DRIFT={timings_end - total_duration:+.2f}s")

        logger.info(f"  [TTS] 문장별 TTS 완료: {len(sentences)}문장, {total_duration:.1f}s")
        return TTSResult(audio_path=output_path, duration_sec=total_duration, sentence_timings=sentence_timings)

    def _stitch_audio_clips(self, clip_paths: List[str], output_path: str, silence_gap: float = 0.3):
        """FFmpeg로 오디오 클립들을 무음 간격과 함께 연결."""
        if len(clip_paths) == 1:
            # Single clip - just copy
            import shutil
            shutil.copy2(clip_paths[0], output_path)
            return

        # Build FFmpeg filter for concat with silence gaps
        inputs = []
        filter_parts = []
        input_idx = 0

        for i, path in enumerate(clip_paths):
            inputs.extend(["-i", path])
            filter_parts.append(f"[{input_idx}:a]")
            input_idx += 1

            # Add silence between clips (not after last)
            if i < len(clip_paths) - 1 and silence_gap > 0:
                inputs.extend(["-f", "lavfi", "-t", str(silence_gap),
                             "-i", "anullsrc=r=44100:cl=mono"])
                filter_parts.append(f"[{input_idx}:a]")
                input_idx += 1

        filter_str = "".join(filter_parts) + f"concat=n={len(filter_parts)}:v=0:a=1[out]"

        cmd = ["ffmpeg", "-y"] + inputs + [
            "-filter_complex", filter_str,
            "-map", "[out]",
            "-c:a", "libmp3lame", "-b:a", "192k",
            output_path
        ]

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
            if result.returncode != 0:
                logger.error(f"  [TTS] FFmpeg stitch failed: {result.stderr[:200]}")
                # Fallback: just use first clip
                import shutil
                shutil.copy2(clip_paths[0], output_path)
        except Exception as e:
            logger.error(f"  [TTS] FFmpeg stitch error: {e}")
            import shutil
            shutil.copy2(clip_paths[0], output_path)

    def _get_audio_duration(self, audio_path: str, narration_text: str = None) -> float:
        """FFprobe로 오디오 실제 길이 측정"""
        try:
            cmd = [
                "ffprobe",
                "-v", "error",
                "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1",
                audio_path
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
            if result.returncode == 0 and result.stdout.strip():
                duration = float(result.stdout.strip())
                if duration > 0:
                    return duration
        except Exception as e:
            logger.error(f"     [Warning] FFprobe duration check failed: {e}")

        # Fallback: 파일 크기 기반 추정 (MP3 128kbps = 16KB/sec)
        try:
            file_size = os.path.getsize(audio_path)
            if file_size > 0:
                estimated = file_size / 16000.0  # 128kbps MP3 ≈ 16KB/sec
                logger.info(f"     [Duration] Estimated from file size: {estimated:.2f}s ({file_size} bytes)")
                return max(1.0, estimated)
        except Exception:
            pass

        # Fallback: 텍스트 기반 추정 (한국어: ~4음절/초)
        if narration_text:
            char_count = len(narration_text.replace(" ", ""))
            estimated = max(2.0, char_count / 4.0)
            logger.info(f"     [Duration] Estimated from text: {estimated:.2f}s ({char_count} chars)")
            return estimated

        return 5.0  # 최종 기본값

    def _call_elevenlabs_api(self, text: str, voice_id: str, output_path: str) -> str:
        """
        Call ElevenLabs API for high-quality TTS (v2.x API).
        alignment 없이 오디오만 생성 (멀티화자 TTS 등 alignment 불필요한 경우).
        """
        try:
            from elevenlabs.client import ElevenLabs

            logger.info(f"     Using ElevenLabs API (Voice: {voice_id[:8]}...)...")
            client = ElevenLabs(api_key=self.elevenlabs_key)

            audio_generator = client.text_to_speech.convert(
                text=text,
                voice_id=voice_id,
                model_id="eleven_multilingual_v2"
            )

            with open(output_path, "wb") as f:
                for chunk in audio_generator:
                    f.write(chunk)

            logger.info(f"     ElevenLabs TTS generated: {output_path}")
            return output_path

        except ImportError as e:
            logger.error(f"     [Warning] ElevenLabs library import failed: {e}")
            raise RuntimeError("ElevenLabs library not properly installed")
        except Exception as e:
            logger.error(f"     [Error] ElevenLabs API call failed: {e}")
            raise

    def _call_elevenlabs_with_alignment(self, text: str, voice_id: str, output_path: str) -> Dict[str, Any]:
        """
        ElevenLabs convert_with_timestamps API 호출.
        오디오 + 캐릭터별 정밀 타이밍을 함께 반환.

        Returns:
            {"audio_path": str, "alignment": {"characters": [...], "start_times": [...], "end_times": [...]}}
        """
        try:
            from elevenlabs.client import ElevenLabs
            import base64

            logger.info(f"     Using ElevenLabs API with alignment (Voice: {voice_id[:8]}...)...")
            client = ElevenLabs(api_key=self.elevenlabs_key)

            response = client.text_to_speech.convert_with_timestamps(
                text=text,
                voice_id=voice_id,
                model_id="eleven_multilingual_v2"
            )

            # base64 오디오 디코딩 → 파일 저장
            audio_bytes = base64.b64decode(response.audio_base_64)
            os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
            with open(output_path, "wb") as f:
                f.write(audio_bytes)

            # alignment 추출
            alignment_data = None
            _align = response.normalized_alignment or response.alignment
            if _align:
                alignment_data = {
                    "characters": list(_align.characters),
                    "start_times": list(_align.character_start_times_seconds),
                    "end_times": list(_align.character_end_times_seconds),
                }
                logger.info(f"     ElevenLabs alignment: {len(_align.characters)} chars, " f"duration {_align.character_end_times_seconds[-1]:.2f}s")
            else:
                logger.warning(f"     [Warning] ElevenLabs returned no alignment data")

            logger.info(f"     ElevenLabs TTS+alignment generated: {output_path}")
            return {"audio_path": output_path, "alignment": alignment_data}

        except ImportError as e:
            logger.error(f"     [Warning] ElevenLabs library import failed: {e}")
            raise RuntimeError("ElevenLabs library not properly installed")
        except Exception as e:
            logger.error(f"     [Error] ElevenLabs API with alignment failed: {e}")
            # fallback: alignment 없이 일반 호출
            logger.info(f"     [Fallback] Trying without alignment...")
            path = self._call_elevenlabs_api(text, voice_id, output_path)
            return {"audio_path": path, "alignment": None}

    def _generate_placeholder_audio(
        self,
        scene_id: int,
        text: str,
        output_path: str
    ) -> str:
        """
        Generate placeholder audio for testing.

        Creates a simple silent audio file with FFmpeg.

        Args:
            scene_id: Scene number
            text: Narration text (for duration estimation)
            output_path: Output file path

        Returns:
            Path to generated placeholder audio
        """
        import subprocess

        # Estimate duration based on text length (rough: 150 words per minute)
        word_count = len(text.split())
        duration = max(3, word_count / 2.5)  # ~150 wpm = 2.5 words/sec

        # Create silent audio with FFmpeg
        cmd = [
            "ffmpeg",
            "-f", "lavfi",
            "-i", f"anullsrc=r=44100:cl=mono",
            "-t", str(duration),
            "-c:a", "libmp3lame",
            "-b:a", "128k",
            output_path,
            "-y"
        ]

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)

        if result.returncode != 0:
            raise RuntimeError(f"Failed to generate placeholder audio: {result.stderr}")

        logger.info(f"     Placeholder audio generated: {output_path}")
        return output_path
