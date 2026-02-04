"""
TTS Agent: Generates narration audio for each scene.
ElevenLabs 전용 — 실패 시 silent placeholder fallback.
"""

import os
import subprocess
from typing import Optional
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()


@dataclass
class TTSResult:
    """TTS 생성 결과"""
    audio_path: str
    duration_sec: float  # 실제 오디오 길이


class TTSAgent:
    """
    Generates narration audio using ElevenLabs TTS.
    Fallback: silent placeholder audio.
    """

    def __init__(self, voice: str = "pNInz6obpgDQGcFmaJgB"):
        """
        Initialize TTS Agent.

        Args:
            voice: ElevenLabs voice ID (default: Adam)
        """
        self.voice = voice
        self.elevenlabs_key = os.getenv("ELEVENLABS_API_KEY")
        if self.elevenlabs_key:
            print("[TTS Agent] Provider: ElevenLabs")
        else:
            print("[TTS Agent] Warning: ELEVENLABS_API_KEY not set — will use silent placeholder")

    def generate_speech(
        self,
        scene_id: int,
        narration: str,
        emotion: str = "neutral"
    ) -> TTSResult:
        """
        Generate narration audio for a scene.

        Args:
            scene_id: Scene identifier
            narration: Text to speak
            emotion: Emotional tone (for compatible TTS services)

        Returns:
            TTSResult with audio_path and duration_sec
        """
        print(f"  [TTS Agent] Generating narration for scene {scene_id}...")
        print(f"     Text: {narration[:60]}...")

        # Build output path
        output_dir = "media/audio"
        os.makedirs(output_dir, exist_ok=True)
        output_path = f"{output_dir}/narration_{scene_id:02d}.mp3"

        audio_path = None

        # 1. Try ElevenLabs
        if self.elevenlabs_key:
            try:
                audio_path = self._call_elevenlabs_api(narration, self.voice, output_path)
            except Exception as e:
                print(f"     [Warning] ElevenLabs TTS failed: {e}")
                audio_path = None

        # 2. Fallback: silent placeholder
        if audio_path is None:
            print(f"     [Fallback] Using silent placeholder audio")
            audio_path = self._generate_placeholder_audio(scene_id, narration, output_path)

        # 측정: 실제 오디오 길이 (FFprobe 사용, 실패 시 텍스트 기반 추정)
        duration_sec = self._get_audio_duration(audio_path, narration_text=narration)

        print(f"     [TTS Agent] Audio saved: {audio_path} (duration: {duration_sec:.2f}s)")
        return TTSResult(audio_path=audio_path, duration_sec=duration_sec)

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
            print(f"     [Warning] FFprobe duration check failed: {e}")

        # Fallback: 파일 크기 기반 추정 (MP3 128kbps = 16KB/sec)
        try:
            file_size = os.path.getsize(audio_path)
            if file_size > 0:
                estimated = file_size / 16000.0  # 128kbps MP3 ≈ 16KB/sec
                print(f"     [Duration] Estimated from file size: {estimated:.2f}s ({file_size} bytes)")
                return max(1.0, estimated)
        except Exception:
            pass

        # Fallback: 텍스트 기반 추정 (한국어: ~4음절/초)
        if narration_text:
            char_count = len(narration_text.replace(" ", ""))
            estimated = max(2.0, char_count / 4.0)
            print(f"     [Duration] Estimated from text: {estimated:.2f}s ({char_count} chars)")
            return estimated

        return 5.0  # 최종 기본값

    def _call_elevenlabs_api(self, text: str, voice_id: str, output_path: str) -> str:
        """
        Call ElevenLabs API for high-quality TTS (v2.x API).

        Args:
            text: Text to convert to speech
            voice_id: ElevenLabs voice ID
            output_path: Where to save the audio

        Returns:
            Path to generated audio file
        """
        try:
            from elevenlabs.client import ElevenLabs

            print(f"     Using ElevenLabs API (Voice: {voice_id[:8]}...)...")

            # Initialize client
            client = ElevenLabs(api_key=self.elevenlabs_key)

            # Generate audio using text_to_speech.convert()
            audio_generator = client.text_to_speech.convert(
                text=text,
                voice_id=voice_id,
                model_id="eleven_multilingual_v2"  # 다국어 지원 모델
            )

            # Save to file
            with open(output_path, "wb") as f:
                for chunk in audio_generator:
                    f.write(chunk)

            print(f"     ElevenLabs TTS generated: {output_path}")
            return output_path

        except ImportError as e:
            print(f"     [Warning] ElevenLabs library import failed: {e}")
            print("     Try: pip install --upgrade elevenlabs")
            raise RuntimeError("ElevenLabs library not properly installed")
        except Exception as e:
            print(f"     [Error] ElevenLabs API call failed: {e}")
            raise

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

        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            raise RuntimeError(f"Failed to generate placeholder audio: {result.stderr}")

        print(f"     Placeholder audio generated: {output_path}")
        return output_path
