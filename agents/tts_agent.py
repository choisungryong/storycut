"""
TTS Agent: Generates narration audio for each scene.
"""

import os
import subprocess
from typing import Optional
from dataclasses import dataclass


@dataclass
class TTSResult:
    """TTS 생성 결과"""
    audio_path: str
    duration_sec: float  # 실제 오디오 길이


class TTSAgent:
    """
    Generates narration audio using Text-to-Speech.

    This is an API adapter that calls external TTS services
    (OpenAI TTS, ElevenLabs, Google Cloud TTS, etc.).
    """

    def __init__(self, api_key: str = None, voice: str = "alloy"):
        """
        Initialize TTS Agent.

        Args:
            api_key: OpenAI API key (or other TTS service)
            voice: Voice ID to use
        """
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.elevenlabs_key = os.getenv("ELEVENLABS_API_KEY")
        self.voice = voice
        
        # Voice ID Mapping (ElevenLabs)
        # 기본: Adam (pre-made), 분위기별 매핑 가능
        self.voice_map = {
            "neutral": "pNInz6obpgDQGcFmaJgB",  # Adam
            "dramatic": "pNInz6obpgDQGcFmaJgB", # Adam (Bold)
            "cheerful": "FGY2WhTYq4u0I1O31p32", # Jess (Energetic)
            "horror": "ErXwobaYiN019PkySvjV",   # Antoni (Deep)
            "mystery": "ErXwobaYiN019PkySvjV",  # Antoni
        }

        # TTS Priority 로그
        if self.elevenlabs_key:
            print("[TTS Agent] Primary: ElevenLabs (High Quality)")
        elif self.api_key:
            print("[TTS Agent] Primary: OpenAI TTS (Good Quality)")
        else:
            print("[TTS Agent] Primary: pyttsx3 (Local, Free)")
            print("[Warning] No TTS API key provided. Using local TTS or placeholder audio.")

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

        # 1. Try ElevenLabs (PRIMARY - High Quality)
        if self.elevenlabs_key:
            try:
                # v2.1 FIX: Use the user-selected voice (self.voice) if available.
                # Only fallback to voice_map (emotion-based) if self.voice is not set or default.
                if self.voice and self.voice != "alloy" and self.voice != "onyx":
                     voice_id = self.voice
                     print(f"     [TTS] Using specific voice ID: {voice_id}")
                else:
                     # Legacy fallback
                     voice_id = self.voice_map.get(emotion, self.voice_map["neutral"])
                     print(f"     [TTS] Using emotion-based fallback voice: {voice_id} ({emotion})")
                
                audio_path = self._call_elevenlabs_api(narration, voice_id, output_path)
            except Exception as e:
                print(f"     [Warning] ElevenLabs TTS failed: {e}")
                audio_path = None

        # 2. Try OpenAI TTS (SECONDARY - Good Quality)
        if audio_path is None and self.api_key:
            try:
                audio_path = self._call_tts_api(narration, output_path)
            except Exception as e:
                print(f"     [Warning] OpenAI TTS failed: {e}")
                audio_path = None

        # 3. Try Local pyttsx3 (TERTIARY - Free, Offline)
        if audio_path is None:
            try:
                audio_path = self._call_pyttsx3_local(scene_id, narration, output_path)
            except Exception as e:
                print(f"     [Warning] pyttsx3 failed: {e}")
                audio_path = None

        # 4. Fallback: Placeholder
        if audio_path is None:
            print(f"     [Fallback] Using silent placeholder audio")
            audio_path = self._generate_placeholder_audio(scene_id, narration, output_path)

        # 측정: 실제 오디오 길이 (FFprobe 사용)
        duration_sec = self._get_audio_duration(audio_path)

        print(f"     [TTS Agent] Audio saved: {audio_path} (duration: {duration_sec:.2f}s)")
        return TTSResult(audio_path=audio_path, duration_sec=duration_sec)

    def _get_audio_duration(self, audio_path: str) -> float:
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
                return float(result.stdout.strip())
        except Exception as e:
            print(f"     [Warning] FFprobe duration check failed: {e}")

        # Fallback: 텍스트 기반 추정 (150 wpm = 2.5 words/sec)
        return 5.0  # 기본값

    def _call_tts_api(self, text: str, output_path: str) -> str:
        """
        Call OpenAI TTS API.

        Args:
            text: Text to convert to speech
            output_path: Where to save the audio

        Returns:
            Path to generated audio file
        """
        try:
            from openai import OpenAI

            print(f"     Using OpenAI TTS API...")

            # Initialize OpenAI client
            client = OpenAI(api_key=self.api_key)

            # Generate speech
            response = client.audio.speech.create(
                model="tts-1",
                voice=self.voice,
                input=text
            )

            # Save audio to file
            response.stream_to_file(output_path)

            print(f"     OpenAI TTS generated: {output_path}")
            return output_path

        except ImportError:
            print("     [Warning] OpenAI library not installed. Try: pip install openai")
            raise RuntimeError("OpenAI library not installed")
        except Exception as e:
            print(f"     [Error] OpenAI TTS call failed: {e}")
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

    def _call_pyttsx3_local(self, scene_id: int, text: str, output_path: str) -> str:
        """
        Use local pyttsx3 library for TTS (FREE, NO API KEY).
        Works on Windows, Mac, Linux.
        """
        try:
            import pyttsx3

            print(f"     Using pyttsx3 (Local TTS)...")

            # Initialize engine
            engine = pyttsx3.init()

            # Settings
            engine.setProperty('rate', 150)  # Speed (150 words/min, good for Korean)
            engine.setProperty('volume', 0.9)

            # Save to file
            engine.save_to_file(text, output_path)
            engine.runAndWait()
            engine.stop()

            print(f"     pyttsx3 generated: {output_path}")
            return output_path

        except ImportError:
            print("     [Info] pyttsx3 not installed. Try: pip install pyttsx3")
            raise RuntimeError("pyttsx3 library not installed")
        except Exception as e:
            print(f"     pyttsx3 error: {e}")
            raise

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
