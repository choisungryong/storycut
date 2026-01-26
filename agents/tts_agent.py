"""
TTS Agent: Generates narration audio for each scene.
"""

import os
from typing import Optional


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
        self.voice = voice

        if not self.api_key:
            print("⚠️  No TTS API key provided. Will use placeholder audio.")

    def generate_speech(
        self,
        scene_id: int,
        narration: str,
        emotion: str = "neutral"
    ) -> str:
        """
        Generate narration audio for a scene.

        Args:
            scene_id: Scene identifier
            narration: Text to speak
            emotion: Emotional tone (for compatible TTS services)

        Returns:
            Path to generated audio file
        """
        print(f"  🎙️  Generating narration for scene {scene_id}...")
        print(f"     Text: {narration[:60]}...")

        # Build output path
        output_dir = "media/audio"
        os.makedirs(output_dir, exist_ok=True)
        output_path = f"{output_dir}/narration_{scene_id:02d}.mp3"

        # Call external TTS API
        if self.api_key:
            audio_path = self._call_tts_api(narration, output_path)
        else:
            # Generate placeholder audio for testing
            audio_path = self._generate_placeholder_audio(scene_id, narration, output_path)

        print(f"     ✅ Audio saved: {audio_path}")
        return audio_path

    def _call_tts_api(self, text: str, output_path: str) -> str:
        """
        Call external TTS API.

        This is a placeholder for the actual API implementation.
        In production, this would call OpenAI TTS, ElevenLabs, etc.

        Args:
            text: Text to convert to speech
            output_path: Where to save the audio

        Returns:
            Path to generated audio file
        """
        try:
            import openai
            openai.api_key = self.api_key

            # Use OpenAI TTS
            response = openai.audio.speech.create(
                model="tts-1",
                voice=self.voice,
                input=text
            )

            # Save audio to file
            response.stream_to_file(output_path)

            return output_path

        except ImportError:
            print("     ⚠️  OpenAI library not available. Using placeholder audio.")
            return self._generate_placeholder_audio(1, text, output_path)
        except Exception as e:
            print(f"     ⚠️  TTS API call failed: {e}. Using placeholder audio.")
            return self._generate_placeholder_audio(1, text, output_path)

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

        return output_path
