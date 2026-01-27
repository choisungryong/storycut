"""
Music Agent: Selects or generates background music.
"""

import os
import shutil
from typing import Optional


class MusicAgent:
    """
    Handles background music selection or generation.

    For MVP, this primarily focuses on music selection.
    Future versions may integrate AI music generation APIs.
    """

    def __init__(self, music_library_path: str = "media/music"):
        """
        Initialize Music Agent.

        Args:
            music_library_path: Path to music library directory
        """
        self.music_library_path = music_library_path
        os.makedirs(music_library_path, exist_ok=True)

    def select_music(
        self,
        genre: str,
        mood: str,
        duration_sec: int
    ) -> Optional[str]:
        """
        Select appropriate background music for the story.

        Args:
            genre: Story genre
            mood: Overall mood
            duration_sec: Required music duration

        Returns:
            Path to selected music file, or None if no music
        """
        print(f"[Music Agent] Selecting background music...")
        print(f"   Genre: {genre}, Mood: {mood}, Duration: {duration_sec}s")

        # For MVP, return None or a placeholder
        # In production, this would:
        # 1. Query a music library database
        # 2. Call a music generation API
        # 3. Select from pre-approved royalty-free music

        music_path = self._get_placeholder_music(duration_sec)

        if music_path:
            print(f"   [Music Agent] Music selected: {music_path}")
        else:
            print(f"   [Info] No background music (silent mode)")

        return music_path

    def _get_placeholder_music(self, duration_sec: int) -> Optional[str]:
        """
        Generate or return placeholder background music.

        For MVP testing, creates a simple tone or returns None.

        Args:
            duration_sec: Music duration

        Returns:
            Path to placeholder music file, or None
        """
        # For MVP, we can skip music entirely or generate a simple ambient tone
        # Let's generate a very quiet ambient tone as placeholder

        import subprocess

        output_path = f"{self.music_library_path}/placeholder_music.mp3"

        # Only generate once
        if os.path.exists(output_path):
            return output_path

        # Create a very quiet ambient tone (200Hz + 300Hz)
        try:
            cmd = [
                "ffmpeg",
                "-y",
                "-f", "lavfi",
                "-i", f"sine=frequency=200:duration={duration_sec}",
                "-i", f"sine=frequency=300:duration={duration_sec}",
                "-filter_complex", "amix=inputs=2:duration=first:dropout_transition=2,volume=0.1",
                "-c:a", "libmp3lame",
                "-b:a", "128k",
                output_path
            ]

            result = subprocess.run(cmd, capture_output=True, text=True)

            if result.returncode == 0:
                return output_path
            else:
                print(f"   [Warning] Could not generate music: {result.stderr}")
                return None

        except Exception as e:
            print(f"   [Warning] Music generation failed: {e}")
            return None
