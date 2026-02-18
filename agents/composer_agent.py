"""
Composer Agent: Composes final video using FFmpeg.
"""

import os
from typing import List
from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from utils.ffmpeg_utils import FFmpegComposer


class ComposerAgent:
    """
    Composes all scene elements into a final YouTube-ready video.

    Uses FFmpeg for all video/audio composition operations.
    """

    def __init__(self, resolution: str = "1920x1080", fps: int = 30):
        """
        Initialize Composer Agent.

        Args:
            resolution: Video resolution (default: 1920x1080)
            fps: Frames per second (default: 30)
        """
        self.composer = FFmpegComposer(resolution=resolution, fps=fps)

    def compose_video(
        self,
        video_clips: List[str],
        narration_clips: List[str],
        music_path: str = None,
        output_path: str = "output/youtube_ready.mp4",
        use_ducking: bool = False,
        scene_durations: List[float] = None
    ) -> str:
        """
        Compose final video from all scene elements.

        Args:
            video_clips: List of video file paths (in scene order)
            narration_clips: List of narration audio paths (in scene order)
            music_path: Background music file path (optional)
            output_path: Final output video path
            use_ducking: 오디오 덕킹 사용 여부
            scene_durations: 씬별 비디오 duration (초). TTS 오디오 패딩에 사용.

        Returns:
            Path to final composed video
        """
        print("\n[Composer] Starting final video composition...")
        if use_ducking:
            print("   [Composer] Audio ducking: ENABLED")

        # Validate inputs
        if len(video_clips) != len(narration_clips):
            raise ValueError(
                f"Mismatch: {len(video_clips)} video clips but {len(narration_clips)} narration clips"
            )

        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # Compose using FFmpeg
        final_video = self.composer.compose_final_video(
            video_clips=video_clips,
            narration_clips=narration_clips,
            music_path=music_path,
            output_path=output_path,
            use_ducking=use_ducking,
            scene_durations=scene_durations
        )

        # Get final video info
        duration = self.composer.get_video_duration(final_video)

        print(f"\n[Composer] Video composition complete!")
        print(f"   Output: {final_video}")
        print(f"   Duration: {duration:.2f} seconds")

        return final_video
