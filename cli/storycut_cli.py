"""
STORYCUT CLI - Command-line interface for story video generation.

업데이트된 기능:
- Feature Flags 지원
- Hook Scene Video 옵션
- Ken Burns / Audio Ducking / Subtitle 옵션
- Context Carry-over
- Optimization Package (제목/썸네일/AB테스트)
"""

import os
import sys
import json
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from schemas import FeatureFlags, ProjectRequest, TargetPlatform
from pipeline import StorycutPipeline


def print_banner():
    """Print STORYCUT banner."""
    banner = """
=====================================================================
   ███████╗████████╗ ██████╗ ██████╗ ██╗   ██╗ ██████╗██╗   ██╗████████╗
   ██╔════╝╚══██╔══╝██╔═══██╗██╔══██╗╚██╗ ██╔╝██╔════╝██║   ██║╚══██╔══╝
   ███████╗   ██║   ██║   ██║██████╔╝ ╚████╔╝ ██║     ██║   ██║   ██║
   ╚════██║   ██║   ██║   ██║██╔══██╗  ╚██╔╝  ██║     ██║   ██║   ██║
   ███████║   ██║   ╚██████╔╝██║  ██║   ██║   ╚██████╗╚██████╔╝   ██║
   ╚══════╝   ╚═╝    ╚═════╝ ╚═╝  ╚═╝   ╚═╝    ╚═════╝ ╚═════╝    ╚═╝

              AI-Powered Story Video Generator (v2.0)
              Optimized for YouTube / YouTube Shorts
=====================================================================
"""
    print(banner)


def load_env():
    """Load environment variables from .env file."""
    try:
        from dotenv import load_dotenv
        load_dotenv()
        print("[OK] Environment variables loaded")
    except ImportError:
        print("[INFO] python-dotenv not installed. Using system environment variables.")


def get_user_input():
    """
    Get story parameters from user via CLI.

    Returns:
        ProjectRequest object
    """
    print("\nLet's create your story video!\n")

    # Topic
    print("Step 1/7: Topic")
    topic = input("Enter topic/theme (or press Enter for AI to decide): ").strip() or None

    # Genre
    print("\nStep 2/7: Genre")
    print("Options: emotional, mystery, thriller, fantasy, horror, comedy")
    genre = input("Enter genre (default: emotional): ").strip() or "emotional"

    # Mood
    print("\nStep 3/7: Mood")
    print("Options: melancholic, suspenseful, heartwarming, dark, cheerful, dramatic")
    mood = input("Enter mood (default: dramatic): ").strip() or "dramatic"

    # Style
    print("\nStep 4/7: Visual Style")
    print("Options: cinematic, realistic, illustration, anime, webtoon")
    style = input("Enter style (default: cinematic, high contrast): ").strip() or "cinematic, high contrast"

    # Duration
    print("\nStep 5/7: Duration")
    print("Recommended: 60-150 seconds for YouTube, 15-60 for Shorts")
    duration_input = input("Enter duration in seconds (default: 60): ").strip()
    try:
        duration = int(duration_input) if duration_input else 60
        if duration < 15 or duration > 300:
            print("[WARNING] Duration out of range. Using 60 seconds.")
            duration = 60
    except ValueError:
        print("[WARNING] Invalid duration. Using 60 seconds.")
        duration = 60

    # Platform
    print("\nStep 6/7: Target Platform")
    print("Options: 1) YouTube Long  2) YouTube Shorts")
    platform_input = input("Enter choice (default: 1): ").strip() or "1"
    target_platform = (
        TargetPlatform.YOUTUBE_SHORTS
        if platform_input == "2"
        else TargetPlatform.YOUTUBE_LONG
    )

    # Feature Flags
    print("\nStep 7/7: Feature Flags")
    print("Configure advanced features (press Enter for defaults):")

    # Hook Scene Video
    hook_input = input("  Enable high-quality Hook video for Scene 1? (y/N): ").strip().lower()
    hook_scene1_video = hook_input == "y"

    # Ken Burns
    kenburns_input = input("  Enable Ken Burns effect for images? (Y/n): ").strip().lower()
    ffmpeg_kenburns = kenburns_input != "n"

    # Audio Ducking
    ducking_input = input("  Enable Audio Ducking (BGM auto-reduce)? (y/N): ").strip().lower()
    ffmpeg_audio_ducking = ducking_input == "y"

    # Subtitles
    subtitle_input = input("  Enable Subtitle burn-in? (Y/n): ").strip().lower()
    subtitle_burn_in = subtitle_input != "n"

    # Context Carry-over
    context_input = input("  Enable Context carry-over between scenes? (Y/n): ").strip().lower()
    context_carry_over = context_input != "n"

    # Optimization Pack
    opt_input = input("  Generate YouTube optimization package? (Y/n): ").strip().lower()
    optimization_pack = opt_input != "n"

    # Create FeatureFlags
    feature_flags = FeatureFlags(
        hook_scene1_video=hook_scene1_video,
        ffmpeg_kenburns=ffmpeg_kenburns,
        ffmpeg_audio_ducking=ffmpeg_audio_ducking,
        subtitle_burn_in=subtitle_burn_in,
        context_carry_over=context_carry_over,
        optimization_pack=optimization_pack,
    )

    # Create ProjectRequest
    return ProjectRequest(
        topic=topic,
        genre=genre,
        mood=mood,
        style_preset=style,
        duration_target_sec=duration,
        target_platform=target_platform,
        voice_over=True,
        bgm=True,
        subtitles=subtitle_burn_in,
        feature_flags=feature_flags,
    )


def print_config(request: ProjectRequest):
    """Print configuration summary."""
    print("\n" + "="*60)
    print("Configuration Summary")
    print("="*60)
    print(f"  Topic: {request.topic or 'AI-generated'}")
    print(f"  Genre: {request.genre}")
    print(f"  Mood: {request.mood}")
    print(f"  Style: {request.style_preset}")
    print(f"  Duration: {request.duration_target_sec} seconds")
    print(f"  Platform: {request.target_platform.value}")
    print()
    print("Feature Flags:")
    print(f"  - Hook Scene Video: {'ON' if request.feature_flags.hook_scene1_video else 'OFF'}")
    print(f"  - Ken Burns Effect: {'ON' if request.feature_flags.ffmpeg_kenburns else 'OFF'}")
    print(f"  - Audio Ducking: {'ON' if request.feature_flags.ffmpeg_audio_ducking else 'OFF'}")
    print(f"  - Subtitle Burn-in: {'ON' if request.feature_flags.subtitle_burn_in else 'OFF'}")
    print(f"  - Context Carry-over: {'ON' if request.feature_flags.context_carry_over else 'OFF'}")
    print(f"  - Optimization Pack: {'ON' if request.feature_flags.optimization_pack else 'OFF'}")
    print("="*60 + "\n")


def main():
    """Main CLI entry point."""
    print_banner()
    load_env()

    # Check for API keys
    if not os.getenv("OPENAI_API_KEY"):
        print("\n[WARNING] OPENAI_API_KEY not found in environment.")
        print("          Story generation and TTS will use placeholders.")
        print("          Set your API key in .env file or environment variables.\n")

    if not os.getenv("RUNWAY_API_KEY"):
        print("[INFO] RUNWAY_API_KEY not set. Hook video will use image+KenBurns.\n")

    # Get user input
    request = get_user_input()

    # Print configuration
    print_config(request)

    confirm = input("Proceed with generation? (y/n): ").strip().lower()
    if confirm != 'y':
        print("[CANCELLED] Generation cancelled.")
        return

    try:
        # Run pipeline
        pipeline = StorycutPipeline()
        manifest = pipeline.run(request)

        # Success summary
        print("\n" + "="*60)
        print("ALL DONE! Your video is ready.")
        print("="*60)
        print(f"Project ID: {manifest.project_id}")
        print(f"Final Video: {manifest.outputs.final_video_path}")
        print(f"Title: {manifest.title}")
        print(f"Scenes: {len(manifest.scenes)}")
        print(f"Execution Time: {manifest.execution_time_sec:.2f}s")
        print(f"Estimated Cost: ${manifest.cost_estimate.estimated_usd:.2f}")

        if manifest.outputs.title_candidates:
            print("\nTitle Candidates:")
            for i, title in enumerate(manifest.outputs.title_candidates, 1):
                print(f"  {i}. {title}")

        if manifest.outputs.thumbnail_texts:
            print("\nThumbnail Texts:")
            for i, text in enumerate(manifest.outputs.thumbnail_texts, 1):
                print(f"  {i}. {text}")

        if manifest.outputs.hashtags:
            print(f"\nHashtags: {' '.join(manifest.outputs.hashtags[:5])}...")

        print("\nNext steps:")
        print("  1. Review the video in outputs/ folder")
        print("  2. Check manifest.json for all metadata")
        print("  3. Use title candidates and hashtags for YouTube")
        print("  4. Generate thumbnail using thumbnail_prompts")
        print("\nThanks for using STORYCUT!\n")

    except KeyboardInterrupt:
        print("\n\n[INTERRUPTED] Generation interrupted by user.")
        sys.exit(1)

    except Exception as e:
        print(f"\n\n[ERROR] {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


def quick_run(topic: str = None, **kwargs):
    """
    Quick run function for programmatic use.

    Args:
        topic: Video topic
        **kwargs: Additional parameters (genre, mood, style, duration, etc.)

    Returns:
        Manifest object
    """
    load_env()

    # Default feature flags
    feature_flags = FeatureFlags(
        hook_scene1_video=kwargs.get("hook_scene1_video", False),
        ffmpeg_kenburns=kwargs.get("ffmpeg_kenburns", True),
        ffmpeg_audio_ducking=kwargs.get("ffmpeg_audio_ducking", False),
        subtitle_burn_in=kwargs.get("subtitle_burn_in", True),
        context_carry_over=kwargs.get("context_carry_over", True),
        optimization_pack=kwargs.get("optimization_pack", True),
    )

    request = ProjectRequest(
        topic=topic,
        genre=kwargs.get("genre", "emotional"),
        mood=kwargs.get("mood", "dramatic"),
        style_preset=kwargs.get("style", "cinematic, high contrast"),
        duration_target_sec=kwargs.get("duration", 60),
        target_platform=kwargs.get("platform", TargetPlatform.YOUTUBE_LONG),
        voice_over=True,
        bgm=True,
        subtitles=True,
        feature_flags=feature_flags,
    )

    pipeline = StorycutPipeline()
    return pipeline.run(request)


if __name__ == "__main__":
    main()
