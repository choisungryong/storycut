
import os
import sys
from pathlib import Path
from dotenv import load_dotenv

env_path = Path(__file__).parent / ".env"
load_dotenv(env_path)

sys.path.append(str(Path(__file__).parent))

from agents.story_agent import StoryAgent

def debug_story_generation():
    print(f"[DEBUG] OPENAI_API_KEY: {os.getenv('OPENAI_API_KEY')[:10]}...")
    
    # Intentionally use 'gpt-5.2' as in current code
    agent = StoryAgent(
        api_key=os.getenv("OPENAI_API_KEY"),
        model="gpt-4o-mini"  # Test with a VALID model first to see if that works?
                             # No, use the one from code to reproduce failure.
        # But wait, code hardcodes "gpt-5.2".
        # Let's instantiate without arg to use default.
    )
    print(f"[DEBUG] Agent initialized with model: {agent.model}")

    try:
        story = agent.generate_story(
            genre="mystery",
            mood="suspense",
            style="cinematic",
            total_duration_sec=60,
            user_idea="Test story for debugging"
        )
        print("[SUCCESS] Story generated:")
        print(story.get("title"))
        if story.get("title") == "테스트 스토리":
            print("[FAILURE] Returned fallback Test Story!")
        else:
            print("[SUCCESS] Returned real story!")

    except Exception as e:
        print(f"[ERROR] Exception caught: {e}")

if __name__ == "__main__":
    debug_story_generation()
