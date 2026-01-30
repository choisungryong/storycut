
import os
import requests
import json
from dotenv import load_dotenv

load_dotenv()

def get_voices():
    api_key = os.getenv("ELEVENLABS_API_KEY")
    if not api_key:
        print("Error: ELEVENLABS_API_KEY not found in .env")
        return

    url = "https://api.elevenlabs.io/v1/voices"
    
    headers = {
        "Accept": "application/json",
        "xi-api-key": api_key
    }

    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        
        data = response.json()
        voices = data.get("voices", [])
        
        print(f"Found {len(voices)} voices:\n")
        print(f"{'Name':<25} | {'Voice ID':<25} | {'Category':<15}")
        print("-" * 70)
        
        # Sort so 'generated' or 'cloned' come first if any
        voices.sort(key=lambda x: x.get("category", "z"), reverse=False)

        for voice in voices:
            name = voice.get("name", "Unknown")
            voice_id = voice.get("voice_id", "Unknown")
            category = voice.get("category", "Unknown")
            print(f"{name:<25} | {voice_id:<25} | {category:<15}")

    except Exception as e:
        print(f"Error fetching voices: {e}")

if __name__ == "__main__":
    get_voices()
