
import os
import sys
from dotenv import load_dotenv

# Add project root to sys.path
sys.path.append(os.getcwd())
load_dotenv()

from agents.tts_agent import TTSAgent

def test_mapping():
    print("Testing Voice Mapping...")
    
    # 1. Initialize Agent with API Key from .env
    agent = TTSAgent()
    
    # 2. Mock the voice map logic (simulating api_server.py behavior)
    # Note: TTSAgent doesn't have the full map, api_server.py does. 
    # But we can test if TTSAgent accepts the ID we mapped to.
    
    # Test Fallback Mappings (Korean Name -> Premade ID)
    # Hyunbin -> Brian (nPcz...), Anna Kim -> Sarah (EXAV...)
    voices_to_test = {
        "Hyunbin (via Fallback)": "nPczCjzI2devNBz1zQrb",
        "Anna Kim (via Fallback)": "EXAVITQu4vr4xnSDxMaL"
    }
    
    for name, voice_id in voices_to_test.items():
        print(f"\nTesting '{name}' -> ID: {voice_id}")
        
        try:
            output_path = f"media/audio/test_{name.replace(' ', '_').lower()}.mp3"
            os.makedirs("media/audio", exist_ok=True)
            
            result = agent._call_elevenlabs_api(
                f"This is a test of the {name} voice.",
                voice_id,
                output_path
            )
            
            if os.path.exists(result):
                print(f"✅ Success! Audio generated at: {result}")
            else:
                print("❌ File not found.")
                
        except Exception as e:
            print(f"❌ Failed: {e}")

if __name__ == "__main__":
    test_mapping()
