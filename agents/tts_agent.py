"""
TTS Agent: Generates narration audio for each scene.
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
        self.voli_key = os.getenv("VOLI_API_KEY")
        self.voice = voice
        
        # VOLI Voice ID Mapping (Verified IDs from User)
        # 0_M-ya_3_high는 존재하지 않아 우선 0_F-ya_3_high로 통일하여 에러 방지
        self.voli_voice_map = {
            "neutral": "0_F-ya_3_high", 
            "female": "0_F-ya_3_high",
            "male": "0_F-ya_3_high",
            "voice_brian": "0_F-ya_3_high", 
            "voice_sarah": "0_F-ya_3_high", 
            "voice_laura": "0_F-ya_3_high", 
        }

        # TTS Priority 로그
        if self.voli_key:
            print("[TTS Agent] Primary: VOLI TTS (Wavedeck)")
        elif self.elevenlabs_key:
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

        # 1. Try VOLI TTS (PRIMARY)
        if self.voli_key:
            try:
                # voice_id mapping
                voice_id = self.voice
                # 기본값이거나 일레븐랩스용 ID인 경우 VOLI용으로 전환
                if not voice_id or voice_id in ["alloy", "onyx", "neutral"] or len(voice_id) > 20: 
                    voice_id = self.voli_voice_map["neutral"]
                
                # emotion mapping
                emotion_id_map = {
                    "neutral": 0, "dramatic": 5, "cheerful": 2, 
                    "horror": 4, "mystery": 4, "sad": 3, "angry": 1
                }
                emotion_id = emotion_id_map.get(emotion, 0)
                
                audio_path = self._call_voli_api(narration, voice_id, emotion_id, output_path)
            except Exception as e:
                print(f"     [Warning] VOLI TTS failed: {e}")
                audio_path = None

        # 2. Try ElevenLabs (SECONDARY - High Quality)
        if audio_path is None and self.elevenlabs_key:
            try:
                # v2.1 FIX: Use the user-selected voice (self.voice) if available.
                if self.voice and self.voice not in ["alloy", "onyx", "neutral"] and "_" not in self.voice:
                     voice_id = self.voice
                else:
                     voice_id = "pNInz6obpgDQGcFmaJgB" # Adam
                
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

    def _call_voli_api(self, text: str, voice_id: str, emotion_id: int, output_path: str) -> str:
        """
        Call VOLI (Wavedeck) TTS API.
        """
        try:
            import requests
            import json

            url = "https://biz-api.voli.ai/v1/conversions/tts"
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.voli_key}"
            }
            
            # script 내 줄바꿈 개수에 맞춰 intonation 배열 생성
            lines = text.split('\n')
            intonation = [1.0] * len(lines)

            payload = {
                "voiceType": "default",
                "voiceId": voice_id,
                "emotionId": emotion_id,
                "script": text,
                "pauseDuration": 0.625,
                "pitch": 0.0,
                "speed": 1.0,
                "needTrim": False,
                "intonation": intonation,
                "language": "kr"
            }

            print(f"     Calling VOLI TTS API (Voice: {voice_id}, Emotion: {emotion_id})...")
            response = requests.post(url, headers=headers, json=payload, timeout=60)

            if response.status_code == 200:
                content_type = response.headers.get("Content-Type", "").lower()
                if "audio" in content_type or "octet-stream" in content_type:
                    with open(output_path, "wb") as f:
                        f.write(response.content)
                    return output_path
                else:
                    # JSON 응답일 경우 (다운로드 URL 포함)
                    res_json = response.json()
                    # 사용자 제공 실제 구조: {"success": true, "data": {"generated_voice": "https://..."}}
                    audio_url = res_json.get("data", {}).get("generated_voice") or res_json.get("data", {}).get("audioUrl") or res_json.get("audioUrl")
                    
                    if audio_url:
                        print(f"     Downloading audio from: {audio_url[:60]}...")
                        audio_resp = requests.get(audio_url, timeout=30)
                        if audio_resp.status_code == 200:
                            with open(output_path, "wb") as f:
                                f.write(audio_resp.content)
                            return output_path
                        else:
                            raise RuntimeError(f"Failed to download audio from VOLI URL: {audio_resp.status_code}")
                    else:
                        raise RuntimeError(f"Unexpected JSON response from VOLI (missing generated_voice/audioUrl): {res_json}")
            else:
                raise RuntimeError(f"VOLI API error: {response.status_code} - {response.text}")

        except Exception as e:
            print(f"     [Error] VOLI TTS call failed: {e}")
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
