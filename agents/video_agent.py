"""
Video Agent: Generates video clips for each scene.

P0 핵심 로직:
- Scene 1 (Hook): Veo 3.1로 고품질 비디오 생성 (Google API 활용)
- Scene 2~N: 이미지 + Ken Burns 효과로 비용 절감
- 실패 시: 이미지 기반으로 자동 폴백

API 우선순위: Veo 3.1 > Runway > Stability > Ken Burns
"""

import os
import subprocess
import time
from typing import Dict, Any, Optional, Tuple
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))

from schemas import FeatureFlags, Scene


class VideoAgent:
    """
    비디오 생성 에이전트

    Scene 1 Hook 전용 고품질 비디오 생성 + 폴백 로직 구현
    """

    def __init__(
        self,
        api_key: str = None,
        service: str = "veo",
        feature_flags: FeatureFlags = None
    ):
        """
        Initialize Video Agent.

        Args:
            api_key: API key for video generation service
            service: Service to use ("veo", "runway", "stability")
            feature_flags: Feature flags configuration
        """
        self.google_api_key = os.getenv("GOOGLE_API_KEY")
        self.api_key = api_key or os.getenv("RUNWAY_API_KEY")
        self.service = service
        self.feature_flags = feature_flags or FeatureFlags()

        # Prioritize Veo 3.1 if Google API key available
        if self.google_api_key:
            self.service = "veo"
            print("  Video Agent: Using Veo 3.1 (Google) for high-quality video generation.")
        elif not self.api_key:
            print("  No video API key provided. Will use image + Ken Burns method.")

    def generate_video(
        self,
        scene_id: int,
        visual_description: str,
        style: str,
        mood: str,
        duration_sec: int,
        scene: Optional[Scene] = None,
        output_dir: str = "media/video"
    ) -> str:
        """
        Generate a video clip for a scene.

        P0 로직:
        - scene_id == 1 and hook_scene1_video == True: 고품질 비디오 생성
        - 그 외: 이미지 + Ken Burns 효과
        - 실패 시: 이미지 기반으로 폴백

        Args:
            scene_id: Scene identifier
            visual_description: Description for video generation
            style: Visual style
            mood: Emotional mood
            duration_sec: Video length in seconds
            scene: Scene object (optional, for enhanced context)
            output_dir: Output directory path

        Returns:
            Path to generated video file
        """
        print(f"  Generating video for scene {scene_id}...")
        print(f"     Description: {visual_description[:60]}...")

        # v2.0: Use image_prompt if available (character reference)
        actual_prompt = visual_description
        if scene and scene.image_prompt:
            actual_prompt = scene.image_prompt
            print(f"     [v2.0] Using image_prompt for character consistency")

        # Output paths
        os.makedirs(output_dir, exist_ok=True)
        video_output_path = f"{output_dir}/scene_{scene_id:02d}.mp4"

        # Determine generation method based on feature flags
        use_high_quality_video = (
            self.feature_flags.hook_scene1_video and
            scene_id == 1 and
            self.api_key
        )

        generation_method = None

        if use_high_quality_video:
            # P0: Scene 1 Hook - 고품질 비디오 생성 시도
            print(f"     [HOOK] Scene 1: Attempting high-quality video generation...")
            try:
                video_path = self._generate_high_quality_video(
                    prompt=actual_prompt,
                    style=style,
                    mood=mood,
                    duration_sec=min(duration_sec, 10),  # Max 10 seconds for cost control
                    output_path=video_output_path
                )
                generation_method = "high_quality_video"
                print(f"     Video saved: {video_path}")

                # Update scene metadata if provided
                if scene:
                    scene.generation_method = generation_method
                    scene.assets.video_path = video_path

                return video_path

            except Exception as e:
                print(f"     High-quality video generation failed: {e}")
                print(f"     Falling back to image + Ken Burns...")
                # Continue to fallback below

        # Default/Fallback: Image + Ken Burns
        video_path, generation_method = self._generate_with_kenburns(
            scene_id=scene_id,
            prompt=actual_prompt,
            style=style,
            duration_sec=duration_sec,
            output_path=video_output_path,
            scene=scene
        )

        # Update scene metadata if provided
        if scene:
            scene.generation_method = generation_method
            scene.assets.video_path = video_path

        print(f"     Video saved: {video_path} (method: {generation_method})")
        return video_path

    def _generate_high_quality_video(
        self,
        prompt: str,
        style: str,
        mood: str,
        duration_sec: int,
        output_path: str
    ) -> str:
        """
        Generate high-quality video using external API (Veo 3.1, Runway, etc.)

        Args:
            prompt: Video generation prompt
            style: Visual style
            mood: Emotional mood
            duration_sec: Duration in seconds
            output_path: Output file path

        Returns:
            Path to generated video
        """
        if self.service == "veo":
            return self._call_veo_api(prompt, style, mood, duration_sec, output_path)
        elif self.service == "runway":
            return self._call_runway_api(prompt, style, duration_sec, output_path)
        elif self.service == "stability":
            return self._call_stability_api(prompt, style, duration_sec, output_path)
        else:
            # Default to placeholder if service not implemented
            print(f"     Service '{self.service}' not implemented yet.")
            raise NotImplementedError(f"Video service '{self.service}' not implemented")

    def _call_veo_api(
        self,
        prompt: str,
        style: str,
        mood: str,
        duration_sec: int,
        output_path: str
    ) -> str:
        """
        Call Google Veo 3.1 API via official google-genai SDK.

        Args:
            prompt: Video generation prompt
            style: Visual style
            mood: Emotional mood
            duration_sec: Duration in seconds
            output_path: Output file path

        Returns:
            Path to generated video
        """
        try:
            from google import genai
            import time
            import requests

            print(f"     Calling Veo 3.1 API (veo-3.1-generate-preview)...")
            
            # Using the official SDK
            client = genai.Client(api_key=self.google_api_key)
            
            # Enhanced prompt
            full_prompt = f"Generate a video of: {style} style, {mood} mood: {prompt}. Cinematic, professional."

            print(f"     Sending request to Veo 3.1...")
            
            # Call generate_videos - Returns an Operation (LRO)
            operation = client.models.generate_videos(
                model="veo-3.1-generate-preview",
                prompt=full_prompt,
                config={
                    'aspect_ratio': '16:9',
                }
            )
            
            print(f"     Operation started: {operation.name}")
            print(f"     Waiting for video generation to complete (this may take a minute)...")
            
            # Poll for completion
            while True:
                # Refresh operation status
                # The name format might need handling if SDK expects short name, but typically full name works
                operation = client.operations.get(operation)
                
                if operation.done:
                    break
                    
                print(f"     ...still generating...")
                time.sleep(5)

            print(f"     Generation confirmed complete.")
            
            # Get result from completed operation
            response = operation.result
            
            # Check for generated_videos attribute (Veo specific)
            if hasattr(response, 'generated_videos') and response.generated_videos:
                video = response.generated_videos[0]
                if hasattr(video, 'video') and hasattr(video.video, 'uri'):
                     video_uri = video.video.uri
                elif hasattr(video, 'uri'):
                     video_uri = video.uri
            
            # Check for candidates architecture (Generic)
            if not video_uri and hasattr(response, 'candidates') and response.candidates:
                candidate = response.candidates[0]
                if hasattr(candidate, 'content') and candidate.content:
                   for part in candidate.content.parts:
                       if hasattr(part, 'video_metadata') and hasattr(part.video_metadata, 'video_uri'):
                           video_uri = part.video_metadata.video_uri
                       elif hasattr(part, 'uri'):
                           video_uri = part.uri
            
            if not video_uri:
                 print(f"     [DEBUG] Response dir: {dir(response)}") 
                 print(f"     [DEBUG] Response str: {str(response)[:500]}")
                 raise NotImplementedError("Could not extract video URI from Veo response")
                 
            print(f"     Downloading video from: {video_uri[:50]}...")
            
            # Append API key if using Google API URL
            download_url = video_uri
            if "generativelanguage.googleapis.com" in video_uri and "key=" not in video_uri:
                separator = "&" if "?" in video_uri else "?"
                download_url = f"{video_uri}{separator}key={self.google_api_key}"
            
            # Download the video
            video_resp = requests.get(download_url, timeout=300)
            if video_resp.status_code == 200:
                with open(output_path, "wb") as f:
                    f.write(video_resp.content)
                print(f"     Video saved: {output_path}")
            else:
                raise RuntimeError(f"Failed to download video from {video_uri}")

            return output_path

        except Exception as e:
            from utils.error_manager import ErrorManager
            ErrorManager.log_error(
                "VideoAgent", 
                "Veo 3.1 API Failed", 
                f"{type(e).__name__}: {str(e)}", 
                severity="error"
            )
            print(f"     Veo 3.1 API integration failed: {str(e)}")
            import traceback
            traceback.print_exc()
            # Raise exception to trigger fallback in generate_video
            raise

    def _call_runway_api(
        self,
        prompt: str,
        style: str,
        duration_sec: int,
        output_path: str
    ) -> str:
        """
        Call Runway ML Gen-2 API for video generation.

        TODO: Implement actual Runway API integration
        """
        # Runway API integration placeholder
        # In production, this would call Runway's Gen-2 API
        #
        # Example structure:
        # client = RunwayClient(api_key=self.api_key)
        # response = client.generate_video(
        #     prompt=f"{style} style: {prompt}",
        #     duration=duration_sec,
        #     aspect_ratio="16:9"
        # )
        # download_video(response.url, output_path)

        print(f"     Runway API integration pending. Using enhanced placeholder.")
        raise NotImplementedError("Runway API not yet integrated")

    def _call_stability_api(
        self,
        prompt: str,
        style: str,
        duration_sec: int,
        output_path: str
    ) -> str:
        """
        Call Stability AI Video API.

        TODO: Implement actual Stability API integration
        """
        print(f"     Stability API integration pending.")
        raise NotImplementedError("Stability API not yet integrated")

    def _generate_with_kenburns(
        self,
        scene_id: int,
        prompt: str,
        style: str,
        duration_sec: int,
        output_path: str,
        scene: Optional[Scene] = None
    ) -> Tuple[str, str]:
        """
        Generate video using image + Ken Burns effect.

        P0 핵심: 이미지 기반 씬은 Ken Burns로 영상처럼 보이게

        Args:
            scene_id: Scene identifier
            prompt: Image generation prompt
            style: Visual style
            duration_sec: Duration in seconds
            output_path: Output file path
            scene: Scene object (optional, for v2.0 character reference)

        Returns:
            Tuple of (video_path, generation_method)
        """
        from agents.image_agent import ImageAgent

        # Step 1: Generate image
        image_agent = ImageAgent()

        # v2.0: Extract character reference info from scene
        seed = None
        character_tokens = None
        character_reference_id = None
        character_reference_path = None
        if scene:
            # character_sheet에서 visual_seed 추출
            seed = getattr(scene, '_seed', None)
            character_tokens = scene.characters_in_scene if scene.characters_in_scene else None

            # v2.0: Extract master_image_path for character reference
            if character_tokens and hasattr(scene, '_character_sheet'):
                first_char_token = character_tokens[0]
                char_data = scene._character_sheet.get(first_char_token, {})
                character_reference_id = char_data.get('master_image_id')  # deprecated
                character_reference_path = char_data.get('master_image_path')
                if character_reference_path:
                    print(f"     [v2.0] Using character reference: {character_reference_path}")

            if seed:
                print(f"     [v2.0] Applying visual seed: {seed}")
            if character_tokens:
                print(f"     [v2.0] Characters: {', '.join(character_tokens)}")

        # Determine image output directory based on video output path
        # If output_path is 'outputs/xyz/media/video/scene.mp4', we want 'outputs/xyz/media/images'
        video_dir = os.path.dirname(output_path)
        # Attempt to keep directory structure clean
        if video_dir.endswith("video"):
             image_dir = video_dir.replace("video", "images") 
        else:
             image_dir = "media/images"

        image_path, image_id = image_agent.generate_image(
            scene_id=scene_id,
            prompt=prompt,
            style=style,
            output_dir=image_dir,
            seed=seed,
            character_tokens=character_tokens,
            character_reference_id=character_reference_id,
            character_reference_path=character_reference_path
        )

        # Step 2: Apply Ken Burns effect if enabled
        if self.feature_flags.ffmpeg_kenburns:
            video_path = self._apply_kenburns_effect(
                image_path=image_path,
                duration_sec=duration_sec,
                output_path=output_path,
                scene_id=scene_id
            )
            return video_path, "image+kenburns"
        else:
            # No Ken Burns - static image as video
            video_path = self._image_to_video(
                image_path=image_path,
                duration_sec=duration_sec,
                output_path=output_path
            )
            return video_path, "image+static"

    def _apply_kenburns_effect(
        self,
        image_path: str,
        duration_sec: int,
        output_path: str,
        scene_id: int = 1
    ) -> str:
        """
        Apply Ken Burns (zoom/pan) effect to an image.

        P0: 이미지 기반 씬에 모션을 추가하여 영상처럼 보이게

        Args:
            image_path: Input image path
            duration_sec: Duration in seconds
            output_path: Output video path
            scene_id: Scene ID for effect variation

        Returns:
            Path to generated video
        """
        from config import get_kenburns_config

        config = get_kenburns_config()
        fps = 30
        total_frames = duration_sec * fps

        # Vary effect based on scene_id for visual diversity
        effects = [
            # Zoom in
            f"zoompan=z='min(zoom+0.001,1.3)':x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':d={total_frames}:s=1920x1080:fps={fps}",
            # Zoom out
            f"zoompan=z='if(lte(zoom,1.0),1.3,max(1.001,zoom-0.001))':x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':d={total_frames}:s=1920x1080:fps={fps}",
            # Pan left to right
            f"zoompan=z='1.1':x='if(lte(on,1),0,min(iw/zoom-iw,x+1))':y='ih/2-(ih/zoom/2)':d={total_frames}:s=1920x1080:fps={fps}",
            # Pan right to left
            f"zoompan=z='1.1':x='if(lte(on,1),iw/zoom-iw,max(0,x-1))':y='ih/2-(ih/zoom/2)':d={total_frames}:s=1920x1080:fps={fps}",
            # Diagonal zoom
            f"zoompan=z='min(zoom+0.001,1.25)':x='if(lte(on,1),0,min(iw/zoom-iw,x+0.5))':y='if(lte(on,1),0,min(ih/zoom-ih,y+0.3))':d={total_frames}:s=1920x1080:fps={fps}",
        ]

        effect = effects[scene_id % len(effects)]

        cmd = [
            "ffmpeg",
            "-y",
            "-loop", "1",
            "-framerate", str(fps),
            "-i", image_path,
            "-vf", effect,
            "-c:v", "libx264",
            "-preset", "ultrafast",
            "-t", str(duration_sec),
            "-pix_fmt", "yuv420p",
            "-r", str(fps),
            "-movflags", "+faststart",
            output_path
        ]

        print(f"     Ken Burns FFmpeg command: {' '.join(cmd[:8])}... (truncated)")

        # Run with 90 second timeout (Ken Burns takes longer)
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=90)
        except subprocess.TimeoutExpired:
            print(f"     Ken Burns timed out, falling back to static image")
            return self._image_to_video(image_path, duration_sec, output_path)

        if result.returncode != 0:
            stderr_lines = result.stderr.split('\n')
            error_summary = '\n'.join(stderr_lines[-5:])
            print(f"     Ken Burns effect failed (last 5 lines):\n{error_summary}")
            # Fallback to static image video
            return self._image_to_video(image_path, duration_sec, output_path)

        return output_path

    def _image_to_video(
        self,
        image_path: str,
        duration_sec: int,
        output_path: str
    ) -> str:
        """
        Convert a static image to video without Ken Burns.

        Args:
            image_path: Input image path
            duration_sec: Duration in seconds
            output_path: Output video path

        Returns:
            Path to generated video
        """
        import os

        # 이미지 파일 존재 확인
        if not os.path.exists(image_path):
            print(f"     ERROR: Image file not found: {image_path}")
            raise RuntimeError(f"Image file not found: {image_path}")

        print(f"     Image file size: {os.path.getsize(image_path)} bytes")

        # 더 안정적인 FFmpeg 명령어 (Railway 환경 호환)
        cmd = [
            "ffmpeg",
            "-y",
            "-loop", "1",
            "-framerate", "30",
            "-i", image_path,
            "-c:v", "libx264",
            "-preset", "ultrafast",  # 빠른 인코딩
            "-tune", "stillimage",   # 정적 이미지 최적화
            "-t", str(duration_sec),
            "-pix_fmt", "yuv420p",
            "-vf", "scale=1920:1080:force_original_aspect_ratio=decrease,pad=1920:1080:(ow-iw)/2:(oh-ih)/2:color=black",
            "-r", "30",
            "-movflags", "+faststart",
            output_path
        ]

        print(f"     FFmpeg command: {' '.join(cmd)}")

        # Run with 60 second timeout
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        except subprocess.TimeoutExpired:
            print(f"     FFmpeg timed out, trying placeholder video")
            # Fallback to placeholder
            return self._generate_placeholder_video(1, duration_sec, output_path)

        if result.returncode != 0:
            # FFmpeg 에러의 마지막 10줄만 출력 (핵심 에러 메시지)
            stderr_lines = result.stderr.split('\n')
            error_summary = '\n'.join(stderr_lines[-10:])
            print(f"     FFmpeg error (last 10 lines):\n{error_summary}")

            # 최종 fallback: placeholder video
            print(f"     Falling back to placeholder video")
            return self._generate_placeholder_video(1, duration_sec, output_path)

        return output_path

    def _generate_placeholder_video(
        self,
        scene_id: int,
        duration: int,
        output_path: str
    ) -> str:
        """
        Generate a placeholder video for testing.

        Creates a simple colored video with FFmpeg.

        Args:
            scene_id: Scene number (for color variation)
            duration: Duration in seconds
            output_path: Output file path

        Returns:
            Path to generated placeholder video
        """
        # Generate different colors for different scenes
        colors = [
            "0x2C3E50",  # Dark blue
            "0x8E44AD",  # Purple
            "0x2980B9",  # Blue
            "0x27AE60",  # Green
            "0xF39C12",  # Orange
            "0xC0392B",  # Red
            "0x7F8C8D",  # Gray
            "0x34495E",  # Dark gray
        ]
        color = colors[scene_id % len(colors)]

        # Create a simple colored video with FFmpeg
        cmd = [
            "ffmpeg",
            "-y",
            "-f", "lavfi",
            "-i", f"color=c={color}:s=1920x1080:d={duration}",
            "-vf", f"drawtext=text='Scene {scene_id}':fontsize=60:fontcolor=white:x=(w-text_w)/2:y=(h-text_h)/2",
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            "-r", "30",
            output_path
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            raise RuntimeError(f"Failed to generate placeholder video: {result.stderr}")

        return output_path

    def get_generation_cost_estimate(
        self,
        scene_id: int,
        duration_sec: int
    ) -> Dict[str, Any]:
        """
        Estimate generation cost for a scene.

        Returns:
            Cost estimate dictionary
        """
        if self.feature_flags.hook_scene1_video and scene_id == 1:
            # High-quality video generation with Veo 3.1 (uses Google API - integrated with Gemini)
            service_name = "Veo 3.1" if self.service == "veo" else self.service
            return {
                "method": f"high_quality_video ({service_name})",
                "estimated_usd": duration_sec * 0.03,  # ~$0.03/sec for Veo 3.1
                "api_calls": 1,
            }
        else:
            # Image + Ken Burns (cheap or free)
            return {
                "method": "image+kenburns",
                "estimated_usd": 0.02 if self.api_key else 0.0,  # DALL-E cost or free placeholder
                "api_calls": 1,
            }
