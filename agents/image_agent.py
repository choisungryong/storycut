"""
Image Agent: Generates still images for scenes.

Scene 2~N에서 사용되며, Ken Burns 효과와 함께 영상처럼 변환됨.
"""

import os
import subprocess
from typing import Optional
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))

from schemas import FeatureFlags


class ImageAgent:
    """
    이미지 생성 에이전트

    비디오 생성이 실패하거나 비용 절감을 위해
    이미지를 생성하고 Ken Burns 효과로 영상화합니다.
    """

    def __init__(self, api_key: str = None, service: str = "dalle"):
        """
        Initialize Image Agent.

        Args:
            api_key: API key for image generation service
            service: Service to use ("dalle", "stability", "midjourney")
        """
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.service = service

        if not self.api_key:
            print("  No image API key provided. Will use placeholder images.")

    def generate_image(
        self,
        scene_id: int,
        prompt: str,
        negative_prompt: Optional[str] = None,
        style: str = "cinematic",
        aspect_ratio: str = "16:9",
        output_dir: str = "media/images"
    ) -> str:
        """
        Generate an image for a scene.

        Args:
            scene_id: Scene identifier
            prompt: Image generation prompt
            negative_prompt: What to avoid in the image
            style: Visual style
            aspect_ratio: Image aspect ratio
            output_dir: Output directory

        Returns:
            Path to generated image file
        """
        print(f"  Generating image for scene {scene_id}...")
        print(f"     Prompt: {prompt[:60]}...")

        os.makedirs(output_dir, exist_ok=True)
        output_path = f"{output_dir}/scene_{scene_id:02d}.png"

        if self.api_key and self.service == "dalle":
            try:
                image_path = self._call_dalle_api(
                    prompt=prompt,
                    style=style,
                    output_path=output_path
                )
                print(f"     Image saved: {image_path}")
                return image_path
            except Exception as e:
                print(f"     DALL-E API failed: {e}")
                print("     Falling back to placeholder image...")

        # Generate placeholder image
        image_path = self._generate_placeholder_image(
            scene_id=scene_id,
            prompt=prompt,
            output_path=output_path
        )
        print(f"     Image saved: {image_path}")
        return image_path

    def _call_dalle_api(
        self,
        prompt: str,
        style: str,
        output_path: str
    ) -> str:
        """
        Call OpenAI DALL-E API for image generation.

        Args:
            prompt: Image generation prompt
            style: Visual style
            output_path: Where to save the image

        Returns:
            Path to generated image
        """
        try:
            from openai import OpenAI
            import requests

            client = OpenAI(api_key=self.api_key)

            # Enhance prompt with style
            enhanced_prompt = f"{style} style: {prompt}"

            response = client.images.generate(
                model="dall-e-3",
                prompt=enhanced_prompt,
                size="1792x1024",  # 16:9 ratio
                quality="standard",
                n=1,
            )

            # Download the image
            image_url = response.data[0].url
            img_response = requests.get(image_url)

            if img_response.status_code == 200:
                with open(output_path, "wb") as f:
                    f.write(img_response.content)
                return output_path
            else:
                raise RuntimeError(f"Failed to download image: {img_response.status_code}")

        except ImportError:
            raise RuntimeError("OpenAI library not installed")

    def _generate_placeholder_image(
        self,
        scene_id: int,
        prompt: str,
        output_path: str
    ) -> str:
        """
        Generate a placeholder image using FFmpeg.

        Creates a gradient background with scene info text.

        Args:
            scene_id: Scene number
            prompt: Original prompt (for display)
            output_path: Output file path

        Returns:
            Path to generated placeholder image
        """
        # Color gradients for different scenes
        gradients = [
            ("0x1a1a2e", "0x16213e"),  # Dark blue gradient
            ("0x2d132c", "0x801336"),  # Purple gradient
            ("0x0f3460", "0x16213e"),  # Blue gradient
            ("0x1e5128", "0x191a19"),  # Green gradient
            ("0x4a3728", "0x2c2c2c"),  # Brown gradient
            ("0x3d0000", "0x1a1a2e"),  # Red gradient
        ]

        color1, color2 = gradients[scene_id % len(gradients)]

        # Truncate prompt for display
        display_text = prompt[:50].replace("'", "\\'").replace('"', '\\"')
        if len(prompt) > 50:
            display_text += "..."

        # Create gradient image with FFmpeg
        cmd = [
            "ffmpeg",
            "-f", "lavfi",
            "-i", f"gradients=s=1920x1080:c0={color1}:c1={color2}:duration=1:speed=0.5",
            "-vf", (
                f"drawtext=text='Scene {scene_id}':fontsize=72:fontcolor=white:"
                f"x=(w-text_w)/2:y=(h-text_h)/2-50,"
                f"drawtext=text='{display_text}':fontsize=28:fontcolor=gray:"
                f"x=(w-text_w)/2:y=(h-text_h)/2+50"
            ),
            "-frames:v", "1",
            output_path,
            "-y"
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            # Fallback to solid color if gradient fails
            fallback_cmd = [
                "ffmpeg",
                "-f", "lavfi",
                "-i", f"color=c={color1}:s=1920x1080",
                "-frames:v", "1",
                output_path,
                "-y"
            ]
            subprocess.run(fallback_cmd, capture_output=True)

        return output_path

    def generate_thumbnail(
        self,
        prompt: str,
        text_overlay: str = None,
        output_path: str = "output/thumbnail.png"
    ) -> str:
        """
        Generate a thumbnail image for YouTube.

        Args:
            prompt: Thumbnail image prompt
            text_overlay: Text to overlay on thumbnail
            output_path: Output file path

        Returns:
            Path to generated thumbnail
        """
        print(f"  Generating thumbnail...")

        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        if self.api_key and self.service == "dalle":
            try:
                return self._call_dalle_api(
                    prompt=f"YouTube thumbnail, eye-catching, bold: {prompt}",
                    style="vibrant, high contrast",
                    output_path=output_path
                )
            except Exception as e:
                print(f"     Thumbnail generation failed: {e}")

        # Placeholder thumbnail
        return self._generate_placeholder_image(
            scene_id=0,
            prompt=prompt,
            output_path=output_path
        )
