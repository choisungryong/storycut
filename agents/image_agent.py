"""
Image Agent: Generates still images for scenes.

Scene 2~N에서 사용되며, Ken Burns 효과와 함께 영상처럼 변환됨.
"""

import os
from typing import Optional
from pathlib import Path
import sys
from PIL import Image, ImageDraw, ImageFont
import textwrap

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
        self._llm_client = None

        if not self.api_key:
            print("[ImageAgent] No image API key provided. Will use placeholder images.")

    @property
    def llm_client(self):
        """Lazy initialization of LLM client for sanitization."""
        if self._llm_client is None and self.api_key:
            try:
                from openai import OpenAI
                self._llm_client = OpenAI(api_key=self.api_key)
            except Exception:
                self._llm_client = None
        return self._llm_client

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
        Generate an image for a scene with Safety Retry.

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

        # 1. Try Generation
        if self.api_key and self.service == "dalle":
            try:
                return self._call_dalle_api(
                    prompt=prompt,
                    style=style,
                    output_path=output_path
                )
            except Exception as e:
                # 2. Safety Retry: If content policy violation, sanitize prompt
                if "content_policy_violation" in str(e) or "content filters" in str(e):
                    print(f"     [Warning] Content Filter triggered. Sanitizing prompt...")
                    try:
                        sanitized_prompt = self._sanitize_prompt(prompt, str(e))
                        print(f"     Sanitized Prompt: {sanitized_prompt[:60]}...")
                        
                        return self._call_dalle_api(
                            prompt=sanitized_prompt,
                            style=style,
                            output_path=output_path
                        )
                    except Exception as retry_e:
                         print(f"     [Error] Retry failed: {retry_e}")

                print(f"     [Error] DALL-E API failed: {e}")
                print("     Falling back to placeholder image...")

        # 3. Fallback: Generate placeholder image (using Pillow)
        image_path = self._generate_placeholder_image(
            scene_id=scene_id,
            prompt=prompt,
            output_path=output_path
        )
        print(f"     Placeholder saved: {image_path}")
        return image_path

    def _sanitize_prompt(self, original_prompt: str, error_msg: str) -> str:
        """
        Sanitize prompt using LLM to bypass content filters safely.
        """
        if not self.llm_client:
            return "A peaceful abstract representation of the concept"

        system_msg = """
        You are a Prompt Engineer. The user's image prompt triggered a safety content filter.
        Rewrite the prompt to completely remove any violent, sexual, or sensitive words while keeping the core artistic meaning.
        Make it abstract and safe.
        Output ONLY the sanitized prompt.
        """
        
        user_msg = f"""
        Original Prompt: {original_prompt}
        Error: {error_msg}
        
        Sanitized Prompt:
        """

        try:
            response = self.llm_client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": user_msg}
                ],
                temperature=0.7,
                max_tokens=200
            )
            return response.choices[0].message.content.strip()
        except Exception:
            return "A mysterious and dramatic atmosphere, cinematic lighting"

    def _call_dalle_api(
        self,
        prompt: str,
        style: str,
        output_path: str
    ) -> str:
        """Call OpenAI DALL-E API with timeout."""
        try:
            from openai import OpenAI
            import requests

            # OpenAI client with 60 second timeout
            client = OpenAI(api_key=self.api_key, timeout=60.0)

            enhanced_prompt = f"{style} style: {prompt}"

            print(f"     Calling DALL-E API (timeout: 60s)...")
            response = client.images.generate(
                model="dall-e-3",
                prompt=enhanced_prompt,
                size="1792x1024",  # 16:9 ratio in DALL-E 3
                quality="standard",
                n=1,
            )

            image_url = response.data[0].url
            print(f"     Downloading image from URL...")
            img_response = requests.get(image_url, timeout=30)

            if img_response.status_code == 200:
                with open(output_path, "wb") as f:
                    f.write(img_response.content)
                print(f"     Image saved: {output_path}")
                return output_path
            else:
                raise RuntimeError(f"Failed to download image: {img_response.status_code}")

        except ImportError:
            raise RuntimeError("OpenAI library not installed")
        except Exception as e:
            print(f"     DALL-E API error: {e}")
            raise

    def _generate_placeholder_image(
        self,
        scene_id: int,
        prompt: str,
        output_path: str
    ) -> str:
        """
        Generate a placeholder image using Pillow (No external dependency).
        """
        width, height = 1920, 1080
        
        # Determine background color based on scene_id
        colors = [
            (26, 26, 46),   # Dark blue
            (45, 19, 44),   # Purple
            (15, 52, 96),   # Blue
            (30, 81, 40),   # Green
            (74, 55, 40),   # Brown
            (61, 0, 0),     # Red
        ]
        bg_color = colors[scene_id % len(colors)]
        
        img = Image.new('RGB', (width, height), color=bg_color)
        draw = ImageDraw.Draw(img)
        
        # Try to load a font, fallback to default
        try:
            # Windows font path usually
            font_title = ImageFont.truetype("arial.ttf", 80)
            font_text = ImageFont.truetype("arial.ttf", 36)
        except IOError:
            font_title = ImageFont.load_default()
            font_text = ImageFont.load_default()

        # Draw Scene ID
        title_text = f"Scene {scene_id}"
        # For default font, getsize might be needed in older Pillow results, but textbbox is newer.
        # Simplification for robustness: approximate centering or just top-left if needed.
        # But let's try standard anchor positioning if available or calculate.
        
        draw.text((width/2, height/2 - 100), title_text, font=font_title, fill="white", anchor="mm")

        # Draw Wrapped Prompt
        wrapper = textwrap.TextWrapper(width=60)
        word_list = wrapper.wrap(text=prompt)
        prompt_text = "\n".join(word_list[:5])  # Limit lines
        
        draw.text((width/2, height/2 + 50), prompt_text, font=font_text, fill="lightgray", anchor="mm", align="center")
        
        # Save
        img.save(output_path)
        return output_path

    def generate_thumbnail(
        self,
        prompt: str,
        text_overlay: str = None,
        output_path: str = "output/thumbnail.png"
    ) -> str:
        """Generate a thumbnail image."""
        print(f"  Generating thumbnail...")

        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        try:
            if self.api_key and self.service == "dalle":
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
