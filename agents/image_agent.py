"""
Image Agent: Generates still images for scenes.

Scene 2~N에서 사용되며, Ken Burns 효과와 함께 영상처럼 변환됨.

v2.0 업데이트:
- 복수 캐릭터 참조 이미지 지원 (character_reference_paths: List[str])
- MultimodalPromptBuilder 통합
"""

import os
from typing import Optional, List
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

    def __init__(self, api_key: str = None, service: str = "nanobana"):
        """
        Initialize Image Agent.

        Args:
            api_key: API key for image generation service
            service: Service to use ("nanobana", "dalle", "replicate")
        """
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.nanobanana_token = os.getenv("GOOGLE_API_KEY")  # NanoBanana uses Google API key
        self.replicate_token = os.getenv("REPLICATE_API_TOKEN")
        self.service = service
        self._llm_client = None

        if not self.api_key and not self.replicate_token and not self.nanobanana_token:
            print("[ImageAgent] No image generation API keys provided. Will use placeholder images.")

        print(f"[ImageAgent] Init - ReplicateToken: {'YES' if self.replicate_token else 'NO'}, NanoToken: {'YES' if self.nanobanana_token else 'NO'}")

        # Prioritize Gemini 2.5 Flash Image (direct API) if available
        if self.nanobanana_token:
            self.service = "nanobana"
            print("[ImageAgent] Using Gemini 2.5 Flash Image (direct API) for image generation.")
        # Fallback to Replicate if available
        elif self.replicate_token:
            self.service = "replicate"
            print("[ImageAgent] Using Replicate for image generation (fallback).")

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
        output_dir: str = "media/images",
        seed: Optional[int] = None,
        character_tokens: Optional[list] = None,
        character_reference_id: Optional[str] = None,
        character_reference_path: Optional[str] = None,
        character_reference_paths: Optional[List[str]] = None,  # v2.0: 복수 참조 이미지
        style_anchor_path: Optional[str] = None,           # v2.1: 스타일 앵커
        environment_anchor_path: Optional[str] = None,     # v2.1: 환경 앵커
        image_model: str = "standard"  # standard / premium
    ) -> tuple:
        """
        Generate an image with specific model strategies.

        Standard: Gemini 2.5 Flash Image -> Replicate -> Placeholder
        Premium: Gemini 2.5 Flash Image (high) -> Replicate -> Placeholder
        """
        print(f"[ImageAgent v2.3] Generating Image for Scene {scene_id} | Model: {image_model}")

        # v2.0: 단일 참조를 복수 참조 리스트로 통합
        if character_reference_paths is None:
            character_reference_paths = []
        if character_reference_path and character_reference_path not in character_reference_paths:
            character_reference_paths.insert(0, character_reference_path)
        
        # Verify arguments to prevent NameError
        try:
            _ = negative_prompt
            _ = character_tokens
        except NameError as ne:
            print(f"[CRITICAL] Arg missing in scope: {ne}")
            # This shouldn't happen if signature is correct, but safe fallback
            negative_prompt = negative_prompt if 'negative_prompt' in locals() else None
            character_tokens = character_tokens if 'character_tokens' in locals() else None

        # Output Path Setup
        if scene_id == 0:
            output_path = output_dir if output_dir.endswith('.png') else f"{output_dir}/master_character.png"
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            print(f"  [Image] Generating MASTER CHARACTER ({image_model})...")
        else:
            os.makedirs(output_dir, exist_ok=True)
            output_path = f"{output_dir}/scene_{scene_id:02d}.png"
            print(f"  [Image] Generating Scene {scene_id} ({image_model})...")

        print(f"     Prompt: {prompt[:60]}...")

        # -------------------------------------------------------------------------
        # Strategy Definition
        # -------------------------------------------------------------------------
        
        # Define attempts list based on model choice
        attempts = []
        
        if image_model == "premium":
            # Premium: Gemini Flash Image first, then Replicate fallback
            attempts.append(("gemini_flash", "high"))
            attempts.append(("replicate", "premium"))
        else:
            # Standard: Gemini Flash Image first, Replicate fallback
            attempts.append(("gemini_flash", "standard"))
            attempts.append(("replicate", "standard"))

        # -------------------------------------------------------------------------
        # Execution Loop
        # -------------------------------------------------------------------------
        
        for service, quality in attempts:
            try:
                if service == "replicate" and self.replicate_token:
                    print(f"     Attempting Replicate ({quality})...")
                    return self._call_replicate_api(
                        prompt=prompt,
                        style=style,
                        aspect_ratio=aspect_ratio,
                        output_path=output_path,
                        seed=seed,
                        character_reference_path=character_reference_path,
                        character_reference_paths=character_reference_paths,  # v2.0: 복수 참조
                        negative_prompt=negative_prompt,
                        character_tokens=character_tokens
                    )
                elif service == "gemini_flash" and self.nanobanana_token:
                    print(f"     Attempting Gemini 2.5 Flash Image ({quality})...")
                    return self._call_nanobana_api(
                        prompt=prompt,
                        style=style,
                        output_path=output_path,
                        seed=seed,
                        character_reference_path=character_reference_path,
                        character_reference_paths=character_reference_paths,
                        style_anchor_path=style_anchor_path,
                        environment_anchor_path=environment_anchor_path
                    )
            except Exception as e:
                error_msg = str(e)
                print(f"     [Error] {service} failed: {error_msg}")
                
                # Censorship Handling (Prompt Softening)
                if "sensitive" in error_msg.lower() or "safety" in error_msg.lower() or "nsfw" in error_msg.lower():
                    print(f"     [Safety] Content filter triggered. Softening prompt...")
                    try:
                        softened_prompt = self._soften_prompt(prompt, error_msg)
                        print(f"     Softened: {softened_prompt[:60]}...")
                        # Retry immediately with softened prompt using same service
                        if service == "replicate" and self.replicate_token:
                             return self._call_replicate_api(
                                prompt=softened_prompt,
                                style=style,
                                aspect_ratio=aspect_ratio,
                                output_path=output_path,
                                seed=seed,
                                character_reference_path=character_reference_path,
                                character_reference_paths=character_reference_paths,  # v2.0: 복수 참조
                                negative_prompt=negative_prompt,
                                character_tokens=character_tokens
                            )
                        elif service == "gemini_flash" and self.nanobanana_token:
                            return self._call_nanobana_api(
                                prompt=softened_prompt,
                                style=style,
                                output_path=output_path,
                                seed=seed,
                                character_reference_path=character_reference_path,
                                character_reference_paths=character_reference_paths,
                                style_anchor_path=style_anchor_path,
                                environment_anchor_path=environment_anchor_path
                            )
                    except Exception as retry_e:
                        print(f"     [Error] Retry with softened prompt failed: {retry_e}")

        # -------------------------------------------------------------------------
        # Fallback: Placeholder (Red Screen)
        # -------------------------------------------------------------------------
        print("     [Fallback] All methods failed. Generating placeholder.")
        image_path, image_id = self._generate_placeholder_image(
            scene_id=scene_id,
            prompt=prompt,
            output_path=output_path
        )
        return (image_path, image_id)

    def _soften_prompt(self, original_prompt: str, error_msg: str) -> str:
        """
        LLM을 사용하여 검열된 프롬프트를 안전하게 순화.
        """
        try:
             # 임시: 간단한 치환 로직 (LLM 호출 비용 절약 및 속도)
             # 실제로는 LLM을 호출하는 것이 가장 좋음
            softened = original_prompt.replace("corpses", "fallen figures")
            softened = softened.replace("blood", "red liquid")
            softened = softened.replace("gruesome", "scary")
            softened = softened.replace("kill", "defeat")
            
            if softened == original_prompt:
                return f"A safe abstract representation of: {original_prompt[:100]}"
            return softened
        except Exception:
            return "A mysterious scene, cinematic lighting"

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

    def _call_nanobana_api(
        self,
        prompt: str,
        style: str,
        output_path: str,
        seed: Optional[int] = None,
        character_reference_path: Optional[str] = None,
        character_reference_paths: Optional[List[str]] = None,  # v2.0: 복수 참조 이미지
        style_anchor_path: Optional[str] = None,           # v2.1: 스타일 앵커
        environment_anchor_path: Optional[str] = None,     # v2.1: 환경 앵커
    ) -> tuple:
        """
        Call Google Gemini 2.5 Flash Image API for image generation.
        v2.1: MultimodalPromptBuilder 통합 - 스타일/환경 앵커 지원

        Args:
            prompt: Image generation prompt
            style: Visual style
            output_path: Path to save the generated image
            seed: Visual seed for consistency
            character_reference_path: Master character image path for reference
            character_reference_paths: v2.0 - List of character reference paths
            style_anchor_path: v2.1 - Style anchor image for visual consistency
            environment_anchor_path: v2.1 - Environment anchor for background consistency

        Returns:
            Tuple of (image_path, image_id)
        """
        try:
            import requests
            import json
            import base64
            from utils.prompt_builder import MultimodalPromptBuilder

            print(f"     Calling Gemini 2.5 Flash Image API...")

            # Google Gemini API endpoint
            api_url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-image:generateContent?key={self.nanobanana_token}"

            headers = {
                "Content-Type": "application/json"
            }

            # v2.0: 단일 참조를 복수 참조 리스트로 통합
            all_reference_paths = []
            if character_reference_paths:
                all_reference_paths.extend(character_reference_paths)
            if character_reference_path and character_reference_path not in all_reference_paths:
                all_reference_paths.insert(0, character_reference_path)
            # Filter to existing files
            all_reference_paths = [p for p in all_reference_paths if p and os.path.exists(p)]

            # v2.1: Use MultimodalPromptBuilder for consistent part ordering
            parts = MultimodalPromptBuilder.build_simple_request(
                prompt=prompt,
                character_reference_paths=all_reference_paths,
                style=style,
                style_anchor_path=style_anchor_path,
                environment_anchor_path=environment_anchor_path,
            )

            # Log what anchors are being used
            if style_anchor_path and os.path.exists(style_anchor_path):
                print(f"     [v2.1] Style anchor included: {os.path.basename(style_anchor_path)}")
            if environment_anchor_path and os.path.exists(environment_anchor_path):
                print(f"     [v2.1] Environment anchor included: {os.path.basename(environment_anchor_path)}")
            if all_reference_paths:
                print(f"     [v2.1] Character references: {len(all_reference_paths)}")

            payload = {
                "contents": [{
                    "parts": parts
                }],
                "generationConfig": {
                    "responseModalities": ["image"],
                    "response_modalities": ["image"]
                }
            }

            # Note: Gemini Flash Image doesn't support seed parameter
            if seed is not None:
                print(f"     [Warning] Seed parameter not supported in Gemini Flash Image")

            response = requests.post(
                api_url,
                headers=headers,
                json=payload,
                timeout=120
            )

            if response.status_code != 200:
                error_text = response.text[:500]
                print(f"     [Gemini DEBUG] Status: {response.status_code}, Response: {error_text}")
                raise RuntimeError(f"Gemini API error: {response.status_code} - {error_text}")

            result = response.json()

            # Extract image from response
            if "candidates" in result and len(result["candidates"]) > 0:
                candidate = result["candidates"][0]
                if "content" in candidate and "parts" in candidate["content"]:
                    for part in candidate["content"]["parts"]:
                        if "inline_data" in part:
                            # Image is base64 encoded in inline_data
                            image_data = base64.b64decode(part["inline_data"]["data"])

                            with open(output_path, "wb") as f:
                                f.write(image_data)

                            print(f"     Image saved: {output_path}")
                            return (output_path, None)  # Gemini doesn't return image ID

            print(f"     [Gemini DEBUG] Raw Result: {json.dumps(result)}")
            raise RuntimeError("No image data found in Gemini API response")

        except ImportError:
            raise RuntimeError("requests library not installed")
        except Exception as e:
            print(f"     Gemini API error: {e}")
            raise

    def _call_dalle_api(
        self,
        prompt: str,
        style: str,
        output_path: str
    ) -> tuple:
        """
        Call OpenAI DALL-E API with timeout.

        Returns:
            Tuple of (image_path, image_id)
        """
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
                return (output_path, None)  # DALL-E doesn't provide stable image IDs
            else:
                raise RuntimeError(f"Failed to download image: {img_response.status_code}")

        except ImportError:
            raise RuntimeError("OpenAI library not installed")
        except Exception as e:
            print(f"     DALL-E API error: {e}")
            raise

    def _call_replicate_api(
        self,
        prompt: str,
        style: str,
        aspect_ratio: str,
        output_path: str,
        seed: Optional[int] = None,
        character_reference_path: Optional[str] = None,
        character_reference_paths: Optional[List[str]] = None,  # v2.0: 복수 참조 이미지
        negative_prompt: Optional[str] = None,
        character_tokens: Optional[list] = None
    ) -> tuple:
        """
        Call Replicate API (Gemini 2.5 Flash Image).
        Using Gemini for high speed and good quality in various styles.

        v2.0 Update:
        - Support multiple character reference images
        - Builds multimodal prompt with all references

        Args:
            character_reference_path: Path to master character image for consistency (legacy)
            character_reference_paths: v2.0 - List of all character reference image paths

        Returns:
            Tuple of (image_path, image_id)
        """
        try:
            import replicate
            import requests
            import base64

            # v2.0: Consolidate all reference paths
            all_reference_paths = []
            if character_reference_paths:
                all_reference_paths.extend(character_reference_paths)
            if character_reference_path and character_reference_path not in all_reference_paths:
                all_reference_paths.insert(0, character_reference_path)
            
            # Filter to existing files only
            all_reference_paths = [p for p in all_reference_paths if p and os.path.exists(p)]

            # Enhanced prompt for better quality
            # Remove generic qualifiers and focus on visual details
            full_prompt = self._enhance_generic_prompt(f"{style} style: {prompt}")

            print(f"     Calling Replicate (Gemini 2.5 Flash Image) API...")
            if all_reference_paths:
                print(f"     [v2.0] Using {len(all_reference_paths)} character reference(s)")

            # Use Gemini 2.5 Flash Image (nano banana)
            model = "google/gemini-2.5-flash-image"

            # v2.1: Advanced Style & Negative Prompt Logic
            # "Realistic" specifically requested to BAN anime/cartoon
            strong_negative_prompt = ""
            if "real" in style.lower() or "cinematic" in style.lower() or "photo" in style.lower():
                strong_negative_prompt = "anime, cartoon, illustration, drawing, painting, sketch, 3d render, plastic, fake, deformed, disfigured"
                print(f"     [Style Enforcement] Enforcing REALISTIC style. Negative: {strong_negative_prompt}")

            # Combine user negative prompt with system negative prompt
            final_negative_prompt = negative_prompt or ""
            if strong_negative_prompt:
                final_negative_prompt = f"{final_negative_prompt}, {strong_negative_prompt}".strip(", ")

            # v2.1: Force character description to be part of the prompt if tokens exist
            # This ensures character consistency better than just 'character_tokens' list
            if character_tokens:
                # Assuming character_tokens is a list of descriptions like ["A young man with glasses", "A woman in a red coat"]
                # We prepend them to make them the MAIN SUBJECT
                char_desc = " ".join(character_tokens)
                full_prompt = f"{char_desc}. {full_prompt}"
                print(f"     [Character Injection] Prepending character info: {char_desc}")

            # v2.0: Build multimodal prompt with character references
            if all_reference_paths:
                # Add character consistency instruction
                ref_instruction = "Using the character reference image(s) provided, maintain exact character consistency including faces, clothing, and distinctive features. "
                full_prompt = f"{ref_instruction}{full_prompt}"
                print(f"     [v2.0] Enhanced prompt with reference instruction")

            input_params = {
                "prompt": full_prompt,
                "aspect_ratio": "16:9",
                "output_format": "png",
            }

            # NOTE: Replicate usually takes 'prompt' only. 'negative_prompt' is for some implementations.
            # We will append exclusionary terms to prompt for maximum compatibility: "..., avoid anime, avoid cartoon"
            if strong_negative_prompt:
                 input_params["prompt"] += f" --no {strong_negative_prompt}" # Common syntax
            
            # v2.0: Add seed for character consistency
            if seed is not None:
                input_params["seed"] = seed
                print(f"     Using seed: {seed} for consistency")

            # v2.0: Add character reference images
            # Replicate's Gemini model expects 'image' or 'images' parameter with URI or base64
            if all_reference_paths:
                # Convert images to data URIs for Replicate
                image_uris = []
                for ref_path in all_reference_paths[:3]:  # Max 3 references
                    try:
                        with open(ref_path, "rb") as f:
                            image_data = base64.b64encode(f.read()).decode('utf-8')
                            ext = os.path.splitext(ref_path)[1].lower()
                            mime_type = "image/png" if ext == ".png" else "image/jpeg"
                            data_uri = f"data:{mime_type};base64,{image_data}"
                            image_uris.append(data_uri)
                            print(f"     Added reference: {os.path.basename(ref_path)}")
                    except Exception as e:
                        print(f"     [Warning] Failed to load reference {ref_path}: {e}")

                if image_uris:
                    # Gemini on Replicate accepts 'image' for single or 'images' for multiple
                    if len(image_uris) == 1:
                        input_params["image"] = image_uris[0]
                    else:
                        input_params["images"] = image_uris

            output = replicate.run(model, input=input_params)

            # Handle different output formats
            if isinstance(output, str):
                image_url = output
            elif isinstance(output, list) and len(output) > 0:
                image_url = output[0]
            elif hasattr(output, 'url'):
                # FileOutput object from Replicate
                image_url = output.url
            else:
                # Try to convert to string
                image_url = str(output)

            print(f"     Downloading image from Replicate...")
            print(f"     Image URL: {image_url[:60]}...")
            img_response = requests.get(image_url, timeout=30)

            if img_response.status_code == 200:
                with open(output_path, "wb") as f:
                    f.write(img_response.content)
                print(f"     Image saved: {output_path}")
                return (output_path, None)  # Replicate doesn't provide stable image IDs
            else:
                raise RuntimeError(f"Failed to download image: {img_response.status_code}")

        except ImportError:
            raise RuntimeError("replicate library not installed")
        except Exception as e:
            error_msg = str(e).lower()
            if any(kw in error_msg for kw in ["quota", "rate", "limit", "429", "exceeded", "billing", "payment"]):
                print(f"     [Replicate] QUOTA/RATE LIMIT exceeded: {e}")
                print(f"     [Replicate] Will fallback to Gemini Flash Image...")
            else:
                print(f"     Replicate API error: {e}")
            raise

    def _enhance_generic_prompt(self, prompt: str) -> str:
        """
        Enhance prompt for better image quality.

        Args:
            prompt: Original prompt

        Returns:
            Enhanced prompt with quality keywords
        """
        # Add quality enhancers
        quality_keywords = (
            "masterpiece, professional artwork, high detail, sharp focus, "
            "cinematic lighting, vibrant colors, intricate details"
        )

        # Avoid redundancy - check if quality keywords already present
        if "masterpiece" not in prompt.lower():
            return f"{prompt}, {quality_keywords}"
        return prompt

    def _generate_placeholder_image(
        self,
        scene_id: int,
        prompt: str,
        output_path: str
    ) -> tuple:
        """
        Generate a placeholder image using Pillow (No external dependency).

        Returns:
            Tuple of (image_path, image_id)
        """
        # CRITICAL: Create directories first!
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

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
        draw.text((width/2, height/2 - 100), title_text, font=font_title, fill="white", anchor="mm")

        # Draw Wrapped Prompt
        wrapper = textwrap.TextWrapper(width=60)
        word_list = wrapper.wrap(text=prompt)
        prompt_text = "\n".join(word_list[:5])  # Limit lines

        draw.text((width/2, height/2 + 50), prompt_text, font=font_text, fill="lightgray", anchor="mm", align="center")

        # Save
        img.save(output_path)
        return (output_path, None)  # Placeholder images don't have IDs

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
