"""
Story Agent: Generates scene-based story JSON using LLM (Gemini 3 Pro).
"""

import json
import os
from typing import Dict, Any
from pathlib import Path


class StoryAgent:
    """
    Generates YouTube-optimized stories in Scene JSON format.

    This agent calls Google Gemini 3 Pro API
    to generate structured story content.
    """

    def __init__(self, api_key: str = None, model: str = "gemini-3-pro-preview"):
        """
        Initialize Story Agent.

        Args:
            api_key: Google Gemini API key
            model: LLM model to use (default: gemini-3-pro-preview)
        """
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY")
        self.model = model

        if not self.api_key:
            raise ValueError("API key is required. Set GOOGLE_API_KEY environment variable.")

        # Load story prompt template
        prompt_path = Path(__file__).parent.parent / "prompts" / "story_prompt.md"
        with open(prompt_path, "r", encoding="utf-8") as f:
            self.system_prompt = f.read()

    def generate_story(
        self,
        genre: str,
        mood: str,
        style: str,
        total_duration_sec: int = 90,
        user_idea: str = None
    ) -> Dict[str, Any]:
        """
        Generate a scene-based story JSON.

        Args:
            genre: Story genre (e.g., "emotional", "mystery")
            mood: Overall mood (e.g., "melancholic", "suspenseful")
            style: Visual style (e.g., "cinematic animation")
            total_duration_sec: Total video length (60-150 seconds)
            user_idea: Optional user-provided story idea

        Returns:
            Story JSON with scenes
        """
        print(f"[Story Agent] Generating story: {genre} / {mood} / {style} / {total_duration_sec}s")

        # Calculate target scene count (approx 1 scene per 5-6 seconds)
        min_scenes = total_duration_sec // 8
        max_scenes = total_duration_sec // 4
        
        # Build user prompt
        user_prompt = f"""CRITICAL REQUIREMENTS:
- Generate scenes to fit TOTAL DURATION: {total_duration_sec} seconds.
- Target Scene Count: Approximately {min_scenes} to {max_scenes} scenes.
- Genre: {genre}
- Mood: {mood}
- Style: {style}

MANDATORY STRUCTURE:
- Ensure the story has a clear beginning (Hook), middle (Build/Climax), and end (Resolution).
- Adjust pacing based on the mood (faster for thriller/action, slower for emotional).

OUTPUT FORMAT - ONLY VALID JSON, NO MARKDOWN:
{{
  "title": "compelling korean title",
  "genre": "{genre}",
  "mood": "{mood}",
  "total_duration_sec": {total_duration_sec},
  "youtube_opt": {{
    "title_candidates": ["Clickbait Title 1", "Searchable Title 2", "Emotional Title 3"],
    "thumbnail_text": "Short Hook Text (Max 10 chars)",
    "hashtags": ["#Tag1", "#Tag2", "#Tag3", "#Shorts", "#Story"]
  }},
  "scenes": [
    {{
      "scene_id": 1,
      "duration_sec": 5,
      "narration": "natural korean spoken language",
      "visual_description": "detailed english description in {style} style, focusing on lighting and atmosphere",
      "mood": "scene mood"
    }},
    ... (continue until total duration uses approx {total_duration_sec}s) ...
  ]
}}
"""

        if user_idea:
            user_prompt += f"\nUser Idea: {user_idea}\n"

        user_prompt += """
OUTPUT ONLY THE JSON. NO EXPLANATIONS. NO MARKDOWN FORMATTING."""

        # Call LLM API
        story_json = self._call_llm_api(user_prompt)

        # Validate and parse JSON
        story_data = self._validate_story_json(story_json)

        print(f"[Story Agent] Story generated: {story_data['title']}")
        print(f"   Scenes: {len(story_data['scenes'])}")

        return story_data

    def _call_llm_api(self, user_prompt: str) -> str:
        """
        Call Gemini 3 Pro API to generate story.

        Args:
            user_prompt: User's story generation request

        Returns:
            LLM response (should be JSON string)
        """
        try:
            import google.generativeai as genai

            print(f"[DEBUG] Calling Gemini 3 Pro API (model: {self.model})", flush=True)
            genai.configure(api_key=self.api_key)

            model = genai.GenerativeModel(
                model_name=self.model,
                system_instruction=self.system_prompt
            )

            # Configure safety settings to allow story generation
            safety_settings = {
                'HARM_CATEGORY_HARASSMENT': 'BLOCK_NONE',
                'HARM_CATEGORY_HATE_SPEECH': 'BLOCK_NONE',
                'HARM_CATEGORY_SEXUALLY_EXPLICIT': 'BLOCK_NONE',
                'HARM_CATEGORY_DANGEROUS_CONTENT': 'BLOCK_NONE',
            }

            response = model.generate_content(
                user_prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.7,
                    max_output_tokens=60000,  # Massively increased for 5min+ scripts
                    response_mime_type="application/json"  # Enforce strict JSON output
                ),
                safety_settings=safety_settings
            )

            print(f"[DEBUG] Gemini API response received", flush=True)
            response_text = response.text.strip()

            # Log the response for debugging
            print(f"[DEBUG] Response length: {len(response_text)} chars", flush=True)
            print(f"[DEBUG] Response preview: {response_text[:200]}...", flush=True)

            return response_text

        except ImportError:
            # Fallback: Return example story for testing
            print("[Warning] google-generativeai library not available. Using example story.")
            return self._get_example_story()
        except Exception as e:
            from utils.error_manager import ErrorManager
            ErrorManager.log_error(
                "StoryAgent", 
                "LLM API Call Failed", 
                f"{type(e).__name__}: {str(e)}", 
                severity="critical"
            )
            print(f"[ERROR] Gemini API call failed: {type(e).__name__}: {str(e)}", flush=True)
            import traceback
            print(f"[ERROR] Traceback: {traceback.format_exc()}", flush=True)
            print(f"[ERROR] Falling back to example story (TEMPORARY)", flush=True)
            return self._get_example_story()

    def _validate_story_json(self, json_string: str) -> Dict[str, Any]:
        """
        Validate and parse story JSON (v2.0 with character reference support).

        Args:
            json_string: JSON string from LLM

        Returns:
            Parsed story dictionary

        Raises:
            ValueError: If JSON is invalid
        """
        # Remove markdown code blocks if present
        json_string = json_string.strip()
        if json_string.startswith("```json"):
            json_string = json_string[7:]
        if json_string.startswith("```"):
            json_string = json_string[3:]
        if json_string.endswith("```"):
            json_string = json_string[:-3]
        json_string = json_string.strip()

        try:
            story_data = json.loads(json_string)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON from LLM: {e}")

        # Validate required fields
        required_fields = ["title", "genre", "total_duration_sec", "scenes"]
        for field in required_fields:
            if field not in story_data:
                raise ValueError(f"Missing required field: {field}")

        # Validate scenes
        if not story_data["scenes"]:
            raise ValueError("No scenes in story")

        scene_count = len(story_data["scenes"])
        print(f"[Story Validation] Scene count: {scene_count}", flush=True)

        # v2.0: Validate character_sheet and global_style (optional but recommended)
        if "character_sheet" in story_data:
            print(f"[Story Validation] Character sheet found: {len(story_data['character_sheet'])} characters", flush=True)
        else:
            print(f"[Story Validation] No character sheet (v1.0 format)", flush=True)

        if "global_style" in story_data:
            print(f"[Story Validation] Global style: {story_data['global_style'].get('art_style', 'N/A')}", flush=True)

        # WARNING: Scene count validation (TEST MODE: 4 scenes expected)
        if scene_count < 3:
            print(f"[WARNING] Only {scene_count} scenes (recommended 4 for test mode)", flush=True)
        elif scene_count > 6:
            print(f"[INFO] {scene_count} scenes generated (test mode expects 4)", flush=True)

        for idx, scene in enumerate(story_data["scenes"], 1):
            # v1.0 필수 필드 (하위 호환성)
            required_scene_fields_v1 = ["scene_id", "narration", "mood", "duration_sec"]

            # visual_description 또는 image_prompt 중 하나는 필수
            has_visual = "visual_description" in scene or "image_prompt" in scene

            for field in required_scene_fields_v1:
                if field not in scene:
                    raise ValueError(f"Scene {idx}: Missing required field '{field}'")

            if not has_visual:
                raise ValueError(f"Scene {idx}: Must have 'visual_description' or 'image_prompt'")

            # v2.0 필드 검증 (선택사항)
            if "image_prompt" in scene:
                print(f"[Scene {idx}] v2.0 format detected (image_prompt present)", flush=True)

            if "characters_in_scene" in scene and scene["characters_in_scene"]:
                print(f"[Scene {idx}] Characters: {', '.join(scene['characters_in_scene'])}", flush=True)

        return story_data

    def _get_example_story(self) -> str:
        """
        Return a TEST example story JSON with 4 scenes (for when Gemini API fails).
        TEMPORARY FALLBACK - Real LLM should generate this.
        """
        example = {
            "title": "테스트 스토리",
            "genre": "mystery",
            "mood": "dramatic",
            "total_duration_sec": 60,
            "scenes": [
                # HOOK: Grab attention
                {"scene_id": 1, "narration": "그날, 그녀의 집에 이상한 편지가 도착했습니다.", "visual_description": "A mysterious envelope on a doorstep, dramatic lighting", "mood": "mysterious", "duration_sec": 15},

                # BUILD: Rising tension
                {"scene_id": 2, "narration": "발신자는 20년 전 사라진 아버지였습니다.", "visual_description": "Woman reading letter with shocked expression, old family photo visible", "mood": "shocking", "duration_sec": 15},

                # CLIMAX: Revelation
                {"scene_id": 3, "narration": "하지만 그것은 아버지가 아니었습니다.", "visual_description": "Dark figure revealed, woman's realization moment", "mood": "terrifying", "duration_sec": 15},

                # RESOLUTION: Ending
                {"scene_id": 4, "narration": "진실은 생각보다 가까운 곳에 있었습니다.", "visual_description": "Police arriving, woman finding closure", "mood": "bittersweet", "duration_sec": 15}
            ],
            # Fallback YouTube Optimization (Test)
            "youtube_opt": {
                "title_candidates": ["The Letter from the Past", "Mystery of the Doorstep", "20 Years Later"],
                "thumbnail_text": "Who Sent This?",
                "hashtags": ["#Mystery", "#Shorts", "#Thriller"]
            }
        }

        print("[WARNING] Using 4-scene example story (Gemini API not available)", flush=True)
        return json.dumps(example, indent=2)

    def save_story(self, story_data: Dict[str, Any], output_path: str = "scenes/story_scenes.json"):
        """
        Save story JSON to file.

        Args:
            story_data: Story dictionary
            output_path: Output file path
        """
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(story_data, f, indent=2, ensure_ascii=False)

        print(f"[Story Agent] Story saved to: {output_path}")
