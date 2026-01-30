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
        Generate a scene-based story JSON using a 2-Step Hierarchical Chain.

        Step 1: Structure & Architecture (Title, Characters, Outline)
        Step 2: Scene-level Detail (Script, Prompts, Camera Work)
        """
        print(f"[Story Agent] Generating story (2-Step Chain): {genre} / {mood} / {style}")

        # Calculate target scene count
        min_scenes = total_duration_sec // 8
        max_scenes = total_duration_sec // 4

        # =================================================================================
        # STEP 1: Story Architecture
        # =================================================================================
        import sys
        print(f"  [Step 1] Planning Story Architecture...", file=sys.stderr, flush=True)
        
        step1_prompt = f"""
ROLE: Professional Storyboard Artist & Director.
TASK: Plan the structure for a {total_duration_sec}-second YouTube Short.
GENRE: {genre}
MOOD: {mood}
STYLE: {style}
SCENE COUNT: Approx {min_scenes}-{max_scenes} scenes.

{'USER IDEA: ' + user_idea if user_idea else ''}

OUTPUT FORMAT (JSON):
{{
  "project_title": "Creative Title",
  "logline": "One sentence summary",
  "global_style": {{
    "art_style": "{style}",
    "color_palette": "e.g., Cyberpunk Neons",
    "visual_seed": 12345
  }},
  "characters": {{
    "Name": {{
      "name": "Name",
      "appearance": "Detailed description",
      "role": "Protagonist/Antagonist"
    }}
  }},
  "outline": [
    {{ "scene_id": 1, "summary": "Brief summary of what happens", "estimated_duration": 5 }}
  ]
}}
"""
        step1_response = self._call_llm_api(step1_prompt)
        try:
            structure_data = json.loads(step1_response)
            print(f"  [Step 1] Structure locked: {structure_data.get('project_title')}", file=sys.stderr, flush=True)
        except Exception as e:
            print(f"  [Step 1] Failed to parse JSON: {e}. Falling back to single-step.", file=sys.stderr, flush=True)
            structure_data = {} 

        # =================================================================================
        # STEP 2: Scene-level Details
        # =================================================================================
        print(f"  [Step 2] Generating Scene Details...", file=sys.stderr, flush=True)
        
        # Context from Step 1
        structure_context = json.dumps(structure_data, ensure_ascii=False, indent=2) if structure_data else "No structure generated."

        step2_prompt = f"""
ROLE: Screenwriter & Visual Director.
TASK: Generate detailed scene specs based on the approved structure.

APPROVED STRUCTURE:
{structure_context}

REQUIREMENTS:
- Follow the outline exactly.
- "narrative": The action description (Korean).
- "tts_script": The spoken line (Korean). Natural, conversational.
- "image_prompt": Visual description for AI Image Generator (English). {style} style.
- "camera_work": Specific camera movement (e.g., "Close-up", "Pan Right", "Drone Shot").

OUTPUT FORMAT (JSON):
{{
  "title": "{structure_data.get('project_title', 'Untitled')}",
  "genre": "{genre}",
  "total_duration_sec": {total_duration_sec},
  "character_sheet": {json.dumps(structure_data.get('characters', {}), ensure_ascii=False)},
  "global_style": {json.dumps(structure_data.get('global_style', {}), ensure_ascii=False)},
  "scenes": [
    {{
      "scene_id": 1,
      "narrative": "주인공이 카페 문을 열고 들어온다.",
      "image_prompt": "A man opening a cafe door, webtoon style, high contrast, cinematic lighting. (Include character appearance)",
      "tts_script": "드디어 이곳인가...",
      "duration_sec": 5,
      "camera_work": "Close-up",
      "mood": "tense",
      "characters_in_scene": ["Name"]
    }}
  ],
  "youtube_opt": {{
    "title_candidates": ["Title 1", "Title 2"],
    "thumbnail_text": "Hook Text",
    "hashtags": ["#Tag1", "#Tag2"]
  }}
}}
"""
        step2_response = self._call_llm_api(step2_prompt)
        print(f"  [Step 2] Response received, starting validation...", file=sys.stderr, flush=True)
        story_data = self._validate_story_json(step2_response)

        print(f"[Story Agent] Story generated successfully.", file=sys.stderr, flush=True)
        return story_data

    def _call_llm_api(self, user_prompt: str) -> str:
        """
        Call Gemini API (v1.0 SDK) to generate content.
        """
        try:
            from google import genai
            from google.genai import types

            client = genai.Client(api_key=self.api_key)
            
            print(f"[DEBUG] Calling Gemini API (model: {self.model})", flush=True)

            response = client.models.generate_content(
                model=self.model,
                contents=user_prompt,
                config=types.GenerateContentConfig(
                    system_instruction=self.system_prompt,
                    temperature=0.7,
                    response_mime_type="application/json"
                )
            )

            response_text = response.text.strip()
            print(f"[DEBUG] Gemini API response received ({len(response_text)} chars)", flush=True)

            return response_text

        except Exception as e:
            from utils.error_manager import ErrorManager
            ErrorManager.log_error(
                "StoryAgent", 
                "Gemini API Call Failed", 
                f"{type(e).__name__}: {str(e)}", 
                severity="critical"
            )
            print(f"[ERROR] Gemini API call failed: {e}", flush=True)
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

        # Field Mapping (v2.0 -> Schema)
        if "project_title" in story_data and "title" not in story_data:
            story_data["title"] = story_data["project_title"]

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
            # -------------------------------------------------------------------------
            # Field Mapping & Normalization (v2.0 -> Schema)
            # -------------------------------------------------------------------------
            
            # 1. TTS Script -> Narration/Sentence
            if "tts_script" in scene:
                scene["narration"] = scene["tts_script"]
                # 'sentence' is required by Schema, map it
                scene["sentence"] = scene["tts_script"]
            
            # 2. Image Prompt -> Visual Description / Prompt
            if "image_prompt" in scene:
                scene["visual_description"] = scene["image_prompt"] # Legacy compatibility
                scene["prompt"] = scene["image_prompt"] # Core field
            
            # 3. Narrative -> Narrative (already matches, but good to be explicit for legacy)
            # 'narrative' is v2.0 field
            
            # 4. Camera Work check
            if "camera_work" in scene:
                # Ensure it matches Enum values roughly or leave for Pydantic to validate
                pass

            # -------------------------------------------------------------------------
            # Validation
            # -------------------------------------------------------------------------
            
            # v1.0 필수 필드 (하위 호환성) - now mapped above
            required_scene_fields_v1 = ["scene_id", "duration_sec"]
            
            # Check for at least one text field
            if "narration" not in scene and "sentence" not in scene and "tts_script" not in scene:
                # If pure visual scene, maybe allowed? But mostly we need text.
                # warning only? No, let's enforce based on schema.
                # But schema says 'sentence' is required.
                if "narrative" in scene:
                     # Fallback: use narrative as sentence if TTS is missing?
                     # No, TTS should be distinct.
                     pass 

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
