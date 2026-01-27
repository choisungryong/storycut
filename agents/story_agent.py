"""
Story Agent: Generates scene-based story JSON using LLM.
"""

import json
import os
from typing import Dict, Any
from pathlib import Path


class StoryAgent:
    """
    Generates YouTube-optimized stories in Scene JSON format.

    This agent calls an external LLM API (OpenAI, Anthropic, etc.)
    to generate structured story content.
    """

    def __init__(self, api_key: str = None, model: str = "gpt-4"):
        """
        Initialize Story Agent.

        Args:
            api_key: OpenAI or Anthropic API key
            model: LLM model to use
        """
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.model = model

        if not self.api_key:
            raise ValueError("API key is required. Set OPENAI_API_KEY environment variable.")

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

        # Build user prompt
        user_prompt = f"""Generate a story with the following parameters:

Genre: {genre}
Mood: {mood}
Style: {style}
Total Duration: {total_duration_sec} seconds
"""

        if user_idea:
            user_prompt += f"\nUser Idea: {user_idea}\n"

        user_prompt += """
Output ONLY valid JSON following the schema in your instructions.
Do not include any markdown formatting, explanations, or extra text.
"""

        # Call LLM API
        story_json = self._call_llm_api(user_prompt)

        # Validate and parse JSON
        story_data = self._validate_story_json(story_json)

        print(f"[Story Agent] Story generated: {story_data['title']}")
        print(f"   Scenes: {len(story_data['scenes'])}")

        return story_data

    def _call_llm_api(self, user_prompt: str) -> str:
        """
        Call external LLM API to generate story.

        This is a placeholder implementation.
        In production, this would call OpenAI, Anthropic, or other LLM APIs.

        Args:
            user_prompt: User's story generation request

        Returns:
            LLM response (should be JSON string)
        """
        # TODO: Implement actual API call
        # For MVP demonstration, we can use OpenAI's chat completion API

        try:
            from openai import OpenAI

            print(f"[DEBUG] Calling OpenAI API (model: {self.model}, timeout: 60s)", flush=True)
            client = OpenAI(api_key=self.api_key, timeout=60.0)

            response = client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.7,
                max_tokens=2000
            )

            print(f"[DEBUG] OpenAI API response received", flush=True)
            return response.choices[0].message.content.strip()

        except ImportError:
            # Fallback: Return example story for testing
            print("[Warning] OpenAI library not available. Using example story.")
            return self._get_example_story()
        except Exception as e:
            print(f"[Warning] LLM API call failed: {e}. Using example story.")
            return self._get_example_story()

    def _validate_story_json(self, json_string: str) -> Dict[str, Any]:
        """
        Validate and parse story JSON.

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

        for scene in story_data["scenes"]:
            required_scene_fields = ["scene_id", "narration", "visual_description", "mood", "duration_sec"]
            for field in required_scene_fields:
                if field not in scene:
                    raise ValueError(f"Missing required scene field: {field}")

        return story_data

    def _get_example_story(self) -> str:
        """
        Return an example story JSON for testing purposes.
        """
        example = {
            "title": "마지막 편지",
            "genre": "emotional",
            "total_duration_sec": 90,
            "scenes": [
                {
                    "scene_id": 1,
                    "narration": "그녀는 20년이나 늦게 그 편지를 발견했습니다.",
                    "visual_description": "A woman's hand holding an old, yellowed envelope in dim light",
                    "mood": "melancholic and regretful",
                    "duration_sec": 5
                },
                {
                    "scene_id": 2,
                    "narration": "어릴 적 살던 집 마루 밑에 숨겨져 있었죠.",
                    "visual_description": "Old wooden floorboards being lifted, dust particles in sunlight",
                    "mood": "mysterious and nostalgic",
                    "duration_sec": 6
                },
                {
                    "scene_id": 3,
                    "narration": "아버지가 실종되기 전날 쓴 것이었습니다.",
                    "visual_description": "A faded photograph of a man, silhouette against window",
                    "mood": "somber and reflective",
                    "duration_sec": 7
                },
                {
                    "scene_id": 4,
                    "narration": "그녀는 떨리는 손으로 봉투를 열었습니다.",
                    "visual_description": "Hands carefully opening an old envelope, close-up",
                    "mood": "tense and emotional",
                    "duration_sec": 5
                },
                {
                    "scene_id": 5,
                    "narration": "그 안에는 딱 한 문장이 적혀 있었습니다.",
                    "visual_description": "A handwritten note with one line of text, slightly blurred",
                    "mood": "suspenseful",
                    "duration_sec": 5
                },
                {
                    "scene_id": 6,
                    "narration": "함께해주지 못해 미안하다. 하지만 넌 혼자가 아니었단다.",
                    "visual_description": "The handwritten words coming into focus",
                    "mood": "heartbreaking and warm",
                    "duration_sec": 7
                },
                {
                    "scene_id": 7,
                    "narration": "그녀는 벽에 걸린 낡은 사진을 올려다보았습니다.",
                    "visual_description": "Woman looking at a family photo on the wall, back view",
                    "mood": "bittersweet",
                    "duration_sec": 6
                },
                {
                    "scene_id": 8,
                    "narration": "그리고 20년 만에 처음으로...",
                    "visual_description": "Soft light filtering through a window, peaceful atmosphere",
                    "mood": "hopeful and healing",
                    "duration_sec": 6
                },
                {
                    "scene_id": 9,
                    "narration": "그녀는 미소 지었습니다.",
                    "visual_description": "A gentle smile, face partially visible in soft lighting",
                    "mood": "peaceful and resolved",
                    "duration_sec": 5
                }
            ]
        }

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
