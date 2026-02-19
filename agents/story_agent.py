"""
Story Agent: Generates scene-based story JSON using LLM (Gemini 3 Pro).
"""

import json
import os
import re
from typing import Dict, Any, List
from pathlib import Path


class StoryAgent:
    """
    Generates YouTube-optimized stories in Scene JSON format.

    This agent calls Google Gemini 3 Pro API
    to generate structured story content.
    """

    def __init__(self, api_key: str = None, model: str = "gpt-5.2"):
        """
        Initialize Story Agent.

        Args:
            api_key: OpenAI API key
            model: LLM model to use (default: gpt-5.2)
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
        user_idea: str = None,
        is_shorts: bool = False,
        include_dialogue: bool = False,
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

        if include_dialogue:
            _step1_dialogue_rule = (
                "[DIALOGUE RULE]\n"
                "- If the user idea mentions dialogue, conversation, or multiple speakers (화자, 대화):\n"
                "  You MUST create at least 2-3 named characters (male + female) with distinct roles.\n"
                "  The tts_script in Step 2 MUST include [male_1], [female_1] speaker tags with actual dialogue lines.\n"
                "  DO NOT make a narrator-only story when dialogue is requested."
            )
        else:
            _step1_dialogue_rule = (
                "[NARRATOR-ONLY MODE - CRITICAL]\n"
                "- This story uses a SINGLE narrator voice only. NO character dialogue allowed.\n"
                "- Do NOT create characters for the purpose of dialogue. A single protagonist is enough.\n"
                "- The tts_script in Step 2 MUST use ONLY [narrator] tags. NO [male_1], [female_1] tags.\n"
                "- Treat this as a documentary or audiobook narration, NOT a drama with conversations."
            )

        step1_prompt = f"""
ROLE: Professional Storyboard Artist & Director.
TASK: Plan the structure for a {total_duration_sec}-second YouTube Short.
GENRE: {genre}
MOOD: {mood}
STYLE: {style}
SCENE COUNT: Approx {min_scenes}-{max_scenes} scenes.

[LANGUAGE RULE - CRITICAL]
- "project_title": 반드시 한국어로 작성 (예: "마지막 편지의 비밀")
- "logline": 반드시 한국어로 작성
- "outline" → "summary": 반드시 한국어로 작성
- character "name": 한국어 이름 사용 (예: 지민, 준혁, 서연)
- character "appearance": 영어로 작성 (이미지 생성용)

{'USER IDEA: ' + user_idea if user_idea else ''}

{_step1_dialogue_rule}

OUTPUT FORMAT (JSON):
{{
  "project_title": "한국어 제목 (Hook처럼 작동)",
  "logline": "한 문장 요약 (한국어)",
  "global_style": {{
    "art_style": "{style}",
    "color_palette": "e.g., Cyberpunk Neons",
    "visual_seed": 12345
  }},
  "characters": {{
    "STORYCUT_HERO_A": {{
      "name": "한국어 이름",
      "gender": "male/female",
      "age": "20s/30s/...",
      "appearance": "hair color+style, eye color, skin tone, face features (English)",
      "clothing_default": "specific outfit worn throughout the story (English)",
      "unique_features": "3+ distinctive marks: scars, tattoos, accessories, unusual eye color, etc. (English)",
      "role": "Protagonist/Antagonist/Supporting"
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

        # DIALOGUE FORMAT 블록: include_dialogue 여부에 따라 분기
        if include_dialogue:
            _dialogue_format = (
                "Multi-speaker dialogue is enabled. Each scene's tts_script MUST use speaker tags:\n"
                "[narrator] 어두운 밤, 한 남자가 골목을 걸어간다.\n"
                "[male_1] 누구야? 거기 서!\n"
                "[female_1] 도망쳐! 빨리!\n"
                "[narrator] 그녀의 목소리는 절박함으로 가득 차 있었다.\n\n"
                "Rules:\n"
                "- [narrator] for narration and description passages\n"
                "- [male_1], [female_1], [male_2], [female_2]... for character dialogue\n"
                "- Keep speaker IDs consistent across ALL scenes (same character = same ID)\n"
                "- Every line MUST start with a speaker tag\n"
                "- If a scene is pure narration with no character dialogue, use only [narrator]\n"
                "- Include emotional cues in parentheses when helpful: [male_1](angry) 이게 무슨 소리야!"
            )
        else:
            _dialogue_format = (
                "[NARRATOR-ONLY MODE] This is a pure narration story. STRICT RULES:\n"
                "- Use ONLY [narrator] tag. NO other speaker tags allowed.\n"
                "- Do NOT write any character dialogue lines whatsoever.\n"
                "- Do NOT use [male_1], [female_1], [male_2], [female_2] or any character tags.\n"
                "- Every single line MUST start with [narrator]\n"
                "- Example:\n"
                "  [narrator] 어두운 밤, 한 남자가 골목을 걸어간다.\n"
                "  [narrator] 그의 발소리가 빗소리에 묻혀 사라졌다.\n"
                "  [narrator] 과연 그는 어디로 향하고 있는 걸까?"
            )

        step2_prompt = f"""
ROLE: Screenwriter & Visual Director.
TASK: Generate detailed scene specs based on the approved structure.

APPROVED STRUCTURE:
{structure_context}

REQUIREMENTS:
- Follow the outline exactly.
- "narrative": 장면 설명 (반드시 한국어). 예: "지민이 카페 문을 열고 들어온다."
- "tts_script": 풍부한 스토리텔링 나레이션 (반드시 한국어, 최소 4-6문장 필수!)
- "image_prompt": Visual description for AI Image Generator (MUST BE English). {style} style.
- "camera_work": Specific camera movement (e.g., "Close-up", "Pan Right", "Drone Shot").

## DIALOGUE FORMAT (CRITICAL)
{_dialogue_format}

[CRITICAL] STORYTELLING NARRATION RULE (스토리텔링 필수):
You are a PROFESSIONAL STORYTELLER for YouTube. Each "tts_script" MUST be rich and immersive!
DO NOT write short, boring narrations like "그는 문을 열었다." ❌

Each "tts_script" MUST include (최소 4-6문장):
1. 상황 묘사: 현재 장면의 분위기와 배경을 생생하게 묘사
2. 감정 전달: 캐릭터의 내면 심리, 두려움, 희망, 절망 등을 표현
3. 긴장감/몰입: 시청자가 다음이 궁금해지도록 서스펜스 조성
4. 디테일: 구체적인 감각 묘사 (소리, 냄새, 촉감, 시각)
5. 시청자 교감: "과연 그녀는...", "하지만 그것은..." 같은 화법으로 몰입 유도

BAD tts_script (짧고 지루함 - FORBIDDEN):
- "그녀는 문을 열었다." ❌
- "편지가 도착했다." ❌
- "그는 놀랐다." ❌

GOOD tts_script (풍부한 스토리텔링 - REQUIRED):
- "빗소리가 창문을 두드리는 그 밤, 지민의 손은 떨리고 있었습니다. 20년 전 사라진 아버지... 그 이름이 적힌 편지를 손에 쥔 순간, 심장이 멎는 것 같았죠. 과연 이 편지는 진짜일까요? 아니면 누군가의 잔인한 장난일까요?" ✓
- "카페 문을 밀어젖히는 순간, 익숙한 커피 향이 코끝을 스쳤습니다. 하지만 지민의 눈에 들어온 건 향긋한 라떼가 아니었어요. 구석 자리에 앉아 있는 그 사람... 분명 죽었다고 들었던 그 사람이 거기 있었습니다." ✓

[LANGUAGE RULE - CRITICAL]
- "narrative"와 "tts_script"는 반드시 한국어로 작성할 것. 영어 금지.
- "image_prompt"만 영어로 작성 (이미지 생성 AI용).
- "title"도 반드시 한국어로 작성.

[STRICT] CHARACTER CONSISTENCY RULE:
- Refer to characters ONLY by their IDs (e.g., STORYCUT_HERO_A) in the "image_prompt".
- DO NOT describe their physical appearance (age, hair, clothes) in "image_prompt". This is already handled by the system.
- Focus ONLY on the scene's action, lighting, and composition.

[CRITICAL] DYNAMIC POSE & ACTION RULE (영상 연출 필수):
You are a FILM DIRECTOR. Each scene MUST have dynamic, cinematic poses. NO static standing poses!
Every "image_prompt" MUST include ALL of the following:
1. BODY ACTION: What is the character physically doing? (running, falling, reaching, kneeling, jumping, crawling, fighting, embracing)
2. BODY POSE: Specific posture details (leaning forward desperately, arms raised in fear, crouching low, body twisted mid-turn)
3. FACIAL EXPRESSION: Emotional state on face (eyes wide with terror, tears streaming down, gritted teeth, shocked open mouth)
4. GESTURE/HANDS: What are the hands doing? (clenched fists, trembling hands reaching out, gripping a weapon, covering mouth in shock)
5. EYE DIRECTION: Where is the character looking? (staring at camera, looking over shoulder, eyes cast downward, glaring at enemy)

BAD (Static - FORBIDDEN):
- "STORYCUT_HERO_A standing in a room" ❌
- "STORYCUT_HERO_A at the door" ❌
- "STORYCUT_HERO_A in the rain" ❌

GOOD (Dynamic - REQUIRED):
- "STORYCUT_HERO_A bursting through the door, body leaning forward mid-stride, eyes wide with desperation, hand reaching out, rain soaking through clothes" ✓
- "STORYCUT_HERO_A collapsed on knees, head thrown back in anguish, tears streaming, fists pounding the ground, dramatic low-angle shot" ✓
- "STORYCUT_HERO_A spinning around in shock, body twisted mid-turn, hand flying to mouth, eyes locked on something off-screen, dramatic backlight" ✓

OUTPUT FORMAT (JSON - title, narrative, tts_script는 반드시 한국어):
{{
  "title": "{structure_data.get('project_title', '제목 없음')}",
  "genre": "{genre}",
  "total_duration_sec": {total_duration_sec},
  "character_sheet": {json.dumps(structure_data.get('characters', {}), ensure_ascii=False)},
  "global_style": {json.dumps(structure_data.get('global_style', {}), ensure_ascii=False)},
  "scenes": [
    {{
      "scene_id": 1,
      "narrative": "STORYCUT_HERO_A가 급하게 카페 문을 밀치며 들어온다.",
      "image_prompt": "STORYCUT_HERO_A bursting through cafe door, body leaning forward in urgent motion, eyes scanning the room desperately, one hand pushing door open while other clutches a crumpled letter, rain-soaked clothes, dramatic side lighting, {style} style.",
      "tts_script": "비에 흠뻑 젖은 채로 카페 문을 밀어젖히는 순간, 지민의 심장은 미친 듯이 뛰고 있었습니다. 손에 꼭 쥔 구겨진 편지... 20년 전 사라진 아버지가 보낸 것이라는 그 편지에는 이 카페의 주소가 적혀 있었죠. 과연 이곳에서 무엇을 발견하게 될까요? 지민은 떨리는 눈으로 카페 안을 훑어보았습니다.",
      "duration_sec": 8,
      "camera_work": "Medium shot, slight low angle",
      "mood": "tense",
      "characters_in_scene": ["STORYCUT_HERO_A"]
    }}
  ],
  "youtube_opt": {{
    "title_candidates": ["한국어 제목 후보 1", "한국어 제목 후보 2"],
    "thumbnail_text": "한국어 Hook 텍스트",
    "hashtags": ["#태그1", "#태그2"]
  }}
}}
"""
        # Shorts: hook_text 필드 추가 요청
        if is_shorts:
            shorts_hook_instruction = (
                '\n[SHORTS HOOK TEXT RULE]\n'
                'This is a YouTube Shorts (9:16 vertical video). You MUST add a "hook_text" field at the top level of the JSON:\n'
                '- A short, curiosity-inducing Korean text (15 characters or less) displayed at the top of the video\n'
                '- Must make viewers want to keep watching\n'
                '- Examples: "이 남자의 정체는?", "반전 주의!", "마지막에 소름", "절대 따라하지 마세요"\n'
                '- Add "hook_text": "..." right after "title" in the output JSON\n'
            )
            step2_prompt += shorts_hook_instruction

        step2_response = self._call_llm_api(step2_prompt)
        print(f"  [Step 2] Response received, starting validation...", file=sys.stderr, flush=True)
        story_data = self._validate_story_json(step2_response)

        print(f"[Story Agent] Story generated successfully.", file=sys.stderr, flush=True)
        return story_data

    def _call_llm_api(self, user_prompt: str) -> str:
        """
        Call OpenAI API to generate content.
        """
        try:
            from openai import OpenAI

            client = OpenAI(api_key=self.api_key)

            print(f"[DEBUG] Calling OpenAI API (model: {self.model})", flush=True)

            response = client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.7,
                response_format={"type": "json_object"}
            )

            response_text = response.choices[0].message.content.strip()
            print(f"[DEBUG] OpenAI API response received ({len(response_text)} chars)", flush=True)

            return response_text

        except Exception as e:
            from utils.error_manager import ErrorManager
            ErrorManager.log_error(
                "StoryAgent",
                "OpenAI API Call Failed",
                f"{type(e).__name__}: {str(e)}",
                severity="critical"
            )
            print(f"[ERROR] OpenAI API call failed: {e}", flush=True)
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

            # v3.0: Parse dialogue lines from tts_script
            tts = scene.get("tts_script", "") or scene.get("narration", "")
            dialogue_lines = StoryAgent.parse_dialogue_lines(tts)
            if dialogue_lines and len(dialogue_lines) > 1:
                scene["dialogue_lines"] = dialogue_lines
                speakers = set(dl["speaker"] for dl in dialogue_lines)
                print(f"[Scene {idx}] Dialogue: {len(dialogue_lines)} lines, speakers: {speakers}", flush=True)

        # Extract detected speakers
        story_data["detected_speakers"] = StoryAgent.extract_speakers(story_data)
        if len(story_data["detected_speakers"]) > 1:
            print(f"[Story Validation] Detected speakers: {story_data['detected_speakers']}", flush=True)

        return story_data

    @staticmethod
    def parse_dialogue_lines(tts_script: str) -> List[Dict[str, str]]:
        """
        [speaker] text 형식의 tts_script를 DialogueLine 딕셔너리 리스트로 파싱.

        태그가 없으면 전체를 narrator로 처리 (하위호환).

        Args:
            tts_script: 화자 태그가 포함된 TTS 스크립트

        Returns:
            [{"speaker": "narrator", "text": "...", "emotion": ""}, ...]
        """
        if not tts_script or not tts_script.strip():
            return []

        lines = []
        # [speaker] 또는 [speaker](emotion) 패턴 매칭
        pattern = re.compile(r'\[([^\]]+)\](?:\(([^)]*)\))?\s*(.*)')

        has_tags = bool(re.search(r'\[[^\]]+\]', tts_script))

        if not has_tags:
            # 태그 없음 → 전체를 narrator로
            return [{"speaker": "narrator", "text": tts_script.strip(), "emotion": ""}]

        # 줄 단위 파싱
        for raw_line in tts_script.strip().split('\n'):
            raw_line = raw_line.strip()
            if not raw_line:
                continue

            match = pattern.match(raw_line)
            if match:
                speaker = match.group(1).strip()
                emotion = (match.group(2) or "").strip()
                text = match.group(3).strip()
                if text:
                    lines.append({"speaker": speaker, "text": text, "emotion": emotion})
            else:
                # 태그 없는 줄 → 이전 화자 또는 narrator
                if lines:
                    lines[-1]["text"] += " " + raw_line
                else:
                    lines.append({"speaker": "narrator", "text": raw_line, "emotion": ""})

        return lines

    @staticmethod
    def extract_speakers(story_data: Dict[str, Any]) -> List[str]:
        """
        story_data의 모든 씬에서 고유 화자 목록을 추출.

        Args:
            story_data: 스토리 JSON

        Returns:
            ["narrator", "male_1", "female_1", ...] (순서 보존)
        """
        speakers = []
        seen = set()
        for scene in story_data.get("scenes", []):
            tts = scene.get("tts_script", "") or scene.get("narration", "")
            for line in StoryAgent.parse_dialogue_lines(tts):
                s = line["speaker"]
                if s not in seen:
                    seen.add(s)
                    speakers.append(s)
        return speakers

    def analyze_script(self, raw_text: str, genre: str = "emotional", mood: str = "dramatic") -> Dict[str, Any]:
        """
        Direct Script 모드: 사용자 텍스트를 Gemini에게 보내 화자를 분석하고 태깅.

        Args:
            raw_text: 사용자 입력 스크립트 텍스트
            genre: 장르
            mood: 분위기

        Returns:
            화자 태깅된 tts_script가 포함된 story_data
        """
        prompt = f"""You are a script analyzer. Analyze the following script and add speaker tags.

SCRIPT:
{raw_text}

TASK:
1. Identify all speakers/characters in the script
2. Add speaker tags to every line: [narrator], [male_1], [female_1], etc.
3. Pure narration/description → [narrator]
4. Character dialogue → [male_1], [female_1], [male_2], etc.
5. Keep consistent speaker IDs
6. Add emotion hints in parentheses when clear: [male_1](angry)

OUTPUT FORMAT (JSON):
{{
  "tagged_script": "[narrator] 설명 텍스트...\\n[male_1] 대사...\\n[narrator] 설명...",
  "detected_speakers": ["narrator", "male_1", "female_1"]
}}

Return ONLY the JSON."""

        try:
            response_text = self._call_llm_api(prompt)
            result = json.loads(response_text)
            return result
        except Exception as e:
            print(f"[StoryAgent] Script analysis failed: {e}. Using narrator fallback.")
            return {
                "tagged_script": f"[narrator] {raw_text}",
                "detected_speakers": ["narrator"]
            }

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
