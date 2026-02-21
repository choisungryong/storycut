"""
Story Agent: Generates scene-based story JSON using LLM (Gemini 3 Pro).
"""

import json
import os
import re
from typing import Dict, Any, List
from pathlib import Path


# 장르별 플롯 씨앗 — 흥미로운 전제를 LLM에 제시하여 진부한 주제를 방지
GENRE_PLOT_SEEDS = {
    "mystery": (
        "진범은 내가 가장 신뢰했던 사람 / 20년 만에 밝혀지는 실종 사건의 진실 / "
        "피해자가 사실 가해자였다 / 내가 기억하는 사건은 처음부터 조작됐다 / "
        "완벽해 보이는 삶 뒤에 숨겨진 비밀"
    ),
    "romance": (
        "10년 뒤 재회했는데 두 사람 사이에 알 수 없는 비밀이 있었다 / "
        "원수라고 생각했던 사람이 내 편이었다 / 그가 나에게 잘해준 진짜 이유 / "
        "포기하려는 순간 상대방도 포기하려 했다 / 첫사랑이 내 기억과 완전히 달랐다"
    ),
    "thriller": (
        "나를 보호해 주던 사람이 사실 위협이었다 / 안전하다고 믿었던 공간이 함정 / "
        "내가 처음부터 조작당하고 있었다 / 도망치는 방향이 사실 함정이었다 / "
        "가장 가까운 사람이 나의 적이었다"
    ),
    "emotional": (
        "상처를 준 사람이 사실 더 큰 상처를 받고 있었다 / "
        "원망했던 사람의 진심을 뒤늦게 알게 됐다 / 마지막 이별이 사실 새로운 시작 / "
        "포기했던 꿈이 전혀 다른 방식으로 이루어졌다 / 오해 때문에 잃었던 10년"
    ),
    "horror": (
        "도움을 구했던 상대가 실제 위협이었다 / 가장 안전해 보이는 인물이 거짓 / "
        "도망치는 방향이 함정이었다 / 내가 피하려 했던 운명이 이미 시작됐다"
    ),
    "fantasy": (
        "세계를 구하러 갔지만 구해야 할 건 자기 자신이었다 / "
        "악당이 사실 또 다른 피해자였다 / 힘이라고 믿었던 것이 저주였다 / "
        "영웅이 선택받은 게 아니라 선택된 것처럼 보였을 뿐"
    ),
    "action": (
        "임무가 처음부터 함정이었다 / 믿었던 동료가 배신자 / "
        "구해야 할 상대가 진짜 적 / 내가 싸우는 이유 자체가 거짓이었다"
    ),
    "comedy": (
        "최악의 상황이 의외의 방식으로 해결됐다 / 연속 오해가 한꺼번에 풀리는 순간 / "
        "포기하려는 순간 행운이 찾아왔다 / 서로 같은 실수를 반복하던 두 사람의 결말"
    ),
    "drama": (
        "진심을 말하지 못한 채로 10년이 흘렀다 / 완벽한 관계 뒤에 숨겨진 균열 / "
        "선택의 기로에서 잘못 고른 것의 대가 / 용서하지 못한 채로 마주한 이별"
    ),
}


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

        # 장르별 플롯 씨앗 참조
        _genre_key = genre.lower().strip()
        _plot_seeds = GENRE_PLOT_SEEDS.get(_genre_key, GENRE_PLOT_SEEDS.get("drama",
            "뒤늦게 알게 되는 진실 / 믿었던 사람의 배신 / 선택의 대가 / 감추어진 비밀"))

        step1_prompt = f"""
ROLE: Master Storyteller & Film Director — 20년 경력의 단편 영상 바이럴 콘텐츠 전문가.
TASK: {total_duration_sec}초짜리 YouTube 영상을 위한 **반전과 서사가 탄탄한** 스토리 구조를 설계하라.
GENRE: {genre}
MOOD: {mood}
STYLE: {style}
SCENE COUNT: {min_scenes}~{max_scenes}개 씬

[LANGUAGE RULE - CRITICAL]
- "project_title": 반드시 한국어로 작성 (예: "마지막 편지의 비밀")
- "logline": 반드시 한국어로 작성
- "outline" → "summary": 반드시 한국어로 작성
- character "name": 한국어 이름 사용 (예: 지민, 준혁, 서연)
- character "appearance": 영어로 작성 (이미지 생성용)

{'USER IDEA: ' + user_idea if user_idea else ''}

[TIME SETTING vs CHARACTER AGE — 필수 구분]
- "N년 뒤", "미래", "2080년" 같은 시간 표현은 **세계관/시대 배경**을 의미한다. 캐릭터 나이가 아니다.
- "50년 뒤의 연인" = 50년 후 미래 세계에 사는 **젊은** 연인이다. 50세 늙은 커플이 아니다.
- "조선시대 전사" = 조선시대 배경의 **젊은** 전사다. 500살 노인이 아니다.
- 시대 배경은 global_style과 씬 비주얼에 반영하고, 캐릭터 나이는 스토리에 적합한 나이(보통 20~30대)로 설정하라.

[CREATIVE TOPIC RULE — 이것이 가장 중요]
- 진부하고 예측 가능한 전제는 절대 금지. 시청자가 "오, 이건 봐야겠다" 느끼게 만들어라.
- 장르별 강력한 플롯 씨앗 (참고, 그대로 쓰지 말 것 — 창의적으로 변형):
  {_plot_seeds}
- 주제 자체에 반전의 방향이 암시되어야 한다.
- 제목은 궁금증을 유발하는 질문형 또는 충격적 진술형이어야 한다.

[MANDATORY STORY ARC — 아웃라인 전에 먼저 정의할 것]
스토리를 설계하기 전에 반드시 5가지 요소를 확정하라:
1. hook_concept: 시청자가 스크롤을 멈추게 만드는 첫 장면/상황 (구체적으로)
2. central_conflict: 이 스토리 전체를 끌고 가는 핵심 드라마틱 질문
3. midpoint_twist: 중반부의 예상치 못한 반전 (이전 씬들의 의미를 바꾸는 것)
4. climax_revelation: 클라이맥스의 충격적 진실/반전 (복선이 쌓여서 터지는 순간)
5. resolution_emotion: 마지막 장면의 지배적 감정 (카타르시스/씁쓸함/희망/충격)

[OUTLINE RULE]
- 각 씬에 "type" 지정 필수: "hook" / "build" / "build_clue" / "twist" / "climax" / "resolution"
- HOOK: 첫 20% 씬 — 즉각적 흡입력
- BUILD: 30% — 긴장과 복선 축적 (clue를 심어라)
- TWIST: 중반 반전 — 모든 것의 의미가 바뀌는 순간
- CLIMAX: 충격 클라이맥스 — 복선이 폭발하는 지점
- RESOLUTION: 마지막 25% — 감정적 착지, 여운

{_step1_dialogue_rule}

OUTPUT FORMAT (JSON):
{{
  "project_title": "한국어 제목 (궁금증 유발 필수)",
  "logline": "한 문장 요약 (한국어) — 반전의 방향이 암시되어야 함",
  "story_arc": {{
    "hook_concept": "구체적인 오프닝 상황",
    "central_conflict": "핵심 드라마틱 질문",
    "midpoint_twist": "중반 반전 내용",
    "climax_revelation": "클라이맥스의 진실/반전",
    "resolution_emotion": "결말의 지배적 감정"
  }},
  "global_style": {{
    "art_style": "{style}",
    "color_palette": "장르/무드에 맞는 색감",
    "visual_seed": 12345
  }},
  "characters": {{
    "STORYCUT_HERO_A": {{
      "name": "한국어 이름",
      "gender": "male/female",
      "age": "20s/30s/...",
      "appearance": "hair color+style, eye color, skin tone, face features (English)",
      "clothing_default": "specific outfit worn throughout the story (English)",
      "unique_features": "REQUIRED 3+ distinctive marks with PRECISE ANATOMICAL POSITION — e.g. 'small mole below left eye', 'silver chain bracelet on right wrist', 'thin scar across left eyebrow', 'rose tattoo on inner right forearm'. Each mark must specify EXACT body location. (English)",
      "role": "Protagonist/Antagonist/Supporting"
    }}
  }},
  "outline": [
    {{ "scene_id": 1, "type": "hook", "summary": "첫 장면 요약", "estimated_duration": 6, "dramatic_purpose": "시청자 주의 집중 + 핵심 질문 제기" }},
    {{ "scene_id": 2, "type": "build_clue", "summary": "복선 심기", "estimated_duration": 5, "dramatic_purpose": "첫 번째 단서 + 긴장감 상승" }},
    {{ "scene_id": 3, "type": "twist", "summary": "중반 반전", "estimated_duration": 7, "dramatic_purpose": "예상 뒤집기 — 이전 씬 의미 변화" }},
    {{ "scene_id": 4, "type": "climax", "summary": "클라이맥스 진실", "estimated_duration": 8, "dramatic_purpose": "복선 폭발 + 최대 감정 충격" }},
    {{ "scene_id": 5, "type": "resolution", "summary": "결말 여운", "estimated_duration": 7, "dramatic_purpose": "감정 착지 + 캐릭터 변화 완성" }}
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

        # 클라이맥스/반전 씬 강화 지시
        _story_arc = structure_data.get("story_arc", {})
        _twist_note = ""
        if _story_arc.get("climax_revelation") or _story_arc.get("midpoint_twist"):
            _twist_note = f"""
[TWIST SCENE AMPLIFICATION — 반전 씬 필수 강화]
이 스토리의 반전 포인트:
- 중반 반전: {_story_arc.get("midpoint_twist", "N/A")}
- 클라이맥스 진실: {_story_arc.get("climax_revelation", "N/A")}

반전 씬(type=twist/climax)의 tts_script는:
✅ 이전 씬들의 의미가 완전히 바뀌는 순간을 강렬하게 표현
✅ "사실은...", "그때서야 깨달았습니다", "모든 것이 달라 보였습니다" 같은 충격 전환 화법 사용
✅ 복선이 폭발하는 느낌 — 시청자가 "아!" 하는 순간을 만들어라
✅ 최소 6-8문장 (다른 씬보다 길게)
✅ 말의 속도와 긴장감이 갑자기 변화하는 느낌을 나레이션으로 표현
✅ 클라이맥스 image_prompt: 충격/각성/감정 폭발을 시각적으로 표현하는 강렬한 장면
"""

        step2_prompt = f"""
ROLE: Award-winning Screenwriter & Visual Director.
TASK: 승인된 스토리 아크를 기반으로 각 씬의 상세 스크립트를 작성하라.
목표: 시청자가 끝까지 보고 감동받거나 충격받는 영상

APPROVED STRUCTURE:
{structure_context}

{_twist_note}

REQUIREMENTS:
- 아웃라인의 outline 순서와 type을 정확히 따를 것.
- "narrative": 장면 설명 (반드시 한국어). 예: "지민이 카페 문을 열고 들어온다."
- "tts_script": 풍부한 스토리텔링 나레이션 (반드시 한국어). 길이는 duration_sec에 맞출 것 (초당 약 4글자).
- "image_prompt": Visual description for AI Image Generator (MUST BE English). {style} style.
- "camera_work": Specific camera movement (e.g., "Close-up", "Pan Right", "Drone Shot").

## DIALOGUE FORMAT (CRITICAL)
{_dialogue_format}

[SCENE CONTINUITY RULE — 씬 연속성 필수. 이것을 어기면 스토리가 산산조각남]
전체 씬은 하나의 연결된 이야기여야 한다. 각 씬을 독립적 에피소드로 쓰지 말 것!
- 씬 N의 tts_script 첫 문장은 씬 N-1의 마지막 상황/감정에서 자연스럽게 이어져야 함
- 모든 씬이 같은 캐릭터, 같은 사건, 같은 시간축 위에 있어야 함
- 각 씬이 전체 이야기의 "그 다음"이어야 함 — 관계없는 새로운 상황 도입 금지
- 긴장감은 씬마다 한 단계씩 높아져야 함 (평탄한 구간 금지)

BAD (독립적 나열 — 절대 금지):
씬1: "지민은 도서관에서 오래된 편지를 발견했습니다."
씬2: "그날 밤 지민은 꿈을 꿨습니다." ← 씬1과 인과관계 없음!
씬3: "다음 날 지민은 카페에 갔습니다." ← 편지 내용 언급 없이 새 장면!

GOOD (자연스러운 연결):
씬1: "지민은 도서관 깊숙한 곳에서 바래진 편지 한 통을 발견했습니다. 발신인은... 10년 전 실종된 언니였습니다."
씬2: "떨리는 손으로 편지를 펼친 지민의 눈에 가장 먼저 들어온 건, '도망쳐'라는 단어였습니다. 그 순간 도서관의 조명이 일제히 꺼졌습니다."
씬3: "어둠 속에서 지민의 휴대폰이 울렸습니다. 화면에 뜬 번호는... 언니의 번호였습니다. 10년간 끊겨 있던 그 번호가."

[NARRATION LENGTH RULE — 가장 중요한 규칙. 반드시 지킬 것!]
목표 영상 길이: {total_duration_sec}초 / 씬 수: {min_scenes}~{max_scenes}개
→ 씬당 목표 길이: 약 {total_duration_sec // max(min_scenes, 1)}~{total_duration_sec // max(max_scenes, 1)}초

한국어 TTS는 초당 약 5.5글자를 읽는다. 각 씬의 tts_script 글자수는 반드시 다음을 지킬 것:
- 공식: tts_script 글자수 = duration_sec × 5.5 (±15% 허용)
- 예: duration_sec=5 → tts_script 23~32자 (2~3문장), duration_sec=8 → 37~51자 (3~4문장)
- 전체 씬의 tts_script 총 글자수 합계 ≈ {total_duration_sec} × 5.5 = 약 {int(total_duration_sec * 5.5)}자
- 짧은 영상({total_duration_sec}초)일수록 간결하게. 핵심만 전달.

[STORYTELLING NARRATION RULE — 필수]
각 tts_script는 시청자가 몰입할 수 있는 구체적이고 생생한 나레이션이어야 한다.
- 추상적 서술 금지 ("그는 슬펐다") → 구체적 장면으로 보여줄 것 ("그의 손이 부들부들 떨리고 있었습니다")
- 각 씬의 tts_script는 2~4개의 완전한 문장으로 구성 (마침표로 끝나는 문장)
- 씬마다 반드시 하나의 새로운 정보/사건/감정 변화가 있어야 함
- 시청자가 "그래서 어떻게 됐어?"라고 궁금해지도록 마무리할 것

핵심 요소:
1. 상황 묘사: 장면의 분위기와 배경을 감각적으로 (소리, 빛, 온도)
2. 감정 전달: 캐릭터의 내면 심리를 행동으로 보여줌
3. 긴장감: 다음이 궁금해지는 서스펜스 ("하지만...", "그 순간...", "과연...")
4. 인과관계: 이전 씬의 결과로 지금 이 상황이 발생했음을 명시

BAD (짧고 맥락 없음 — 절대 금지):
"그녀는 문을 열었다." ❌  "편지가 도착했다." ❌  "그는 놀랐다." ❌
"지민은 슬펐다. 세상은 냉정했다." ❌ (추상적, 구체성 없음)

GOOD (duration_sec=8 → 약 44자, 2~3문장, 몰입감):
"빗소리가 창문을 두드리는 그 밤, 지민의 손에 쥔 편지에는 사라진 아버지의 이름이 적혀 있었습니다. 10년간 감춰져 있던 비밀이 마침내 수면 위로 올라오는 순간이었죠." ✓

GOOD (duration_sec=5 → 약 28자, 2문장):
"그 순간, 편지 속 진실이 지민의 세계를 무너뜨렸습니다. 아버지는 사라진 게 아니었습니다." ✓

[LANGUAGE RULE - CRITICAL]
- "narrative"와 "tts_script"는 반드시 한국어로 작성할 것. 영어 금지.
- "image_prompt"만 영어로 작성 (이미지 생성 AI용).
- "title"도 반드시 한국어로 작성.

[STRICT] IMAGE PROMPT RULE:
- Do NOT use character token IDs (e.g., STORYCUT_HERO_A) in "image_prompt". The system injects character visuals automatically.
- Do NOT describe character appearance (hair, clothes, face) — the reference image handles this.
- Describe ONLY the scene action, body pose, facial expression, lighting, and composition.

[CRITICAL] DYNAMIC POSE & ACTION RULE (영상 연출 필수):
You are a FILM DIRECTOR. Each scene MUST have dynamic, cinematic poses. NO static standing poses!
Every "image_prompt" MUST include ALL of the following:
1. BODY ACTION: What is the character physically doing? (running, falling, reaching, kneeling, jumping, crawling, fighting, embracing)
2. BODY POSE: Specific posture details (leaning forward desperately, arms raised in fear, crouching low, body twisted mid-turn)
3. FACIAL EXPRESSION: Emotional state on face (eyes wide with terror, tears streaming down, gritted teeth, shocked open mouth)
4. GESTURE/HANDS: What are the hands doing? (clenched fists, trembling hands reaching out, gripping a weapon, covering mouth in shock)
5. EYE DIRECTION: Where is the character looking? (staring at camera, looking over shoulder, eyes cast downward, glaring at enemy)

BAD (Static or uses character token - FORBIDDEN):
- "a young woman standing in a room" ❌
- "a man at the door" ❌
- "a person standing in the rain" ❌

GOOD (Dynamic action, no character token - REQUIRED):
- "figure bursting through the door, body leaning forward mid-stride, eyes wide with desperation, hand reaching out, rain soaking through clothes, dramatic side lighting" ✓
- "person collapsed on knees, head thrown back in anguish, tears streaming, fists pounding the ground, dramatic low-angle shot" ✓
- "silhouette spinning around in shock, body twisted mid-turn, hand flying to mouth, eyes locked on something off-screen, dramatic backlight" ✓

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
      "narrative": "지민이 급하게 카페 문을 밀치며 들어온다.",
      "image_prompt": "figure bursting through cafe door, body leaning forward in urgent motion, eyes scanning room desperately, one hand pushing door open while other clutches a crumpled letter, dramatic side lighting, {style} style.",
      "tts_script": "비에 젖은 채 카페 문을 밀어젖히는 순간, 지민의 심장이 멎었습니다. 구겨진 편지 속 주소가 바로 이곳이었죠.",
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
                temperature=0.75,
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
