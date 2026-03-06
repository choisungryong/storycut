"""
Story Agent: Generates scene-based story JSON using LLM (Gemini 3 Pro).
"""

import json
import os
import re
from typing import Dict, Any, List
from pathlib import Path
from utils.logger import get_logger
from utils.llm_utils import parse_llm_json
logger = get_logger("story_agent")



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


# content_type → 기존 genre 시스템 매핑
CONTENT_TYPE_TO_GENRE = {
    "folktale": "fantasy",
    "myth": "fantasy",
    "historical": "drama",
    "economy": "drama",
    "documentary": "drama",
    "fiction": "emotional",
    "mystery": "mystery",
    "romance": "romance",
    "sf": "fantasy",
    "horror": "horror",
    "fairytale": "comedy",
    "educational": "drama",
}

# 콘텐츠 유형별 규칙 딕셔너리 (3중 방어 체계의 핵심)
CONTENT_TYPE_RULES = {
    "folktale": {
        "story_structure": (
            "[CONTENT TYPE: 전래동화]\n"
            "- 전통 서사 구조를 따를 것: 권선징악, 교훈, 인과응보.\n"
            "- 옛이야기 화법 사용: '옛날 옛적에...', '~했답니다', '~했다고 합니다'.\n"
            "- 한국 전래동화의 전형적 캐릭터 사용: 나무꾼, 도깨비, 선녀, 호랑이, 임금님, 효자/효녀 등.\n"
            "- 시대 배경: 조선시대 또는 그 이전 한국 전통 시대. 절대 현대 배경 금지.\n"
            "- 톤: 밝고 따뜻하며 교훈적. 공포/괴기/호러/잔인한 톤 절대 금지.\n"
            "- 결말은 반드시 교훈이 담긴 해피엔딩 또는 권선징악 구조."
        ),
        "era_context": (
            "MANDATORY VISUAL CONTEXT for image_prompt: "
            "Traditional Korean Joseon-era setting. "
            "Thatched-roof houses (초가집) or traditional Korean hanok with curved tile roofs. "
            "Korean countryside landscape with rice paddies, bamboo groves, mountain paths. "
            "Characters wear hanbok (한복). Traditional Korean props: 절구, 도리깨, 부채, 등잔. "
            "CRITICAL — Korean folklore creature/character visual dictionary (image_prompt에 반드시 이 묘사를 사용할 것): "
            "도깨비(dokkaebi) = Korean mythical being with horn(s) on head, muscular humanoid, "
            "wearing tiger-skin loincloth (호피 바지), carrying a magic wooden club (도깨비 방망이), "
            "playful tricksters with reddish or bluish skin. NOT Western goblins, NOT green goblins. "
            "호랑이 = Korean folklore tiger, large majestic Korean tiger with warm amber fur, "
            "often depicted as wise or humorous, sometimes wearing a traditional hat or smoking a pipe (담뱃대), "
            "anthropomorphized with expressive human-like facial expressions. NOT a ferocious wild beast. "
            "선녀(seonnyeo) = Korean celestial fairy/maiden, ethereal beauty wearing flowing pastel-colored "
            "cheonuimu (천의무봉) heavenly garments with long trailing ribbons, floating gracefully, "
            "with a serene gentle expression. NOT a Western fairy with butterfly wings. "
            "까치(magpie) = Korean magpie (까치), black and white plumage with iridescent blue-green tail, "
            "often depicted as a helpful messenger bird in Korean folklore. "
            "토끼 = Korean folklore rabbit, cute white rabbit often depicted as clever trickster, "
            "anthropomorphized with human-like expressions, sometimes standing upright. "
            "용왕(Dragon King) = Korean sea dragon king, dignified elderly figure wearing "
            "royal dragon-embroidered hanbok robes and a golden crown, with dragon motifs, "
            "residing in an underwater palace. NOT a Western dragon. "
            "구미호(gumiho) = Korean nine-tailed fox spirit, beautiful woman in elegant hanbok "
            "with fox ears and nine tails visible, seductive but dangerous. NOT a Japanese kitsune style."
        ),
        "image_context": (
            "traditional Korean Joseon-era village, hanok architecture, hanbok clothing, Korean countryside, "
            "Korean folklore characters in traditional Korean visual style"
        ),
        "avoid": (
            "modern buildings, cars, smartphones, European architecture, Western clothing, suits, neon lights, skyscrapers, "
            "Western goblin, European goblin, green goblin, orc, troll, elf, dwarf, Western fairy with wings, "
            "Western dragon, Japanese yokai style, Western fantasy creatures, "
            "dark horror atmosphere, grotesque imagery, gore, blood, scary monsters"
        ),
        "character_design": (
            "- 이 영상은 어린이가 주 시청자이다. 캐릭터 외형은 아이들이 친근하게 느낄 수 있도록 따뜻하고 부드럽게 묘사할 것.\n"
            "- appearance 필드에 추상적/성인 표현 절대 금지: 'idol-level', 'celebrity-level', 'stunning', 'gorgeous', 'seductive', 'sexy' 등 사용 금지.\n"
            "- appearance 필드에는 아이들이 이해할 수 있는 구체적이고 쉬운 표현 사용:\n"
            "  좋은 예: 'kind-looking young man with bright round eyes and a warm smile, neat black topknot hair'\n"
            "  나쁜 예: 'idol-level handsome Joseon-era woodcutter with intense gaze and defined jawline'\n"
            "- 인간 캐릭터는 반드시 한복(hanbok) 착용. 현대 의상 절대 금지.\n"
            "- 전통 한국식 헤어스타일: 남성은 상투/갓, 여성은 댕기/비녀.\n"
            "- clothing_default에 한복 종류를 구체적으로 명시 (예: 'white jeogori with indigo chima', 'blue dopo overcoat with gat hat').\n"
            "- appearance 필드에 반드시 해당 캐릭터의 한국 전래동화 시각 특징을 영어로 상세 기술할 것:\n"
            "  도깨비 → 'Korean dokkaebi, horn on head, muscular, reddish skin, tiger-skin loincloth, magic wooden club'\n"
            "  호랑이 → 'Korean folklore tiger, majestic warm amber fur, wise expressive face, anthropomorphized'\n"
            "  선녀 → 'Korean celestial maiden (seonnyeo), ethereal, flowing heavenly garments with ribbons'\n"
            "  까치 → 'Korean magpie, black-white with iridescent blue-green tail, helpful messenger'\n"
            "  토끼 → 'Korean folklore rabbit, cute white, clever, anthropomorphized with human expressions'\n"
            "  용왕 → 'Korean Dragon King, dignified elderly in royal dragon-embroidered hanbok, golden crown'\n"
            "  구미호 → 'Korean gumiho, beautiful woman in elegant hanbok, fox ears, nine tails'\n"
            "- 전래동화 캐릭터 외형은 친근하고 따뜻한 느낌. 무섭거나 괴기한 외형 금지.\n"
            "- 동물 캐릭터는 의인화하여 표정이 풍부하고 감정이 드러나야 한다."
        ),
        "plot_seeds": (
            "마법의 보은 / 동물이 은혜를 갚다 / 게으른 자의 응징 / "
            "현명한 판결 / 도깨비의 장난 / 효도의 기적"
        ),
        # --- 프롬프트 슬롯 오버라이드 ---
        "role_override": (
            "한국 전래동화 전문 이야기꾼. "
            "아이들에게 옛이야기를 따뜻하고 재미있게 들려주는 20년 경력의 동화 작가."
        ),
        "task_override": (
            "한국 전래동화 영상을 위한 **권선징악과 교훈이 담긴** 옛이야기 구조를 설계하라."
        ),
        "topic_note_override": (
            "- 원작의 교훈과 메시지를 충실히 살려라.\n"
            "- 제목은 전래동화다운 따뜻한 제목 (예: '콩쥐팥쥐', '해와 달이 된 오누이').\n"
            "- 자극적이거나 충격적인 제목 금지."
        ),
        "step1_rules_override": (
            "[전래동화 서사 구조 — 반드시 따를 것]\n"
            "스토리를 설계하기 전에 반드시 5가지 요소를 확정하라:\n"
            "1. hook_concept: 이야기의 도입 상황 ('옛날 옛적에...'로 시작하는 배경 설정)\n"
            "2. central_conflict: 주인공이 겪는 핵심 시련이나 과제\n"
            "3. midpoint_twist: 도움을 받거나 시련이 심화되는 전환점\n"
            "4. climax_revelation: 선행/악행의 결과가 드러나는 권선징악의 순간\n"
            "5. resolution_emotion: 결말의 교훈과 감정 (따뜻함/뿌듯함/안도)\n\n"
            "[OUTLINE RULE]\n"
            "- 각 씬에 'type' 지정 필수: 'hook' / 'build' / 'build_clue' / 'twist' / 'climax' / 'resolution'\n"
            "- HOOK: 도입 — '옛날 옛적에' 상황 설정과 인물 소개\n"
            "- BUILD: 전개 — 시련/과제 발생, 캐릭터의 선택과 행동\n"
            "- TWIST: 전환 — 도움(동물, 신선 등)이 나타나거나 상황 반전\n"
            "- CLIMAX: 절정 — 선행의 보상 또는 악행의 응징이 드러남\n"
            "- RESOLUTION: 교훈 — 따뜻한 마무리와 이야기의 교훈 전달\n\n"
            "[전래동화 서사 규칙]\n"
            "- 매 씬은 이야기 흐름에 기여해야 한다. 단, 전래동화 특유의 여유로운 전개 허용.\n"
            "- 배경/인물 소개 씬은 전래동화에서 자연스럽다. 무리한 사건 삽입 금지.\n"
            "- 교훈과 인과응보가 명확해야 한다: 착한 행동 → 보상, 나쁜 행동 → 벌.\n"
            "- 잔인하거나 무서운 벌 금지. 아이들이 무섭지 않게 표현.\n"
            "- 복선(foreshadowing)은 의무가 아니다. 전래동화는 단순명쾌한 구조가 미덕.\n"
            "- 결말은 반드시 교훈이 담긴 해피엔딩 또는 권선징악 마무리."
        ),
        "step2_rules_override": (
            "[전래동화 나레이션 톤 — 반드시 통일할 것]\n"
            "전체 영상에서 나레이션 어미를 옛이야기체로 통일. 혼용 절대 금지!\n"
            "→ 기본 어미: '~했답니다', '~했다고 합니다', '~였대요'\n"
            "→ 도입: '옛날 옛적에...', '아주 먼 옛날...'\n"
            "→ 전환: '그런데 말이에요,', '그때였답니다,', '바로 그때,'\n"
            "→ 마무리: '~했답니다', '그래서 ~하게 되었답니다'\n"
            "→ 절대 금지: 같은 씬에서 '~했다'(해라체)와 '~했답니다' 혼용\n"
            "BAD: '콩쥐가 문을 열었다. 그 안에는 선물이 있었답니다.' ❌ (해라체+옛이야기체 혼용)\n"
            "GOOD: '콩쥐가 문을 열었답니다. 그 안에는 반짝반짝 빛나는 선물이 있었대요.' ✓\n\n"
            "[씬 끝 연결 규칙]\n"
            "- 매 씬 끝은 이야기꾼이 다음을 들려주는 자연스러운 연결:\n"
            "  '그런데 말이에요...' / '그래서 어떻게 되었냐면요...' / '그 다음에는요...'\n"
            "  (서스펜스/충격 전환이 아닌, 이야기꾼의 따뜻한 연결)\n"
            "- resolution 씬: 교훈을 명확히 전달하는 따뜻한 마무리.\n\n"
            "[감정 흐름]\n"
            "- 전래동화는 감정이 급격히 상승하지 않는다. 자연스럽고 따뜻한 흐름.\n"
            "- 공포/충격/극도의 긴장 대신: 호기심, 걱정, 안도, 기쁨, 뿌듯함.\n"
            "- 클라이맥스도 '충격'이 아닌 '교훈이 드러나는 순간'으로 표현.\n\n"
            "[나레이션 어휘 수준]\n"
            "- 어려운 한자어나 추상적 표현 금지. 아이들이 이해할 수 있는 쉬운 말.\n"
            "- 구체적이고 감각적인 묘사: '예쁜 꽃'보다 '빨간 장미꽃이 활짝 피어 있었답니다'\n"
            "BAD: '운명의 전환점이 도래했습니다.' ❌ (한자어 과다, 아동 부적합)\n"
            "GOOD: '바로 그때, 신기한 일이 벌어졌답니다.' ✓ (쉬운 말, 옛이야기체)"
        ),
        "enable_twist_note": False,
        "appearance_guide": (
            "Warm, child-friendly appearance in English. "
            "Describe: traditional Korean hairstyle (상투/댕기/비녀), "
            "face features (round kind face, bright round eyes, warm smile), "
            "hanbok type and color. "
            "NO idol-level, celebrity, stunning, gorgeous, seductive. "
            "Must feel approachable and warm to children."
        ),
        "age_guide": "story-appropriate (child/young/elderly/non-human creature)",
        "features_guide": (
            "2-3 simple visual markers a child would notice — "
            "e.g. 'red hair ribbon', 'wooden club in hand', 'tiger-stripe pattern', "
            "'blue hanbok with white trim'. NO scars, tattoos, moles, birthmarks."
        ),
    },
    "myth": {
        "story_structure": (
            "[CONTENT TYPE: 신화/전설]\n"
            "- 신화적 서사 구조: 영웅의 여정, 신들의 개입, 운명과 시련.\n"
            "- 서사시적 화법 사용: 장엄하고 격식 있는 톤.\n"
            "- TOPIC에서 문화권을 추론하라:\n"
            "  한국 신화(단군, 주몽, 삼국유사) → 한국 고대 배경\n"
            "  그리스 신화 → 고대 그리스 배경\n"
            "  북유럽 신화 → 바이킹/북유럽 배경\n"
            "- 해당 문화권의 시대와 세계관을 반드시 유지."
        ),
        "era_context": (
            "MANDATORY VISUAL CONTEXT for image_prompt: "
            "Ancient mythological setting matching the story's cultural origin. "
            "If Korean myth: ancient Korean kingdom with palace architecture, traditional armor, sacred mountains. "
            "If Greek myth: marble temples, olive groves, ancient Mediterranean. "
            "If Norse myth: fjords, longhouses, runic carvings."
        ),
        "image_context": "ancient mythological setting, epic scale, sacred/divine atmosphere",
        "avoid": "modern elements, contemporary clothing, urban environments, technology",
        "character_design": (
            "- 해당 문화권의 전통 의상/갑옷 착용 필수.\n"
            "- 신적 존재는 신성한 아우라/후광 등 시각적 표현.\n"
            "- clothing_default에 문화권에 맞는 구체적 의상 명시."
        ),
        "plot_seeds": (
            "신의 시련을 통과하는 영웅 / 금기를 어긴 대가 / 신과 인간의 사랑 / "
            "예언된 운명에 맞서는 자 / 세계를 창조하는 희생"
        ),
    },
    "historical": {
        "story_structure": (
            "[CONTENT TYPE: 역사 이야기]\n"
            "- 시대 정확성 필수. TOPIC에서 시대를 추론하라.\n"
            "- 역사적 사실을 바탕으로 하되, 드라마적 각색 허용.\n"
            "- 해당 시대의 사회 구조, 문화, 언어 습관을 반영.\n"
            "- 시대착오적 요소 절대 금지 (해당 시대에 없는 기술/문화/사물)."
        ),
        "era_context": (
            "MANDATORY VISUAL CONTEXT for image_prompt: "
            "Historically accurate setting matching the story's time period. "
            "Architecture, clothing, props, and landscape must be period-appropriate. "
            "Infer the era from TOPIC and apply correct historical visual details."
        ),
        "image_context": "historically accurate period setting, era-appropriate architecture and clothing",
        "avoid": "anachronistic elements, modern technology in historical settings, wrong-era architecture",
        "character_design": (
            "- 해당 시대의 의상 착용 필수.\n"
            "- 사회적 지위에 맞는 복장 (왕족, 무사, 평민 등).\n"
            "- clothing_default에 시대와 신분에 맞는 의상 구체적 명시."
        ),
        "plot_seeds": (
            "역사의 전환점에 선 인물의 선택 / 알려지지 않은 영웅의 이야기 / "
            "권력과 의리의 충돌 / 시대를 앞서간 인물의 비극"
        ),
    },
    "economy": {
        "story_structure": (
            "[CONTENT TYPE: 경제/비즈니스]\n"
            "- 다큐멘터리 해설체 사용. 데이터와 사실 기반.\n"
            "- 경제 사건/기업 스토리를 극적으로 재구성.\n"
            "- 수치, 금액, 시장 상황 등 구체적 디테일 포함.\n"
            "- 현대 배경이 기본. TOPIC에서 시대를 추론."
        ),
        "era_context": (
            "MANDATORY VISUAL CONTEXT for image_prompt: "
            "Modern business/financial environment. "
            "City skylines, office buildings, trading floors, boardrooms. "
            "Professional business attire, graphs and charts as visual elements."
        ),
        "image_context": "modern business environment, financial district, professional corporate setting",
        "avoid": "fantasy elements, medieval settings, magic, supernatural",
        "character_design": (
            "- 비즈니스 복장 (정장, 셔츠, 비즈니스 캐주얼).\n"
            "- 현대적이고 프로페셔널한 외형.\n"
            "- clothing_default에 비즈니스 복장 명시."
        ),
        "plot_seeds": (
            "누구도 예상 못 한 시장의 붕괴 / 작은 회사가 거인을 이기다 / "
            "한 번의 결정이 수조 원을 날리다 / 위기 속에서 기회를 잡은 자"
        ),
    },
    "documentary": {
        "story_structure": (
            "[CONTENT TYPE: 실화/다큐]\n"
            "- 사실 기반, 차분하고 객관적인 해설체.\n"
            "- 실제 사건을 극적으로 재구성하되 사실 왜곡 금지.\n"
            "- 포토저널리스틱 스타일의 시각적 접근."
        ),
        "era_context": (
            "MANDATORY VISUAL CONTEXT for image_prompt: "
            "Photojournalistic style. Realistic, documentary-grade lighting. "
            "Setting matches the actual event's time and location."
        ),
        "image_context": "photojournalistic documentary style, realistic setting, natural lighting",
        "avoid": "fantasy elements, exaggerated expressions, cartoon style, supernatural",
        "character_design": (
            "- 실제 시대/상황에 맞는 현실적 의상.\n"
            "- 과장된 외형이나 판타지 요소 금지."
        ),
        "plot_seeds": (
            "은폐된 진실이 드러나는 순간 / 평범한 사람의 비범한 행동 / "
            "재난 속 인간의 선택 / 역사가 감춘 이야기"
        ),
        # --- 프롬프트 슬롯 오버라이드 ---
        "role_override": (
            "다큐멘터리 나레이터 겸 작가. "
            "사실의 무게감으로 시청자를 사로잡는 전문가."
        ),
        "task_override": (
            "다큐멘터리 영상을 위한 **사실 기반의 탄탄한** 서사 구조를 설계하라."
        ),
        "topic_note_override": (
            "- 사실에 기반한 스토리를 구성하라. 사실 왜곡 금지.\n"
            "- 제목은 사건의 핵심을 담되 시청자의 궁금증을 유발해야 한다.\n"
            "- 허구적 반전보다 사실 자체의 무게감으로 몰입시켜라."
        ),
        "step2_rules_override": (
            "[다큐멘터리 나레이션 톤]\n"
            "→ 기본 어미: '~였습니다', '~합니다' (차분한 해설체)\n"
            "→ 강조: '~였죠', '~거든요' (감정 절제하면서 사실 전달)\n"
            "→ 혼용 금지: '~했다'(해라체)와 '~했습니다' 혼용 금지\n\n"
            "[씬 연결]\n"
            "- 매 씬 끝은 다음 사실/정보로의 자연스러운 연결.\n"
            "  '하지만 이건 시작에 불과했습니다.' / '더 놀라운 사실이 기다리고 있었죠.'\n"
            "- 서스펜스보다 사실의 무게감으로 다음 씬을 당겨라.\n\n"
            "[감정 흐름]\n"
            "- 차분한 해설 속에서 사실의 충격이 자연스럽게 드러나는 흐름.\n"
            "- 과장된 감정 표현 지양. 사실 자체가 감정을 만들게 하라."
        ),
        "appearance_guide": (
            "Realistic, natural appearance in English. "
            "Describe: realistic hair, face, build. "
            "NO exaggerated beauty, idol-level, celebrity. "
            "Ordinary, believable people."
        ),
    },
    "fiction": {
        "story_structure": "",
        "era_context": "",
        "image_context": "",
        "avoid": "",
        "character_design": "",
        "plot_seeds": "",
    },
    "mystery": {
        "story_structure": (
            "[CONTENT TYPE: 미스터리/스릴러]\n"
            "- 단서→추리→반전 구조. 복선 배치 필수.\n"
            "- 서스펜스와 긴장감 유지. 모든 씬에 미스터리 요소."
        ),
        "era_context": (
            "MANDATORY VISUAL CONTEXT for image_prompt: "
            "Dark, atmospheric setting with dramatic shadows and mystery ambience. "
            "Dim lighting, rain, fog, noir-style visual elements."
        ),
        "image_context": "dark atmospheric mystery setting, dramatic shadows, suspenseful mood",
        "avoid": "bright cheerful colors, cartoon style, cute elements",
        "character_design": (
            "- 느와르/서스펜스에 어울리는 의상: 트렌치코트, 어두운 색조 정장, 비에 젖은 외투 등.\n"
            "- 표정에 긴장감/경계심/의심 반영.\n"
            "- clothing_default에 어두운 톤의 구체적 의상 명시 (예: 'dark trench coat over black turtleneck')."
        ),
        "plot_seeds": "",
    },
    "romance": {
        "story_structure": (
            "[CONTENT TYPE: 로맨스/감성]\n"
            "- 감성적 서사. 캐릭터 간 감정 변화가 핵심.\n"
            "- 따뜻하고 감성적인 톤. 섬세한 감정 묘사."
        ),
        "era_context": (
            "MANDATORY VISUAL CONTEXT for image_prompt: "
            "Warm, emotionally resonant setting. Soft golden lighting, warm color tones. "
            "Romantic atmosphere with cherry blossoms, sunset, cafe, rain."
        ),
        "image_context": "warm romantic atmosphere, soft golden lighting, emotionally resonant setting",
        "avoid": "gore, horror elements, dark oppressive atmosphere",
        "character_design": (
            "- 감성적이고 따뜻한 외형. 부드러운 표정과 눈빛.\n"
            "- 세련되고 감각적인 의상: 캐주얼 세미포멀, 니트, 코트, 원피스 등.\n"
            "- clothing_default에 따뜻한 색조의 구체적 의상 명시 (예: 'cream knit sweater with camel wool coat')."
        ),
        "plot_seeds": "",
    },
    "sf": {
        "story_structure": (
            "[CONTENT TYPE: SF]\n"
            "- 미래적 세계관 필수. 과학/기술 기반 설정.\n"
            "- TOPIC에서 SF 하위 장르 추론: 사이버펑크, 우주, AI, 시간여행 등.\n"
            "- 세계관 설정이 스토리 전반에 일관되게 유지되어야 함."
        ),
        "era_context": (
            "MANDATORY VISUAL CONTEXT for image_prompt: "
            "Futuristic sci-fi setting. Neon-lit cityscapes, holographic displays, "
            "advanced technology, sleek futuristic architecture, space stations, starships."
        ),
        "image_context": "futuristic sci-fi setting, advanced technology, neon-lit cityscape or space environment",
        "avoid": "medieval elements, traditional/historical clothing, horses, castles, swords, magic wands",
        "character_design": (
            "- TOPIC에서 SF 서브장르를 추론하여 의상/외형을 결정할 것:\n"
            "  사이버펑크 → 네온 액센트 의상, 사이버네틱 임플란트, 테크웨어, LED 장식\n"
            "  우주/스페이스 → 스페이스슈트, 선내복, 함장 유니폼, 우주선 배지\n"
            "  디스토피아 → 낡은 전투복, 저항군 복장, 패치워크 방호복\n"
            "  AI/로봇 → 메탈릭 요소, 홀로그램 인터페이스, 첨단 유니폼\n"
            "- appearance 필드에 반드시 'futuristic' 키워드 포함.\n"
            "- clothing_default에 SF 세계관에 맞는 구체적 의상 명시 (예: 'black techwear jacket with neon blue trim and cybernetic arm implant').\n"
            "- 중세/전통 의상 절대 금지. 미래적이지 않은 일반 캐주얼 의상도 지양."
        ),
        "plot_seeds": (
            "AI가 인간을 넘어서는 순간 / 시간여행의 패러독스 / "
            "외계 문명과의 첫 접촉 / 디스토피아에서의 저항"
        ),
    },
    "horror": {
        "story_structure": (
            "[CONTENT TYPE: 호러/공포]\n"
            "- 공포와 불안감 조성이 핵심. 점진적 공포 상승.\n"
            "- 심리적 공포 > 물리적 공포. 분위기와 암시 활용."
        ),
        "era_context": (
            "MANDATORY VISUAL CONTEXT for image_prompt: "
            "Dark, unsettling horror atmosphere. Deep shadows, eerie lighting, "
            "decayed/abandoned environments, fog, moonlight, isolated locations."
        ),
        "image_context": "dark eerie horror atmosphere, deep shadows, unsettling environment",
        "avoid": "bright cheerful colors, cute cartoon style, comedy elements",
        "character_design": (
            "- 불안감을 주는 외형 요소: 창백한 피부, 어두운 눈 밑 그림자, 긴장된 표정.\n"
            "- 어둡고 칙칙한 톤의 의상: 낡은 옷, 피 묻은 흔적, 찢어진 의복 등.\n"
            "- clothing_default에 호러 분위기의 구체적 의상 명시 (예: 'worn pale hospital gown with dark stains')."
        ),
        "plot_seeds": "",
    },
    "fairytale": {
        "story_structure": (
            "[CONTENT TYPE: 동화]\n"
            "- 아동 친화적 동화체 사용: '옛날 옛적에...', 따뜻하고 교훈적.\n"
            "- 밝고 긍정적인 결말 필수. 교훈/메시지 포함.\n"
            "- 폭력, 공포, 어두운 요소 절대 금지.\n"
            "- 쉽고 간결한 문장. 아이들이 이해할 수 있는 수준."
        ),
        "era_context": (
            "MANDATORY VISUAL CONTEXT for image_prompt: "
            "Bright, colorful storybook illustration style. "
            "Warm pastel colors, soft lighting, whimsical and cute elements. "
            "Friendly characters, rounded shapes, magical sparkles."
        ),
        "image_context": "bright colorful storybook style, warm pastel colors, cute whimsical elements",
        "avoid": "violence, blood, horror, dark shadows, scary elements, weapons, death, gore",
        "character_design": (
            "- 귀엽고 친근한 외형. 과장된 큰 눈, 둥근 얼굴.\n"
            "- 밝고 화사한 색감의 의상.\n"
            "- 무서운 외형 요소 금지."
        ),
        "plot_seeds": (
            "용기를 찾아 떠나는 모험 / 서로 다른 친구들의 우정 / "
            "작은 친절이 만드는 큰 기적 / 두려움을 극복하는 이야기"
        ),
        # --- 프롬프트 슬롯 오버라이드 ---
        "role_override": (
            "동화 전문 이야기꾼. "
            "아이들에게 꿈과 희망을 주는 따뜻한 이야기를 만드는 전문가."
        ),
        "task_override": (
            "아이들을 위한 동화 영상을 위한 **꿈과 교훈이 담긴** 이야기 구조를 설계하라."
        ),
        "topic_note_override": (
            "- 이야기의 교훈과 메시지를 살려라.\n"
            "- 제목은 동화다운 따뜻하고 상상력 넘치는 제목이어야 한다.\n"
            "- 자극적이거나 충격적인 제목 금지."
        ),
        "step1_rules_override": (
            "[동화 서사 구조 — 반드시 따를 것]\n"
            "스토리를 설계하기 전에 반드시 5가지 요소를 확정하라:\n"
            "1. hook_concept: 이야기의 시작 상황 (흥미로운 세계/캐릭터 소개)\n"
            "2. central_conflict: 주인공이 해결해야 할 과제나 모험\n"
            "3. midpoint_twist: 예상치 못한 도움이나 새로운 도전\n"
            "4. climax_revelation: 용기/우정/사랑으로 문제를 해결하는 순간\n"
            "5. resolution_emotion: 결말의 교훈과 감정 (기쁨/뿌듯함/따뜻함)\n\n"
            "[OUTLINE RULE]\n"
            "- 각 씬에 'type' 지정: 'hook' / 'build' / 'twist' / 'climax' / 'resolution'\n"
            "- HOOK: 도입 — 매력적인 세계와 캐릭터 소개\n"
            "- BUILD: 전개 — 모험/도전 시작, 친구 만남\n"
            "- TWIST: 전환 — 예상 밖의 도움이나 시련\n"
            "- CLIMAX: 절정 — 용기와 지혜로 문제 해결\n"
            "- RESOLUTION: 교훈 — 따뜻한 마무리와 성장\n\n"
            "[동화 서사 규칙]\n"
            "- 밝고 긍정적인 톤 유지. 어두운 분위기 금지.\n"
            "- 교훈이 자연스럽게 드러나야 한다. 직접적 설교 금지.\n"
            "- 복선은 의무가 아니다. 단순하고 명쾌한 구조가 동화의 미덕.\n"
            "- 결말은 반드시 밝고 희망적인 마무리."
        ),
        "step2_rules_override": (
            "[동화 나레이션 톤 — 반드시 통일할 것]\n"
            "전체 영상에서 나레이션 어미를 동화체로 통일. 혼용 절대 금지!\n"
            "→ 기본 어미: '~했어요', '~였어요', '~이에요'\n"
            "→ 도입: '옛날 옛적에...', '어느 날...'\n"
            "→ 전환: '그런데요,', '그때였어요,', '바로 그 순간,'\n"
            "→ 마무리: '~했어요', '~하게 되었어요'\n"
            "→ 절대 금지: '~했다'(해라체)와 '~했어요' 혼용\n\n"
            "[씬 끝 연결 규칙]\n"
            "- 매 씬 끝은 자연스러운 이야기 연결:\n"
            "  '그런데요...' / '그래서 어떻게 되었냐면요...' / '그 다음에는요...'\n"
            "- resolution 씬: 교훈과 희망의 따뜻한 마무리.\n\n"
            "[감정 흐름]\n"
            "- 밝고 따뜻한 감정 흐름. 공포/충격/극도의 긴장 금지.\n"
            "- 호기심 → 설렘 → 걱정 → 안도 → 기쁨의 자연스러운 순환.\n\n"
            "[나레이션 어휘 수준]\n"
            "- 아이들이 이해할 수 있는 쉬운 말. 한자어/추상 표현 금지.\n"
            "BAD: '운명적 만남이 성사되었습니다.' ❌\n"
            "GOOD: '드디어 둘은 만나게 되었어요!' ✓"
        ),
        "enable_twist_note": False,
        "appearance_guide": (
            "Cute, friendly appearance in English. "
            "Big expressive eyes, round soft face, bright colorful outfit. "
            "Describe: hair color+style, eye shape, clothing color/type. "
            "NO idol-level, celebrity, stunning, gorgeous. Child-friendly only."
        ),
        "age_guide": "story-appropriate (child/young/animal character)",
        "features_guide": (
            "2-3 simple visual markers — e.g. 'sparkly blue hat', "
            "'red cape', 'flower-shaped hairpin'. NO scars, tattoos, moles."
        ),
    },
    "educational": {
        "story_structure": (
            "[CONTENT TYPE: 교육 콘텐츠]\n"
            "- 학습 목적. 쉬운 설명과 명확한 구조.\n"
            "- 핵심 개념을 스토리에 자연스럽게 녹여낼 것.\n"
            "- 시청자가 무언가를 배우고 가야 함."
        ),
        "era_context": (
            "MANDATORY VISUAL CONTEXT for image_prompt: "
            "Clean, bright educational visual style. "
            "Clear compositions, well-lit scenes, infographic-like clarity."
        ),
        "image_context": "clean bright educational style, clear well-lit compositions",
        "avoid": "violence, horror, dark atmosphere, complex abstract imagery",
        "character_design": (
            "- 깨끗하고 밝은 이미지의 캐릭터.\n"
            "- 학습 환경에 적합한 의상."
        ),
        "plot_seeds": (
            "궁금증에서 시작되는 탐험 / 실수에서 배우는 교훈 / "
            "과학 원리가 만드는 마법 같은 순간"
        ),
        # --- 프롬프트 슬롯 오버라이드 ---
        "role_override": (
            "교육 콘텐츠 전문 작가. "
            "지식을 재미있는 스토리에 녹여 시청자가 자연스럽게 배우게 만드는 전문가."
        ),
        "task_override": (
            "교육 영상을 위한 **학습 목표가 자연스럽게 녹아든** 스토리 구조를 설계하라."
        ),
        "topic_note_override": (
            "- 핵심 학습 목표를 스토리에 자연스럽게 녹여라.\n"
            "- 제목은 궁금증을 유발하되 학습 주제가 드러나야 한다.\n"
            "- 반전보다 '아하!' 하는 깨달음의 순간을 만들어라."
        ),
        "step1_rules_override": (
            "[교육 콘텐츠 서사 구조]\n"
            "스토리를 설계하기 전에 반드시 5가지 요소를 확정하라:\n"
            "1. hook_concept: 궁금증을 유발하는 질문이나 상황\n"
            "2. central_conflict: 해결해야 할 학습 과제/질문\n"
            "3. midpoint_twist: 새로운 정보나 관점이 등장하는 전환점\n"
            "4. climax_revelation: '아하!' 핵심 개념이 이해되는 순간\n"
            "5. resolution_emotion: 배움의 기쁨과 성취감\n\n"
            "[OUTLINE RULE]\n"
            "- 각 씬에 'type' 지정: 'hook' / 'build' / 'build_clue' / 'twist' / 'climax' / 'resolution'\n"
            "- HOOK: 흥미로운 질문/현상 제시\n"
            "- BUILD: 개념 탐구, 단서 제시\n"
            "- TWIST: 예상 밖의 정보나 실험 결과\n"
            "- CLIMAX: 핵심 원리 이해의 순간\n"
            "- RESOLUTION: 배운 것의 정리와 적용\n\n"
            "[교육 서사 규칙]\n"
            "- 개념 설명 씬은 허용. 단, 시각적 비유와 예시를 반드시 활용.\n"
            "- 복선보다 '단서 제시 → 종합' 구조가 적합.\n"
            "- 매 씬이 학습 목표에 기여해야 한다."
        ),
        "step2_rules_override": (
            "[교육 나레이션 톤]\n"
            "→ 기본 어미: '~입니다', '~합니다' (명확한 해설체)\n"
            "→ 친근한 톤: '~인데요,', '~거든요', '~죠?' 허용\n"
            "→ 혼용 금지: '~했다'(해라체)와 '~입니다' 혼용 금지\n\n"
            "[씬 연결]\n"
            "- 매 씬 끝은 다음 학습 단계로의 자연스러운 연결.\n"
            "  '그렇다면...', '여기서 한 가지 더...', '그런데 재미있는 건요...'\n\n"
            "[감정 흐름]\n"
            "- 호기심 → 탐구 → 놀라움 → 이해 → 성취감의 흐름.\n"
            "- 공포/충격/긴장보다 궁금증과 발견의 즐거움."
        ),
        "appearance_guide": (
            "Clean, bright, approachable appearance in English. "
            "Friendly face, neat clothing appropriate to the learning context. "
            "NO idol-level, celebrity, stunning. Natural and relatable."
        ),
    },
}

# ── 프롬프트 슬롯 기본값 (오버라이드가 없을 때 사용) ──
_DEFAULT_ROLE_STEP1 = "Master Storyteller & Film Director — 20년 경력의 단편 영상 바이럴 콘텐츠 전문가."
_DEFAULT_TASK_STEP1 = "YouTube 영상을 위한 **반전과 서사가 탄탄한** 스토리 구조를 설계하라."
_DEFAULT_ROLE_STEP2 = "Award-winning Screenwriter & Visual Director."
_DEFAULT_TASK_STEP2 = (
    "승인된 스토리 아크를 기반으로 각 씬의 상세 스크립트를 작성하라.\n"
    "목표: 시청자가 끝까지 보고 감동받거나 충격받는 영상. \"얘기하다 그만둔\" 느낌이 들면 실패."
)
_DEFAULT_TOPIC_NOTE = (
    "- 주제 안에서 반전의 방향을 찾아라.\n"
    "- 제목은 궁금증을 유발하는 질문형 또는 충격적 진술형이어야 한다."
)
_DEFAULT_STEP1_RULES = (
    "[MANDATORY STORY ARC — 아웃라인 전에 먼저 정의할 것]\n"
    "스토리를 설계하기 전에 반드시 5가지 요소를 확정하라:\n"
    "1. hook_concept: 시청자가 스크롤을 멈추게 만드는 첫 장면/상황 (구체적으로)\n"
    "2. central_conflict: 이 스토리 전체를 끌고 가는 핵심 드라마틱 질문\n"
    "3. midpoint_twist: 중반부의 예상치 못한 반전 (이전 씬들의 의미를 바꾸는 것)\n"
    "4. climax_revelation: 클라이맥스의 충격적 진실/반전 (복선이 쌓여서 터지는 순간)\n"
    "5. resolution_emotion: 마지막 장면의 지배적 감정 (카타르시스/씁쓸함/희망/충격)\n\n"
    "[OUTLINE RULE]\n"
    "- 각 씬에 \"type\" 지정 필수: \"hook\" / \"build\" / \"build_clue\" / \"twist\" / \"climax\" / \"resolution\"\n"
    "- HOOK: 첫 20% 씬 — 즉각적 흡입력\n"
    "- BUILD: 30% — 긴장과 복선 축적 (clue를 심어라)\n"
    "- TWIST: 중반 반전 — 모든 것의 의미가 바뀌는 순간\n"
    "- CLIMAX: 충격 클라이맥스 — 복선이 폭발하는 지점\n"
    "- RESOLUTION: 마지막 25% — 감정적 착지, 여운\n\n"
    "[ZERO BOREDOM RULE — 지루한 구간 제로]\n"
    "- 매 씬은 반드시 새로운 정보, 사건, 또는 감정 변화를 포함해야 한다.\n"
    "- \"설명만 하는 씬\", \"분위기만 깔리는 씬\"은 절대 금지. 매 씬에 사건(event)이 있어야 한다.\n"
    "- 긴장도는 오직 상승 곡선만 허용. 평탄하거나 하강하는 구간이 있으면 실패.\n"
    "- 시청자가 어떤 씬에서든 \"지루하다\"고 느끼는 순간 영상은 끝난다.\n\n"
    "[FORESHADOWING RULE — 복선 의무화]\n"
    "- build_clue 타입 씬에는 반드시 \"clue_name\" 필드 추가: 나중에 회수될 구체적 단서 이름.\n"
    "- 예: \"clue_name\": \"깨진 시계\" → climax에서 \"깨진 시계의 의미가 밝혀진다\"\n"
    "- 복선 없는 build 씬은 금지. 모든 복선은 반드시 twist 또는 climax에서 회수되어야 한다."
)
_DEFAULT_STEP2_RULES = (
    "[SCENE-END HOOK RULE — 매 씬 끝 서스펜스 의무]\n"
    "- 매 씬의 tts_script 마지막 문장은 반드시 \"다음이 궁금한\" 서스펜스를 포함해야 한다.\n"
    "- 마지막 문장이 사건/정보를 열어야 함 — 단순 감상/정리로 끝내지 말 것.\n"
    "- 기법: 미완의 정보 (\"하지만 그건 시작에 불과했습니다\"), 예고 (\"그런데 문제는 그 다음이었습니다\"), "
    "충격 전환 (\"바로 그때, 전화벨이 울렸습니다\")\n"
    "- resolution 씬만 예외: 확실한 감정 착지로 마무리.\n\n"
    "[EMOTIONAL INTENSITY ESCALATION — 감정 강도 상승 필수]\n"
    "- 씬 1의 감정 강도 < 씬 2 < 씬 3 < ... < 클라이맥스\n"
    "- 매 씬의 tts_script는 이전 씬보다 더 강한 감정/긴장/충격을 담아야 함\n"
    "- 평탄한 감정 씬은 절대 금지. 모든 씬이 이전보다 한 단계 더 강해야 함\n\n"
    "[NARRATION TONE CONSISTENCY — 어미 통일 필수]\n"
    "전체 영상에서 나레이션 어미를 반드시 하나로 통일할 것. 혼용 절대 금지!\n"
    "→ 기본 어미: \"~했습니다/~였습니다/~입니다\" (해요체 금지, 해라체 금지)\n"
    "→ 감정 강조 시 \"~였죠\", \"~거든요\", \"~거예요\" 허용 (단, 기본 어미와 자연스럽게 섞일 것)\n"
    "→ 절대 금지: 같은 씬 안에서 \"~했다\" (해라체)와 \"~했습니다\" (합쇼체) 혼용\n"
    "BAD: \"서연이 문을 열었다. 그 안에는 편지가 있었습니다.\" ❌ (~다 + ~습니다 혼용)\n"
    "GOOD: \"서연이 문을 열었습니다. 그 안에는 편지가 있었죠.\" ✓ (합쇼체 통일)"
)
_DEFAULT_APPEARANCE_GUIDE = (
    "MUST be celebrity/idol-level stunning visuals. "
    "Describe: hair color+style, eye color, skin tone, face features. "
    "All characters must be exceptionally attractive like top actors/idols. (English)"
)
_DEFAULT_AGE_GUIDE = "20s/30s/..."
_DEFAULT_FEATURES_GUIDE = (
    "REQUIRED 3+ distinctive marks with PRECISE ANATOMICAL POSITION — "
    "e.g. 'silver chain bracelet on right wrist', 'distinctive sharp arched eyebrows', "
    "'round glasses with gold frames', 'red hair ribbon on right side of head', "
    "'heterochromia (left eye blue, right eye amber)'. "
    "Each mark must specify EXACT body location. "
    "NEVER use moles, scars, tattoos, birthmarks, blemishes, or wounds as identifying features. (English)"
)


class StoryAgent:
    """
    Generates YouTube-optimized stories in Scene JSON format.

    This agent calls OpenAI API (gpt-5.2)
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
        content_type: str = "fiction",
    ) -> Dict[str, Any]:
        """
        Generate a scene-based story JSON using a 2-Step Hierarchical Chain.

        Step 1: Structure & Architecture (Title, Characters, Outline)
        Step 2: Scene-level Detail (Script, Prompts, Camera Work)
        """
        logger.info(f"[Story Agent] Generating story (2-Step Chain): {genre} / {mood} / {style} / content_type={content_type}")
        logger.info(f"[Story Agent] duration={total_duration_sec}s, is_shorts={is_shorts}, user_idea={user_idea}")

        # content_type 규칙 로드
        _ct_rules = CONTENT_TYPE_RULES.get(content_type, CONTENT_TYPE_RULES["fiction"])
        _ct_story_structure = _ct_rules.get("story_structure", "")
        _ct_era_context = _ct_rules.get("era_context", "")
        _ct_character_design = _ct_rules.get("character_design", "")
        _ct_avoid = _ct_rules.get("avoid", "")
        _ct_plot_seeds = _ct_rules.get("plot_seeds", "")

        # 프롬프트 슬롯 로드 (오버라이드 있으면 사용, 없으면 기본값)
        _role_step1 = _ct_rules.get("role_override", _DEFAULT_ROLE_STEP1)
        _task_step1 = _ct_rules.get("task_override", _DEFAULT_TASK_STEP1)
        _role_step2 = _ct_rules.get("role_override", _DEFAULT_ROLE_STEP2)
        _task_step2 = _ct_rules.get("task_override", _DEFAULT_TASK_STEP2)
        _topic_note = _ct_rules.get("topic_note_override", _DEFAULT_TOPIC_NOTE)
        _step1_rules = _ct_rules.get("step1_rules_override", _DEFAULT_STEP1_RULES)
        _step2_rules = _ct_rules.get("step2_rules_override", _DEFAULT_STEP2_RULES)
        _enable_twist = _ct_rules.get("enable_twist_note", True)
        _appearance_guide = _ct_rules.get("appearance_guide", _DEFAULT_APPEARANCE_GUIDE)
        _age_guide = _ct_rules.get("age_guide", _DEFAULT_AGE_GUIDE)
        _features_guide = _ct_rules.get("features_guide", _DEFAULT_FEATURES_GUIDE)

        # content_type 전용 플롯 씨앗이 있으면 genre 플롯 씨앗보다 우선
        if _ct_plot_seeds:
            _genre_key_for_seeds = CONTENT_TYPE_TO_GENRE.get(content_type, genre.lower().strip())
        else:
            _genre_key_for_seeds = genre.lower().strip()

        # Calculate target scene count
        if is_shorts:
            min_scenes = 3
            max_scenes = 4
        else:
            min_scenes = max(5, total_duration_sec // 8)
            max_scenes = total_duration_sec // 4
        logger.info(f"[Story Agent] Target scene count: {min_scenes}~{max_scenes}")

        # =================================================================================
        # STEP 1: Story Architecture
        # =================================================================================
        logger.info(f"  [Step 1] Planning Story Architecture...")

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

        # 장르별 플롯 씨앗 참조 (content_type 전용 씨앗 우선)
        _genre_key = genre.lower().strip()
        if _ct_plot_seeds:
            _plot_seeds = _ct_plot_seeds
        else:
            _plot_seeds = GENRE_PLOT_SEEDS.get(_genre_key, GENRE_PLOT_SEEDS.get("drama",
                "뒤늦게 알게 되는 진실 / 믿었던 사람의 배신 / 선택의 대가 / 감추어진 비밀"))

        # ── 공통 블록 ──
        _common_language_rule = f"""
[LANGUAGE RULE - CRITICAL]
- "project_title": 반드시 한국어로 작성 (예: "마지막 편지의 비밀")
- "logline": 반드시 한국어로 작성
- "outline" → "summary": 반드시 한국어로 작성
- character "name": 한국어 이름 사용 (예: 지민, 준혁, 서연)
- character "appearance": 영어로 작성 (이미지 생성용)

{('USER IDEA (최우선 반영 — 이 주제/소재를 반드시 중심으로 스토리를 구성할 것): ' + user_idea) if user_idea else ''}

[TIME SETTING vs CHARACTER AGE — 필수 구분]
- "N년 뒤", "미래", "2080년" 같은 시간 표현은 **세계관/시대 배경**을 의미한다. 캐릭터 나이가 아니다.
- "50년 뒤의 연인" = 50년 후 미래 세계에 사는 **젊은** 연인이다. 50세 늙은 커플이 아니다.
- "조선시대 전사" = 조선시대 배경의 **젊은** 전사다. 500살 노인이 아니다.
- 시대 배경은 global_style과 씬 비주얼에 반영하고, 캐릭터 나이는 스토리에 적합한 나이(보통 20~30대)로 설정하라.

[CREATIVE TOPIC RULE]
{'- TOPIC이 주어졌으면 그 주제/소재/세계관을 반드시 스토리의 핵심으로 사용하라. 주제를 무시하거나 다른 이야기로 바꾸는 것은 절대 금지.' if user_idea else '- 진부하고 예측 가능한 전제는 절대 금지. 시청자가 "오, 이건 봐야겠다" 느끼게 만들어라.'}
- 장르별 강력한 플롯 씨앗 (참고, 그대로 쓰지 말 것 — 창의적으로 변형):
  {_plot_seeds}
{_topic_note}

[WORLDBUILDING PRESERVATION RULE — 원작 세계관 보존 필수]
- TOPIC이 전래동화, 신화, 설화, 고전문학, 역사적 이야기를 언급하면 (예: 도깨비 방망이, 콩쥐팥쥐, 선녀와 나무꾼, 그리스 신화, 삼국지 등):
  → 원작의 시대배경과 세계관을 반드시 유지하라. 현대로 리메이크하거나 배경을 바꾸는 것은 절대 금지.
  → 도깨비 방망이 = 조선시대/옛날 시골 마을 배경. 병원, 스마트폰, 현대 도시 등장 불가.
  → 캐릭터 의상/소품/배경도 원작 시대에 맞춰야 한다.
- TOPIC이 "현대판 도깨비", "SF 버전 콩쥐팥쥐"처럼 명시적으로 리메이크를 요청한 경우에만 시대 변경 허용.
- 원작을 모르면 가장 널리 알려진 전통적 버전을 기준으로 스토리를 구성하라.

{_ct_story_structure}
"""

        _common_characters_block = f"""
  "characters": {{
    "STORYCUT_HERO_A": {{
      "name": "한국어 이름",
      "gender": "male/female",
      "age": "{_age_guide}",
      "appearance": "{_appearance_guide}",
      "clothing_default": "specific outfit worn throughout the story (English)",
      "unique_features": "{_features_guide}",
      "role": "Protagonist/Antagonist/Supporting"
    }}
  }},"""

        # content_type 캐릭터 디자인 가이드 주입 (characters_block 뒤에 추가)
        if _ct_character_design:
            _common_characters_block += f"""
[CHARACTER DESIGN GUIDE — content_type={content_type}]
{_ct_character_design}
"""

        # ── 숏폼 vs 롱폼 Step 1 분기 ──
        if is_shorts:
            step1_prompt = f"""
ROLE: Master Storyteller — 유튜브 쇼츠 바이럴 전문가. 30~60초 안에 시청자를 못 놓게 만드는 전문가.
TASK: {total_duration_sec}초짜리 YouTube Shorts를 위한 **극한 밀도의 4단 구조** 스토리를 설계하라.
{('TOPIC (필수 반영): ' + user_idea) if user_idea else ''}
GENRE: {genre}
MOOD: {mood}
STYLE: {style}
SCENE COUNT: 3~4개 씬 (절대 초과 금지)

{_common_language_rule}

[SHORTS 4-BEAT STRUCTURE — 이 구조를 반드시 따라라]
숏폼은 롱폼과 완전히 다른 문법이다. 30~60초 안에 기승전결을 완성해야 한다.

BEAT 1 — HOOK (첫 3~5초):
  - 스크롤을 멈추게 하는 충격적 상황/질문/이미지
  - "왜?", "어떻게?", "말도 안 돼" 반응을 즉시 유발
  - 설명 금지. 상황 한 방으로 시작.
  예: "그녀의 결혼식 하객 명단에 3년 전 죽은 남자의 이름이 있었습니다."

BEAT 2 — ESCALATION (5~10초):
  - 상황이 급격히 악화/심화. 새로운 정보가 모든 것을 바꿈.
  - 시청자가 "이거 어떻게 되는 거야?" 느끼는 순간.
  예: "더 충격적인 건, 그 남자가 직접 축의금을 보냈다는 겁니다."

BEAT 3 — TWIST (5~10초):
  - 완전한 반전. 예상을 180도 뒤집는 정보.
  - 이전 BEAT들의 의미가 완전히 달라지는 순간.
  예: "사실 그녀가 죽인 건 남자가 아니라, 남자의 쌍둥이였습니다."

BEAT 4 — PUNCHLINE (3~5초):
  - 소름/감동/충격 한 줄 마무리. 여운을 남기되 흐지부지 금지.
  - 시청자가 다시 처음부터 보고 싶게 만드는 결말.
  예: "그리고 오늘, 진짜 그 남자가 그녀 앞에 나타났습니다."

[MANDATORY STORY ARC]
스토리 설계 전 반드시 확정:
1. hook_concept: 스크롤 멈추는 첫 상황 (구체적으로)
2. central_conflict: 핵심 드라마틱 질문
3. twist: 모든 것을 뒤집는 반전
4. punchline: 마지막 한 줄의 충격/여운

{_step1_dialogue_rule}

OUTPUT FORMAT (JSON):
{{
  "project_title": "한국어 제목 (궁금증 유발 필수)",
  "logline": "한 문장 요약 (한국어) — 반전 암시",
  "story_arc": {{
    "hook_concept": "구체적인 오프닝 충격",
    "central_conflict": "핵심 질문",
    "midpoint_twist": "반전 내용",
    "climax_revelation": "클라이맥스 진실",
    "resolution_emotion": "결말 감정"
  }},
  "global_style": {{
    "art_style": "{style}",
    "color_palette": "장르/무드에 맞는 색감",
    "visual_seed": 12345
  }},
{_common_characters_block}
  "outline": [
    {{ "scene_id": 1, "type": "hook", "summary": "HOOK 요약", "estimated_duration": {total_duration_sec // 4}, "dramatic_purpose": "스크롤 정지 + 질문 제기" }},
    {{ "scene_id": 2, "type": "escalation", "summary": "ESCALATION 요약", "estimated_duration": {total_duration_sec // 3}, "dramatic_purpose": "상황 급격 악화" }},
    {{ "scene_id": 3, "type": "twist", "summary": "TWIST 요약", "estimated_duration": {total_duration_sec // 3}, "dramatic_purpose": "예상 완전 뒤집기" }},
    {{ "scene_id": 4, "type": "punchline", "summary": "PUNCHLINE 요약", "estimated_duration": {total_duration_sec // 6}, "dramatic_purpose": "소름/충격 마무리" }}
  ]
}}
"""
        else:
            step1_prompt = f"""
ROLE: {_role_step1}
TASK: {total_duration_sec}초짜리 {_task_step1}
{('TOPIC (필수 반영): ' + user_idea) if user_idea else ''}
GENRE: {genre}
MOOD: {mood}
STYLE: {style}
SCENE COUNT: {min_scenes}~{max_scenes}개 씬

{_common_language_rule}

{_step1_rules}

[RESOLUTION COMPLETENESS — 결말 완성도]
- resolution 씬에서 central_conflict의 핵심 질문에 반드시 명확한 답을 제공할 것.
- "열린 결말", "상상에 맡긴다", "애매한 여운"은 금지. 확실한 마무리.
- 시청자가 "그래서 어떻게 된 거야?"라고 불만을 느끼면 실패.

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
{_common_characters_block}
  "outline": [
    {{ "scene_id": 1, "type": "hook", "summary": "첫 장면 요약", "estimated_duration": 6, "dramatic_purpose": "시청자 주의 집중 + 핵심 질문 제기" }},
    {{ "scene_id": 2, "type": "build_clue", "summary": "복선 심기", "estimated_duration": 5, "clue_name": "회수될 단서 이름", "dramatic_purpose": "첫 번째 단서 + 긴장감 상승" }},
    {{ "scene_id": 3, "type": "twist", "summary": "중반 반전", "estimated_duration": 7, "dramatic_purpose": "예상 뒤집기 — 이전 씬 의미 변화" }},
    {{ "scene_id": 4, "type": "climax", "summary": "클라이맥스 진실", "estimated_duration": 8, "dramatic_purpose": "복선 폭발 + 최대 감정 충격" }},
    {{ "scene_id": 5, "type": "resolution", "summary": "결말 — central_conflict의 답", "estimated_duration": 7, "dramatic_purpose": "감정 착지 + 확실한 마무리" }}
  ]
}}
"""
        step1_response = self._call_llm_api(step1_prompt)
        try:
            structure_data = json.loads(step1_response)
            logger.info(f"  [Step 1] Structure locked: {structure_data.get('project_title')}")
        except Exception as e:
            logger.error(f"  [Step 1] Failed to parse JSON: {e}. Falling back to single-step.")
            structure_data = {} 

        # =================================================================================
        # STEP 2: Scene-level Details
        # =================================================================================
        logger.info(f"  [Step 2] Generating Scene Details...")
        
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

        # 클라이맥스/반전 씬 강화 지시 (content_type에 따라 비활성화 가능)
        _story_arc = structure_data.get("story_arc", {})
        _twist_note = ""
        if _enable_twist and (_story_arc.get("climax_revelation") or _story_arc.get("midpoint_twist")):
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

        # ── Step 2 공통 블록 ──
        _common_requirements = f"""
REQUIREMENTS:
- 아웃라인의 outline 순서와 type을 정확히 따를 것.
- "narrative": 장면 설명 (반드시 한국어). 예: "지민이 카페 문을 열고 들어온다."
- "tts_script": 풍부한 스토리텔링 나레이션 (반드시 한국어). 길이는 duration_sec에 맞출 것 (초당 약 4글자).
- "image_prompt": Visual description for AI Image Generator (MUST BE English). {style} style.
- "camera_work": Specific camera movement (e.g., "Close-up", "Pan Right", "Drone Shot").

## DIALOGUE FORMAT (CRITICAL)
{_dialogue_format}
"""

        # content_type 시각 컨텍스트 블록 (f-string 밖에서 조립)
        _ct_visual_block = ""
        if _ct_era_context:
            _ct_visual_block = (
                "[CONTENT TYPE VISUAL CONTEXT — image_prompt에 반드시 반영할 것]\n"
                + _ct_era_context + "\n"
                + "AVOID in image_prompt: " + _ct_avoid + "\n"
            )

        _common_language_and_image_rules = f"""
[LANGUAGE RULE - CRITICAL]
- "narrative"와 "tts_script"는 반드시 한국어로 작성할 것. 영어 금지.
- "image_prompt"만 영어로 작성 (이미지 생성 AI용).
- "title"도 반드시 한국어로 작성.

{_ct_visual_block}

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
"""

        _common_output_format = f"""
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

        # ── 숏폼 vs 롱폼 Step 2 분기 ──
        if is_shorts:
            step2_prompt = f"""
ROLE: 유튜브 쇼츠 스크립트 전문 작가. 30~60초 안에 시청자를 못 놓게 하는 기술의 달인.
TASK: 승인된 4-BEAT 구조를 기반으로 각 씬의 상세 스크립트를 작성하라.
목표: 30~60초 안에 시청자가 소름 돋거나 충격받는 영상. 흐지부지 끝나면 실패.

APPROVED STRUCTURE:
{structure_context}

{_twist_note}

{_common_requirements}

[SHORTS NARRATION STYLE — 숏폼 전용 톤]
숏폼은 매 문장이 펀치다. 군더더기 없이 핵심만.
- 짧고 강렬한 문장. 한 문장 = 하나의 정보/충격.
- 전환어 의무 사용: "그런데,", "바로 그 순간,", "문제는,", "하지만,", "더 충격적인 건,"
- 마지막 씬(PUNCHLINE)의 마지막 문장 = 소름/충격/여운. 이것이 영상의 모든 것.
- 열린 결말 절대 금지. "과연 어떻게 됐을까요?" 같은 도망치는 결말 금지.
- 매 씬 끝이 다음 씬의 시작을 당기는 힘이 있어야 함.

BAD (숏폼에서 절대 금지):
"오늘 이야기를 시작해 보겠습니다." ❌ (도입부 낭비)
"여러분은 어떻게 생각하시나요?" ❌ (도망치는 결말)
"그 뒤로 그녀는 행복하게 살았습니다." ❌ (진부한 마무리)

GOOD (숏폼 스타일):
HOOK: "3년 전 죽은 남자에게서 택배가 도착했습니다." ✓ (즉시 충격)
ESCALATION: "그런데, 택배 안에는 내일 날짜의 신문이 들어 있었습니다." ✓ (상황 악화)
TWIST: "바로 그 순간, 현관문 비밀번호가 눌리는 소리가 들렸습니다." ✓ (반전)
PUNCHLINE: "문이 열리고, 거울 속 자신의 얼굴을 한 남자가 서 있었습니다." ✓ (소름 마무리)

[NARRATION LENGTH RULE]
목표 영상 길이: {total_duration_sec}초 / 씬 수: 3~4개
한국어 TTS는 초당 약 5.5글자. 전체 tts_script 총 글자수 ≈ {int(total_duration_sec * 5.5)}자
- HOOK: 짧고 강렬 (1~2문장)
- ESCALATION/TWIST: 중간 길이 (2~3문장)
- PUNCHLINE: 짧고 임팩트 (1~2문장)

[NARRATION TONE CONSISTENCY — 어미 통일 필수]
→ 기본 어미: "~했습니다/~였습니다/~입니다"
→ 감정 강조 시 "~였죠", "~거든요" 허용
→ 혼용 절대 금지

{_common_language_and_image_rules}

{_common_output_format}
"""
        else:
            step2_prompt = f"""
ROLE: {_role_step2}
TASK: {_task_step2}

APPROVED STRUCTURE:
{structure_context}

{_twist_note}

{_common_requirements}

[SCENE CONTINUITY RULE — 씬 연속성 필수. 이것을 어기면 스토리가 산산조각남]
전체 씬은 하나의 연결된 이야기여야 한다. 각 씬을 독립적 에피소드로 쓰지 말 것!
- 씬 N의 tts_script 첫 문장은 씬 N-1의 마지막 상황/감정에서 자연스럽게 이어져야 함
- 모든 씬이 같은 캐릭터, 같은 사건, 같은 시간축 위에 있어야 함
- 각 씬이 전체 이야기의 "그 다음"이어야 함 — 관계없는 새로운 상황 도입 금지

{_step2_rules}

[RESOLUTION COMPLETENESS — 결말 완성도 필수]
- 마지막 씬(resolution)에서 central_conflict의 핵심 질문에 반드시 명확한 답 제공
- "열린 결말", "상상에 맡긴다"는 금지. 시청자가 만족하는 확실한 마무리.

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
- 추상적 서술 금지 ("그는 슬펐다") → 구체적 장면으로 보여줄 것
- 각 씬의 tts_script는 2~4개의 완전한 문장으로 구성 (마침표로 끝나는 문장)
- 씬마다 반드시 하나의 새로운 정보/사건/감정 변화가 있어야 함

{_common_language_and_image_rules}

{_common_output_format}
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
        logger.info(f"  [Step 2] Response received, starting validation...")
        story_data = self._validate_story_json(step2_response)

        logger.info(f"[Story Agent] Story generated successfully.")
        return story_data

    def _call_llm_api(self, user_prompt: str) -> str:
        """
        Call OpenAI API to generate content.
        """
        try:
            from openai import OpenAI

            client = OpenAI(api_key=self.api_key)

            logger.debug(f"[DEBUG] Calling OpenAI API (model: {self.model})")

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
            logger.debug(f"[DEBUG] OpenAI API response received ({len(response_text)} chars)")

            return response_text

        except Exception as e:
            ErrorManager.log_error(
                "StoryAgent",
                "OpenAI API Call Failed",
                f"{type(e).__name__}: {str(e)}",
                severity="critical"
            )
            logger.error(f"[ERROR] OpenAI API call failed: {e}")
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
        try:
            story_data = parse_llm_json(json_string)
        except (json.JSONDecodeError, ValueError) as e:
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
        logger.info(f"[Story Validation] Scene count: {scene_count}")

        # v2.0: Validate character_sheet and global_style (optional but recommended)
        if "character_sheet" in story_data:
            logger.info(f"[Story Validation] Character sheet found: {len(story_data['character_sheet'])} characters")
        else:
            logger.info(f"[Story Validation] No character sheet (v1.0 format)")

        if "global_style" in story_data:
            logger.info(f"[Story Validation] Global style: {story_data['global_style'].get('art_style', 'N/A')}")

        # WARNING: Scene count validation (TEST MODE: 4 scenes expected)
        if scene_count < 3:
            logger.warning(f"[WARNING] Only {scene_count} scenes (recommended 4 for test mode)")
        elif scene_count > 6:
            logger.info(f"[INFO] {scene_count} scenes generated (test mode expects 4)")

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
                logger.info(f"[Scene {idx}] v2.0 format detected (image_prompt present)")

            if "characters_in_scene" in scene and scene["characters_in_scene"]:
                logger.info(f"[Scene {idx}] Characters: {', '.join(scene['characters_in_scene'])}")

            # v3.0: Parse dialogue lines from tts_script
            tts = scene.get("tts_script", "") or scene.get("narration", "")
            dialogue_lines = StoryAgent.parse_dialogue_lines(tts)
            if dialogue_lines and len(dialogue_lines) > 1:
                scene["dialogue_lines"] = dialogue_lines
                speakers = set(dl["speaker"] for dl in dialogue_lines)
                logger.info(f"[Scene {idx}] Dialogue: {len(dialogue_lines)} lines, speakers: {speakers}")

        # Extract detected speakers
        story_data["detected_speakers"] = StoryAgent.extract_speakers(story_data)
        if len(story_data["detected_speakers"]) > 1:
            logger.info(f"[Story Validation] Detected speakers: {story_data['detected_speakers']}")

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
            logger.error(f"[StoryAgent] Script analysis failed: {e}. Using narrator fallback.")
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

        logger.warning("[WARNING] Using 4-scene example story (Gemini API not available)")
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

        logger.info(f"[Story Agent] Story saved to: {output_path}")
