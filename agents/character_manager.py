"""
Character Manager: Master anchor image generation and management.

v2.0 핵심 기능:
- 캐릭터 캐스팅: 스토리 생성 후 각 캐릭터의 마스터 앵커 이미지 생성
- 멀티 캐릭터 참조: 씬에 등장하는 캐릭터들의 이미지 경로 반환
- 멀티포즈 Anchor Set: 캐릭터당 3~6포즈, 포즈별 2장 후보 생성 후 best 선택
"""

import os
from typing import Dict, List, Optional, Any, TYPE_CHECKING
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))

from schemas import CharacterSheet, GlobalStyle, AnchorSet, PoseAnchor, PoseType

if TYPE_CHECKING:
    from schemas.mv_models import MVCharacter, MVProject


# 포즈별 프롬프트 설정
POSE_CONFIGS = {
    "front": "close-up portrait, face and shoulders only, front facing, centered, looking directly at camera, head-and-shoulders framing, hands NOT visible in frame, arms below frame edge, NO hands touching face or collar",
    "three_quarter": "upper body portrait, turned 45 degrees to the right, body angled away from camera, looking over shoulder, waist-up framing, arms relaxed at sides",
    "side": "portrait, face and upper body, side profile view, looking to the side",
    "full_body": "full body portrait from head to feet, arms relaxed naturally at sides, feet visible, medium-wide shot with slight space above and below, same style as close-up shots, idealized body proportions (7.5-8 heads tall), slim and well-proportioned figure, NOT stubby NOT short-limbed NOT chibi NOT cartoonish proportions",
    "emotion_neutral": "portrait, neutral expression, calm, relaxed",
    "emotion_intense": "portrait, intense emotional expression, dramatic lighting",
}


class CharacterManager:
    """
    마스터 앵커 이미지 생성/관리

    v2.0: 캐릭터 일관성 확보를 위한 마스터 이미지 시스템
    - 스토리 생성 후 별도 단계로 마스터 앵커 이미지 생성
    - 씬 이미지 생성 시 활성 캐릭터의 마스터 이미지 참조
    - 멀티포즈 Anchor Set으로 씬 맥락에 맞는 포즈 선택
    """

    def __init__(self, image_agent=None):
        """
        Initialize Character Manager.

        Args:
            image_agent: ImageAgent instance for image generation
        """
        if image_agent is None:
            from agents.image_agent import ImageAgent
            image_agent = ImageAgent()
        self.image_agent = image_agent

        # Gemini Vision 클라이언트 (포즈 품질 평가용)
        self._vision_client = None
        self.google_api_key = os.getenv("GOOGLE_API_KEY")

    @property
    def vision_client(self):
        """Lazy initialization of Gemini Vision client."""
        if self._vision_client is None:
            try:
                import google.generativeai as genai
                if self.google_api_key:
                    genai.configure(api_key=self.google_api_key)
                    self._vision_client = genai.GenerativeModel(model_name="gemini-2.0-flash")
            except Exception as e:
                print(f"  [Warning] Failed to init Gemini Vision for scoring: {e}")
        return self._vision_client

    def cast_characters(
        self,
        character_sheet: Dict[str, CharacterSheet],
        global_style: Optional[GlobalStyle],
        project_dir: str,
        poses: Optional[List[str]] = None,
        candidates_per_pose: int = 2,
        ethnicity: str = "auto",
        progress_callback: Optional[callable] = None
    ) -> Dict[str, str]:
        """
        모든 캐릭터의 마스터 앵커 이미지 생성.

        각 캐릭터에 대해 멀티포즈 이미지를 생성하여
        이후 씬 이미지 생성 시 참조 이미지로 사용합니다.

        Args:
            character_sheet: 캐릭터 시트 딕셔너리 (token -> CharacterSheet)
            global_style: 글로벌 스타일 설정
            project_dir: 프로젝트 디렉토리 (outputs/<project_id>)
            poses: 생성할 포즈 목록 (기본: front, three_quarter, full_body)
            candidates_per_pose: 포즈당 후보 이미지 수

        Returns:
            캐릭터 토큰 -> best_pose 마스터 이미지 경로 딕셔너리
        """
        print(f"\n{'='*60}")
        print(f"[CharacterManager] Casting Characters (Multi-Pose)")
        print(f"{'='*60}")

        if not character_sheet:
            print("  No characters to cast.")
            return {}

        if poses is None:
            poses = ["front", "three_quarter", "full_body"]

        # 스타일 정보 추출
        art_style = "cinematic animation, high contrast, dramatic lighting"
        color_palette = ""
        visual_seed = 12345

        if global_style:
            if isinstance(global_style, GlobalStyle):
                art_style = global_style.art_style
                color_palette = global_style.color_palette
                visual_seed = global_style.visual_seed
            elif isinstance(global_style, dict):
                art_style = global_style.get("art_style", art_style)
                color_palette = global_style.get("color_palette", "")
                visual_seed = global_style.get("visual_seed", visual_seed)

        # 스타일 directive 강화 (MV와 동일): LLM이 생성한 art_style 대신 명시적 스타일 지시
        _STYLE_DIRECTIVES = {
            "cinematic": "cinematic film still, dramatic chiaroscuro lighting, shallow depth of field, color graded like a Hollywood blockbuster",
            "anime": "Japanese anime cel-shaded illustration, bold black outlines, vibrant saturated colors, anime character proportions, NOT a photograph, NOT photorealistic",
            "webtoon": "Korean webtoon manhwa digital art, clean sharp lines, flat color blocks, manhwa character design, NOT a photograph, NOT photorealistic",
            "realistic": "hyperrealistic photograph, DSLR quality, natural lighting, photojournalistic, sharp focus, real-world textures, visible skin pores, natural asymmetry, NOT anime, NOT cartoon, NOT illustration, NOT AI-generated look, NOT plastic skin",
            "illustration": "digital painting illustration, painterly brushstrokes, concept art quality, rich color palette, NOT a photograph, NOT photorealistic",
            "abstract": "abstract expressionist art, surreal dreamlike imagery, bold geometric shapes, non-representational",
            "game_anime": "3D cel-shaded toon-rendered character, modern anime action RPG game quality (Genshin Impact style), high-fidelity 3D model with toon shader, cel-shading outlines, rim lighting with bloom, Unreal Engine quality, NOT photorealistic, NOT flat 2D, NOT western cartoon",
        }
        _art_lower = art_style.lower()
        # "NOT anime" 같은 부정 표현 속 키워드 오탐 방지:
        # 부정 접두사(not/no/non) 뒤의 단어는 제외한 후 매칭
        import re as _re
        _negated = set()
        for _m in _re.finditer(r'\b(?:not|no|non)[- _](\w+)', _art_lower):
            _negated.add(_m.group(1))
        _matched_style = None
        for _key, _directive in _STYLE_DIRECTIVES.items():
            _search_key = _key.replace("_", " ")
            # 부정 표현으로 언급된 키는 스킵
            if _key in _negated or _search_key in _negated:
                continue
            if _key in _art_lower or _search_key in _art_lower:
                _matched_style = (_key, _directive)
                break
        if _matched_style:
            art_style = _matched_style[1]
            print(f"  [Style] Applied directive for '{_matched_style[0]}'")
        del _negated, _matched_style

        character_images = {}
        total_chars = len(character_sheet)

        for idx, (token, char_data) in enumerate(character_sheet.items(), 1):
            print(f"\n  [{idx}/{total_chars}] Casting: {token}")

            # CharacterSheet 또는 dict 처리
            if isinstance(char_data, CharacterSheet):
                name = char_data.name
                appearance = char_data.appearance
                gender = char_data.gender
                age = char_data.age
                clothing = char_data.clothing_default
                char_seed = char_data.visual_seed or visual_seed
            elif isinstance(char_data, dict):
                name = char_data.get("name", token)
                appearance = char_data.get("appearance", "")
                gender = char_data.get("gender", "unknown")
                age = char_data.get("age", "unknown")
                clothing = char_data.get("clothing_default", "")
                char_seed = char_data.get("visual_seed", visual_seed)
            else:
                print(f"    [Warning] Invalid character data for {token}, skipping.")
                continue

            # 인종 키워드를 appearance에 주입 (씬 이미지와 동일하게)
            _ETH_KW = {
                "korean": "Korean", "japanese": "Japanese", "chinese": "Chinese",
                "southeast_asian": "Southeast Asian", "european": "European",
                "black": "Black", "hispanic": "Hispanic", "mixed": "Mixed ethnicity",
            }
            eth_keyword = _ETH_KW.get(str(ethnicity).lower(), "")
            if eth_keyword and eth_keyword.lower() not in appearance.lower():
                appearance = f"{eth_keyword}, {appearance}"
                print(f"    [Ethnicity] Injected '{eth_keyword}' into appearance")

            print(f"    Name: {name}")
            print(f"    Appearance: {appearance[:60]}..." if len(appearance) > 60 else f"    Appearance: {appearance}")
            print(f"    Seed: {char_seed}")
            print(f"    Poses: {poses}")

            # 캐릭터별 디렉토리 생성
            char_dir = f"{project_dir}/media/characters/{token}"
            os.makedirs(char_dir, exist_ok=True)

            # unique_features 추출
            if isinstance(char_data, CharacterSheet):
                _unique_features = getattr(char_data, 'unique_features', '') or ""
            elif isinstance(char_data, dict):
                _unique_features = char_data.get("unique_features", "")
            else:
                _unique_features = ""

            # 멀티포즈 앵커 생성
            anchor_set = self._generate_pose_candidates(
                token=token,
                name=name,
                appearance=appearance,
                gender=gender,
                age=age,
                clothing=clothing,
                art_style=art_style,
                color_palette=color_palette,
                char_seed=char_seed,
                poses=poses,
                candidates_per_pose=candidates_per_pose,
                char_dir=char_dir,
                unique_features=_unique_features,
            )

            # CharacterSheet에 anchor_set 저장
            if isinstance(char_data, CharacterSheet):
                char_data.anchor_set = anchor_set
                # 하위호환: best_pose 이미지를 master_image_path에 설정
                best_image = anchor_set.get_pose_image(anchor_set.best_pose)
                if best_image:
                    char_data.master_image_path = best_image
            elif isinstance(char_data, dict):
                best_image = anchor_set.get_pose_image(anchor_set.best_pose)
                if best_image:
                    char_data["master_image_path"] = best_image

            character_images[token] = anchor_set.get_pose_image(anchor_set.best_pose) or ""
            print(f"    [OK] Anchor set complete: {len(anchor_set.poses)} poses")

            # 진행률 콜백
            if progress_callback:
                progress_callback(idx, total_chars, name)

        print(f"\n[CharacterManager] Casting complete: {len(character_images)}/{total_chars} characters")
        return character_images

    def cast_mv_characters(
        self,
        characters: List["MVCharacter"],
        project: "MVProject",
        project_dir: str,
        candidates_per_pose: int = 2
    ) -> Dict[str, str]:
        """
        MV 파이프라인용 어댑터: MVCharacter → CharacterSheet 변환 후 cast_characters() 호출.

        Args:
            characters: MVCharacter 리스트 (VisualBible.characters)
            project: MVProject 객체 (style/mood/genre/ethnicity 참조)
            project_dir: 프로젝트 디렉토리
            candidates_per_pose: 포즈당 후보 수 (기본 2)

        Returns:
            role → anchor_image_path 딕셔너리
        """
        import re as _re

        print(f"\n{'='*60}")
        print(f"[CharacterManager] MV Character Casting (Scored Multi-Candidate)")
        print(f"{'='*60}")

        if not characters:
            print("  No MV characters to cast.")
            return {}, {}

        # MVCharacter → CharacterSheet 변환
        character_sheet: Dict[str, CharacterSheet] = {}
        role_to_token: Dict[str, str] = {}  # token → role 역매핑용

        for char in characters:
            # role을 안전한 토큰으로 변환
            token = _re.sub(r'[^\w\-]', '_', char.role)[:20]
            role_to_token[token] = char.role

            # description에서 gender 휴리스틱 추출
            desc_lower = char.description.lower()
            gender = "unknown"
            for kw, g in [("female", "female"), ("woman", "female"), ("girl", "female"),
                          ("여성", "female"), ("여자", "female"), ("소녀", "female"),
                          ("male", "male"), ("man", "male"), ("boy", "male"),
                          ("남성", "male"), ("남자", "male"), ("소년", "male")]:
                if kw in desc_lower:
                    gender = g
                    break

            # 인종 키워드를 appearance에 주입
            appearance = char.description
            ethnicity_val = getattr(project, 'character_ethnicity', None)
            if ethnicity_val and str(ethnicity_val.value) != "auto":
                _eth_kw = {
                    "korean": "Korean", "japanese": "Japanese", "chinese": "Chinese",
                    "southeast_asian": "Southeast Asian", "european": "European",
                    "black": "Black", "hispanic": "Hispanic", "mixed": "Mixed ethnicity",
                }
                eth_keyword = _eth_kw.get(str(ethnicity_val.value), "")
                if eth_keyword and eth_keyword.lower() not in appearance.lower():
                    appearance = f"{eth_keyword}, {appearance}"

            # Era/time period injection for character appearance
            era_setting = getattr(project, 'era_setting', '')
            if era_setting:
                era_lower = era_setting.lower()
                if 'joseon' in era_lower:
                    appearance = f"Joseon dynasty era, {appearance}"
                    if char.outfit and 'hanbok' not in char.outfit.lower():
                        char.outfit = f"traditional hanbok, {char.outfit}"
                elif 'medieval' in era_lower:
                    appearance = f"medieval European era, {appearance}"
                elif 'victorian' in era_lower:
                    appearance = f"Victorian era, {appearance}"
                elif 'ancient' in era_lower or 'roman' in era_lower or 'greek' in era_lower:
                    appearance = f"ancient era, {appearance}"
                elif era_lower not in ('modern', 'contemporary', 'modern_urban'):
                    appearance = f"{era_setting} era, {appearance}"

            sheet = CharacterSheet(
                name=char.role,
                gender=gender,
                age="unknown",
                appearance=appearance,
                clothing_default=char.outfit or "",
                unique_features=getattr(char, 'unique_features', '') or "",
                visual_seed=42,
            )
            character_sheet[token] = sheet

        # GlobalStyle 합성 (MVProject 메타데이터로부터)
        color_palette = ""
        if project.visual_bible and project.visual_bible.color_palette:
            color_palette = ", ".join(project.visual_bible.color_palette[:5])

        # 스타일별 전용 캐릭터 앵커 directive
        _style_anchor_directives = {
            "cinematic": "cinematic film still, dramatic chiaroscuro lighting, shallow depth of field, color graded like a Hollywood blockbuster",
            "anime": "Japanese anime cel-shaded illustration, bold black outlines, vibrant saturated colors, anime character proportions, NOT a photograph, NOT photorealistic",
            "webtoon": "Korean webtoon manhwa digital art, clean sharp lines, flat color blocks, manhwa character design, NOT a photograph, NOT photorealistic",
            "realistic": "hyperrealistic photograph, DSLR quality, natural lighting, photojournalistic, sharp focus, real-world textures, visible skin pores, natural asymmetry, NOT anime, NOT cartoon, NOT illustration, NOT AI-generated look, NOT plastic skin",
            "illustration": "digital painting illustration, painterly brushstrokes, concept art quality, rich color palette, NOT a photograph",
            "abstract": "abstract expressionist art, surreal dreamlike imagery, bold geometric shapes, non-representational",
            "game_anime": "3D cel-shaded toon-rendered character, modern anime action RPG game quality (Genshin Impact, Honkai Star Rail, Wuthering Waves style), high-fidelity 3D model with cartoon/toon shader, crisp cel-shading outlines, strong rim lighting with bloom, dynamic hair and cloth physics, Unreal Engine quality toon rendering, vibrant saturated colors, NOT photorealistic, NOT flat 2D hand-drawn, NOT western cartoon, NOT watercolor",
        }
        style_directive = _style_anchor_directives.get(project.style.value, f"{project.style.value} style")

        global_style = GlobalStyle(
            art_style=f"{style_directive}, {project.mood.value} mood",
            color_palette=color_palette,
            visual_seed=42,
        )

        mv_poses = ["front", "three_quarter", "full_body"]
        print(f"  Characters: {len(character_sheet)}")
        print(f"  Style: {global_style.art_style}")
        print(f"  Poses: {mv_poses}, Candidates: {candidates_per_pose}")

        # cast_characters 호출 (front + three_quarter + full_body 포즈)
        token_to_path = self.cast_characters(
            character_sheet=character_sheet,
            global_style=global_style,
            project_dir=project_dir,
            poses=mv_poses,
            candidates_per_pose=candidates_per_pose,
        )

        # token → role 역매핑 + 전체 포즈 경로 수집
        role_to_path: Dict[str, str] = {}
        role_to_poses: Dict[str, Dict[str, str]] = {}
        for token, path in token_to_path.items():
            role = role_to_token.get(token, token)
            role_to_path[role] = path
            # AnchorSet에서 전체 포즈 경로 추출
            char_data = character_sheet.get(token)
            if char_data and isinstance(char_data, CharacterSheet) and char_data.anchor_set:
                poses_dict = {}
                for pose_name, pose_anchor in char_data.anchor_set.poses.items():
                    if pose_anchor.image_path and os.path.exists(pose_anchor.image_path):
                        poses_dict[pose_name] = pose_anchor.image_path
                if poses_dict:
                    role_to_poses[role] = poses_dict

        print(f"\n[CharacterManager] MV casting complete: {len(role_to_path)} characters")
        for role, poses in role_to_poses.items():
            print(f"    {role}: {list(poses.keys())}")
        return role_to_path, role_to_poses

    def _generate_pose_candidates(
        self,
        token: str,
        name: str,
        appearance: str,
        gender: str,
        age: str,
        clothing: str,
        art_style: str,
        color_palette: str,
        char_seed: int,
        poses: List[str],
        candidates_per_pose: int,
        char_dir: str,
        unique_features: str = "",
    ) -> AnchorSet:
        """
        포즈별 후보 이미지 생성 및 best 선택.

        Args:
            token: 캐릭터 토큰
            name: 캐릭터 이름
            appearance: 외형 상세
            gender: 성별
            age: 나이대
            clothing: 기본 의상
            art_style: 아트 스타일
            color_palette: 색상 팔레트
            char_seed: 캐릭터 시드
            poses: 생성할 포즈 목록
            candidates_per_pose: 포즈당 후보 수
            char_dir: 캐릭터 이미지 디렉토리

        Returns:
            AnchorSet 객체
        """
        import threading
        import shutil
        from concurrent.futures import ThreadPoolExecutor

        anchor_set = AnchorSet(character_token=token)
        _best_score = [-1.0]
        _best_pose_key = [poses[0] if poses else "three_quarter"]
        _lock = threading.Lock()

        def _process_pose(pose_key: str, reference_image_path: str = None):
            """단일 포즈 이미지 생성.
            reference_image_path: 첫 포즈 완성 후 나머지 포즈에 캐릭터 참조 이미지 전달"""
            from agents.image_agent import ImageAgent as _IA
            pose_desc = POSE_CONFIGS.get(pose_key, pose_key)
            print(f"      [Pose] Generating: {pose_key} ({pose_desc})" +
                  (f" [ref: {os.path.basename(reference_image_path)}]" if reference_image_path else ""))

            best_candidate_path = None
            best_candidate_score = -1.0

            for cand_idx in range(candidates_per_pose):
                candidate_seed = char_seed + cand_idx * 7
                casting_prompt = self._build_casting_prompt(
                    name=name,
                    appearance=appearance,
                    gender=gender,
                    age=age,
                    clothing=clothing,
                    art_style=art_style,
                    color_palette=color_palette,
                    pose_description=pose_desc,
                    unique_features=unique_features,
                )
                # 참조 이미지가 있으면 "포즈는 무시, 얼굴/의상만 참조" 지시 추가
                if reference_image_path:
                    casting_prompt = (
                        f"IMPORTANT: The reference image is for FACE and OUTFIT identity ONLY. "
                        f"IGNORE the arm/hand pose from the reference. "
                        f"Use the NEW pose described below instead. "
                        f"Do NOT copy arm positions from the reference image. "
                        f"{casting_prompt}"
                    )
                output_path = f"{char_dir}/{pose_key}_cand{cand_idx}.jpg"
                try:
                    _agent = _IA()
                    image_path, _ = _agent.generate_image(
                        scene_id=0,
                        prompt=casting_prompt,
                        style=art_style,
                        output_dir=output_path,
                        seed=candidate_seed,
                        image_model="standard",
                        character_reference_path=reference_image_path,  # 일관성 참조
                    )
                    # 후보가 1개면 스코어링 생략 (Gemini Vision API 호출 절약)
                    if candidates_per_pose > 1:
                        score = self._score_candidate(
                            image_path=image_path,
                            pose_key=pose_key,
                            appearance=appearance,
                            art_style=art_style
                        )
                    else:
                        score = 0.5
                    print(f"        [Pose:{pose_key}] cand{cand_idx} score={score:.2f}")
                    if score > best_candidate_score:
                        best_candidate_score = score
                        best_candidate_path = image_path
                except Exception as e:
                    print(f"        [Pose:{pose_key}] cand{cand_idx} failed: {e}")

            if best_candidate_path:
                final_path = f"{char_dir}/{pose_key}.jpg"
                try:
                    shutil.copy2(best_candidate_path, final_path)
                except Exception:
                    final_path = best_candidate_path
                pose_anchor = PoseAnchor(
                    pose=PoseType(pose_key),
                    image_path=final_path,
                    score=best_candidate_score
                )
                with _lock:
                    anchor_set.poses[pose_key] = pose_anchor
                    if best_candidate_score > _best_score[0]:
                        _best_score[0] = best_candidate_score
                        _best_pose_key[0] = pose_key

        # 1단계: 첫 포즈(기준 앵커)를 먼저 단독 생성
        first_pose = poses[0]
        remaining_poses = poses[1:]
        _process_pose(first_pose, reference_image_path=None)

        # 첫 포즈 이미지를 reference로 추출
        first_ref_path = None
        if first_pose in anchor_set.poses:
            first_ref_path = anchor_set.poses[first_pose].image_path
            print(f"      [Reference] First pose ready: {os.path.basename(first_ref_path) if first_ref_path else 'None'}")

        # 2단계: 나머지 포즈 — reference 이미지 사용하되,
        # Gemini가 reference를 그대로 복사하는 문제 방지를 위해
        # three_quarter만 reference 사용, 나머지는 프롬프트로만 일관성 확보
        REF_POSES = {"three_quarter", "full_body"}  # 스타일/배경 일관성을 위해 reference 사용
        if remaining_poses:
            with ThreadPoolExecutor(max_workers=len(remaining_poses)) as executor:
                executor.map(
                    lambda p: _process_pose(p, first_ref_path if p in REF_POSES else None),
                    remaining_poses
                )

        anchor_set.best_pose = _best_pose_key[0]
        return anchor_set

    def _score_candidate(
        self,
        image_path: str,
        pose_key: str,
        appearance: str,
        art_style: str
    ) -> float:
        """
        Gemini Vision으로 후보 이미지 품질 점수 측정.

        Args:
            image_path: 후보 이미지 경로
            pose_key: 요청한 포즈 유형
            appearance: 캐릭터 외형 설명
            art_style: 아트 스타일

        Returns:
            0~1 품질 점수
        """
        if not self.vision_client or not os.path.exists(image_path):
            # Vision 클라이언트 없으면 기본 점수
            return 0.5

        try:
            import base64
            import json

            with open(image_path, "rb") as f:
                image_data = base64.b64encode(f.read()).decode("utf-8")

            ext = Path(image_path).suffix.lower()
            mime_type = "image/jpeg" if ext in [".jpg", ".jpeg"] else "image/png"

            prompt = f"""Score this character image on a scale of 0.0 to 1.0 for the following criteria:
1. face_clarity: Is the face clearly visible and well-defined? (0.0-1.0)
2. pose_accuracy: Does it match the requested pose "{pose_key}"? (0.0-1.0)
3. style_match: Does it match the art style "{art_style}"? (0.0-1.0)

Respond ONLY with JSON: {{"face_clarity": 0.0, "pose_accuracy": 0.0, "style_match": 0.0, "overall": 0.0}}"""

            response = self.vision_client.generate_content(
                [
                    {"inline_data": {"mime_type": mime_type, "data": image_data}},
                    {"text": prompt}
                ],
                generation_config={"temperature": 0.1, "max_output_tokens": 200}
            )

            content = response.text.strip()
            if content.startswith("```"):
                content = content.split("```")[1]
                if content.startswith("json"):
                    content = content[4:]
            scores = json.loads(content)
            return float(scores.get("overall", 0.5))

        except Exception as e:
            print(f"        [Warning] Scoring failed: {e}")
            return 0.5

    def _build_casting_prompt(
        self,
        name: str,
        appearance: str,
        gender: str,
        age: str,
        clothing: str,
        art_style: str,
        color_palette: str,
        pose_description: str = "three-quarter view or front facing",
        unique_features: str = "",
    ) -> str:
        """
        캐스팅 이미지 생성을 위한 프롬프트 구성.

        Args:
            name: 캐릭터 이름
            appearance: 외형 상세
            gender: 성별
            age: 나이대
            clothing: 기본 의상
            art_style: 아트 스타일
            color_palette: 색상 팔레트
            pose_description: 포즈 설명
            unique_features: 고유 식별 특징 (점, 흉터, 문신 등)

        Returns:
            캐스팅 프롬프트 문자열
        """
        # art_style을 최상단에 배치 (Gemini가 앞부분을 더 중시하므로 스타일 일관성 확보)
        prompt_parts = []
        prompt_parts.append(art_style)
        if appearance:
            prompt_parts.append(appearance)
        if gender and gender != "unknown":
            prompt_parts.append(f"{gender}")
        if age and age != "unknown":
            prompt_parts.append(f"{age}")
        prompt_parts.extend([
            "Character portrait for reference sheet",
            pose_description,
            "neutral pose",
            "clean background, studio lighting",
        ])
        if clothing:
            prompt_parts.append(f"wearing {clothing}")

        # unique_features를 정확한 위치 고정 지시와 함께 주입
        if unique_features:
            prompt_parts.append(
                f"MANDATORY IDENTIFYING MARKS at EXACT positions: {unique_features}. "
                f"These marks must appear at the PRECISE locations described — never shifted, mirrored, or relocated"
            )

        if color_palette:
            prompt_parts.append(color_palette)

        prompt_parts.append("high quality, detailed, character design")

        return ", ".join(prompt_parts)

    def get_active_character_images(
        self,
        active_characters: List[str],
        character_sheet: Dict[str, Any]
    ) -> List[str]:
        """
        씬에 등장하는 캐릭터 이미지 경로만 반환.

        Args:
            active_characters: 이 씬에 등장하는 캐릭터 토큰 목록
            character_sheet: 캐릭터 시트 딕셔너리

        Returns:
            마스터 이미지 경로 목록
        """
        if not active_characters or not character_sheet:
            return []

        image_paths = []

        for token in active_characters:
            char_data = character_sheet.get(token)
            if not char_data:
                continue

            if isinstance(char_data, CharacterSheet):
                path = char_data.master_image_path
            elif isinstance(char_data, dict):
                path = char_data.get("master_image_path")
            else:
                continue

            if path and os.path.exists(path):
                image_paths.append(path)
            else:
                print(f"  [Warning] Master image not found for {token}: {path}")

        return image_paths

    def get_pose_appropriate_image(
        self,
        token: str,
        character_sheet: Dict[str, Any],
        scene_context: Optional[str] = None
    ) -> Optional[str]:
        """
        씬 맥락에 맞는 포즈 이미지 반환.

        Args:
            token: 캐릭터 토큰
            character_sheet: 캐릭터 시트 딕셔너리
            scene_context: 씬 맥락 힌트 (예: "close-up", "emotional", "action")

        Returns:
            적절한 포즈 이미지 경로 (없으면 master_image_path 폴백)
        """
        char_data = character_sheet.get(token)
        if not char_data:
            return None

        # AnchorSet 추출
        anchor_set = None
        if isinstance(char_data, CharacterSheet):
            anchor_set = char_data.anchor_set
            fallback_path = char_data.master_image_path
        elif isinstance(char_data, dict):
            fallback_path = char_data.get("master_image_path")
            # dict에서도 anchor_set 추출
            _as = char_data.get("anchor_set")
            if _as and isinstance(_as, dict):
                from schemas.models import AnchorSet as _AnchorSet
                try:
                    anchor_set = _AnchorSet(**_as)
                except Exception:
                    anchor_set = None
        else:
            return None

        if not anchor_set:
            return fallback_path

        # 씬 맥락 기반 포즈 선택
        if scene_context:
            context_lower = scene_context.lower()
            if any(kw in context_lower for kw in ["close-up", "face", "portrait", "눈", "얼굴"]):
                return anchor_set.get_pose_image("front") or fallback_path
            elif any(kw in context_lower for kw in ["emotional", "intense", "dramatic", "분노", "슬픔"]):
                return anchor_set.get_pose_image("emotion_intense") or fallback_path
            elif any(kw in context_lower for kw in ["full", "standing", "walking", "전신"]):
                return anchor_set.get_pose_image("full_body") or fallback_path
            elif any(kw in context_lower for kw in ["side", "profile", "옆"]):
                return anchor_set.get_pose_image("side") or fallback_path

        # 기본: best_pose
        return anchor_set.get_pose_image(anchor_set.best_pose) or fallback_path

    def get_character_descriptions(
        self,
        active_characters: List[str],
        character_sheet: Dict[str, Any]
    ) -> List[str]:
        """
        씬에 등장하는 캐릭터의 상세 묘사 반환.

        Args:
            active_characters: 이 씬에 등장하는 캐릭터 토큰 목록
            character_sheet: 캐릭터 시트 딕셔너리

        Returns:
            캐릭터 상세 묘사 목록
        """
        if not active_characters or not character_sheet:
            return []

        descriptions = []

        for token in active_characters:
            char_data = character_sheet.get(token)
            if not char_data:
                continue

            if isinstance(char_data, CharacterSheet):
                name = char_data.name
                appearance = char_data.appearance
                gender = char_data.gender
                age = char_data.age
            elif isinstance(char_data, dict):
                name = char_data.get("name", token)
                appearance = char_data.get("appearance", "")
                gender = char_data.get("gender", "")
                age = char_data.get("age", "")
            else:
                continue

            desc_parts = [name]
            if appearance:
                desc_parts.append(appearance)
            if gender and gender != "unknown":
                desc_parts.append(gender)
            if age and age != "unknown":
                desc_parts.append(age)

            descriptions.append(f"{token} ({', '.join(desc_parts)})")

        return descriptions
