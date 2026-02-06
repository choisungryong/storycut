"""
Character Manager: Master anchor image generation and management.

v2.0 핵심 기능:
- 캐릭터 캐스팅: 스토리 생성 후 각 캐릭터의 마스터 앵커 이미지 생성
- 멀티 캐릭터 참조: 씬에 등장하는 캐릭터들의 이미지 경로 반환
- 멀티포즈 Anchor Set: 캐릭터당 3~6포즈, 포즈별 2장 후보 생성 후 best 선택
"""

import os
from typing import Dict, List, Optional, Any
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))

from schemas import CharacterSheet, GlobalStyle, AnchorSet, PoseAnchor, PoseType


# 포즈별 프롬프트 설정
POSE_CONFIGS = {
    "front": "front facing, centered, looking directly at camera",
    "three_quarter": "three-quarter view, slight angle",
    "side": "side profile view",
    "full_body": "full body shot, standing",
    "emotion_neutral": "neutral expression, calm",
    "emotion_intense": "intense expression, dramatic",
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
        candidates_per_pose: int = 2
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

            print(f"    Name: {name}")
            print(f"    Appearance: {appearance[:50]}..." if len(appearance) > 50 else f"    Appearance: {appearance}")
            print(f"    Seed: {char_seed}")
            print(f"    Poses: {poses}")

            # 캐릭터별 디렉토리 생성
            char_dir = f"{project_dir}/media/characters/{token}"
            os.makedirs(char_dir, exist_ok=True)

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
                char_dir=char_dir
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

        print(f"\n[CharacterManager] Casting complete: {len(character_images)}/{total_chars} characters")
        return character_images

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
        char_dir: str
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
        anchor_set = AnchorSet(character_token=token)
        best_score = -1.0
        best_pose_key = poses[0] if poses else "three_quarter"

        for pose_key in poses:
            pose_desc = POSE_CONFIGS.get(pose_key, pose_key)
            print(f"      Generating pose: {pose_key} ({pose_desc})")

            best_candidate_path = None
            best_candidate_score = -1.0

            for cand_idx in range(candidates_per_pose):
                # 후보별 시드 변경
                candidate_seed = char_seed + cand_idx * 7

                # 포즈별 캐스팅 프롬프트
                casting_prompt = self._build_casting_prompt(
                    name=name,
                    appearance=appearance,
                    gender=gender,
                    age=age,
                    clothing=clothing,
                    art_style=art_style,
                    color_palette=color_palette,
                    pose_description=pose_desc
                )

                output_path = f"{char_dir}/{pose_key}_cand{cand_idx}.jpg"

                try:
                    image_path, _ = self.image_agent.generate_image(
                        scene_id=0,
                        prompt=casting_prompt,
                        style=art_style,
                        output_dir=output_path,
                        seed=candidate_seed,
                        image_model="standard"
                    )

                    # 품질 점수 측정
                    score = self._score_candidate(
                        image_path=image_path,
                        pose_key=pose_key,
                        appearance=appearance,
                        art_style=art_style
                    )

                    print(f"        Candidate {cand_idx}: score={score:.2f}")

                    if score > best_candidate_score:
                        best_candidate_score = score
                        best_candidate_path = image_path

                except Exception as e:
                    print(f"        Candidate {cand_idx} failed: {e}")
                    continue

            # 최고 후보를 최종 포즈 이미지로 복사
            if best_candidate_path:
                final_path = f"{char_dir}/{pose_key}.jpg"
                try:
                    import shutil
                    shutil.copy2(best_candidate_path, final_path)
                except Exception:
                    final_path = best_candidate_path

                pose_anchor = PoseAnchor(
                    pose=PoseType(pose_key),
                    image_path=final_path,
                    score=best_candidate_score
                )
                anchor_set.poses[pose_key] = pose_anchor

                if best_candidate_score > best_score:
                    best_score = best_candidate_score
                    best_pose_key = pose_key

        anchor_set.best_pose = best_pose_key
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
        pose_description: str = "three-quarter view or front facing"
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

        Returns:
            캐스팅 프롬프트 문자열
        """
        prompt_parts = [
            "Character portrait for reference sheet",
            pose_description,
            "neutral pose",
            "clean background, studio lighting",
        ]

        if appearance:
            prompt_parts.append(appearance)
        if gender and gender != "unknown":
            prompt_parts.append(f"{gender}")
        if age and age != "unknown":
            prompt_parts.append(f"{age}")
        if clothing:
            prompt_parts.append(f"wearing {clothing}")

        prompt_parts.append(art_style)
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
