"""
Character Manager: Master anchor image generation and management.

v2.0 핵심 기능:
- 캐릭터 캐스팅: 스토리 생성 후 각 캐릭터의 마스터 앵커 이미지 생성
- 멀티 캐릭터 참조: 씬에 등장하는 캐릭터들의 이미지 경로 반환
"""

import os
from typing import Dict, List, Optional, Any
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))

from schemas import CharacterSheet, GlobalStyle


class CharacterManager:
    """
    마스터 앵커 이미지 생성/관리

    v2.0: 캐릭터 일관성 확보를 위한 마스터 이미지 시스템
    - 스토리 생성 후 별도 단계로 마스터 앵커 이미지 생성
    - 씬 이미지 생성 시 활성 캐릭터의 마스터 이미지 참조
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

    def cast_characters(
        self,
        character_sheet: Dict[str, CharacterSheet],
        global_style: Optional[GlobalStyle],
        project_dir: str
    ) -> Dict[str, str]:
        """
        모든 캐릭터의 마스터 앵커 이미지 생성.

        각 캐릭터에 대해 정면/3/4 샷, 중립 포즈의 캐스팅 이미지를 생성하여
        이후 씬 이미지 생성 시 참조 이미지로 사용합니다.

        Args:
            character_sheet: 캐릭터 시트 딕셔너리 (token -> CharacterSheet)
            global_style: 글로벌 스타일 설정
            project_dir: 프로젝트 디렉토리 (outputs/<project_id>)

        Returns:
            캐릭터 토큰 -> 마스터 이미지 경로 딕셔너리
        """
        print(f"\n{'='*60}")
        print(f"[CharacterManager] Casting Characters")
        print(f"{'='*60}")

        if not character_sheet:
            print("  No characters to cast.")
            return {}

        # 캐릭터 이미지 저장 디렉토리
        characters_dir = f"{project_dir}/media/characters"
        os.makedirs(characters_dir, exist_ok=True)

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

            # 캐스팅 프롬프트 생성 (정면/3/4 샷, 중립 포즈)
            casting_prompt = self._build_casting_prompt(
                name=name,
                appearance=appearance,
                gender=gender,
                age=age,
                clothing=clothing,
                art_style=art_style,
                color_palette=color_palette
            )

            print(f"    Name: {name}")
            print(f"    Appearance: {appearance[:50]}..." if len(appearance) > 50 else f"    Appearance: {appearance}")
            print(f"    Seed: {char_seed}")

            # 이미지 출력 경로
            output_path = f"{characters_dir}/{token}.png"

            try:
                # ImageAgent로 마스터 이미지 생성
                image_path, image_id = self.image_agent.generate_image(
                    scene_id=0,  # 0 indicates master character image
                    prompt=casting_prompt,
                    style=art_style,
                    output_dir=output_path,  # Direct path for scene_id=0
                    seed=char_seed,
                    image_model="standard"
                )

                # 캐릭터 시트에 마스터 이미지 경로 업데이트
                if isinstance(char_data, CharacterSheet):
                    char_data.master_image_path = image_path
                elif isinstance(char_data, dict):
                    char_data["master_image_path"] = image_path

                character_images[token] = image_path
                print(f"    ✓ Master image saved: {image_path}")

            except Exception as e:
                print(f"    ✗ Failed to cast {token}: {e}")
                # 실패해도 계속 진행 (다른 캐릭터 처리)
                continue

        print(f"\n[CharacterManager] Casting complete: {len(character_images)}/{total_chars} characters")
        return character_images

    def _build_casting_prompt(
        self,
        name: str,
        appearance: str,
        gender: str,
        age: str,
        clothing: str,
        art_style: str,
        color_palette: str
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

        Returns:
            캐스팅 프롬프트 문자열
        """
        # 기본 구조: 정면 또는 3/4 샷, 중립 포즈, 캐릭터 시트 스타일
        prompt_parts = [
            "Character portrait for reference sheet",
            "three-quarter view or front facing",
            "neutral expression, neutral pose",
            "clean background, studio lighting",
        ]

        # 외형 정보 추가
        if appearance:
            prompt_parts.append(appearance)

        # 성별/나이 추가
        if gender and gender != "unknown":
            prompt_parts.append(f"{gender}")
        if age and age != "unknown":
            prompt_parts.append(f"{age}")

        # 의상 추가
        if clothing:
            prompt_parts.append(f"wearing {clothing}")

        # 스타일 적용
        prompt_parts.append(art_style)
        if color_palette:
            prompt_parts.append(color_palette)

        # 품질 키워드
        prompt_parts.append("high quality, detailed, character design, full body or upper body shot")

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

            # CharacterSheet 또는 dict에서 master_image_path 추출
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

            # CharacterSheet 또는 dict에서 정보 추출
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

            # 상세 묘사 구성
            desc_parts = [name]
            if appearance:
                desc_parts.append(appearance)
            if gender and gender != "unknown":
                desc_parts.append(gender)
            if age and age != "unknown":
                desc_parts.append(age)

            descriptions.append(f"{token} ({', '.join(desc_parts)})")

        return descriptions
