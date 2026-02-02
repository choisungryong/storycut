"""
Multimodal Prompt Builder for Gemini 2.5 Flash Image.

v2.0 핵심 기능:
- 복수 캐릭터 참조 이미지 + 텍스트 리스트 구성
- Gemini API 스펙에 맞는 멀티모달 요청 빌더
"""

import os
import base64
from typing import Dict, List, Any, Optional
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))

from schemas import Scene, GlobalStyle, CharacterSheet


class MultimodalPromptBuilder:
    """
    Gemini 2.5 Flash Image용 멀티모달 요청 빌더

    v2.0: 복수 캐릭터 참조 이미지와 텍스트 프롬프트를 결합하여
    Gemini API 형식의 멀티모달 요청을 구성합니다.
    """

    @staticmethod
    def build_request(
        scene: Scene,
        character_sheet: Dict[str, Any],
        global_style: Optional[GlobalStyle] = None,
        max_reference_images: int = 3
    ) -> List[Dict]:
        """
        씬 이미지 생성을 위한 멀티모달 요청 구성.

        Returns:
        [
            {"inline_data": {"mime_type": "image/png", "data": base64_hero}},
            {"inline_data": {"mime_type": "image/png", "data": base64_villain}},
            {"text": "Character Reference for Hero: ..."},
            {"text": "Character Reference for Villain: ..."},
            {"text": "Scene Description: ..."},
            {"text": "Style/Cinematography: ..."}
        ]

        Args:
            scene: Scene 객체
            character_sheet: 캐릭터 시트 딕셔너리
            global_style: 글로벌 스타일 설정
            max_reference_images: 최대 참조 이미지 수 (API 제한 고려)

        Returns:
            Gemini API parts 리스트
        """
        parts = []

        # 1. 활성 캐릭터의 참조 이미지 추가
        active_characters = scene.characters_in_scene or []
        character_descriptions = []
        added_images = 0

        for token in active_characters:
            if added_images >= max_reference_images:
                break

            char_data = character_sheet.get(token)
            if not char_data:
                continue

            # 마스터 이미지 경로 추출
            if isinstance(char_data, CharacterSheet):
                image_path = char_data.master_image_path
                name = char_data.name
                appearance = char_data.appearance
            elif isinstance(char_data, dict):
                image_path = char_data.get("master_image_path")
                name = char_data.get("name", token)
                appearance = char_data.get("appearance", "")
            else:
                continue

            # 이미지 추가
            if image_path and os.path.exists(image_path):
                image_data = MultimodalPromptBuilder._encode_image(image_path)
                if image_data:
                    # MIME type 결정
                    mime_type = MultimodalPromptBuilder._get_mime_type(image_path)
                    parts.append({
                        "inline_data": {
                            "mime_type": mime_type,
                            "data": image_data
                        }
                    })
                    added_images += 1

            # 캐릭터 설명 추가
            desc = f"Character '{name}' ({token}): {appearance}"
            character_descriptions.append(desc)

        # 2. 캐릭터 참조 텍스트 추가
        if character_descriptions:
            char_ref_text = "Character References:\n" + "\n".join(
                f"- {desc}" for desc in character_descriptions
            )
            parts.append({"text": char_ref_text})

        # 3. 씬 설명 추가
        scene_description = MultimodalPromptBuilder._build_scene_description(scene)
        parts.append({"text": scene_description})

        # 4. 스타일/시네마토그래피 추가
        style_text = MultimodalPromptBuilder._build_style_text(global_style)
        parts.append({"text": style_text})

        # 5. 최종 생성 지시 추가
        generation_instruction = MultimodalPromptBuilder._build_generation_instruction(
            scene, character_descriptions
        )
        parts.append({"text": generation_instruction})

        return parts

    @staticmethod
    def build_simple_request(
        prompt: str,
        character_reference_paths: Optional[List[str]] = None,
        style: str = "cinematic"
    ) -> List[Dict]:
        """
        간단한 멀티모달 요청 구성 (단순 프롬프트 + 참조 이미지).

        Args:
            prompt: 이미지 생성 프롬프트
            character_reference_paths: 캐릭터 참조 이미지 경로 목록
            style: 스타일 문자열

        Returns:
            Gemini API parts 리스트
        """
        parts = []

        # 1. 참조 이미지 추가
        if character_reference_paths:
            for path in character_reference_paths[:3]:  # 최대 3개
                if path and os.path.exists(path):
                    image_data = MultimodalPromptBuilder._encode_image(path)
                    if image_data:
                        mime_type = MultimodalPromptBuilder._get_mime_type(path)
                        parts.append({
                            "inline_data": {
                                "mime_type": mime_type,
                                "data": image_data
                            }
                        })

        # 2. 참조 지시문 (이미지가 있는 경우)
        if parts:
            parts.append({
                "text": "Using the above character reference image(s), maintain character consistency in the generated image."
            })

        # 3. 메인 프롬프트
        full_prompt = f"Generate a high-quality image: {style} style. {prompt}. Aspect ratio 16:9, cinematic, professional photography."
        parts.append({"text": full_prompt})

        return parts

    @staticmethod
    def _encode_image(image_path: str) -> Optional[str]:
        """이미지 파일을 base64로 인코딩."""
        try:
            with open(image_path, "rb") as f:
                return base64.b64encode(f.read()).decode("utf-8")
        except Exception as e:
            print(f"  [Warning] Failed to encode image {image_path}: {e}")
            return None

    @staticmethod
    def _get_mime_type(image_path: str) -> str:
        """파일 확장자에서 MIME type 추론."""
        ext = Path(image_path).suffix.lower()
        mime_map = {
            ".png": "image/png",
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".gif": "image/gif",
            ".webp": "image/webp",
        }
        return mime_map.get(ext, "image/png")

    @staticmethod
    def _build_scene_description(scene: Scene) -> str:
        """씬 설명 텍스트 구성."""
        parts = ["Scene Description:"]

        # image_prompt 우선 사용
        if scene.image_prompt:
            parts.append(f"Visual: {scene.image_prompt}")
        elif scene.visual_description:
            parts.append(f"Visual: {scene.visual_description}")
        elif scene.prompt:
            parts.append(f"Visual: {scene.prompt}")

        # narrative 추가
        if scene.narrative:
            parts.append(f"Narrative: {scene.narrative}")

        # mood 추가
        if scene.mood:
            parts.append(f"Mood: {scene.mood}")

        # 엔티티 정보 추가
        if scene.entities:
            if scene.entities.location:
                parts.append(f"Location: {scene.entities.location}")
            if scene.entities.action:
                parts.append(f"Action: {scene.entities.action}")

        return "\n".join(parts)

    @staticmethod
    def _build_style_text(global_style: Optional[GlobalStyle]) -> str:
        """스타일 텍스트 구성."""
        if not global_style:
            return "Style: cinematic animation, high contrast, dramatic lighting, 16:9 aspect ratio"

        # GlobalStyle 객체 또는 dict 처리
        if isinstance(global_style, GlobalStyle):
            art_style = global_style.art_style
            color_palette = global_style.color_palette
            aspect_ratio = global_style.aspect_ratio
        elif isinstance(global_style, dict):
            art_style = global_style.get("art_style", "cinematic")
            color_palette = global_style.get("color_palette", "")
            aspect_ratio = global_style.get("aspect_ratio", "16:9")
        else:
            return "Style: cinematic animation, high contrast, dramatic lighting, 16:9 aspect ratio"

        parts = ["Style/Cinematography:"]
        parts.append(f"Art Style: {art_style}")
        if color_palette:
            parts.append(f"Color Palette: {color_palette}")
        parts.append(f"Aspect Ratio: {aspect_ratio}")

        return "\n".join(parts)

    @staticmethod
    def _build_generation_instruction(
        scene: Scene,
        character_descriptions: List[str]
    ) -> str:
        """최종 생성 지시문 구성."""
        instruction_parts = [
            "Generation Instructions:",
            "- Maintain exact character appearance from reference images",
            "- Keep consistent clothing, hair, and facial features",
            "- Match the described mood and atmosphere",
            "- Use professional cinematographic composition",
            "- Output a single cohesive scene image",
        ]

        # 활성 캐릭터가 있으면 일관성 강조
        if character_descriptions:
            instruction_parts.append(
                f"- Characters in this scene: {', '.join(scene.characters_in_scene or [])}"
            )
            instruction_parts.append(
                "- CRITICAL: Character faces, body proportions, and distinctive features must match the reference images exactly"
            )

        return "\n".join(instruction_parts)


def build_multimodal_parts(
    prompt: str,
    character_reference_paths: Optional[List[str]] = None,
    style: str = "cinematic"
) -> List[Dict]:
    """
    편의 함수: 멀티모달 parts 리스트 빌드.

    Args:
        prompt: 이미지 생성 프롬프트
        character_reference_paths: 캐릭터 참조 이미지 경로 목록
        style: 스타일 문자열

    Returns:
        Gemini API parts 리스트
    """
    return MultimodalPromptBuilder.build_simple_request(
        prompt=prompt,
        character_reference_paths=character_reference_paths,
        style=style
    )
