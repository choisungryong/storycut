"""
Style Anchor Agent: Generates style and environment anchor images.

v2.0 핵심 기능:
- StyleAnchor: 프로젝트 전체 룩 이미지 1장 생성 (캐릭터 없음)
- EnvironmentAnchor: 씬별 배경 기준 이미지 생성 (캐릭터 없음)
"""

import os
import re
from typing import Dict, List, Optional, Any
from pathlib import Path


from schemas import GlobalStyle, Scene
from utils.logger import get_logger
logger = get_logger("style_anchor")



class StyleAnchorAgent:
    """
    스타일/환경 앵커 이미지 생성 에이전트

    v2.0: 프로젝트 전체의 시각적 일관성 기준 이미지를 생성합니다.
    - StyleAnchor: 순수 스타일 시연 이미지 (캐릭터 없음, 특정 씬 없음)
    - EnvironmentAnchor: 씬별 배경 환경 기준 이미지 (캐릭터 없음)
    """

    def __init__(self, image_agent=None):
        """
        Initialize Style Anchor Agent.

        Args:
            image_agent: ImageAgent instance for image generation
        """
        if image_agent is None:
            from agents.image_agent import ImageAgent
            image_agent = ImageAgent()
        self.image_agent = image_agent

    def generate_style_anchor(
        self,
        global_style: Optional[GlobalStyle],
        project_dir: str
    ) -> str:
        """
        프로젝트 전체 룩 이미지 1장 생성.

        순수 스타일 시연 이미지 (캐릭터 없음, 특정 씬 없음).

        Args:
            global_style: 글로벌 스타일 설정
            project_dir: 프로젝트 디렉토리

        Returns:
            스타일 앵커 이미지 경로
        """
        logger.info(f"\n[StyleAnchor] Generating style anchor image...")

        output_path = f"{project_dir}/media/style_anchor.png"
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        prompt = self._build_style_anchor_prompt(global_style)
        logger.info(f"  Prompt: {prompt[:80]}...")

        # 스타일 정보 추출
        art_style = "cinematic animation"
        visual_seed = 12345
        if global_style:
            if isinstance(global_style, GlobalStyle):
                art_style = global_style.art_style
                visual_seed = global_style.visual_seed
            elif isinstance(global_style, dict):
                art_style = global_style.get("art_style", art_style)
                visual_seed = global_style.get("visual_seed", visual_seed)

        try:
            image_path, _ = self.image_agent.generate_image(
                scene_id=0,
                prompt=prompt,
                style=art_style,
                output_dir=output_path,  # .png로 끝나므로 그대로 파일 경로로 사용됨
                seed=visual_seed,
                image_model="standard"
            )

            # GlobalStyle에 경로 저장
            if isinstance(global_style, GlobalStyle):
                global_style.style_anchor_path = image_path

            logger.info(f"  [StyleAnchor] Style anchor saved: {image_path}")
            return image_path

        except Exception as e:
            logger.error(f"  [StyleAnchor] Failed to generate style anchor: {e}")
            return ""

    _MAX_ENV_ANCHORS = 3  # 최대 환경앵커 수 (API 호출 절약)

    def generate_environment_anchors(
        self,
        scenes: List[Dict[str, Any]],
        global_style: Optional[GlobalStyle],
        project_dir: str
    ) -> Dict[int, str]:
        """
        환경 기준 이미지 생성 (유사 배경 그룹화).

        동일/유사한 장소의 씬은 1장의 앵커를 공유하여 API 호출 절약.
        최대 _MAX_ENV_ANCHORS 장까지만 생성.

        Args:
            scenes: 씬 데이터 리스트
            global_style: 글로벌 스타일 설정
            project_dir: 프로젝트 디렉토리

        Returns:
            씬 ID -> 환경 앵커 이미지 경로 딕셔너리
        """
        logger.info(f"\n[EnvAnchor] Generating environment anchor images (max {self._MAX_ENV_ANCHORS})...")

        os.makedirs(f"{project_dir}/media", exist_ok=True)

        art_style = "cinematic animation"
        visual_seed = 12345
        if global_style:
            if isinstance(global_style, GlobalStyle):
                art_style = global_style.art_style
                visual_seed = global_style.visual_seed
            elif isinstance(global_style, dict):
                art_style = global_style.get("art_style", art_style)
                visual_seed = global_style.get("visual_seed", visual_seed)

        # --- 씬 그룹화: location 키워드 기반 ---
        groups = self._group_scenes_by_location(scenes)
        logger.info(f"[EnvAnchor] {len(scenes)} scenes → {len(groups)} location groups")
        for gid, (rep_scene, member_ids) in enumerate(groups):
            logger.info(f"  Group {gid}: scenes {member_ids} (rep=scene_{rep_scene.get('scene_id', 0)})")

        env_anchors = {}
        import threading
        from concurrent.futures import ThreadPoolExecutor
        _lock = threading.Lock()

        def _generate_one(group_data):
            rep_scene, member_scene_ids = group_data
            rep_id = rep_scene.get("scene_id", 0)
            output_path = f"{project_dir}/media/env_anchor_scene_{rep_id:02d}.png"
            prompt = self._build_environment_prompt(rep_scene, global_style)
            logger.info(f"  EnvAnchor group (rep scene {rep_id}, covers {member_scene_ids}): {prompt[:60]}...")
            try:
                from agents.image_agent import ImageAgent as _IA
                image_path, _ = _IA().generate_image(
                    scene_id=0,
                    prompt=prompt,
                    style=art_style,
                    output_dir=output_path,
                    seed=visual_seed + rep_id,
                    image_model="standard"
                )
                with _lock:
                    for sid in member_scene_ids:
                        env_anchors[sid] = image_path
                logger.info(f"  [EnvAnchor] Group done: {image_path} → scenes {member_scene_ids}")
            except Exception as e:
                logger.error(f"  [EnvAnchor] Group (rep scene {rep_id}) failed: {e}")

        with ThreadPoolExecutor(max_workers=min(4, len(groups))) as executor:
            executor.map(_generate_one, groups)

        # GlobalStyle에 저장
        if isinstance(global_style, GlobalStyle):
            global_style.environment_anchors = env_anchors

        logger.info(f"[EnvAnchor] Generated {len(set(env_anchors.values()))} unique anchors → mapped to {len(env_anchors)}/{len(scenes)} scenes")
        return env_anchors

    def _group_scenes_by_location(
        self,
        scenes: List[Dict[str, Any]]
    ) -> List[tuple]:
        """
        씬을 장소(location) 키워드 기반으로 그룹화.

        Returns:
            [(대표씬_dict, [scene_id, ...]), ...] 최대 _MAX_ENV_ANCHORS 그룹
        """
        # 각 씬에서 location 키워드 추출
        scene_locations = []
        for s in scenes:
            loc = (s.get("location") or "").strip().lower()
            if not loc:
                # image_prompt에서 장소 힌트 추출 (첫 명사구)
                visual = s.get("image_prompt") or s.get("visual_description") or ""
                visual = re.sub(r'STORYCUT_\w+', '', visual)
                # 장소 관련 핵심 단어만 추출 (실내/실외/자연/건물 등)
                loc = self._extract_location_key(visual)
            scene_locations.append((s, loc))

        # 같은 location끼리 그룹화
        from collections import OrderedDict
        loc_groups: OrderedDict = OrderedDict()
        for scene_data, loc_key in scene_locations:
            sid = scene_data.get("scene_id", 0)
            if loc_key in loc_groups:
                loc_groups[loc_key][1].append(sid)
            else:
                loc_groups[loc_key] = (scene_data, [sid])

        groups = list(loc_groups.values())

        # 그룹 수가 _MAX_ENV_ANCHORS 초과하면 작은 그룹들을 가장 가까운 그룹에 병합
        if len(groups) > self._MAX_ENV_ANCHORS:
            # 씬 수 기준 내림차순 정렬 → 상위 N-1개 유지 + 나머지를 마지막에 병합
            groups.sort(key=lambda g: len(g[1]), reverse=True)
            keep = groups[:self._MAX_ENV_ANCHORS - 1]
            merge_into = groups[self._MAX_ENV_ANCHORS - 1]
            for extra in groups[self._MAX_ENV_ANCHORS:]:
                merge_into[1].extend(extra[1])
            groups = keep + [merge_into]

        return groups

    @staticmethod
    def _extract_location_key(visual_text: str) -> str:
        """image_prompt에서 장소 키워드를 추출하여 그룹핑 키 생성."""
        text = visual_text.lower()
        # 대표적인 장소 키워드 매칭
        _LOCATION_KEYWORDS = [
            "forest", "mountain", "village", "house", "room", "palace",
            "castle", "river", "ocean", "sea", "beach", "cave", "temple",
            "market", "street", "city", "school", "hospital", "office",
            "garden", "field", "bridge", "road", "path", "kitchen",
            "bedroom", "library", "church", "tower", "gate", "cliff",
            "lake", "pond", "farm", "barn", "shop", "hall", "court",
            # 한국어 장소
            "마을", "집", "방", "궁", "산", "강", "바다", "숲", "동굴",
            "사찰", "시장", "거리", "학교", "병원", "정원", "밭", "들판",
        ]
        for kw in _LOCATION_KEYWORDS:
            if kw in text:
                return kw
        # 매칭 안 되면 첫 20자를 키로 (대략적 구분)
        return text[:20].strip() if text else "default"

    def _build_style_anchor_prompt(self, global_style: Optional[GlobalStyle]) -> str:
        """
        스타일 앵커 프롬프트 생성.

        순수 스타일 시연 (캐릭터 없음, 특정 씬 없음).

        Args:
            global_style: 글로벌 스타일 설정

        Returns:
            스타일 앵커 프롬프트
        """
        art_style = "cinematic animation, high contrast, dramatic lighting"
        color_palette = "rich vibrant blues and warm amber highlights"

        if global_style:
            if isinstance(global_style, GlobalStyle):
                art_style = global_style.art_style
                color_palette = global_style.color_palette
            elif isinstance(global_style, dict):
                art_style = global_style.get("art_style", art_style)
                color_palette = global_style.get("color_palette", color_palette)

        return (
            f"Cinematic establishing shot, {art_style}, {color_palette}, "
            "no people, no text, no characters, empty scene, "
            "atmospheric, professional lighting, 16:9 aspect ratio, "
            "high quality, detailed environment"
        )

    def _build_environment_prompt(
        self,
        scene_data: Dict[str, Any],
        global_style: Optional[GlobalStyle]
    ) -> str:
        """
        환경 앵커 프롬프트 생성.

        캐릭터 토큰을 제거한 순수 배경/환경 이미지.

        Args:
            scene_data: 씬 데이터 딕셔너리
            global_style: 글로벌 스타일 설정

        Returns:
            환경 앵커 프롬프트
        """
        # visual_description 또는 image_prompt에서 환경 정보 추출
        visual = scene_data.get("image_prompt") or scene_data.get("visual_description") or ""

        # 캐릭터 토큰 제거 (STORYCUT_* 패턴)
        visual = re.sub(r'STORYCUT_\w+', '', visual)

        # 캐릭터 관련 표현 제거
        visual = re.sub(r'\b(character|person|man|woman|boy|girl|he|she|they)\b', '', visual, flags=re.IGNORECASE)

        # 정리
        visual = re.sub(r'\s+', ' ', visual).strip()
        visual = re.sub(r',\s*,', ',', visual)
        visual = visual.strip(', ')

        # 스타일 정보
        art_style = "cinematic animation"
        color_palette = ""
        if global_style:
            if isinstance(global_style, GlobalStyle):
                art_style = global_style.art_style
                color_palette = global_style.color_palette
            elif isinstance(global_style, dict):
                art_style = global_style.get("art_style", art_style)
                color_palette = global_style.get("color_palette", "")

        parts = [
            f"Environment establishing shot, {art_style}",
            "no people, no characters, empty background",
        ]

        if visual:
            parts.append(visual)
        if color_palette:
            parts.append(color_palette)

        parts.append("atmospheric, 16:9 aspect ratio, high quality")

        return ", ".join(parts)
