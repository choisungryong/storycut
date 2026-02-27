"""
STORYCUT Configuration Loader
"""

import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional

# 기본 설정 디렉토리
CONFIG_DIR = Path(__file__).parent


def load_feature_flags(config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Feature flags 설정 로드

    Args:
        config_path: 설정 파일 경로 (기본: config/feature_flags.yaml)

    Returns:
        Feature flags 딕셔너리
    """
    if config_path is None:
        config_path = CONFIG_DIR / "feature_flags.yaml"

    if not os.path.exists(config_path):
        # 기본값 반환
        return get_default_feature_flags()

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    return config


def get_default_feature_flags() -> Dict[str, Any]:
    """기본 feature flags 반환"""
    return {
        "hook_scene1_video": False,
        "ffmpeg_kenburns": True,
        "ffmpeg_audio_ducking": False,
        "subtitle_burn_in": True,
        "context_carry_over": True,
        "optimization_pack": True,
        "topic_finding": False,
    }


def get_kenburns_config() -> Dict[str, Any]:
    """Ken Burns 효과 설정 반환"""
    config = load_feature_flags()
    return config.get("kenburns", {
        "zoom_range": [1.0, 1.3],
        "pan_speed": 0.5,
        "direction": "random",
    })


def get_ducking_config() -> Dict[str, Any]:
    """Audio Ducking 설정 반환"""
    config = load_feature_flags()
    return config.get("ducking", {
        "threshold": 0.02,
        "ratio": 10,
        "attack_ms": 20,
        "release_ms": 200,
        "bgm_volume_normal": 0.3,
        "bgm_volume_ducked": 0.1,
    })


def get_subtitle_style() -> Dict[str, Any]:
    """자막 스타일 설정 반환"""
    config = load_feature_flags()
    return config.get("subtitle_style", {
        "font_name": "Arial",
        "font_size": 24,
        "primary_color": "&HFFFFFF",
        "outline_color": "&H000000",
        "outline_width": 2,
        "position": "bottom",
        "margin_v": 50,
    })


def load_veo_policy() -> Dict[str, Any]:
    """
    Veo I2V 정책 설정 로드.

    Returns:
        Veo 정책 딕셔너리 (없으면 기본값)
    """
    config_path = CONFIG_DIR / "veo_policy.yaml"

    if not os.path.exists(config_path):
        return get_default_veo_policy()

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    return config.get("veo_policy", get_default_veo_policy())


def get_default_veo_policy() -> Dict[str, Any]:
    """기본 Veo 정책 반환."""
    return {
        "mode": "image_to_video_only",
        "clip_length": {
            "character_min_sec": 2,
            "character_max_sec": 4,
            "broll_max_sec": 6,
        },
        "allowed_motions": {
            "camera": ["slow zoom in", "slow zoom out", "gentle pan left"],
            "subject": ["subtle head turn", "hair blowing in wind", "gentle breathing"],
            "ambient": ["dust particles floating", "light flickering", "leaves rustling"],
        },
        "forbidden_motions": ["jump", "run", "fight", "swing", "kick", "fly", "explode"],
        "forbidden_content_tokens": ["race", "ethnicity", "skin color"],
    }


def load_style_tokens() -> Dict[str, Any]:
    """
    스타일 토큰 화이트리스트 로드.

    Returns:
        스타일 토큰 딕셔너리 (없으면 기본값)
    """
    config_path = CONFIG_DIR / "style_tokens.yaml"

    if not os.path.exists(config_path):
        return get_default_style_tokens()

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    return config.get("style_tokens", get_default_style_tokens())


def get_default_style_tokens() -> Dict[str, Any]:
    """기본 스타일 토큰 반환."""
    return {
        "art_styles": ["cinematic animation", "cinematic", "photorealistic", "illustration"],
        "lighting": ["dramatic lighting", "soft lighting", "natural lighting", "high contrast"],
        "composition": ["rule of thirds", "wide angle", "close-up", "establishing shot"],
        "quality": ["high quality", "detailed", "4k", "professional"],
    }


def load_genre_profiles() -> Dict[str, Any]:
    """
    장르별 비주얼 프로필 로드.

    Returns:
        장르 프로필 딕셔너리 (없으면 빈 dict → 기존 full-LLM 동작)
    """
    config_path = CONFIG_DIR / "genre_profiles.yaml"

    if not os.path.exists(config_path):
        return get_default_genre_profiles()

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    return config.get("genre_profiles", get_default_genre_profiles())


def get_default_genre_profiles() -> Dict[str, Any]:
    """기본 장르 프로필 반환 (빈 dict → YAML 없으면 기존 동작 유지)."""
    return {}
