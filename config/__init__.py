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


def get_provider_config(provider_type: str) -> Dict[str, Any]:
    """
    프로바이더 설정 반환

    Args:
        provider_type: "video_provider", "image_provider", "llm_provider"

    Returns:
        프로바이더 설정 딕셔너리
    """
    config = load_feature_flags()
    defaults = {
        "video_provider": {"primary": "runway", "fallback": "placeholder"},
        "image_provider": {"primary": "dalle", "fallback": "placeholder"},
        "llm_provider": {"primary": "openai", "model": "gpt-4"},
    }
    return config.get(provider_type, defaults.get(provider_type, {}))
