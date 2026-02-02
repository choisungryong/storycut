"""
STORYCUT Agents Package

에이전트 기반 아키텍처:
- StoryAgent: 스토리 생성
- SceneOrchestrator: Scene 조율 (맥락 상속 포함)
- VideoAgent: 영상 생성 (Hook 비디오 + Ken Burns)
- ImageAgent: 이미지 생성
- TTSAgent: 음성 생성
- MusicAgent: 음악 선택
- ComposerAgent: 최종 합성
- OptimizationAgent: 유튜브 최적화 (제목/썸네일/AB테스트)
- CharacterManager: v2.0 마스터 앵커 이미지 생성/관리
"""

from .story_agent import StoryAgent
from .video_agent import VideoAgent
from .image_agent import ImageAgent
from .tts_agent import TTSAgent
from .music_agent import MusicAgent
from .composer_agent import ComposerAgent
from .scene_orchestrator import SceneOrchestrator
from .optimization_agent import OptimizationAgent
from .character_manager import CharacterManager

__all__ = [
    "StoryAgent",
    "VideoAgent",
    "ImageAgent",
    "TTSAgent",
    "MusicAgent",
    "ComposerAgent",
    "SceneOrchestrator",
    "OptimizationAgent",
    "CharacterManager",
]
