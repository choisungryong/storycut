"""
Music Agent: Selects background music based on genre and mood.

BGM Library: media/music/library/
  - {category}.mp3 (fallback)
  - {category}_01.mp3, {category}_02.mp3, {category}_03.mp3 (random selection)

License: CC BY-SA 4.0 (Internet Archive)
"""

import os
import random
from typing import Optional, List


# 장르/분위기 → BGM 카테고리 매핑
_MOOD_MAP = {
    "dramatic": "emotional_dramatic",
    "emotional": "emotional_dramatic",
    "sad": "emotional_dramatic",
    "melancholic": "emotional_dramatic",
    "romantic": "emotional_dramatic",
    "nostalgic": "emotional_dramatic",

    "happy": "happy_upbeat",
    "upbeat": "happy_upbeat",
    "cheerful": "happy_upbeat",
    "fun": "happy_upbeat",
    "energetic": "happy_upbeat",
    "bright": "happy_upbeat",
    "comedic": "happy_upbeat",

    "suspenseful": "suspense_thriller",
    "tense": "suspense_thriller",
    "thriller": "suspense_thriller",
    "mysterious": "suspense_thriller",
    "dark": "suspense_thriller",
    "intense": "suspense_thriller",

    "calm": "calm_peaceful",
    "peaceful": "calm_peaceful",
    "relaxing": "calm_peaceful",
    "gentle": "calm_peaceful",
    "serene": "calm_peaceful",
    "meditative": "calm_peaceful",
    "neutral": "calm_peaceful",

    "epic": "epic_cinematic",
    "cinematic": "epic_cinematic",
    "heroic": "epic_cinematic",
    "grand": "epic_cinematic",
    "action": "epic_cinematic",
    "adventure": "epic_cinematic",
    "inspiring": "epic_cinematic",

    "horror": "horror_eerie",
    "eerie": "horror_eerie",
    "creepy": "horror_eerie",
    "scary": "horror_eerie",
    "haunting": "horror_eerie",
    "sinister": "horror_eerie",
}

_GENRE_MAP = {
    "emotional": "emotional_dramatic",
    "drama": "emotional_dramatic",
    "romance": "emotional_dramatic",

    "comedy": "happy_upbeat",
    "slice_of_life": "happy_upbeat",

    "thriller": "suspense_thriller",
    "mystery": "suspense_thriller",
    "crime": "suspense_thriller",

    "documentary": "calm_peaceful",
    "education": "calm_peaceful",

    "action": "epic_cinematic",
    "fantasy": "epic_cinematic",
    "sci_fi": "epic_cinematic",
    "historical": "epic_cinematic",

    "horror": "horror_eerie",
}


class MusicAgent:
    """
    Handles background music selection based on genre and mood.
    Randomly picks from available tracks per category.
    """

    def __init__(self, music_library_path: str = "media/music"):
        self.music_library_path = music_library_path
        self.library_path = os.path.join(music_library_path, "library")
        os.makedirs(self.library_path, exist_ok=True)

    def _find_tracks(self, category: str) -> List[str]:
        """카테고리에 해당하는 모든 트랙 파일 경로를 반환."""
        tracks = []
        # numbered tracks: {category}_01.mp3, {category}_02.mp3, ...
        for f in os.listdir(self.library_path):
            if f.startswith(category + "_") and f.endswith(".mp3"):
                tracks.append(os.path.join(self.library_path, f))
        # fallback: {category}.mp3
        base = os.path.join(self.library_path, f"{category}.mp3")
        if os.path.exists(base) and base not in tracks:
            tracks.append(base)
        return tracks

    def select_music(
        self,
        genre: str,
        mood: str,
        duration_sec: int
    ) -> Optional[str]:
        """
        Select appropriate background music based on genre and mood.
        Randomly picks one track from available options.

        Args:
            genre: Story genre (e.g., "emotional", "thriller")
            mood: Overall mood (e.g., "dramatic", "happy")
            duration_sec: Required music duration (for future use)

        Returns:
            Path to selected music file, or None
        """
        print(f"[Music Agent] Selecting BGM...")
        print(f"   Genre: {genre}, Mood: {mood}, Duration: {duration_sec}s")

        # 1. mood로 카테고리 매칭
        category = _MOOD_MAP.get(mood.lower())

        # 2. mood 매칭 실패 -> genre fallback
        if not category:
            category = _GENRE_MAP.get(genre.lower())

        # 3. 둘 다 실패 -> 기본값
        if not category:
            category = "calm_peaceful"
            print(f"   [Music Agent] No match for genre='{genre}', mood='{mood}' -> default: {category}")

        # 카테고리에서 사용 가능한 트랙 탐색
        tracks = self._find_tracks(category)

        if tracks:
            selected = random.choice(tracks)
            print(f"   [Music Agent] Category: {category} ({len(tracks)} tracks available)")
            print(f"   [Music Agent] Selected: {os.path.basename(selected)}")
            return selected

        # 전체 fallback
        print(f"   [Warning] No tracks found for category '{category}'")
        placeholder = os.path.join(self.music_library_path, "placeholder_music.mp3")
        if os.path.exists(placeholder):
            print(f"   [Music Agent] Fallback: placeholder_music.mp3")
            return placeholder
        return None
