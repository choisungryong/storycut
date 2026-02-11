"""
Pexels B-roll Agent - 무료 스톡 영상으로 intro/outro/bridge 씬 대체
"""

import os
import requests
from typing import Optional, List, Dict, Any


# 장르별 검색 키워드
GENRE_KEYWORDS: Dict[str, List[str]] = {
    "fantasy": ["mystical forest", "ethereal nature", "magical landscape"],
    "romance": ["city sunset", "bokeh lights", "rain street"],
    "action": ["urban night", "motion blur", "dynamic skyline"],
    "scifi": ["futuristic neon", "technology abstract", "space nebula"],
    "horror": ["dark fog", "abandoned building", "eerie shadows"],
    "drama": ["emotional atmosphere", "window rain", "solitary figure silhouette"],
    "comedy": ["bright colorful", "playful atmosphere", "sunny day"],
    "abstract": ["abstract motion", "flowing colors", "geometric patterns"],
}

# 세그먼트별 검색 보조 키워드
SEGMENT_MODIFIERS: Dict[str, str] = {
    "intro": "establishing shot cinematic",
    "outro": "sunset fade ending",
    "bridge": "transition atmospheric",
}


class PexelsAgent:
    """Pexels Video Search API를 통해 B-roll 영상을 검색/다운로드"""

    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.getenv("PEXELS_API_KEY")
        self.base_url = "https://api.pexels.com/videos"

    def search_videos(
        self,
        query: str,
        per_page: int = 15,
        orientation: str = "landscape",
        min_duration: int = 3,
        max_duration: int = 30,
    ) -> List[Dict[str, Any]]:
        """Pexels Video Search API 호출

        Args:
            query: 검색 키워드
            per_page: 결과 수 (최대 80)
            orientation: landscape/portrait/square
            min_duration: 최소 길이 (초)
            max_duration: 최대 길이 (초)

        Returns:
            비디오 결과 리스트
        """
        if not self.api_key:
            return []

        headers = {"Authorization": self.api_key}
        params = {
            "query": query,
            "per_page": per_page,
            "orientation": orientation,
        }

        try:
            resp = requests.get(
                f"{self.base_url}/search",
                headers=headers,
                params=params,
                timeout=15,
            )
            if not resp.ok:
                print(f"    [Pexels] Search failed: HTTP {resp.status_code}")
                return []

            data = resp.json()
            videos = data.get("videos", [])

            # duration 필터
            filtered = [
                v for v in videos
                if min_duration <= v.get("duration", 0) <= max_duration
            ]
            return filtered

        except Exception as e:
            print(f"    [Pexels] Search error: {e}")
            return []

    def download_video(
        self,
        video_data: Dict[str, Any],
        out_path: str,
        target_quality: str = "hd",
    ) -> Optional[str]:
        """비디오 다운로드 (HD 우선, SD 폴백)

        Args:
            video_data: Pexels API video object
            out_path: 저장 경로
            target_quality: hd or sd

        Returns:
            다운로드 경로 또는 None
        """
        video_files = video_data.get("video_files", [])
        if not video_files:
            return None

        # HD(1920x1080) landscape 우선
        hd_files = [
            f for f in video_files
            if f.get("quality") == "hd"
            and f.get("width", 0) >= 1280
            and f.get("height", 0) >= 720
        ]
        sd_files = [
            f for f in video_files
            if f.get("quality") == "sd"
            and f.get("width", 0) >= 640
        ]

        chosen = None
        if target_quality == "hd" and hd_files:
            # 1920x1080에 가장 가까운 것
            hd_files.sort(key=lambda f: abs(f.get("width", 0) - 1920))
            chosen = hd_files[0]
        elif sd_files:
            sd_files.sort(key=lambda f: -f.get("width", 0))
            chosen = sd_files[0]
        elif hd_files:
            chosen = hd_files[0]
        elif video_files:
            # 아무거나
            chosen = video_files[0]

        if not chosen:
            return None

        download_url = chosen.get("link")
        if not download_url:
            return None

        try:
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            resp = requests.get(download_url, stream=True, timeout=30)
            if not resp.ok:
                print(f"    [Pexels] Download failed: HTTP {resp.status_code}")
                return None

            with open(out_path, "wb") as f:
                for chunk in resp.iter_content(chunk_size=8192):
                    f.write(chunk)

            if os.path.exists(out_path) and os.path.getsize(out_path) > 0:
                return out_path
            return None

        except Exception as e:
            print(f"    [Pexels] Download error: {e}")
            return None

    def build_query(
        self,
        genre: str,
        mood: str,
        segment_type: str,
        visual_bible=None,
    ) -> str:
        """장르 + 세그먼트 + Visual Bible 기반 검색 쿼리 생성

        Args:
            genre: MVGenre value (e.g. "fantasy")
            mood: MVMood value (e.g. "epic")
            segment_type: "intro", "outro", "bridge"
            visual_bible: VisualBible 객체 (Optional)

        Returns:
            Pexels 검색 쿼리
        """
        parts = []

        # 장르 키워드 (첫 번째)
        genre_kw = GENRE_KEYWORDS.get(genre, ["cinematic landscape"])
        parts.append(genre_kw[0])

        # 세그먼트 모디파이어
        modifier = SEGMENT_MODIFIERS.get(segment_type, "cinematic")
        parts.append(modifier)

        # 분위기
        parts.append(mood)

        # Visual Bible 모티프 (상위 2개)
        if visual_bible:
            motifs = getattr(visual_bible, "recurring_motifs", None)
            if motifs and isinstance(motifs, list):
                for m in motifs[:2]:
                    parts.append(m)

        return " ".join(parts)

    def fetch_broll(
        self,
        query: str,
        duration_sec: float,
        out_path: str,
    ) -> Optional[str]:
        """B-roll 영상 검색 + 다운로드 통합 메서드

        Args:
            query: 검색 쿼리
            duration_sec: 목표 길이 (초)
            out_path: 저장 경로

        Returns:
            다운로드된 파일 경로 또는 None
        """
        videos = self.search_videos(
            query=query,
            per_page=15,
            orientation="landscape",
            min_duration=max(1, int(duration_sec) - 3),
            max_duration=int(duration_sec) + 15,
        )

        if not videos:
            return None

        # duration 매칭: 목표 길이에 가장 가까운 것 선택
        videos.sort(key=lambda v: abs(v.get("duration", 0) - duration_sec))

        # 상위 3개 중 다운로드 시도
        for video in videos[:3]:
            result = self.download_video(video, out_path, target_quality="hd")
            if result:
                return result

        return None
