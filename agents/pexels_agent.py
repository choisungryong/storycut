"""
Pexels B-roll Agent - 무료 스톡 영상으로 intro/outro/bridge 씬 대체
LLM 기반 스톡 검색 쿼리 생성 + 멀티 쿼리 순차 시도
"""

import os
import json
import random
import requests
from typing import Optional, List, Dict, Any, Set


# 장르별 폴백 검색 키워드 (LLM 실패 시 사용)
GENRE_KEYWORDS: Dict[str, List[str]] = {
    "fantasy": ["mystical forest b-roll", "ethereal nature cinematic", "magical landscape drone"],
    "romance": ["city sunset b-roll", "bokeh lights slow motion", "rain street cinematic"],
    "action": ["urban night b-roll", "motion blur cinematic", "dynamic skyline drone"],
    "scifi": ["futuristic neon b-roll", "technology abstract slow motion", "space nebula cinematic"],
    "horror": ["dark fog b-roll", "abandoned building cinematic", "eerie shadows slow motion"],
    "drama": ["emotional atmosphere b-roll", "window rain close-up", "solitary silhouette cinematic"],
    "comedy": ["bright colorful b-roll", "playful atmosphere cinematic", "sunny day slow motion"],
    "abstract": ["abstract motion b-roll", "flowing colors slow motion", "geometric patterns cinematic"],
}

SEGMENT_MODIFIERS: Dict[str, str] = {
    "intro": "establishing shot",
    "outro": "sunset ending",
    "bridge": "transition atmospheric",
}

# LLM 스톡 쿼리 생성 프롬프트
_STOCK_QUERY_SYSTEM_PROMPT = """You are generating STOCK SEARCH QUERIES for short b-roll video clips (Pexels).
Input: lyric line + scene description + shot role + genre + mood + environment context.
Output: JSON with 3-8 English queries.

Rules:
- Queries must be 2-6 words each.
- Add "b-roll" to at least half of the queries.
- Include at least 1 query with a camera/style term: "cinematic", "slow motion", "handheld", "drone", "close-up", "silhouette", "bokeh".
- role is always "broll" (never "hero"). Avoid identity-specific queries (no character names, no celebrity).
- Prefer environment/texture/detail shots over faces.
- Include variety: (place/object) + (time/weather) + (camera/style) combos.

ENVIRONMENT MATCHING (critical):
- If concept or era_setting is given, ALL queries MUST match that setting. Medieval → castles/stone/candlelight. Futuristic → neon/cyber/hologram.
- If locale is given (e.g. "Korean", "Japanese"), prefer that region's urban/cultural aesthetics. Korean → Seoul city, hanok, neon alley. Japanese → Tokyo, shrine. European → cobblestone, cathedral.
- If time_of_day is given, ALL queries must match: "night" → neon/streetlight/moonlight, "dawn" → sunrise/golden hour, "day" → bright daylight.
- If weather/climate is given, include it: "rain" → rain drops/wet street, "snow" → snowfall/winter, "fog" → misty/hazy.
- If lighting is given, match it: "warm golden hour" → golden light, "cold blue" → blue tone night.
- If location is given, use it directly: "rooftop" → city rooftop b-roll, "forest" → forest b-roll.
- IMPORTANT: Do NOT default to tropical/beach/palm tree videos. Unless concept explicitly mentions "tropical"/"beach"/"island", avoid tropical imagery. Prefer URBAN, INDOOR, or WEATHER shots as generic fillers.

Return ONLY valid JSON, no markdown:
{"stock_query":[...],"notes":"brief reasoning"}"""


class PexelsAgent:
    """Pexels Video Search API를 통해 B-roll 영상을 검색/다운로드"""

    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.getenv("PEXELS_API_KEY")
        self.base_url = "https://api.pexels.com/videos"
        # 세션 내 중복 방지: 이미 사용된 영상 ID 추적
        self.used_video_ids: Set[int] = set()

    def search_videos(
        self,
        query: str,
        per_page: int = 15,
        orientation: str = "landscape",
        min_duration: int = 3,
        max_duration: int = 30,
    ) -> List[Dict[str, Any]]:
        """Pexels Video Search API 호출"""
        if not self.api_key:
            return []

        headers = {"Authorization": self.api_key}
        # 랜덤 페이지(1-3)로 다양한 결과 확보
        page = random.randint(1, 3)
        params = {
            "query": query,
            "per_page": per_page,
            "orientation": orientation,
            "page": page,
        }

        try:
            resp = requests.get(
                f"{self.base_url}/search",
                headers=headers,
                params=params,
                timeout=15,
            )
            if not resp.ok:
                print(f"    [Pexels] Search failed: HTTP {resp.status_code} (page={page})")
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
        """비디오 다운로드 (HD 우선, SD 폴백)"""
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
            hd_files.sort(key=lambda f: abs(f.get("width", 0) - 1920))
            chosen = hd_files[0]
        elif sd_files:
            sd_files.sort(key=lambda f: -f.get("width", 0))
            chosen = sd_files[0]
        elif hd_files:
            chosen = hd_files[0]
        elif video_files:
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

    # ------------------------------------------------------------------
    # LLM 기반 스톡 검색 쿼리 생성
    # ------------------------------------------------------------------

    def generate_stock_queries(
        self,
        scene_prompt: str = None,
        lyrics_text: str = None,
        segment_type: str = "intro",
        genre: str = "drama",
        mood: str = "epic",
        concept: str = None,
        era_setting: str = None,
        color_palette: List[str] = None,
        locale: str = None,
        time_of_day: str = None,
        weather: str = None,
        lighting: str = None,
        location: str = None,
        atmosphere: str = None,
    ) -> List[str]:
        """Gemini LLM으로 Pexels 검색 쿼리 3-8개 생성

        Args:
            scene_prompt: 디렉터 LLM이 생성한 씬 프롬프트
            lyrics_text: 해당 씬의 가사 (없을 수 있음)
            segment_type: intro/outro/bridge
            genre: 장르
            mood: 분위기
            concept: 비주얼 컨셉 (예: "중세 유럽 성", "네온 도시")
            era_setting: 시대/배경 (예: "medieval", "futuristic")
            color_palette: 색상 팔레트 (예: ["deep blue", "gold"])
            locale: 지역/인종 (예: "Korean", "Japanese")
            time_of_day: 시간대 (예: "night", "dawn", "golden hour")
            weather: 날씨 (예: "rain", "snow", "fog")
            lighting: 조명 스타일 (예: "warm golden hour", "cold blue neon")
            location: 장소 (예: "rooftop", "forest", "city street")
            atmosphere: 전체 분위기 (예: "melancholic urban decay")

        Returns:
            검색 쿼리 리스트 (실패 시 폴백 쿼리)
        """
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key or not scene_prompt:
            return self._fallback_queries(genre, segment_type, mood)

        try:
            from google import genai
            from google.genai import types

            client = genai.Client(api_key=api_key)

            context_lines = []
            if concept:
                context_lines.append(f"concept={concept}")
            if era_setting:
                context_lines.append(f"era_setting={era_setting}")
            if color_palette:
                context_lines.append(f"color_palette={', '.join(color_palette[:4])}")
            if locale:
                context_lines.append(f"locale={locale}")
            if time_of_day:
                context_lines.append(f"time_of_day={time_of_day}")
            if weather:
                context_lines.append(f"weather={weather}")
            if lighting:
                context_lines.append(f"lighting={lighting}")
            if location:
                context_lines.append(f"location={location}")
            if atmosphere:
                context_lines.append(f"atmosphere={atmosphere}")
            context_str = "\n".join(context_lines)

            user_prompt = (
                f"role=broll\n"
                f"lyric={lyrics_text or '(instrumental)'}\n"
                f"scene={scene_prompt}\n"
                f"genre={genre}\n"
                f"mood={mood}\n"
                f"segment={segment_type}"
            )
            if context_str:
                user_prompt += f"\n{context_str}"

            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=user_prompt,
                config=types.GenerateContentConfig(
                    system_instruction=_STOCK_QUERY_SYSTEM_PROMPT,
                    temperature=0.7,
                )
            )

            text = response.text.strip()
            # JSON 파싱 (마크다운 래퍼 제거)
            if text.startswith("```"):
                text = text.split("```")[1]
                if text.startswith("json"):
                    text = text[4:]
                text = text.strip()

            data = json.loads(text)
            queries = data.get("stock_query", [])
            notes = data.get("notes", "")

            if queries and isinstance(queries, list):
                # tropical/palm/beach 쿼리 필터링 (concept에 명시적으로 없는 경우)
                _concept_lower = (concept or "").lower()
                _allow_tropical = any(kw in _concept_lower for kw in ["tropical", "beach", "island", "hawaii", "bali", "caribbean"])
                if not _allow_tropical:
                    _BANNED = {"tropical", "palm", "palm tree", "beach", "coconut", "ocean wave", "island"}
                    filtered = [q for q in queries if not any(b in q.lower() for b in _BANNED)]
                    if filtered:
                        removed = len(queries) - len(filtered)
                        if removed > 0:
                            print(f"    [Pexels] Filtered {removed} tropical queries")
                        queries = filtered
                print(f"    [Pexels] LLM generated {len(queries)} queries: {queries[:3]}...")
                if notes:
                    print(f"    [Pexels] Reasoning: {notes[:80]}")
                return queries

        except Exception as e:
            print(f"    [Pexels] LLM query generation failed: {e}")

        return self._fallback_queries(genre, segment_type, mood)

    def _fallback_queries(self, genre: str, segment_type: str, mood: str) -> List[str]:
        """LLM 실패 시 장르/세그먼트 기반 폴백 쿼리"""
        keywords = GENRE_KEYWORDS.get(genre, ["cinematic landscape b-roll"])
        modifier = SEGMENT_MODIFIERS.get(segment_type, "cinematic")
        return keywords + [f"{mood} {modifier} b-roll"]

    # ------------------------------------------------------------------
    # 통합 B-roll Fetch (멀티 쿼리 순차 시도)
    # ------------------------------------------------------------------

    def fetch_broll(
        self,
        queries: List[str],
        duration_sec: float,
        out_path: str,
    ) -> Optional[str]:
        """여러 검색 쿼리를 순차 시도하여 B-roll 다운로드

        Args:
            queries: 검색 쿼리 리스트 (우선순위 순)
            duration_sec: 목표 길이 (초)
            out_path: 저장 경로

        Returns:
            다운로드된 파일 경로 또는 None
        """
        for qi, query in enumerate(queries):
            videos = self.search_videos(
                query=query,
                per_page=15,
                orientation="landscape",
                min_duration=max(1, int(duration_sec) - 3),
                max_duration=int(duration_sec) + 15,
            )

            if not videos:
                continue

            # 이미 사용된 영상 제외
            fresh = [v for v in videos if v.get("id") not in self.used_video_ids]
            if not fresh:
                fresh = videos  # 모두 사용된 경우 전체에서 다시 선택

            # duration 매칭: 목표 길이에 가장 가까운 순 정렬 후 상위 5개 중 랜덤 선택
            fresh.sort(key=lambda v: abs(v.get("duration", 0) - duration_sec))
            candidates = fresh[:5]
            random.shuffle(candidates)

            for video in candidates:
                result = self.download_video(video, out_path, target_quality="hd")
                if result:
                    vid_id = video.get("id")
                    if vid_id:
                        self.used_video_ids.add(vid_id)
                    print(f"    [Pexels] Found with query #{qi+1}: '{query}' (id={vid_id}, used={len(self.used_video_ids)})")
                    return result

        return None
