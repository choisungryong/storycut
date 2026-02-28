"""
StoryCut 공통 상수 모듈

프로젝트 전체에서 반복 사용되는 상수를 단일 소스로 관리합니다.
"""
import os

# ─── Backend URL ───────────────────────────────────────────
BACKEND_BASE_URL = os.getenv(
    "RAILWAY_PUBLIC_DOMAIN",
    "https://web-production-bb6bf.up.railway.app",
)
if BACKEND_BASE_URL and not BACKEND_BASE_URL.startswith("http"):
    BACKEND_BASE_URL = f"https://{BACKEND_BASE_URL}"

# ─── Gemini 모델명 ────────────────────────────────────────
MODEL_GEMINI_FLASH = "gemini-2.5-flash"
MODEL_GEMINI_PRO = "gemini-2.5-pro"
MODEL_GEMINI_FLASH_IMAGE = "gemini-2.5-flash-image"

# ─── 인종 키워드 매핑 (공용) ──────────────────────────────
ETH_KEYWORD_MAP = {
    "korean": "Korean",
    "japanese": "Japanese",
    "chinese": "Chinese",
    "southeast_asian": "Southeast Asian",
    "south_asian": "South Asian",
    "middle_eastern": "Middle Eastern",
    "african": "African",
    "european": "European",
    "black": "Black",
    "hispanic": "Hispanic",
    "hispanic_latino": "Hispanic/Latino",
    "caucasian": "Caucasian",
    "pacific_islander": "Pacific Islander",
    "mixed": "Mixed ethnicity",
}

# ─── 인종별 프롬프트 지시문 ───────────────────────────────
ETH_INSTRUCTIONS = {
    "korean": "All characters MUST be Korean. Describe them with Korean facial features, skin tone, and names.",
    "japanese": "All characters MUST be Japanese. Describe them with Japanese facial features, skin tone, and names.",
    "chinese": "All characters MUST be Chinese. Describe them with Chinese facial features, skin tone, and names.",
    "southeast_asian": "All characters MUST be Southeast Asian. Describe them with Southeast Asian facial features and skin tone.",
    "european": "All characters MUST be European/Caucasian. Describe them with European facial features, light skin, and Western names.",
    "black": "All characters MUST be Black/African. Describe them with African facial features, dark skin tone, and appropriate names.",
    "hispanic": "All characters MUST be Hispanic/Latino. Describe them with Latin American features and Spanish names.",
    "mixed": "Characters should be a MIX of different ethnicities. Make each character a different race for diversity.",
}

# ─── 인종별 규칙 텍스트 (pipeline.py 용) ─────────────────
ETH_RULES = {
    "korean": "All characters MUST be described as 'Korean' (e.g., 'Korean man', 'Korean woman').",
    "japanese": "All characters MUST be described as 'Japanese'.",
    "chinese": "All characters MUST be described as 'Chinese'.",
    "southeast_asian": "All characters MUST be described as 'Southeast Asian'.",
    "european": "All characters MUST be described as 'European/Caucasian'.",
    "black": "All characters MUST be described as 'Black/African'.",
    "hispanic": "All characters MUST be described as 'Hispanic/Latino'.",
    "mixed": "Each character's specific ethnicity MUST be stated explicitly.",
}
