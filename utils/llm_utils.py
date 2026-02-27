"""
LLM 응답 파싱 유틸리티

LLM(Gemini, OpenAI 등)이 반환하는 마크다운 래핑 JSON을 안전하게 파싱합니다.
"""
import json


def parse_llm_json(text: str) -> dict:
    """LLM 응답에서 마크다운 코드블록 제거 후 JSON 파싱.

    지원 패턴:
      - ```json ... ```
      - ``` ... ```
      - 순수 JSON
    """
    text = text.strip()
    if text.startswith("```"):
        # split 방식: ```json\n{...}\n```  →  ["", "json\n{...}\n", ""]
        parts = text.split("```")
        if len(parts) >= 3:
            inner = parts[1]
            if inner.startswith("json"):
                inner = inner[4:]
            text = inner.strip()
        else:
            # 닫는 ``` 없는 경우 fallback
            text = text.split("\n", 1)[1] if "\n" in text else text[3:]
            if text.endswith("```"):
                text = text[:-3]
            text = text.strip()
    return json.loads(text)
