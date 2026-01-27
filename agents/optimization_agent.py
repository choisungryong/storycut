"""
Optimization Agent: 제목/썸네일/AB 테스트 패키지 생성

P2 핵심 기능:
- 유튜브 제목 후보 3종 (충격/질문/요약 스타일)
- 썸네일 문구 3종
- 썸네일 이미지 프롬프트 2종
- 해시태그 10개
- AB 테스트 메타데이터 JSON
"""

import os
import json
from typing import Dict, Any, List, Optional
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))

from schemas import FeatureFlags, Scene, ProjectRequest


class OptimizationAgent:
    """
    유튜브 최적화 에이전트

    P2: 제목/썸네일/AB 테스트 패키지 생성
    - 생성 후 끝이 아니라, 게시/실험까지 지원
    """

    def __init__(self, api_key: str = None):
        """
        Initialize Optimization Agent.

        Args:
            api_key: OpenAI API key
        """
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self._llm_client = None

    @property
    def llm_client(self):
        """Lazy initialization of LLM client."""
        if self._llm_client is None and self.api_key:
            try:
                from openai import OpenAI
                self._llm_client = OpenAI(api_key=self.api_key)
            except Exception as e:
                print(f"  Failed to initialize LLM client: {e}")
                self._llm_client = None
        return self._llm_client

    def run(
        self,
        topic: str,
        script: str,
        scenes: List[Scene],
        request: ProjectRequest
    ) -> Dict[str, Any]:
        """
        최적화 패키지 생성.

        P2: 제목/썸네일/AB 테스트 메타데이터 생성

        Args:
            topic: 영상 주제
            script: 전체 스크립트
            scenes: Scene 목록
            request: ProjectRequest

        Returns:
            최적화 패키지 딕셔너리
        """
        print("\n  [OPTIMIZATION] Generating YouTube optimization package...")

        if not request.feature_flags.optimization_pack:
            print("  [OPTIMIZATION] Feature disabled. Skipping.")
            return {}

        # 스크립트 요약
        script_summary = self._summarize_script(script)

        # LLM이 없으면 기본값 반환
        if not self.llm_client:
            print("  [OPTIMIZATION] No LLM client. Using default package.")
            return self._get_default_package(topic, script_summary)

        # LLM으로 최적화 패키지 생성
        try:
            result = self._generate_with_llm(
                topic=topic,
                script_summary=script_summary,
                target_platform=request.target_platform.value,
                genre=request.genre,
                mood=request.mood
            )
            print("  [OPTIMIZATION] Package generated successfully.")
            return result

        except Exception as e:
            print(f"  [OPTIMIZATION] LLM generation failed: {e}")
            return self._get_default_package(topic, script_summary)

    def _summarize_script(self, script: str, max_length: int = 200) -> str:
        """스크립트 요약."""
        if len(script) <= max_length:
            return script
        return script[:max_length] + "..."

    def _generate_with_llm(
        self,
        topic: str,
        script_summary: str,
        target_platform: str,
        genre: str = None,
        mood: str = None
    ) -> Dict[str, Any]:
        """
        LLM을 사용하여 최적화 패키지 생성.

        Args:
            topic: 영상 주제
            script_summary: 스크립트 요약
            target_platform: 대상 플랫폼
            genre: 장르
            mood: 분위기

        Returns:
            최적화 패키지
        """
        prompt = f"""
너는 유튜브 그로스 매니저다.

주제: {topic}
스크립트 요약: {script_summary}
타깃 플랫폼: {target_platform}
장르: {genre or "미지정"}
분위기: {mood or "미지정"}

다음 JSON 형식으로 출력해줘:

{{
    "title_candidates": [
        "충격형 제목 (자극적, 클릭 유도)",
        "질문형 제목 (호기심 유발)",
        "요약형 제목 (정보 전달)"
    ],
    "thumbnail_texts": [
        "썸네일 문구 1 (2-5단어, 임팩트)",
        "썸네일 문구 2 (2-5단어)",
        "썸네일 문구 3 (2-5단어)"
    ],
    "thumbnail_prompts": [
        "썸네일 이미지 프롬프트 1 (과장된 감정, 높은 대비, 영어로)",
        "썸네일 이미지 프롬프트 2 (다른 스타일, 영어로)"
    ],
    "hashtags": ["#태그1", "#태그2", "#태그3", "#태그4", "#태그5", "#태그6", "#태그7", "#태그8", "#태그9", "#태그10"],
    "description": "유튜브 영상 설명문 (2-3문장, 해시태그 포함)",
    "ab_test_meta": {{
        "titleA": "충격형 제목",
        "titleB": "질문형 제목",
        "thumbnailA": "썸네일 프롬프트 1",
        "thumbnailB": "썸네일 프롬프트 2",
        "test_hypothesis": "AB 테스트 가설"
    }}
}}

주의:
- 제목은 40자 이내로
- 썸네일 문구는 5단어 이내로
- 썸네일 프롬프트는 영어로, 과장된 표정/감정 강조
- 해시태그는 관련성 높은 순으로
- 허위/명예훼손/불법 유도 금지
"""

        response = self.llm_client.chat.completions.create(
            model="gpt-4",
            messages=[
                {
                    "role": "system",
                    "content": "JSON만 출력하세요. 마크다운 코드 블록 없이 순수 JSON만."
                },
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=1000
        )

        content = response.choices[0].message.content.strip()

        # JSON 파싱
        if content.startswith("```"):
            lines = content.split("\n")
            json_lines = []
            in_json = False
            for line in lines:
                if line.startswith("```json"):
                    in_json = True
                    continue
                if line.startswith("```"):
                    in_json = False
                    continue
                if in_json:
                    json_lines.append(line)
            content = "\n".join(json_lines)

        result = json.loads(content)
        return result

    def _get_default_package(
        self,
        topic: str,
        script_summary: str
    ) -> Dict[str, Any]:
        """
        기본 최적화 패키지 반환.

        Args:
            topic: 영상 주제
            script_summary: 스크립트 요약

        Returns:
            기본 패키지
        """
        short_topic = topic[:30] if topic else "이 영상"

        return {
            "title_candidates": [
                f"충격! {short_topic}의 진실이 밝혀졌다",
                f"{short_topic}, 당신도 몰랐던 사실?",
                f"{short_topic} 완벽 정리",
            ],
            "thumbnail_texts": [
                "충격 반전",
                "이게 실화?",
                "몰랐던 진실",
            ],
            "thumbnail_prompts": [
                f"Dramatic close-up portrait, shocked expression, high contrast lighting, "
                f"cinematic, bold colors, YouTube thumbnail style, 4K quality",
                f"Split image composition, before and after concept, dramatic lighting, "
                f"emotional face expression, vibrant colors, eye-catching design",
            ],
            "hashtags": [
                "#유튜브", "#쇼츠", "#shorts", "#viral",
                "#충격", "#반전", "#실화", "#핫이슈",
                "#추천", "#인기"
            ],
            "description": f"{short_topic}에 대한 모든 것을 담았습니다. "
                          f"끝까지 시청해주세요! #유튜브 #쇼츠",
            "ab_test_meta": {
                "titleA": f"충격! {short_topic}의 진실이 밝혀졌다",
                "titleB": f"{short_topic}, 당신도 몰랐던 사실?",
                "thumbnailA": "Dramatic close-up style",
                "thumbnailB": "Split composition style",
                "test_hypothesis": "충격형 제목이 질문형보다 CTR이 높을 것으로 예상"
            }
        }

    def generate_titles(
        self,
        topic: str,
        style: str = "mixed"
    ) -> List[str]:
        """
        제목 후보만 생성.

        Args:
            topic: 영상 주제
            style: 스타일 (shock, question, summary, mixed)

        Returns:
            제목 후보 목록
        """
        if not self.llm_client:
            return self._get_default_package(topic, "")["title_candidates"]

        style_instructions = {
            "shock": "충격적이고 자극적인 제목 3개",
            "question": "호기심을 유발하는 질문형 제목 3개",
            "summary": "핵심 정보를 전달하는 요약형 제목 3개",
            "mixed": "충격형 1개, 질문형 1개, 요약형 1개"
        }

        prompt = f"""
주제: {topic}
요청: {style_instructions.get(style, style_instructions['mixed'])}

JSON 배열로 출력:
["제목1", "제목2", "제목3"]

주의: 40자 이내, 클릭 유도, 허위/불법 금지
"""

        try:
            response = self.llm_client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "JSON 배열만 출력"},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.8,
                max_tokens=300
            )
            content = response.choices[0].message.content.strip()
            return json.loads(content)
        except Exception:
            return self._get_default_package(topic, "")["title_candidates"]

    def generate_thumbnail_prompts(
        self,
        topic: str,
        mood: str = "dramatic"
    ) -> List[str]:
        """
        썸네일 이미지 프롬프트 생성.

        Args:
            topic: 영상 주제
            mood: 분위기

        Returns:
            썸네일 프롬프트 목록
        """
        if not self.llm_client:
            return self._get_default_package(topic, "")["thumbnail_prompts"]

        prompt = f"""
주제: {topic}
분위기: {mood}

유튜브 썸네일용 이미지 프롬프트 2개를 영어로 작성해줘.

요구사항:
- 과장된 감정/표정
- 높은 대비, 선명한 색상
- 클릭을 유도하는 비주얼
- 4K 품질

JSON 배열로 출력:
["prompt1", "prompt2"]
"""

        try:
            response = self.llm_client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "JSON 배열만 출력"},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=400
            )
            content = response.choices[0].message.content.strip()
            return json.loads(content)
        except Exception:
            return self._get_default_package(topic, "")["thumbnail_prompts"]

    def generate_hashtags(
        self,
        topic: str,
        platform: str = "youtube"
    ) -> List[str]:
        """
        해시태그 생성.

        Args:
            topic: 영상 주제
            platform: 플랫폼

        Returns:
            해시태그 목록
        """
        if not self.llm_client:
            return self._get_default_package(topic, "")["hashtags"]

        prompt = f"""
주제: {topic}
플랫폼: {platform}

관련 해시태그 10개를 생성해줘.
- 관련성 높은 순서로
- 인기 있는 태그 포함
- **주로 한글 해시태그**를 사용 (필요한 경우에만 영어 태그 1-2개 포함)

JSON 배열로 출력:
["#태그1", "#태그2", ...]
"""

        try:
            response = self.llm_client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "JSON 배열만 출력"},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.6,
                max_tokens=200
            )
            content = response.choices[0].message.content.strip()
            return json.loads(content)
        except Exception:
            return self._get_default_package(topic, "")["hashtags"]

    def save_optimization_package(
        self,
        package: Dict[str, Any],
        output_dir: str,
        project_id: str
    ) -> str:
        """
        최적화 패키지를 JSON 파일로 저장.

        Args:
            package: 최적화 패키지
            output_dir: 출력 디렉토리
            project_id: 프로젝트 ID

        Returns:
            저장된 파일 경로
        """
        os.makedirs(output_dir, exist_ok=True)
        output_path = f"{output_dir}/optimization_{project_id}.json"

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(package, f, ensure_ascii=False, indent=2)

        print(f"  [OPTIMIZATION] Saved to: {output_path}")
        return output_path

    def get_ab_test_config(
        self,
        package: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        AB 테스트 설정 추출.

        Args:
            package: 최적화 패키지

        Returns:
            AB 테스트 설정
        """
        ab_meta = package.get("ab_test_meta", {})

        return {
            "variants": [
                {
                    "id": "A",
                    "title": ab_meta.get("titleA", package.get("title_candidates", [""])[0]),
                    "thumbnail_prompt": ab_meta.get("thumbnailA", ""),
                },
                {
                    "id": "B",
                    "title": ab_meta.get("titleB", package.get("title_candidates", ["", ""])[1] if len(package.get("title_candidates", [])) > 1 else ""),
                    "thumbnail_prompt": ab_meta.get("thumbnailB", ""),
                }
            ],
            "hypothesis": ab_meta.get("test_hypothesis", ""),
            "metrics": ["ctr", "watch_time", "engagement"],
            "duration_days": 7,
        }
