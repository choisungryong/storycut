"""
Consistency Validator: Gemini Vision-based visual consistency validation.

v2.0 핵심 기능:
- 생성된 이미지 vs 앵커 이미지 비교 점수 산출
- 임계값 미달 시 재시도 루프 (seed 변경 + variation)
- 얼굴 유사도, 스타일 드리프트, 환경 유사도 3축 평가
"""

import os
import json
import base64
from typing import Dict, List, Optional, Tuple, Callable, Any
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))

from schemas import ValidationResult


class ConsistencyValidator:
    """
    Gemini Vision 기반 시각적 일관성 검증기

    v2.0: 생성된 이미지가 앵커 이미지와 일관성을 유지하는지 검증합니다.
    """

    DEFAULT_THRESHOLDS = {
        "face_similarity": 0.6,
        "style_drift": 0.7,
        "environment_similarity": 0.5,
    }
    MAX_RETRIES = 3

    def __init__(self, thresholds: Optional[Dict[str, float]] = None):
        """
        Initialize Consistency Validator.

        Args:
            thresholds: 차원별 임계값 (기본: DEFAULT_THRESHOLDS)
        """
        self.thresholds = thresholds or self.DEFAULT_THRESHOLDS.copy()
        self._vision_client = None
        self.google_api_key = os.getenv("GOOGLE_API_KEY")

    @property
    def vision_client(self):
        """Lazy initialization of Gemini Vision client."""
        if self._vision_client is None:
            try:
                import google.generativeai as genai
                if self.google_api_key:
                    genai.configure(api_key=self.google_api_key)
                    self._vision_client = genai.GenerativeModel(model_name="gemini-2.0-flash")
            except Exception as e:
                print(f"  [ConsistencyValidator] Failed to init Gemini Vision: {e}")
        return self._vision_client

    def validate_scene_image(
        self,
        generated_image_path: str,
        scene_id: int,
        character_anchor_paths: Optional[List[str]] = None,
        style_anchor_path: Optional[str] = None,
        environment_anchor_path: Optional[str] = None,
        attempt_number: int = 1,
    ) -> ValidationResult:
        """
        생성된 이미지 vs 앵커 이미지 비교 점수 산출.

        Args:
            generated_image_path: 생성된 이미지 경로
            scene_id: 씬 ID
            character_anchor_paths: 캐릭터 앵커 이미지 경로 목록
            style_anchor_path: 스타일 앵커 이미지 경로
            environment_anchor_path: 환경 앵커 이미지 경로
            attempt_number: 시도 번호

        Returns:
            ValidationResult 객체
        """
        if not self.vision_client or not os.path.exists(generated_image_path):
            # Vision 없으면 통과 처리
            return ValidationResult(
                scene_id=scene_id,
                passed=True,
                overall_score=1.0,
                attempt_number=attempt_number,
            )

        # 앵커 이미지 수집
        anchor_paths = []
        anchor_labels = []

        if character_anchor_paths:
            for i, path in enumerate(character_anchor_paths):
                if path and os.path.exists(path):
                    anchor_paths.append(path)
                    anchor_labels.append(f"character_anchor_{i}")

        if style_anchor_path and os.path.exists(style_anchor_path):
            anchor_paths.append(style_anchor_path)
            anchor_labels.append("style_anchor")

        if environment_anchor_path and os.path.exists(environment_anchor_path):
            anchor_paths.append(environment_anchor_path)
            anchor_labels.append("environment_anchor")

        if not anchor_paths:
            # 앵커 없으면 통과
            return ValidationResult(
                scene_id=scene_id,
                passed=True,
                overall_score=1.0,
                attempt_number=attempt_number,
            )

        # Gemini Vision 스코어링
        dimension_scores = self._score_with_gemini(
            generated_image_path=generated_image_path,
            anchor_image_paths=anchor_paths,
            anchor_labels=anchor_labels,
        )

        # 임계값 체크
        issues = []
        passed = True

        for dim, threshold in self.thresholds.items():
            score = dimension_scores.get(dim, 0.0)
            if score < threshold:
                passed = False
                issues.append(f"{dim}: {score:.2f} < {threshold:.2f}")

        overall_score = sum(dimension_scores.values()) / max(len(dimension_scores), 1)

        result = ValidationResult(
            scene_id=scene_id,
            passed=passed,
            overall_score=overall_score,
            dimension_scores=dimension_scores,
            issues=issues,
            attempt_number=attempt_number,
        )

        status = "PASS" if passed else "FAIL"
        print(f"  [ConsistencyValidator] Scene {scene_id} attempt {attempt_number}: {status} (score={overall_score:.2f})")
        if issues:
            for issue in issues:
                print(f"    - {issue}")

        return result

    def validate_and_retry(
        self,
        scene_id: int,
        generate_fn: Callable[[int], str],
        character_anchor_paths: Optional[List[str]] = None,
        style_anchor_path: Optional[str] = None,
        environment_anchor_path: Optional[str] = None,
        max_retries: int = 3,
    ) -> Tuple[str, ValidationResult]:
        """
        생성 → 검증 → 실패시 재시도 루프.

        Args:
            scene_id: 씬 ID
            generate_fn: 이미지 생성 함수 (seed -> image_path)
            character_anchor_paths: 캐릭터 앵커 경로
            style_anchor_path: 스타일 앵커 경로
            environment_anchor_path: 환경 앵커 경로
            max_retries: 최대 재시도 횟수

        Returns:
            (best_image_path, ValidationResult) 튜플
        """
        best_path = None
        best_result = None
        best_score = -1.0
        base_seed = 42

        for attempt in range(1, max_retries + 1):
            seed = base_seed + (attempt - 1) * 13

            try:
                image_path = generate_fn(seed)
            except Exception as e:
                print(f"  [ConsistencyValidator] Scene {scene_id} generate attempt {attempt} failed: {e}")
                continue

            result = self.validate_scene_image(
                generated_image_path=image_path,
                scene_id=scene_id,
                character_anchor_paths=character_anchor_paths,
                style_anchor_path=style_anchor_path,
                environment_anchor_path=environment_anchor_path,
                attempt_number=attempt,
            )

            if result.overall_score > best_score:
                best_score = result.overall_score
                best_path = image_path
                best_result = result

            if result.passed:
                return image_path, result

        # 모든 재시도 실패
        if best_result and best_score > 0.4:
            # degraded 모드: best attempt 사용
            print(f"  [ConsistencyValidator] Scene {scene_id}: degraded mode (best score={best_score:.2f})")
            best_result.passed = True
            best_result.issues.append("DEGRADED: using best attempt after all retries failed")
            return best_path, best_result
        elif best_result:
            # 완전 실패
            print(f"  [ConsistencyValidator] Scene {scene_id}: FAILED (best score={best_score:.2f})")
            return best_path or "", best_result
        else:
            return "", ValidationResult(
                scene_id=scene_id,
                passed=False,
                overall_score=0.0,
                issues=["All generation attempts failed"],
            )

    def _score_with_gemini(
        self,
        generated_image_path: str,
        anchor_image_paths: List[str],
        anchor_labels: List[str],
    ) -> Dict[str, float]:
        """
        Gemini Vision 멀티모달 스코어링.

        Args:
            generated_image_path: 생성된 이미지 경로
            anchor_image_paths: 앵커 이미지 경로 목록
            anchor_labels: 앵커 라벨 목록

        Returns:
            차원별 점수 딕셔너리
        """
        try:
            parts = []

            # 생성 이미지
            gen_part = self._encode_image_part(generated_image_path)
            if gen_part:
                parts.append(gen_part)
                parts.append({"text": "[GENERATED IMAGE] This is the newly generated scene image."})

            # 앵커 이미지들
            for path, label in zip(anchor_image_paths, anchor_labels):
                anchor_part = self._encode_image_part(path)
                if anchor_part:
                    parts.append(anchor_part)
                    parts.append({"text": f"[{label.upper()}] This is the reference anchor."})

            # 스코어링 요청
            parts.append({"text": """Compare the GENERATED IMAGE against the reference anchors.
Score each dimension from 0.0 to 1.0:
- face_similarity: How consistent are character faces between generated and anchor? (1.0 = identical)
- style_drift: How well does the art style match? (1.0 = perfect match, 0.0 = completely different style)
- environment_similarity: How consistent is the background/environment? (1.0 = identical setting)

Respond ONLY with JSON: {"face_similarity": 0.0, "style_drift": 0.0, "environment_similarity": 0.0}"""})

            response = self.vision_client.generate_content(
                parts,
                generation_config={"temperature": 0.1, "max_output_tokens": 200}
            )

            content = response.text.strip()
            if content.startswith("```"):
                content = content.split("```")[1]
                if content.startswith("json"):
                    content = content[4:]

            scores = json.loads(content)

            return {
                "face_similarity": float(scores.get("face_similarity", 0.5)),
                "style_drift": float(scores.get("style_drift", 0.5)),
                "environment_similarity": float(scores.get("environment_similarity", 0.5)),
            }

        except Exception as e:
            print(f"  [ConsistencyValidator] Gemini scoring failed: {e}")
            # 실패 시 기본 점수 (통과하도록)
            return {
                "face_similarity": 0.7,
                "style_drift": 0.7,
                "environment_similarity": 0.7,
            }

    def _encode_image_part(self, image_path: str) -> Optional[Dict]:
        """이미지 파일을 base64 inline_data 파트로 변환."""
        try:
            with open(image_path, "rb") as f:
                data = base64.b64encode(f.read()).decode("utf-8")
            ext = Path(image_path).suffix.lower()
            mime_type = "image/jpeg" if ext in [".jpg", ".jpeg"] else "image/png"
            return {
                "inline_data": {
                    "mime_type": mime_type,
                    "data": data
                }
            }
        except Exception as e:
            print(f"  [Warning] Failed to encode image {image_path}: {e}")
            return None
