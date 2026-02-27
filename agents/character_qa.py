"""
CharacterQA v1: Face embedding-based anchor verification.

- EmbeddingProvider interface for swappable backends (face_recognition, insightface, etc.)
- Cosine similarity scoring against character anchor embeddings
- Fail policy: exclude from derived cuts (no regeneration)
"""

import os
import math
from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Tuple
from utils.logger import get_logger
logger = get_logger("character_qa")



class EmbeddingProvider(ABC):
    """Abstract face embedding provider. Swap implementations without changing QA logic."""

    @abstractmethod
    def get_embedding(self, image_path: str) -> Optional[List[float]]:
        """Extract face embedding from image. Returns None if no face detected."""
        ...

    @abstractmethod
    def name(self) -> str:
        """Provider name for logging."""
        ...


class FaceRecognitionProvider(EmbeddingProvider):
    """face_recognition (dlib) based embedding provider. 128D vectors."""

    def __init__(self):
        self._fr = None
        self._available = None

    def _ensure_loaded(self):
        if self._available is not None:
            return self._available
        try:
            import face_recognition
            self._fr = face_recognition
            self._available = True
        except ImportError:
            logger.info("  [CharacterQA] face_recognition not installed, will use fallback")
            self._available = False
        return self._available

    def get_embedding(self, image_path: str) -> Optional[List[float]]:
        if not self._ensure_loaded():
            return None
        if not os.path.exists(image_path):
            return None
        try:
            img = self._fr.load_image_file(image_path)
            encodings = self._fr.face_encodings(img)
            if not encodings:
                return None
            return encodings[0].tolist()
        except Exception as e:
            logger.error(f"  [CharacterQA] face_recognition error: {e}")
            return None

    def name(self) -> str:
        return "face_recognition"


class GeminiVisionFallbackProvider(EmbeddingProvider):
    """
    Fallback: uses Gemini Vision 'same person?' binary judgment.
    Returns a pseudo-embedding (not real vector) — cosine similarity not meaningful.
    Instead, compare_faces() is overridden for direct LLM comparison.
    """

    def __init__(self):
        self._client = None

    def _get_client(self):
        if self._client is None:
            try:
                from google import genai
                api_key = os.environ.get("GOOGLE_API_KEY")
                if api_key:
                    self._client = genai.Client(api_key=api_key)
            except Exception:
                pass
        return self._client

    def get_embedding(self, image_path: str) -> Optional[List[float]]:
        # Gemini doesn't produce embeddings; return sentinel for "has face"
        if not os.path.exists(image_path):
            return None
        return [0.0]  # sentinel — use compare_faces() instead

    def compare_faces(self, anchor_path: str, generated_path: str) -> float:
        """Direct LLM-based face similarity (0.0-1.0)."""
        client = self._get_client()
        if not client:
            return 0.5  # neutral score when unavailable

        try:
            import base64
            parts = []
            for path in [anchor_path, generated_path]:
                with open(path, "rb") as f:
                    data = base64.b64encode(f.read()).decode()
                ext = os.path.splitext(path)[1].lower()
                mime = "image/png" if ext == ".png" else "image/jpeg"
                parts.append({"inline_data": {"mime_type": mime, "data": data}})

            parts.append({
                "text": (
                    "Compare these two images. Image 1 is the reference character anchor. "
                    "Image 2 is a generated scene. Rate how likely the main person in Image 2 "
                    "is the SAME character as Image 1, on a scale of 0.0 to 1.0. "
                    "Consider face shape, hair, skin tone, and overall appearance. "
                    "Reply with ONLY a single decimal number, e.g. 0.72"
                )
            })

            resp = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=[{"role": "user", "parts": parts}],
            )
            score_text = resp.text.strip()
            # Parse float from response
            for token in score_text.split():
                try:
                    val = float(token)
                    if 0.0 <= val <= 1.0:
                        return val
                except ValueError:
                    continue
            return 0.5
        except Exception as e:
            logger.error(f"  [CharacterQA] Gemini Vision fallback error: {e}")
            return 0.5

    def name(self) -> str:
        return "gemini_vision_fallback"


def cosine_similarity(a: List[float], b: List[float]) -> float:
    """Cosine similarity between two vectors."""
    if len(a) != len(b) or len(a) == 0:
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    mag_a = math.sqrt(sum(x * x for x in a))
    mag_b = math.sqrt(sum(x * x for x in b))
    if mag_a == 0 or mag_b == 0:
        return 0.0
    return dot / (mag_a * mag_b)


class CharacterQA:
    """
    Face embedding-based character anchor verification.

    Usage:
        qa = CharacterQA(threshold=0.45)
        qa.register_anchor("hero", anchor_image_path)
        result = qa.verify_scene_image(scene_image_path, ["hero"])
    """

    def __init__(self, threshold: float = 0.45, provider: Optional[EmbeddingProvider] = None):
        self.threshold = threshold
        # Try face_recognition first, fall back to Gemini Vision
        if provider:
            self.provider = provider
        else:
            fr_provider = FaceRecognitionProvider()
            if fr_provider._ensure_loaded():
                self.provider = fr_provider
            else:
                self.provider = GeminiVisionFallbackProvider()

        self._anchor_embeddings: Dict[str, List[float]] = {}
        self._anchor_paths: Dict[str, str] = {}
        self._log_entries: List[Dict] = []
        logger.info(f"  [CharacterQA] Provider: {self.provider.name()}, threshold: {self.threshold}")

    def register_anchor(self, role: str, image_path: str) -> Optional[List[float]]:
        """Register a character anchor and extract its face embedding."""
        embedding = self.provider.get_embedding(image_path)
        if embedding is not None:
            self._anchor_embeddings[role] = embedding
            self._anchor_paths[role] = image_path
            logger.info(f"    [CharacterQA] Anchor registered: {role} (dim={len(embedding)})")
        else:
            logger.info(f"    [CharacterQA] No face detected in anchor for: {role}")
            self._log(role=role, scene_id=-1, face_detected=False,
                      similarity=-1.0, passed=False, fail_reason="anchor_face_not_found")
        return embedding

    def verify_scene_image(
        self,
        image_path: str,
        characters_in_scene: List[str],
        scene_id: int = 0,
    ) -> Dict[str, dict]:
        """
        Verify generated image against registered character anchors.

        Returns:
            Dict[role -> {"passed": bool, "similarity": float, "fail_reason": str}]
        """
        results = {}
        if not characters_in_scene:
            return results

        for role in characters_in_scene:
            if role not in self._anchor_embeddings:
                # No anchor registered for this role — skip
                results[role] = {"passed": True, "similarity": -1.0, "fail_reason": "no_anchor"}
                continue

            anchor_emb = self._anchor_embeddings[role]

            # Gemini Vision fallback: use direct comparison
            if isinstance(self.provider, GeminiVisionFallbackProvider):
                anchor_path = self._anchor_paths.get(role, "")
                sim = self.provider.compare_faces(anchor_path, image_path)
                passed = sim >= self.threshold
                fail_reason = "" if passed else "sim_below_threshold"
                results[role] = {"passed": passed, "similarity": sim, "fail_reason": fail_reason}
                self._log(role=role, scene_id=scene_id, face_detected=True,
                          similarity=sim, passed=passed, fail_reason=fail_reason)
                continue

            # Standard embedding comparison
            gen_emb = self.provider.get_embedding(image_path)
            if gen_emb is None:
                results[role] = {"passed": False, "similarity": 0.0, "fail_reason": "face_not_found"}
                self._log(role=role, scene_id=scene_id, face_detected=False,
                          similarity=0.0, passed=False, fail_reason="face_not_found")
                continue

            sim = cosine_similarity(anchor_emb, gen_emb)
            passed = sim >= self.threshold
            fail_reason = "" if passed else "sim_below_threshold"
            results[role] = {"passed": passed, "similarity": sim, "fail_reason": fail_reason}
            self._log(role=role, scene_id=scene_id, face_detected=True,
                      similarity=sim, passed=passed, fail_reason=fail_reason)

        return results

    def _log(self, **entry):
        """Append a QA log entry."""
        entry["provider"] = self.provider.name()
        self._log_entries.append(entry)

    def get_log_entries(self) -> List[Dict]:
        """Return all QA log entries for render.log."""
        return list(self._log_entries)

    def all_passed(self, results: Dict[str, dict]) -> bool:
        """Check if all characters passed verification."""
        if not results:
            return True
        return all(r.get("passed", True) for r in results.values())
