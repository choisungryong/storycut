"""
Scene Orchestrator: Manages scene-by-scene processing with context carry-over.

P1 핵심 기능:
- 이전 장면의 핵심 키워드(인물/장소/감정/행동)를 다음 장면 프롬프트에 상속
- Scene 간 일관성 유지
"""

import os
import json
import re
from typing import Dict, Any, List, Optional
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))

from agents.video_agent import VideoAgent
from agents.tts_agent import TTSAgent
from agents.music_agent import MusicAgent
from agents.composer_agent import ComposerAgent
from schemas import FeatureFlags, Scene, SceneEntities, ProjectRequest, SceneStatus, CameraWork


class SceneOrchestrator:
    """
    Scene 단위 처리 오케스트레이터

    P1 핵심: Context Carry-over (맥락 상속)
    - 이전 장면의 핵심 키워드를 다음 장면 프롬프트에 강제 포함
    - 인물/장소/감정/행동 일관성 유지
    """

    def __init__(self, feature_flags: FeatureFlags = None):
        """
        Initialize Scene Orchestrator with all sub-agents.

        Args:
            feature_flags: Feature flags configuration
        """
        self.feature_flags = feature_flags or FeatureFlags()
        self.video_agent = VideoAgent(feature_flags=self.feature_flags)
        self.tts_agent = TTSAgent()
        self.music_agent = MusicAgent()
        self.composer_agent = ComposerAgent()

        # LLM 클라이언트 (맥락 추출용)
        self._llm_client = None
        self.google_api_key = os.getenv("GOOGLE_API_KEY")

    @property
    def llm_client(self):
        """Lazy initialization of LLM client (Gemini 3 Pro)."""
        if self._llm_client is None:
            try:
                import google.generativeai as genai
                if self.google_api_key:
                    genai.configure(api_key=self.google_api_key)
                    self._llm_client = genai.GenerativeModel(model_name="gemini-3-pro-preview")
                else:
                    print("[WARNING] GOOGLE_API_KEY not set. LLM features disabled.")
                    self._llm_client = None
            except Exception as e:
                print(f"[WARNING] Failed to initialize Gemini client: {e}")
                self._llm_client = None
        return self._llm_client

    # =========================================================================
    # 캐릭터 동작/표정 추론 (MV 포팅 #1)
    # =========================================================================

    # mood → expression 매핑 (MV의 VisualBible.scene_blocking.expression 대체)
    _MOOD_TO_EXPRESSION = {
        "hopeful": "hopeful smile, bright warm eyes",
        "happy": "genuine smile, joyful expression",
        "joyful": "bright beaming smile, sparkling eyes",
        "sad": "downcast eyes, melancholy expression",
        "melancholy": "wistful gaze, subtle sadness in eyes",
        "tense": "furrowed brows, intense focused gaze",
        "angry": "fierce scowl, burning eyes",
        "fearful": "wide eyes, tense jaw, anxious expression",
        "mysterious": "enigmatic half-smile, knowing eyes",
        "romantic": "soft tender gaze, gentle smile",
        "dramatic": "intense emotional expression, dramatic eyes",
        "peaceful": "serene calm expression, gentle closed-mouth smile",
        "excited": "wide bright eyes, enthusiastic grin",
        "contemplative": "thoughtful distant gaze, pensive expression",
        "determined": "set jaw, resolute fierce gaze",
        "nostalgic": "bittersweet smile, distant wistful eyes",
        "dark": "shadowed intense gaze, grim expression",
        "warm": "warm genuine smile, kind eyes",
        "lonely": "distant empty gaze, solitary expression",
        "triumphant": "proud confident smile, victorious expression",
    }

    # visual_description/prompt에서 action pose 추출용 패턴
    _ACTION_PATTERNS = [
        (r'\b(?:running|sprinting|dashing|chasing)\b', "running dynamically, body in forward motion"),
        (r'\b(?:sitting|seated|crouching)\b', "sitting naturally, relaxed posture"),
        (r'\b(?:walking|strolling|wandering)\b', "walking forward, mid-stride natural gait"),
        (r'\b(?:crying|weeping|sobbing)\b', "hunched slightly, hands near face, emotional"),
        (r'\b(?:fighting|attacking|punching|swinging)\b', "dynamic action pose, mid-combat"),
        (r'\b(?:dancing|twirling|spinning)\b', "dancing gracefully, body in fluid motion"),
        (r'\b(?:hugging|embracing|holding)\b', "embracing warmly, arms around"),
        (r'\b(?:reaching|stretching|grasping)\b', "reaching outward with one arm extended"),
        (r'\b(?:kneeling|bowing|praying)\b', "kneeling on one knee, reverent posture"),
        (r'\b(?:leaning|resting|lounging)\b', "leaning casually against surface"),
        (r'\b(?:pointing|gesturing|waving)\b', "gesturing expressively with hands"),
        (r'\b(?:looking up|gazing up|staring at sky)\b', "head tilted upward, gazing at sky"),
        (r'\b(?:looking down|head bowed)\b', "head bowed downward, contemplative"),
        (r'\b(?:turning|looking back|glancing)\b', "turning head slightly, three-quarter view"),
        (r'\b(?:lying|collapsed|fallen)\b', "lying down, body on ground"),
        (r'\b(?:jumping|leaping)\b', "mid-jump, dynamic airborne pose"),
        # 한국어 패턴
        (r'(?:달리|뛰)', "running dynamically, body in forward motion"),
        (r'(?:앉|쪼그)', "sitting naturally, relaxed posture"),
        (r'(?:걸어|걷)', "walking forward, mid-stride"),
        (r'(?:울|눈물)', "hunched slightly, emotional, tears"),
        (r'(?:싸우|공격|때리)', "dynamic action pose, mid-combat"),
        (r'(?:춤|춤추)', "dancing gracefully"),
        (r'(?:안|껴안|포옹)', "embracing warmly"),
        (r'(?:무릎|꿇)', "kneeling, reverent posture"),
        (r'(?:기대|눕)', "leaning or lying down"),
    ]

    def _derive_expression_from_mood(self, mood: str) -> str:
        """씬의 mood에서 캐릭터 표정 토큰 추론 (MV의 scene_blocking.expression 대체)."""
        if not mood:
            return ""
        mood_lower = mood.lower().strip()
        # 직접 매칭
        if mood_lower in self._MOOD_TO_EXPRESSION:
            return self._MOOD_TO_EXPRESSION[mood_lower]
        # 부분 매칭 (e.g., "slightly sad" → "sad")
        for key, expr in self._MOOD_TO_EXPRESSION.items():
            if key in mood_lower:
                return expr
        return ""

    def _derive_action_pose(self, visual_desc: str, narration: str, image_prompt: str) -> str:
        """visual_description/narration/image_prompt에서 action pose 추론 (MV의 scene_blocking.action_pose 대체)."""
        import re
        combined = f"{visual_desc or ''} {narration or ''} {image_prompt or ''}"
        if not combined.strip():
            return ""
        for pattern, pose in self._ACTION_PATTERNS:
            if re.search(pattern, combined, re.IGNORECASE):
                return pose
        return ""

    # =========================================================================
    # P1: Context Carry-over (맥락 상속)
    # =========================================================================

    def extract_entities(
        self,
        sentence: str,
        inherited_keywords: List[str] = None
    ) -> SceneEntities:
        """
        문장에서 엔티티(인물/장소/감정/행동) 추출.

        P1: 맥락 상속을 위한 엔티티 추출

        Args:
            sentence: 장면 문장
            inherited_keywords: 이전 장면에서 상속받은 키워드

        Returns:
            SceneEntities 객체
        """
        if not self.llm_client:
            # LLM 없으면 기본 엔티티 반환
            return SceneEntities(
                characters=[],
                location=None,
                props=[],
                mood=None,
                action=None
            )

        inherited_context = ", ".join(inherited_keywords) if inherited_keywords else "없음"

        prompt = f"""
다음 문장에서 핵심 엔티티를 추출하세요:

문장: {sentence}
이전 장면 맥락: {inherited_context}

JSON 형식으로 출력:
{{
    "characters": ["인물1", "인물2"],
    "location": "장소",
    "props": ["소품1", "소품2"],
    "mood": "분위기/감정",
    "action": "주요 행동"
}}

주의:
- 이전 장면 맥락과 일관성을 유지하세요
- 뜬금없는 인물/장소 변경 감지 시 이전 맥락 우선
- 명시적 언급이 없으면 null 사용
"""

        try:
            system_prompt = "JSON만 출력하세요. 다른 설명 없이."
            full_prompt = f"{system_prompt}\n\n{prompt}"

            response = self.llm_client.generate_content(
                full_prompt,
                generation_config={
                    "temperature": 0.3,
                    "max_output_tokens": 300
                }
            )

            content = response.text.strip()
            # JSON 파싱
            if content.startswith("```"):
                content = content.split("```")[1]
                if content.startswith("json"):
                    content = content[4:]

            data = json.loads(content)
            return SceneEntities(**data)

        except Exception as e:
            print(f"  Entity extraction failed: {e}")
            return SceneEntities()

    def _shorten_narration(self, narration: str, target_duration: float) -> str:
        """LLM으로 narration을 target_duration에 맞게 축약. 스토리 핵심은 보존."""
        if not self.llm_client:
            return narration

        target_chars = int(target_duration * 4)  # 한국어 TTS 초당 ~4글자
        prompt = (
            f"다음 나레이션을 {target_chars}자 이내로 축약하세요.\n"
            f"규칙:\n"
            f"- 스토리의 핵심 사건과 감정은 반드시 보존\n"
            f"- 수식어, 반복, 부가 설명만 제거\n"
            f"- 자연스러운 문장 유지\n"
            f"- 축약된 나레이션만 출력 (설명 없이)\n\n"
            f"원문 ({len(narration)}자):\n{narration}"
        )
        try:
            response = self.llm_client.generate_content(
                prompt,
                generation_config={"temperature": 0.3, "max_output_tokens": 1000}
            )
            shortened = response.text.strip()
            print(f"     [Narration] LLM raw response: {shortened[:200]}")

            # Gemini 메타데이터 누출 제거: (36chars), *Refining:*, 마크다운 등
            shortened = re.sub(r'\(\d+chars?\)', '', shortened)
            shortened = re.sub(r'\*[A-Za-z_]+:?\*\s*', '', shortened)
            shortened = re.sub(r'^[\s\-\*#>:`]+', '', shortened).strip()
            # 여러 줄이면 첫 번째 비어있지 않은 줄부터 사용 (메타 헤더 제거)
            lines = [l for l in shortened.split('\n') if l.strip()]
            if lines:
                shortened = '\n'.join(lines)

            if shortened and len(shortened) < len(narration):
                print(f"     [Narration] Shortened: {len(narration)}자 → {len(shortened)}자 (target: {target_chars}자)")
                print(f"     [Narration] Result: {shortened[:100]}")
                return shortened
            else:
                print(f"     [Narration] Shorten result invalid (len={len(shortened) if shortened else 0}), keeping original")
        except Exception as e:
            print(f"     [Narration] Shorten failed: {e}")
        return narration

    def _resolve_image_to_local(self, img_path: str, project_dir: str, scene_id) -> str:
        """URL 또는 /media/ 경로를 로컬 파일 경로로 변환/다운로드.

        Returns:
            로컬 파일 경로 (성공 시) 또는 None (실패 시)
        """
        import urllib.request

        # /media/{project_id}/media/images/scene_XX.png → outputs/{project_id}/media/images/scene_XX.png
        if img_path.startswith("/media/"):
            local_candidate = "outputs/" + img_path[len("/media/"):]
            if os.path.exists(local_candidate):
                return local_candidate

        # HTTP(S) URL → 로컬 다운로드
        if img_path.startswith(("http://", "https://")):
            os.makedirs(f"{project_dir}/media/images", exist_ok=True)
            local_path = f"{project_dir}/media/images/scene_{int(scene_id):02d}.png"
            if os.path.exists(local_path):
                return local_path
            try:
                print(f"  [v2.3] Downloading image: {img_path[:80]}...")
                urllib.request.urlretrieve(img_path, local_path)
                if os.path.exists(local_path) and os.path.getsize(local_path) > 0:
                    print(f"  [v2.3] Downloaded to: {local_path}")
                    return local_path
            except Exception as e:
                print(f"  [v2.3] Download failed: {e}")

        return None

    def summarize_prev_scene(self, scene: Scene) -> str:
        """
        이전 장면 요약 생성.

        Args:
            scene: 이전 Scene 객체

        Returns:
            요약 문자열
        """
        parts = []

        if scene.entities.characters:
            parts.append(f"인물: {', '.join(scene.entities.characters)}")
        if scene.entities.location:
            parts.append(f"장소: {scene.entities.location}")
        if scene.entities.mood:
            parts.append(f"분위기: {scene.entities.mood}")
        if scene.entities.action:
            parts.append(f"행동: {scene.entities.action}")

        return " / ".join(parts) if parts else scene.sentence[:50]

    def extract_key_terms(self, scene: Scene) -> List[str]:
        """
        이전 장면에서 핵심 키워드 추출.

        P1: 다음 장면 프롬프트에 상속할 키워드

        Args:
            scene: 이전 Scene 객체

        Returns:
            키워드 목록
        """
        keywords = []

        # 엔티티에서 키워드 추출
        if scene.entities.characters:
            keywords.extend(scene.entities.characters[:2])  # 최대 2명
        if scene.entities.location:
            keywords.append(scene.entities.location)
        if scene.entities.mood:
            keywords.append(scene.entities.mood)
        if scene.entities.action:
            keywords.append(scene.entities.action)

        return keywords[:5]  # 최대 5개 키워드

    def build_prompt(
        self,
        sentence: str,
        inherited: List[str],
        entities: SceneEntities,
        style: str = None
    ) -> str:
        """
        영상 생성 프롬프트 구성.

        P1: inherited 키워드는 반드시 포함

        Args:
            sentence: 장면 문장
            inherited: 이전 장면에서 상속받은 키워드
            entities: 장면 엔티티
            style: 영상 스타일

        Returns:
            영상 생성 프롬프트
        """
        if style == "webtoon":
            # Webtoon Style (Primary target)
            style_prompt = "Premium Webtoon Style, manhwa aesthetics, 2D cel shaded, vibrant colors, clean lines, high quality anime art"
        elif style == "realistic":
            style_prompt = "Cinematic Lighting, 4k, detailed texture, photorealistic, photography"
        else:
            # Fallback but biased towards illustration for safety
            style_prompt = f"{style}, cinematic animation, high contrast"

        inherited_str = ", ".join(inherited) if inherited else "none"

        # 엔티티를 문자열로 변환
        entities_parts = []
        if entities.characters:
            entities_parts.append(f"Characters: {', '.join(entities.characters)}")
        if entities.location:
            entities_parts.append(f"Location: {entities.location}")
        if entities.props:
            entities_parts.append(f"Props: {', '.join(entities.props)}")
        if entities.mood:
            entities_parts.append(f"Mood: {entities.mood}")
        if entities.action:
            entities_parts.append(f"Action: {entities.action}")

        entities_str = " | ".join(entities_parts) if entities_parts else "N/A"

        prompt = f"""[STYLE] {style}
[INHERITED CONTEXT] {inherited_str}
[SCENE SENTENCE] {sentence}
[ENTITIES] {entities_str}
[RULES]
- 이전 장면과 동일 인물/공간/톤을 유지한다.
- 뜬금없는 배경/소품 변경 금지.
- 감정은 과장하되 개연성 유지."""

        return prompt

    def build_negative_prompt(self, style: str = None) -> str:
        """
        네거티브 프롬프트 생성.

        Args:
            style: 영상 스타일

        Returns:
            네거티브 프롬프트
        """
        base_negative = (
            "blurry, low quality, distorted, disfigured, "
            "watermark, text, logo, bad anatomy, extra limbs, "
            "mutant, deformed, ugly, missing fingers, extra fingers, "
            "inconsistent characters, changing clothes, different face, morphing features, cropped head"
        )
        
        if style == "webtoon":
            # Webtoon specific negatives
            return f"{base_negative}, photorealistic, 3d render, uncanny valley, realistic texture"
        else:
            return base_negative

    # =========================================================================
    # 메인 처리 로직
    # =========================================================================

    def process_story(
        self,
        story_data: Dict[str, Any],
        output_path: str = "output/youtube_ready.mp4",
        request: ProjectRequest = None,
        progress_callback: Any = None,
        style_anchor_path: Optional[str] = None,
        environment_anchors: Optional[Dict[int, str]] = None,
    ) -> str:
        """
        Scene JSON에서 최종 영상까지 전체 처리.

        P1: 맥락 상속 적용

        Args:
            story_data: Story JSON (scenes 포함)
            output_path: 최종 영상 출력 경로
            request: ProjectRequest (feature flags 포함)
            progress_callback: 진행 콜백
            style_anchor_path: 스타일 앵커 이미지 경로
            environment_anchors: 씬별 환경 앵커 이미지 딕셔너리

        Returns:
            최종 영상 파일 경로
        """
        print(f"\n{'='*60}")
        print(f"STORYCUT - Processing Story: {story_data['title']}")
        print(f"{'='*60}\n")

        # Feature flags 업데이트
        if request:
            self.feature_flags = request.feature_flags
            self.video_agent.feature_flags = request.feature_flags

        # Platform 기반 해상도 결정
        _req_platform = getattr(request, 'target_platform', None) if request else None
        _req_platform_val = _req_platform.value if _req_platform else 'youtube_long'
        _is_shorts = _req_platform_val == 'youtube_shorts'
        _resolution = "1080x1920" if _is_shorts else "1920x1080"

        # ComposerAgent를 해상도에 맞게 재생성
        if _is_shorts:
            self.composer_agent = ComposerAgent(resolution=_resolution)

        scenes = story_data["scenes"]
        total_scenes = len(scenes)
        style = story_data.get("style", request.style_preset if request else "cinematic")
        
        # TTS Voice 설정
        if request and hasattr(request, 'voice_id'):
            self.tts_agent.voice = request.voice_id
            print(f"TTS Voice set to: {self.tts_agent.voice}")

        # v2.0: 글로벌 스타일 가이드 추출
        global_style = story_data.get("global_style")
        character_sheet = story_data.get("character_sheet", {})

        print(f"Total scenes: {total_scenes}")
        print(f"Target duration: {story_data.get('total_duration_sec', 60)} seconds")
        print(f"Context carry-over: {'ON' if self.feature_flags.context_carry_over else 'OFF'}")
        
        # 프로젝트 베이스 디렉토리 설정 (final_video.mp4 경로 기반)
        # output_path: outputs/<project_id>/final_video.mp4
        project_dir = os.path.dirname(output_path)
        print(f"Project Directory: {project_dir}")

        # v2.2: Load existing images from manifest (이미지 생성 스킵용)
        existing_images = {}  # scene_id -> image_path
        manifest_path = os.path.join(project_dir, "manifest.json")
        if os.path.exists(manifest_path):
            try:
                with open(manifest_path, 'r', encoding='utf-8') as f:
                    manifest_data = json.load(f)
                if manifest_data.get('status') == 'images_ready' or manifest_data.get('_images_pregenerated'):
                    for sc in manifest_data.get('scenes', []):
                        sc_id = sc.get('scene_id')
                        img_path = sc.get('assets', {}).get('image_path') if isinstance(sc.get('assets'), dict) else None
                        if not img_path and isinstance(sc.get('assets'), dict):
                            img_path = sc['assets'].get('image_path')
                        if sc_id and img_path:
                            # 로컬 파일 존재 체크 또는 URL(http/https, /media/) 허용
                            if os.path.exists(img_path):
                                existing_images[sc_id] = img_path
                            elif img_path.startswith(("http://", "https://", "/media/")):
                                # URL → 로컬 경로 변환 시도
                                local_path = self._resolve_image_to_local(img_path, project_dir, sc_id)
                                if local_path:
                                    existing_images[sc_id] = local_path
                                else:
                                    existing_images[sc_id] = img_path  # URL 그대로 사용
                    if existing_images:
                        print(f"\n[v2.2] Loaded {len(existing_images)} existing images from manifest (skipping regeneration)")
            except Exception as e:
                print(f"[v2.2] Failed to load manifest: {e}")

        # v2.3: story_data에서도 이미지 경로 로드 (프론트엔드가 전달한 경우)
        if not existing_images and (story_data.get('_images_pregenerated') or any(
            isinstance(sc.get('assets'), dict) and sc.get('assets', {}).get('image_path')
            for sc in scenes
        )):
            print(f"\n[v2.3] Checking story_data scenes for pre-generated image paths...")
            for sc in scenes:
                sc_id = sc.get('scene_id')
                sc_assets = sc.get('assets', {})
                img_path = sc_assets.get('image_path') if isinstance(sc_assets, dict) else None
                if sc_id and img_path:
                    if os.path.exists(img_path):
                        existing_images[sc_id] = img_path
                    elif img_path.startswith(("http://", "https://", "/media/")):
                        local_path = self._resolve_image_to_local(img_path, project_dir, sc_id)
                        if local_path:
                            existing_images[sc_id] = local_path
                        else:
                            existing_images[sc_id] = img_path
            if existing_images:
                print(f"[v2.3] Loaded {len(existing_images)} images from story_data (skipping regeneration)")

        # v2.0: 글로벌 스타일 정보 출력
        if global_style:
            print(f"\n[Global Style Guide]")
            print(f"  Art Style: {global_style.get('art_style', 'N/A')}")
            print(f"  Color Palette: {global_style.get('color_palette', 'N/A')}")
            print(f"  Visual Seed: {global_style.get('visual_seed', 'N/A')}")
            print(f"  Aspect Ratio: {global_style.get('aspect_ratio', '16:9')}")

        if character_sheet:
            print(f"\n[Character Sheet]")
            for token, char_data in character_sheet.items():
                print(f"  {token}: {char_data.get('name')} (seed: {char_data.get('visual_seed')})")

        # v2.0: 앵커 정보 로깅
        if style_anchor_path:
            print(f"\n[StyleAnchor] Path: {style_anchor_path}")
        if environment_anchors:
            print(f"[EnvAnchors] {len(environment_anchors)} scenes: {list(environment_anchors.keys())}")

        # v2.0: ConsistencyValidator 초기화
        consistency_validator = None
        if self.feature_flags.consistency_validation:
            from agents.consistency_validator import ConsistencyValidator
            consistency_validator = ConsistencyValidator()
            print(f"[ConsistencyValidator] Enabled (max_retries={self.feature_flags.consistency_max_retries})")

        print()

        # Scene 처리
        video_clips = []
        narration_clips = []
        processed_scenes = []
        prev_scene = None

        for i, scene_data in enumerate(scenes, 1):
            print(f"\n{'─'*60}")
            print(f"Processing Scene {i}/{total_scenes} (ID: {scene_data['scene_id']})")
            print(f"{'─'*60}")
            print(f"  [DEBUG] Starting scene {i} processing...")

            # Scene 객체 생성
            scene = Scene(
                index=i,
                scene_id=scene_data["scene_id"],
                sentence=scene_data.get("narration", ""),
                narration=scene_data.get("narration"),
                visual_description=scene_data.get("visual_description"),
                mood=scene_data.get("mood"),
                duration_sec=scene_data.get("duration_sec", 5),
                # v2.0 필드
                narrative=scene_data.get("narrative"),
                image_prompt=scene_data.get("image_prompt"),
                characters_in_scene=scene_data.get("characters_in_scene", []),
            )

            # v2.2: Set existing image path if available (skip image generation)
            if scene_data["scene_id"] in existing_images:
                scene.assets.image_path = existing_images[scene_data["scene_id"]]
                print(f"  [v2.2] Using existing image: {scene.assets.image_path}")

            # v2.0: Character reference 로그 및 시드 추출
            scene_seed = None
            if scene.image_prompt:
                print(f"  [v2.0] Using image_prompt (character reference enabled)")
            if scene.characters_in_scene:
                print(f"  [v2.0] Characters: {', '.join(scene.characters_in_scene)}")

                # 모든 캐릭터의 visual_seed 결합
                if character_sheet and scene.characters_in_scene:
                    all_seeds = []
                    for char_token in scene.characters_in_scene:
                        if char_token in character_sheet:
                            s = character_sheet[char_token].get("visual_seed")
                            if s is not None:
                                all_seeds.append(s)
                    if all_seeds:
                        _base = sum(all_seeds) % (2**31) if len(all_seeds) > 1 else all_seeds[0]
                        # 씬 인덱스를 반영하여 같은 캐릭터 조합이라도 씬마다 다른 시드 생성
                        scene_seed = (_base + scene.scene_id * 31337) % (2**31)
                        print(f"  [v2.0] Combined visual_seed: {scene_seed} (from {len(all_seeds)} characters: {all_seeds}, scene_id: {scene.scene_id})")

            # v2.0: Scene에 메타데이터 저장 (video_agent가 활용)
            if not hasattr(scene, '_seed'):
                scene._seed = scene_seed
            if not hasattr(scene, '_global_style'):
                scene._global_style = global_style
            if not hasattr(scene, '_character_sheet'):
                scene._character_sheet = character_sheet
            if not hasattr(scene, '_style_anchor_path'):
                scene._style_anchor_path = style_anchor_path
            if not hasattr(scene, '_env_anchor_path'):
                env_path = environment_anchors.get(scene.scene_id) if environment_anchors else None
                scene._env_anchor_path = env_path

            # P1: Context Carry-over
            if self.feature_flags.context_carry_over and prev_scene:
                scene.context_summary = self.summarize_prev_scene(prev_scene)
                scene.inherited_keywords = self.extract_key_terms(prev_scene)
                print(f"  [CONTEXT] Inherited: {scene.inherited_keywords}")
            else:
                scene.inherited_keywords = []

            # 엔티티 추출
            scene.entities = self.extract_entities(
                scene.sentence,
                scene.inherited_keywords
            )

            # 프롬프트 생성
            # v2.0: image_prompt가 있으면 우선 사용, 없으면 기존 방식
            if scene.image_prompt:
                # image_prompt에 global_style 정보 추가
                if global_style:
                    style_suffix = f", {global_style.get('art_style', '')}, {global_style.get('color_palette', '')}"
                    scene.prompt = scene.image_prompt + style_suffix
                else:
                    scene.prompt = scene.image_prompt
                print(f"  [v2.0] Using pre-defined image_prompt")
            else:
                # v1.0 방식: build_prompt로 생성
                scene.prompt = self.build_prompt(
                    sentence=scene.sentence,
                    inherited=scene.inherited_keywords,
                    entities=scene.entities,
                    style=style
                )

            scene.negative_prompt = self.build_negative_prompt(style)

            # 캐릭터 외형 설명 주입 — MV 파이프라인 방식 (렌즈/손/의상/unique_features 포함)
            if scene.characters_in_scene and character_sheet:
                import re as _re
                _char_descs = []
                _outfit_locks = []
                for _tok in scene.characters_in_scene[:3]:
                    _cd = character_sheet.get(_tok)
                    if not _cd:
                        continue
                    _name = _cd.get("name", _tok) if isinstance(_cd, dict) else getattr(_cd, "name", _tok)
                    _app = _cd.get("appearance", "") if isinstance(_cd, dict) else getattr(_cd, "appearance", "")
                    _cloth = _cd.get("clothing_default", "") if isinstance(_cd, dict) else getattr(_cd, "clothing_default", "")
                    _uniq = _cd.get("unique_features", "") if isinstance(_cd, dict) else getattr(_cd, "unique_features", "")
                    if _app:
                        _parts = [_app]
                        if _cloth:
                            _parts.append(f"wearing {_cloth}")
                        if _uniq:
                            _parts.append(f"IDENTIFYING MARKS at EXACT positions (DO NOT relocate/mirror): {_uniq}")
                        _char_descs.append(f"[{_name}] {', '.join(_parts)}")
                    if _cloth:
                        _outfit_locks.append(f"OUTFIT LOCK: {_name} MUST wear {_cloth} in EVERY scene.")
                if _char_descs:
                    _action_pose = self._derive_action_pose(
                        scene.visual_description, scene.narration, scene.image_prompt
                    )
                    _expr = self._derive_expression_from_mood(scene.mood)
                    _is_cu = any(kw in scene.prompt.lower() for kw in ["close-up", "close up", "portrait", "face"])
                    _lens = "cinematic 50mm lens, natural facial proportions" if _is_cu else "portrait 85mm lens, natural facial proportions"
                    _hand = "natural hands, correct fingers, anatomically correct hands" if len(scene.characters_in_scene) >= 2 else ""
                    _pfx = []
                    if _action_pose:
                        _pfx.append(f"POSE: {_action_pose}")
                    if _expr:
                        _pfx.append(f"EXPRESSION: {_expr}")
                    _pfx.append(_lens)
                    if _hand:
                        _pfx.append(_hand)
                    _pfx.append(" | ".join(_char_descs))
                    _outfit_str = " ".join(_outfit_locks)
                    _clean = _re.sub(r'STORYCUT_\w+', '', scene.prompt).strip().strip(',').strip()
                    scene.prompt = f"{'. '.join(_pfx)}.{' ' + _outfit_str if _outfit_str else ''} {_clean}"
                    # 네거티브 강화 (MV 포팅)
                    _lens_neg = "wide-angle distortion, fisheye, exaggerated facial features"
                    _hand_neg = "extra fingers, deformed hands, fused fingers, missing fingers"
                    _id_neg = "different face, identity change, age change, ethnicity change, different outfit, wardrobe change"
                    scene.negative_prompt = f"{_lens_neg}, {_hand_neg}, {_id_neg}, {scene.negative_prompt}"
                else:
                    _no_people = "random person, unnamed person, bystander, stranger, human figure"
                    scene.negative_prompt = f"{_no_people}, {scene.negative_prompt}"

            # 인종 런타임 주입
            _eth_ps = getattr(request, 'character_ethnicity', 'auto') if request else 'auto'
            _ETH_MAP = {"korean": "Korean", "japanese": "Japanese", "chinese": "Chinese",
                        "southeast_asian": "Southeast Asian", "european": "European",
                        "black": "Black", "hispanic": "Hispanic"}
            _eth_kw_ps = _ETH_MAP.get(_eth_ps, "")
            if _eth_kw_ps and _eth_kw_ps.lower() not in scene.prompt.lower():
                scene.prompt = f"{_eth_kw_ps} characters, {scene.prompt}"

            # 카메라 워크 할당 (다양화)
            camera_works = list(CameraWork)
            scene.camera_work = camera_works[i % len(camera_works)]

            try:
                # Phase 1: TTS 먼저 생성하여 실제 duration 확보
                scene.status = SceneStatus.GENERATING_TTS

                # v3.0: 멀티 화자 TTS 지원
                scene_dict = scenes[i - 1] if isinstance(scenes[i - 1], dict) else {}
                character_voices = story_data.get("character_voices", [])

                # dialogue_lines를 현재 narration에서 재파싱 (프론트엔드 편집 반영)
                _current_narration = scene.narration or ""
                dialogue_lines = []
                if character_voices and re.search(r'\[[^\]]+\]', _current_narration):
                    from agents.story_agent import StoryAgent
                    dialogue_lines = StoryAgent.parse_dialogue_lines(_current_narration)

                # 프로젝트별 TTS 출력 경로 (동시 생성 시 파일 충돌 방지)
                _audio_dir = f"{os.path.dirname(output_path)}/media/audio"
                os.makedirs(_audio_dir, exist_ok=True)
                _audio_path = f"{_audio_dir}/narration_{scene.scene_id:02d}.mp3"

                print(f"     [TTS INPUT] Scene {scene.scene_id} narration: {scene.narration[:150]}")
                if dialogue_lines:
                    print(f"     [TTS INPUT] Scene {scene.scene_id} dialogue_lines: {dialogue_lines[:3]}")

                if dialogue_lines and character_voices:
                    # 멀티 화자 TTS (line_timings 포함)
                    tts_result = self.tts_agent.generate_dialogue_audio(
                        dialogue_lines=dialogue_lines,
                        character_voices=character_voices,
                        output_path=_audio_path,
                    )
                else:
                    # alignment 포함 TTS 생성 (정밀 자막 싱크)
                    tts_result = self.tts_agent.generate_speech(
                        scene_id=scene.scene_id,
                        narration=scene.narration,
                        emotion=scene.mood,
                        output_path=_audio_path
                    )
                scene.assets.narration_path = tts_result.audio_path
                scene.tts_duration_sec = tts_result.duration_sec
                # 실제 발화 타이밍 저장 (SRT 생성에 사용)
                if tts_result.sentence_timings:
                    scene._sentence_timings = tts_result.sentence_timings
                # ElevenLabs 캐릭터별 정밀 alignment 저장
                if getattr(tts_result, 'char_alignment', None):
                    scene._char_alignment = tts_result.char_alignment
                    print(f"     [TTS] Scene {scene.scene_id}: char_alignment 저장 ({len(tts_result.char_alignment['characters'])} chars)")

                # TTS 기반으로 duration 조정 — 나레이션 원본 보존, duration만 TTS에 맞춤
                if tts_result.duration_sec > 0:
                    import math
                    original_dur = scene.duration_sec
                    tts_dur = math.ceil(tts_result.duration_sec) + 1

                    # 씬 duration을 TTS에 맞춤 (최소 3초)
                    scene.duration_sec = max(3, tts_dur)
                    print(f"     [Duration] Final: {scene.duration_sec}s (TTS: {scene.tts_duration_sec:.1f}s, original: {original_dur}s)")

                # 영상 생성 (업데이트된 duration 사용)
                scene.status = SceneStatus.GENERATING_VIDEO

                # 프로젝트 구조에 맞는 비디오/이미지 출력 경로 설정
                video_output_dir = f"{os.path.dirname(output_path)}/media/video"

                video_path = self.video_agent.generate_video(
                    scene_id=scene.scene_id,
                    visual_description=scene.visual_description or scene.prompt,
                    style=style,
                    mood=scene.mood,
                    duration_sec=scene.duration_sec,
                    scene=scene,
                    output_dir=video_output_dir,
                    resolution=_resolution
                )
                # video_clips.append(video_path) -> REMOVED: 나중에 한꺼번에 수집
                scene.assets.video_path = video_path

                # v2.0: ConsistencyValidator 검증 (이미지 생성 후, 비디오 합성 전)
                if consistency_validator and scene.assets.image_path:
                    # 캐릭터 앵커 경로 수집 (포즈 선택 활성화)
                    char_anchor_paths = []
                    if scene.characters_in_scene and character_sheet:
                        from agents.character_manager import CharacterManager
                        cm = CharacterManager.__new__(CharacterManager)
                        for char_token in scene.characters_in_scene:
                            pose_path = cm.get_pose_appropriate_image(
                                char_token, character_sheet, scene.prompt
                            )
                            if pose_path and os.path.exists(pose_path):
                                char_anchor_paths.append(pose_path)
                            else:
                                print(f"  [WARNING] Anchor missing for character '{char_token}' in scene {scene.scene_id}")
                        if len(char_anchor_paths) < len(scene.characters_in_scene):
                            print(f"  [WARNING] Only {len(char_anchor_paths)}/{len(scene.characters_in_scene)} character anchors found")

                    env_anchor = environment_anchors.get(scene.scene_id) if environment_anchors else None

                    # Anchor audit log
                    print(f"  [ANCHOR AUDIT] Scene {scene.scene_id}: chars={[os.path.basename(p) for p in char_anchor_paths]}, style={os.path.basename(style_anchor_path) if style_anchor_path else 'None'}, env={os.path.basename(env_anchor) if env_anchor else 'None'}")

                    val_result = consistency_validator.validate_scene_image(
                        generated_image_path=scene.assets.image_path,
                        scene_id=scene.scene_id,
                        character_anchor_paths=char_anchor_paths,
                        style_anchor_path=style_anchor_path,
                        environment_anchor_path=env_anchor,
                    )

                    if not val_result.passed and val_result.overall_score <= 0.4:
                        print(f"     [ConsistencyValidator] Scene {i} FAILED validation (score={val_result.overall_score:.2f})")
                        scene.status = SceneStatus.FAILED
                        scene.error_message = f"Consistency validation failed: {val_result.issues}"
                        processed_scenes.append(scene)
                        prev_scene = scene
                        continue

                # [Fix] Generate & Burn-in Subtitles
                try:
                    # 1. Generate SRT
                    subtitle_dir = f"{os.path.dirname(output_path)}/media/subtitles"
                    self.generate_subtitle_files([scene], subtitle_dir)
                    
                    # 2. Burn-in if enabled
                    if getattr(self.feature_flags, 'subtitle_burn_in', True):
                        print(f"     [Subtitle] Burning in subtitles for scene {i}...")
                        subtitled_video_path = video_path.replace(".mp4", "_sub.mp4")
                        
                        result_path, subtitle_success = self.composer_agent.composer.overlay_subtitles(
                            video_in=video_path,
                            srt_path=scene.assets.subtitle_srt_path,
                            out_path=subtitled_video_path,
                             style={
                                "font_size": 16,
                                "margin_v": 30
                            }
                        )
                        
                        # Check actual subtitle application result
                        if subtitle_success and os.path.exists(result_path):
                            scene.assets.video_path = result_path
                            print(f"     [Subtitle] Subtitles burned successfully: {result_path}")
                        else:
                            print(f"     [Warning] Subtitle burn-in failed (OOM?), using original video without subtitles.")
                            # Keep original video path (fallback was already copied)
                            scene.assets.video_path = result_path
                            
                except Exception as sub_e:
                     print(f"     [Warning] Subtitle processing failed: {sub_e}")
                     # Do not fail the scene, just proceed without subtitles

                # 완료
                scene.status = SceneStatus.COMPLETED

            except Exception as e:
                scene.status = SceneStatus.FAILED
                scene.error_message = str(e)
                scene.retry_count += 1
                print(f"     [ERROR] Scene {i} failed: {e}")
                # 계속 진행 (실패한 씬은 나중에 재생성 가능)

            processed_scenes.append(scene)
            prev_scene = scene

            print(f"Scene {i} complete (status: {scene.status})\n")

            if progress_callback:
                try:
                    import asyncio
                    if asyncio.iscoroutinefunction(progress_callback):
                        # Async callback - skip in sync context
                        print(f"  [DEBUG] Skipping async progress_callback")
                        pass
                    else:
                        # Sync callback
                        print(f"  [DEBUG] Calling progress_callback for scene {i}")
                        progress_callback(scene, i)
                except Exception as cb_error:
                    print(f"  [WARNING] Progress callback failed: {cb_error}")

        # =================================================================
        # ROBUSTNESS FIX: Collect clips only from successfully completed scenes
        # =================================================================
        print(f"\n[Composer] Collecting clips from completed scenes...")
        video_clips = []
        narration_clips = []
        scene_durations = []

        for s in processed_scenes:
            if s.status == SceneStatus.COMPLETED and s.assets.video_path and s.assets.narration_path:
                video_clips.append(s.assets.video_path)
                narration_clips.append(s.assets.narration_path)
                scene_durations.append(float(s.duration_sec) if s.duration_sec else 5.0)
                print(f"  + Added Scene {s.scene_id}")
            else:
                print(f"  - Skipped Scene {s.scene_id} (Status: {s.status})")

        if not video_clips:
            raise RuntimeError("No scenes were successfully generated. Cannot compose video.")

        # 배경 음악 선택
        print(f"{'─'*60}")
        music_path = self.music_agent.select_music(
            genre=story_data.get("genre", "emotional"),
            mood=story_data.get("mood", "neutral"),
            duration_sec=story_data.get("total_duration_sec", 60)
        )
        print(f"{'─'*60}\n")

        # 최종 영상 합성
        _use_ducking = getattr(self.feature_flags, 'ffmpeg_audio_ducking', False)
        final_video = self.composer_agent.compose_video(
            video_clips=video_clips,
            narration_clips=narration_clips,
            music_path=music_path,
            output_path=output_path,
            use_ducking=_use_ducking,
            scene_durations=scene_durations
        )

        print(f"\n{'='*60}")
        print(f"SUCCESS! Video ready for YouTube upload")
        print(f"{'='*60}")
        print(f"File: {os.path.abspath(final_video)}\n")

        return final_video

    def compose_scenes_from_images(
        self,
        story_data: Dict[str, Any],
        output_path: str = "output/youtube_ready.mp4",
        request: ProjectRequest = None,
        progress_callback: Any = None,
        style_anchor_path: Optional[str] = None,
        environment_anchors: Optional[Dict[int, str]] = None,
    ) -> str:
        """
        이미 생성된 이미지로부터 영상을 합성하는 전용 경로.

        process_story()와 달리 이미지 생성/스킵 로직이 전혀 없음.
        각 씬에 image_path가 필수로 존재해야 함.

        흐름: TTS -> Ken Burns -> 자막 -> 클립 수집 -> BGM -> 최종 합성
        """
        print(f"\n{'='*60}")
        print(f"STORYCUT - Compose from Pre-generated Images: {story_data['title']}")
        print(f"{'='*60}\n")

        # Feature flags 업데이트
        if request:
            self.feature_flags = request.feature_flags
            self.video_agent.feature_flags = request.feature_flags

        # Platform 기반 해상도 결정
        _req_platform = getattr(request, 'target_platform', None) if request else None
        _req_platform_val = _req_platform.value if _req_platform else 'youtube_long'
        _is_shorts = _req_platform_val == 'youtube_shorts'
        _resolution = "1080x1920" if _is_shorts else "1920x1080"

        if _is_shorts:
            self.composer_agent = ComposerAgent(resolution=_resolution)

        scenes = story_data["scenes"]
        total_scenes = len(scenes)

        # TTS Voice 설정
        if request and hasattr(request, 'voice_id'):
            self.tts_agent.voice = request.voice_id

        project_dir = os.path.dirname(output_path)

        # 이미지 경로 수집 (씬 데이터 또는 매니페스트에서)
        image_map = {}  # scene_id -> local image path

        def _resolve_to_local(img_path, sc_id):
            """이미지 경로를 로컬 파일 경로로 정규화"""
            if not img_path:
                return None
            # 이미 로컬에 존재하면 그대로
            if os.path.exists(img_path):
                return img_path
            # outputs/ 상대경로가 project_dir과 다른 경우 보정
            if "media/images/" in img_path:
                local_candidate = os.path.join(project_dir, "media/images", f"scene_{int(sc_id):02d}.png")
                if os.path.exists(local_candidate):
                    return local_candidate
            # /media/{project_id}/... → outputs/{project_id}/... 변환
            if img_path.startswith("/media/"):
                local_candidate = "outputs/" + img_path[len("/media/"):]
                if os.path.exists(local_candidate):
                    return local_candidate
            # /api/asset/{project_id}/... → outputs/{project_id}/... 변환
            if "/api/asset/" in img_path:
                _after = img_path.split("/api/asset/", 1)[-1]
                local_candidate = f"outputs/{_after}"
                if os.path.exists(local_candidate):
                    return local_candidate
            # HTTP URL → 로컬 다운로드 시도
            if img_path.startswith(("http://", "https://")):
                local_path = self._resolve_image_to_local(img_path, project_dir, sc_id)
                if local_path and os.path.exists(local_path):
                    return local_path
            # 최종 fallback: project_dir 기준 직접 탐색
            fallback = os.path.join(project_dir, "media", "images", f"scene_{int(sc_id):02d}.png")
            if os.path.exists(fallback):
                return fallback
            return None

        manifest_path = os.path.join(project_dir, "manifest.json")
        if os.path.exists(manifest_path):
            try:
                with open(manifest_path, 'r', encoding='utf-8') as f:
                    manifest_data = json.load(f)
                for sc in manifest_data.get('scenes', []):
                    sc_id = sc.get('scene_id')
                    img_path = sc.get('assets', {}).get('image_path') if isinstance(sc.get('assets'), dict) else None
                    if sc_id and img_path:
                        resolved = _resolve_to_local(img_path, sc_id)
                        if resolved:
                            image_map[sc_id] = resolved
            except Exception as e:
                print(f"[compose] Failed to load manifest images: {e}")

        # story_data scenes에서도 보충
        for sc in scenes:
            sc_id = sc.get('scene_id')
            if sc_id not in image_map:
                sc_assets = sc.get('assets', {})
                img_path = sc_assets.get('image_path') if isinstance(sc_assets, dict) else None
                if sc_id and img_path:
                    resolved = _resolve_to_local(img_path, sc_id)
                    if resolved:
                        image_map[sc_id] = resolved

        print(f"[compose] {len(image_map)}/{total_scenes} scenes have pre-generated images")

        # 카메라 워크 목록
        camera_works = list(CameraWork)

        # Scene 처리: TTS → Ken Burns → 자막
        processed_scenes = []

        for i, scene_data in enumerate(scenes, 1):
            scene_id = scene_data["scene_id"]
            print(f"\n{'---'*20}")
            print(f"[compose] Scene {i}/{total_scenes} (ID: {scene_id})")

            image_path = image_map.get(scene_id)
            if not image_path or not os.path.exists(str(image_path)):
                # 최종 fallback: 디스크에서 직접 탐색
                fallback_path = os.path.join(project_dir, "media", "images", f"scene_{int(scene_id):02d}.png")
                if os.path.exists(fallback_path):
                    image_path = fallback_path
                    print(f"  [compose] Fallback image found: {fallback_path}")
                else:
                    print(f"  [SKIP] No valid image for scene {scene_id} (path={image_path}), skipping")
                    continue

            scene = Scene(
                index=i,
                scene_id=scene_id,
                sentence=scene_data.get("narration", ""),
                narration=scene_data.get("narration"),
                visual_description=scene_data.get("visual_description"),
                mood=scene_data.get("mood"),
                duration_sec=scene_data.get("duration_sec", 5),
                narrative=scene_data.get("narrative"),
                image_prompt=scene_data.get("image_prompt"),
                characters_in_scene=scene_data.get("characters_in_scene", []),
            )
            scene.assets.image_path = image_path
            scene.camera_work = camera_works[i % len(camera_works)]

            try:
                # Phase 1: TTS
                scene.status = SceneStatus.GENERATING_TTS

                scene_dict = scenes[i - 1] if isinstance(scenes[i - 1], dict) else {}
                character_voices = story_data.get("character_voices", [])

                # dialogue_lines를 현재 narration에서 재파싱 (프론트엔드 편집 반영)
                _current_narration = scene.narration or ""
                dialogue_lines = []
                if character_voices and re.search(r'\[[^\]]+\]', _current_narration):
                    from agents.story_agent import StoryAgent
                    dialogue_lines = StoryAgent.parse_dialogue_lines(_current_narration)

                # 프로젝트별 TTS 출력 경로 (동시 생성 시 파일 충돌 방지)
                _audio_dir = f"{project_dir}/media/audio"
                os.makedirs(_audio_dir, exist_ok=True)
                _audio_path = f"{_audio_dir}/narration_{scene.scene_id:02d}.mp3"

                print(f"     [TTS INPUT] Scene {scene.scene_id} narration: {scene.narration[:150]}")
                if dialogue_lines:
                    print(f"     [TTS INPUT] Scene {scene.scene_id} dialogue_lines: {dialogue_lines[:3]}")

                if dialogue_lines and character_voices:
                    tts_result = self.tts_agent.generate_dialogue_audio(
                        dialogue_lines=dialogue_lines,
                        character_voices=character_voices,
                        output_path=_audio_path,
                    )
                else:
                    # alignment 포함 TTS 생성 (정밀 자막 싱크)
                    tts_result = self.tts_agent.generate_speech(
                        scene_id=scene.scene_id,
                        narration=scene.narration,
                        emotion=scene.mood,
                        output_path=_audio_path
                    )
                scene.assets.narration_path = tts_result.audio_path
                scene.tts_duration_sec = tts_result.duration_sec
                # 실제 발화 타이밍 저장 (SRT 생성에 사용)
                if tts_result.sentence_timings:
                    scene._sentence_timings = tts_result.sentence_timings
                if getattr(tts_result, 'char_alignment', None):
                    scene._char_alignment = tts_result.char_alignment

                # TTS 기반으로 duration 조정 — 나레이션 원본 보존, duration만 TTS에 맞춤
                if tts_result.duration_sec > 0:
                    import math
                    original_dur = scene.duration_sec
                    tts_dur = math.ceil(tts_result.duration_sec) + 1

                    scene.duration_sec = max(3, tts_dur)
                    print(f"  [Duration] Final: {scene.duration_sec}s (TTS: {scene.tts_duration_sec:.1f}s, original: {original_dur}s)")

                # Phase 2: Ken Burns (이미지 → 비디오)
                scene.status = SceneStatus.GENERATING_VIDEO
                video_output_dir = f"{project_dir}/media/video"
                os.makedirs(video_output_dir, exist_ok=True)
                video_out = f"{video_output_dir}/scene_{scene.scene_id:02d}.mp4"

                video_path = self.video_agent.apply_kenburns(
                    image_path=image_path,
                    duration_sec=scene.duration_sec,
                    output_path=video_out,
                    scene_id=scene.scene_id,
                    camera_work=scene.camera_work.value if hasattr(scene.camera_work, 'value') else None,
                    resolution=_resolution,
                )
                scene.assets.video_path = video_path

                # Phase 3: 자막
                try:
                    subtitle_dir = f"{project_dir}/media/subtitles"
                    self.generate_subtitle_files([scene], subtitle_dir)

                    if getattr(self.feature_flags, 'subtitle_burn_in', True):
                        subtitled_path = video_path.replace(".mp4", "_sub.mp4")
                        result_path, subtitle_success = self.composer_agent.composer.overlay_subtitles(
                            video_in=video_path,
                            srt_path=scene.assets.subtitle_srt_path,
                            out_path=subtitled_path,
                            style={"font_size": 16, "margin_v": 30}
                        )
                        if subtitle_success and os.path.exists(result_path):
                            scene.assets.video_path = result_path
                except Exception as sub_e:
                    print(f"  [Warning] Subtitle processing failed: {sub_e}")

                scene.status = SceneStatus.COMPLETED

            except Exception as e:
                scene.status = SceneStatus.FAILED
                scene.error_message = str(e)
                print(f"  [ERROR] Scene {i} failed: {e}")

            processed_scenes.append(scene)
            print(f"  Scene {i} complete (status: {scene.status})")

            if progress_callback:
                try:
                    import asyncio
                    if not asyncio.iscoroutinefunction(progress_callback):
                        progress_callback(scene, i)
                except Exception as cb_error:
                    print(f"  [WARNING] Progress callback failed: {cb_error}")

        # 클립 수집
        video_clips = []
        narration_clips = []
        scene_durations = []

        for s in processed_scenes:
            if s.status == SceneStatus.COMPLETED and s.assets.video_path and s.assets.narration_path:
                video_clips.append(s.assets.video_path)
                narration_clips.append(s.assets.narration_path)
                scene_durations.append(float(s.duration_sec) if s.duration_sec else 5.0)

        if not video_clips:
            raise RuntimeError("No scenes were successfully composed. Cannot produce video.")

        # BGM + 최종 합성
        music_path = self.music_agent.select_music(
            genre=story_data.get("genre", "emotional"),
            mood=story_data.get("mood", "neutral"),
            duration_sec=story_data.get("total_duration_sec", 60)
        )

        _use_ducking = getattr(self.feature_flags, 'ffmpeg_audio_ducking', False)
        final_video = self.composer_agent.compose_video(
            video_clips=video_clips,
            narration_clips=narration_clips,
            music_path=music_path,
            output_path=output_path,
            use_ducking=_use_ducking,
            scene_durations=scene_durations
        )

        print(f"\n{'='*60}")
        print(f"SUCCESS! Video composed from pre-generated images")
        print(f"File: {os.path.abspath(final_video)}\n")

        return final_video

    def generate_images_for_scenes(
        self,
        story_data: Dict[str, Any],
        project_dir: str,
        request: ProjectRequest = None,
        style_anchor_path: Optional[str] = None,
        environment_anchors: Optional[Dict[int, str]] = None,
        on_scene_complete: Any = None,
    ) -> List[Dict[str, Any]]:
        """
        이미지만 생성 (TTS, 비디오 스킵).
        
        사용자가 이미지를 검토한 후 재생성/I2V 변환 가능.
        
        Args:
            story_data: Story JSON
            project_dir: 프로젝트 디렉토리
            request: ProjectRequest
            style_anchor_path: 스타일 앵커 경로
            environment_anchors: 환경 앵커 딕셔너리
            on_scene_complete: 각 씬 이미지 완료 시 콜백 (scene_dict, scene_index, total)

        Returns:
            Scene 데이터 목록 (이미지 경로 포함)
        """
        print(f"\n[SceneOrchestrator] Generating IMAGES ONLY")
        
        if request:
            self.feature_flags = request.feature_flags
            self.video_agent.feature_flags = request.feature_flags
        
        scenes = story_data["scenes"]
        total_scenes = len(scenes)
        style = story_data.get("style", request.style_preset if request else "cinematic")
        
        global_style = story_data.get("global_style")
        character_sheet = story_data.get("character_sheet", {})

        print(f"Total scenes: {total_scenes}")
        print(f"Style: {style}")
        # 앵커 디버그: character_sheet에 anchor_set과 master_image_path가 있는지 확인
        for _dbg_token, _dbg_cs in character_sheet.items():
            if isinstance(_dbg_cs, dict):
                _has_anchor = bool(_dbg_cs.get("anchor_set"))
                _has_master = bool(_dbg_cs.get("master_image_path"))
                _master_exists = os.path.exists(_dbg_cs.get("master_image_path", "")) if _has_master else False
                print(f"  [DEBUG] {_dbg_token}: anchor_set={_has_anchor}, master_image_path={_has_master} (exists={_master_exists})")
                if _has_anchor:
                    _poses = _dbg_cs["anchor_set"].get("poses", {})
                    for _pk, _pv in _poses.items():
                        _pp = _pv.get("image_path", "") if isinstance(_pv, dict) else ""
                        _pe = os.path.exists(_pp) if _pp else False
                        print(f"    [DEBUG] pose={_pk}: path={_pp}, exists={_pe}")
        print()
        
        processed_scenes = [None] * total_scenes  # pre-allocate for ordered results

        # Image output directory
        image_output_dir = f"{project_dir}/media/images"
        os.makedirs(image_output_dir, exist_ok=True)

        # --- 공통 설정 (인종, 에이전트 등) ---
        from agents.image_agent import ImageAgent
        from agents.character_manager import CharacterManager
        _eth = getattr(request, 'character_ethnicity', 'auto') if request else 'auto'
        _ETH_KW = {
            "korean": "Korean", "japanese": "Japanese", "chinese": "Chinese",
            "southeast_asian": "Southeast Asian", "european": "European",
            "black": "Black", "hispanic": "Hispanic",
        }
        _eth_kw = _ETH_KW.get(_eth, "")

        import threading
        from concurrent.futures import ThreadPoolExecutor, as_completed
        _result_lock = threading.Lock()
        _completed_count = [0]

        # pHash 중복 감지 레지스트리 (scene_id -> pHash)
        _phash_registry = {}
        _phash_lock = threading.Lock()

        def _process_single_scene(idx, scene_data, prev_scene_ref=None):
            """단일 씬 이미지 생성 (스레드에서 실행 가능)"""
            i = idx + 1  # 1-based for display

            print(f"\n{'---'*20}")
            print(f"Generating Image for Scene {i}/{total_scenes} (ID: {scene_data['scene_id']})")
            print(f"{'---'*20}")

            # Scene 객체 생성
            scene = Scene(
                index=i,
                scene_id=scene_data["scene_id"],
                sentence=scene_data.get("narration", ""),
                narration=scene_data.get("narration"),
                visual_description=scene_data.get("visual_description"),
                mood=scene_data.get("mood"),
                duration_sec=scene_data.get("duration_sec", 5),
                narrative=scene_data.get("narrative"),
                image_prompt=scene_data.get("image_prompt"),
                characters_in_scene=scene_data.get("characters_in_scene", []),
            )

            # Seed 추출 (모든 캐릭터의 visual_seed 결합)
            scene_seed = None
            if scene.characters_in_scene and character_sheet:
                all_seeds = []
                for char_token in scene.characters_in_scene:
                    if char_token in character_sheet:
                        s = character_sheet[char_token].get("visual_seed")
                        if s is not None:
                            all_seeds.append(s)
                if all_seeds:
                    _base = sum(all_seeds) % (2**31) if len(all_seeds) > 1 else all_seeds[0]
                    # 씬 인덱스를 반영하여 같은 캐릭터 조합이라도 씬마다 다른 시드 생성
                    scene_seed = (_base + scene.scene_id * 31337) % (2**31)

            # 메타데이터 저장
            scene._seed = scene_seed
            scene._global_style = global_style
            scene._character_sheet = character_sheet
            scene._style_anchor_path = style_anchor_path
            scene._env_anchor_path = environment_anchors.get(scene.scene_id) if environment_anchors else None

            # Context Carry-over (직렬 처리 씬만 적용)
            if self.feature_flags.context_carry_over and prev_scene_ref:
                scene.context_summary = self.summarize_prev_scene(prev_scene_ref)
                scene.inherited_keywords = self.extract_key_terms(prev_scene_ref)
            else:
                scene.inherited_keywords = []

            # 엔티티 추출 — image_prompt가 이미 있으면 건너뛰기 (불필요한 LLM 호출 방지)
            if not scene.image_prompt:
                scene.entities = self.extract_entities(scene.sentence, scene.inherited_keywords)

            # 프롬프트 생성
            if scene.image_prompt:
                if global_style:
                    style_suffix = f", {global_style.get('art_style', '')}, {global_style.get('color_palette', '')}"
                    scene.prompt = scene.image_prompt + style_suffix
                else:
                    scene.prompt = scene.image_prompt
            else:
                scene.prompt = self.build_prompt(
                    sentence=scene.sentence,
                    inherited=scene.inherited_keywords,
                    entities=scene.entities,
                    style=style
                )

            # 캐릭터 외형 설명 주입 — MV 파이프라인 방식 포팅 (렌즈/손/의상 잠금 포함)
            if scene.characters_in_scene and character_sheet:
                char_descs = []
                outfit_locks = []
                for char_token in scene.characters_in_scene[:3]:
                    char_data = character_sheet.get(char_token)
                    if not char_data:
                        continue
                    name = char_data.get("name", char_token) if isinstance(char_data, dict) else getattr(char_data, "name", char_token)
                    appearance = char_data.get("appearance", "") if isinstance(char_data, dict) else getattr(char_data, "appearance", "")
                    clothing = char_data.get("clothing_default", "") if isinstance(char_data, dict) else getattr(char_data, "clothing_default", "")
                    unique_features = char_data.get("unique_features", "") if isinstance(char_data, dict) else getattr(char_data, "unique_features", "")
                    if appearance:
                        parts = [appearance]
                        if clothing:
                            parts.append(f"wearing {clothing}")
                        if unique_features:
                            parts.append(f"IDENTIFYING MARKS at EXACT positions (DO NOT relocate/mirror): {unique_features}")
                        char_descs.append(f"[{name}] {', '.join(parts)}")
                    # 의상 잠금 (MV 포팅 #4)
                    if clothing:
                        outfit_locks.append(f"OUTFIT LOCK: {name} MUST wear {clothing} in EVERY scene.")
                if char_descs:
                    # action_pose / expression 추론 (MV 포팅 #1)
                    action_pose = self._derive_action_pose(
                        scene.visual_description, scene.narration, scene.image_prompt
                    )
                    expression = self._derive_expression_from_mood(scene.mood)

                    # 렌즈 왜곡 방지 토큰 (MV 포팅 #2)
                    prompt_lower = scene.prompt.lower()
                    is_closeup = any(kw in prompt_lower for kw in ["close-up", "close up", "portrait", "face"])
                    lens_token = "cinematic 50mm lens, natural facial proportions" if is_closeup else "portrait 85mm lens, natural facial proportions"

                    # 손 품질 강화 토큰 (MV 포팅 #3)
                    hand_token = ""
                    if len(scene.characters_in_scene) >= 2:
                        hand_token = "natural hands, correct fingers, anatomically correct hands"

                    # 프롬프트 조립: MV 방식 (pose → expression → lens → hand → characters → outfit lock → prompt)
                    prefix_parts = []
                    if action_pose:
                        prefix_parts.append(f"POSE: {action_pose}")
                    if expression:
                        prefix_parts.append(f"EXPRESSION: {expression}")
                    prefix_parts.append(lens_token)
                    if hand_token:
                        prefix_parts.append(hand_token)
                    char_block = " | ".join(char_descs)
                    prefix_parts.append(char_block)
                    outfit_lock_str = " ".join(outfit_locks)
                    # STORYCUT 토큰 제거 (캐릭터 설명으로 대체됐으므로 불필요)
                    import re as _re
                    clean_prompt = _re.sub(r'STORYCUT_\w+', '', scene.prompt).strip().strip(',').strip()
                    scene.prompt = f"{'. '.join(prefix_parts)}.{' ' + outfit_lock_str if outfit_lock_str else ''} {clean_prompt}"

                    if action_pose or expression:
                        print(f"  [MV포팅#1] POSE={action_pose or 'N/A'}, EXPR={expression or 'N/A'}")
                    # 앵커 이미지의 직립 자세 복제 방지 네거티브
                    if action_pose:
                        scene._action_pose_injected = True

            # 인종 런타임 주입 (MV 파이프라인과 동일 방식)
            if _eth_kw and _eth_kw.lower() not in scene.prompt.lower():
                scene.prompt = f"{_eth_kw} characters, {scene.prompt}"

            # 네거티브 프롬프트 — MV 파이프라인 방식 포팅 (렌즈/손/ID드리프트/인물방지)
            _scene_neg = self.build_negative_prompt(style)
            if scene.characters_in_scene:
                # 렌즈 왜곡 + 손 품질 네거티브 (MV 포팅 #2, #3)
                _lens_neg = "wide-angle distortion, fisheye, exaggerated facial features"
                _hand_neg = "extra fingers, deformed hands, fused fingers, missing fingers"
                _scene_neg = f"{_lens_neg}, {_hand_neg}, {_scene_neg}"
                # ID 드리프트 방지 네거티브 (MV 포팅 #5)
                _scene_neg = f"different face, identity change, age change, ethnicity change, different outfit, wardrobe change, wrong clothes, {_scene_neg}"
            else:
                # 캐릭터 없는 씬: 인물 방지 네거티브 (MV 포팅 #6)
                _no_people = "random person, unnamed person, elderly man, old man, old woman, young woman, young man, bystander, stranger, human figure, person standing, woman standing, man standing, silhouette of person, crowd"
                _scene_neg = f"{_no_people}, {_scene_neg}"
            scene.negative_prompt = _scene_neg

            try:
                # Generate IMAGE ONLY (no TTS, no video)
                image_agent = ImageAgent()

                # Character references (포즈 선택 활성화)
                char_refs = []
                if scene.characters_in_scene and character_sheet:
                    cm = CharacterManager.__new__(CharacterManager)
                    for char_token in scene.characters_in_scene:
                        pose_path = cm.get_pose_appropriate_image(
                            char_token, character_sheet, scene.prompt
                        )
                        if pose_path and os.path.exists(pose_path):
                            char_refs.append(pose_path)
                        elif pose_path and pose_path.startswith("http"):
                            # R2 URL → 로컬 다운로드
                            try:
                                import requests as _req
                                _resp = _req.get(pose_path, timeout=30)
                                if _resp.status_code == 200:
                                    _dl_dir = f"{project_dir}/media/characters"
                                    os.makedirs(_dl_dir, exist_ok=True)
                                    _dl_path = f"{_dl_dir}/{os.path.basename(pose_path.split('?')[0])}"
                                    with open(_dl_path, "wb") as _f:
                                        _f.write(_resp.content)
                                    char_refs.append(_dl_path)
                                    print(f"  [R2->Local] Downloaded anchor for '{char_token}': {os.path.basename(_dl_path)}")
                                else:
                                    print(f"  [WARNING] Anchor download failed for '{char_token}': HTTP {_resp.status_code}")
                            except Exception as _dl_err:
                                print(f"  [WARNING] Anchor download failed for '{char_token}': {_dl_err}")
                        else:
                            # master_image_path 또는 master_image_url 폴백
                            _char_data = character_sheet.get(char_token, {})
                            _fallback = None
                            if isinstance(_char_data, dict):
                                _fallback = _char_data.get("master_image_url") or _char_data.get("master_image_path")
                            elif hasattr(_char_data, 'master_image_path'):
                                _fallback = _char_data.master_image_path
                            if _fallback and os.path.exists(_fallback):
                                char_refs.append(_fallback)
                            elif _fallback and _fallback.startswith("http"):
                                try:
                                    import requests as _req
                                    _resp = _req.get(_fallback, timeout=30)
                                    if _resp.status_code == 200:
                                        _dl_dir = f"{project_dir}/media/characters"
                                        os.makedirs(_dl_dir, exist_ok=True)
                                        _dl_path = f"{_dl_dir}/{char_token}_master.jpg"
                                        with open(_dl_path, "wb") as _f:
                                            _f.write(_resp.content)
                                        char_refs.append(_dl_path)
                                        print(f"  [R2->Local] Downloaded master anchor for '{char_token}'")
                                except Exception:
                                    pass
                            else:
                                print(f"  [WARNING] Anchor missing for character '{char_token}' in scene {scene.scene_id}")
                    print(f"  [ANCHOR] Scene {scene.scene_id}: chars={scene.characters_in_scene}, refs={len(char_refs)} paths={char_refs}")
                    if len(char_refs) < len(scene.characters_in_scene):
                        print(f"  [WARNING] Only {len(char_refs)}/{len(scene.characters_in_scene)} character anchors found")

                # v2.1: Extract anchors for this scene
                # 캐릭터 미등장 씬: 스타일 앵커 제외 — 고스트 방지 (MV 포팅 #7)
                scene_style_anchor = style_anchor_path if scene.characters_in_scene else None
                scene_env_anchor = environment_anchors.get(scene.scene_id) if environment_anchors else None

                # Anchor audit log
                print(f"  [ANCHOR AUDIT] Scene {scene.scene_id}: chars={[os.path.basename(p) for p in char_refs]}, style={os.path.basename(scene_style_anchor) if scene_style_anchor else 'None'}, env={os.path.basename(scene_env_anchor) if scene_env_anchor else 'None'}")

                # Platform 기반 aspect ratio
                _platform = getattr(request, 'target_platform', None)
                _platform_val = _platform.value if _platform else 'youtube_long'
                _aspect = "9:16" if _platform_val == 'youtube_shorts' else "16:9"

                # camera_directive 추론 (MV 포팅: shot_type → prompt_builder 프레이밍 활성화)
                _camera_directive = None
                _pl = scene.prompt.lower()
                for _shot_kw, _shot_val in [
                    ("extreme close-up", "extreme-close-up"), ("extreme-close-up", "extreme-close-up"),
                    ("close-up", "close-up"), ("close up", "close-up"),
                    ("medium shot", "medium"), ("medium", "medium"),
                    ("wide shot", "wide"), ("wide", "wide"), ("full body", "wide"),
                ]:
                    if _shot_kw in _pl:
                        _camera_directive = _shot_val
                        break

                # genre / mood 추출 (request에서 가져옴)
                _genre = getattr(request, 'genre', None) if request else None
                _mood = scene.mood or (getattr(request, 'mood', None) if request else None)

                # Generate image
                image_path, image_id = image_agent.generate_image(
                    scene_id=scene.scene_id,
                    prompt=scene.prompt,
                    negative_prompt=scene.negative_prompt,
                    style=style,
                    aspect_ratio=_aspect,
                    output_dir=image_output_dir,
                    seed=scene_seed,
                    character_reference_paths=char_refs,
                    style_anchor_path=scene_style_anchor,
                    environment_anchor_path=scene_env_anchor,
                    image_model=getattr(self.feature_flags, 'image_model', 'standard'),
                    genre=_genre,
                    mood=_mood,
                    camera_directive=_camera_directive,
                )

                scene.assets.image_path = image_path

                # pHash 중복 감지: 유사 이미지 반복 생성 방지
                try:
                    import imagehash
                    from PIL import Image as _PILImage
                    with _PILImage.open(image_path) as _img:
                        _new_hash = imagehash.phash(_img)
                    _dup_scene_id = None
                    with _phash_lock:
                        for _prev_sid, _prev_hash in _phash_registry.items():
                            if _new_hash - _prev_hash <= 8:  # hamming distance ≤ 8 → 중복
                                _dup_scene_id = _prev_sid
                                break
                        if _dup_scene_id is None:
                            _phash_registry[scene.scene_id] = _new_hash
                    if _dup_scene_id is not None:
                        print(f"  [pHash] Scene {scene.scene_id} too similar to Scene {_dup_scene_id} (dist≤8). Regenerating...")
                        _varied_prompt = f"unique composition, different angle, {scene.prompt}"
                        _new_seed = (scene_seed + 42) if scene_seed else None
                        image_path, image_id = image_agent.generate_image(
                            scene_id=scene.scene_id,
                            prompt=_varied_prompt,
                            negative_prompt=scene.negative_prompt,
                            style=style,
                            aspect_ratio=_aspect,
                            output_dir=image_output_dir,
                            seed=_new_seed,
                            character_reference_paths=char_refs,
                            style_anchor_path=scene_style_anchor,
                            environment_anchor_path=scene_env_anchor,
                            image_model=getattr(self.feature_flags, 'image_model', 'standard'),
                            genre=_genre,
                            mood=_mood,
                            camera_directive=_camera_directive,
                        )
                        scene.assets.image_path = image_path
                        with _phash_lock:
                            with _PILImage.open(image_path) as _img:
                                _phash_registry[scene.scene_id] = imagehash.phash(_img)
                        print(f"  [pHash] Regenerated: {image_path}")
                except ImportError:
                    pass  # imagehash 미설치 시 무시
                except Exception as _ph_err:
                    print(f"  [pHash] Hash check skipped: {_ph_err}")

                scene.status = SceneStatus.COMPLETED

                print(f"  [OK] Image generated: {image_path}")

            except Exception as e:
                scene.status = SceneStatus.FAILED
                scene.error_message = str(e)
                print(f"  [FAIL] Image generation failed: {e}")

            # Scene 데이터를 딕셔너리로 변환하여 저장
            scene_dict = scene_data.copy()
            scene_dict["assets"] = {
                "image_path": scene.assets.image_path if scene.assets else None
            }
            scene_dict["status"] = scene.status
            scene_dict["prompt"] = scene.prompt

            with _result_lock:
                processed_scenes[idx] = scene_dict
                _completed_count[0] += 1

            # 콜백 호출 (프로그레시브 로딩용) — scene_id(1-based)를 전달
            if on_scene_complete:
                try:
                    _actual_scene_id = scene_data.get("scene_id", idx + 1)
                    on_scene_complete(scene_dict, _actual_scene_id, total_scenes)
                except Exception as cb_err:
                    print(f"  [WARNING] on_scene_complete callback error: {cb_err}")

            print(f"Scene {i} image complete\n")
            return scene

        # Scene 1: 직렬 처리 (스타일 앵커 확보 + context carry-over 기준점)
        prev_scene = None
        if scenes:
            prev_scene = _process_single_scene(0, scenes[0], prev_scene_ref=None)

        # Scene 2+: 병렬 처리 (max_workers=4)
        if len(scenes) > 1:
            remaining = list(enumerate(scenes[1:], start=1))
            print(f"\n[Parallel] Processing {len(remaining)} remaining scenes with 4 workers...")
            with ThreadPoolExecutor(max_workers=4) as executor:
                futures = {
                    executor.submit(_process_single_scene, idx, sd, prev_scene): sd
                    for idx, sd in remaining
                }
                for future in as_completed(futures):
                    try:
                        future.result()
                    except Exception as exc:
                        print(f"  [ERROR] Scene thread exception: {exc}")

        # None 제거 (실패 시 방어) + scene_id 순서 복원 (병렬 처리 시 완료 순서 ≠ 원래 순서)
        processed_scenes = [s for s in processed_scenes if s is not None]
        processed_scenes.sort(key=lambda s: s.get("scene_id", 0) if isinstance(s, dict) else s.scene_id)

        print(f"\n[SUCCESS] {len(processed_scenes)} images generated!")
        print(f"[DEBUG-SYNC] processed_scenes order: {[s.get('scene_id') if isinstance(s, dict) else s.scene_id for s in processed_scenes]}")
        return processed_scenes


    def process_scenes_from_script(
        self,
        script_text: str,
        request: ProjectRequest
    ) -> List[Scene]:
        """
        스크립트 텍스트에서 Scene 목록 생성.

        P1: 맥락 상속 적용

        Args:
            script_text: 전체 스크립트 텍스트
            request: ProjectRequest (feature flags 포함)

        Returns:
            Scene 객체 목록
        """
        # 문장 단위 분할
        sentences = self._split_into_sentences(script_text)
        scenes = []
        prev_scene = None

        for idx, sentence in enumerate(sentences, start=1):
            scene = Scene(
                index=idx,
                scene_id=idx,
                sentence=sentence,
                narration=sentence,
            )

            # P1: Context Carry-over
            if request.feature_flags.context_carry_over and prev_scene:
                scene.context_summary = self.summarize_prev_scene(prev_scene)
                scene.inherited_keywords = self.extract_key_terms(prev_scene)
            else:
                scene.inherited_keywords = []

            # 엔티티 추출
            scene.entities = self.extract_entities(
                sentence,
                scene.inherited_keywords
            )

            # 프롬프트 생성
            scene.prompt = self.build_prompt(
                sentence=sentence,
                inherited=scene.inherited_keywords,
                entities=scene.entities,
                style=request.style_preset
            )
            scene.negative_prompt = self.build_negative_prompt(request.style_preset)

            scenes.append(scene)
            prev_scene = scene

        return scenes

    def _split_into_sentences(self, text: str) -> List[str]:
        """
        텍스트를 문장 단위로 분할.

        Args:
            text: 전체 텍스트

        Returns:
            문장 목록
        """
        import re

        # 한국어 및 영어 문장 분할
        # 마침표, 물음표, 느낌표 기준
        sentences = re.split(r'(?<=[.!?])\s+', text.strip())

        # 빈 문장 제거 및 정리
        sentences = [s.strip() for s in sentences if s.strip()]

        # 너무 긴 문장은 분할
        result = []
        for s in sentences:
            if len(s) > 100:
                # 쉼표나 연결어 기준으로 추가 분할
                parts = re.split(r'(?<=,)\s+|(?<=그리고)\s+|(?<=하지만)\s+', s)
                result.extend([p.strip() for p in parts if p.strip()])
            else:
                result.append(s)

        return result

    def retry_scene(
        self,
        scene: Dict[str, Any],
        story_style: str = "cinematic",
        project_dir: str = None
    ) -> tuple[str, str]:
        """
        단일 Scene 재처리.

        Args:
            scene: Scene 데이터
            story_style: 영상 스타일
            project_dir: 프로젝트 디렉토리 (TTS 파일 격리용)

        Returns:
            (video_path, audio_path) 튜플
        """
        print(f"Retrying scene {scene['scene_id']}...")

        video_path = self.video_agent.generate_video(
            scene_id=scene["scene_id"],
            visual_description=scene["visual_description"],
            style=story_style,
            mood=scene["mood"],
            duration_sec=scene["duration_sec"]
        )

        # 프로젝트별 TTS 경로
        _tts_output = None
        if project_dir:
            _audio_dir = f"{project_dir}/media/audio"
            os.makedirs(_audio_dir, exist_ok=True)
            _tts_output = f"{_audio_dir}/narration_{scene['scene_id']:02d}.mp3"

        tts_result = self.tts_agent.generate_speech(
            scene_id=scene["scene_id"],
            narration=scene["narration"],
            emotion=scene["mood"],
            output_path=_tts_output
        )
        audio_path = tts_result.audio_path

        return video_path, audio_path

    def generate_subtitle_files(
        self,
        scenes: List[Scene],
        output_dir: str = "media/subtitles"
    ) -> List[str]:
        """
        각 Scene에 대한 SRT 자막 파일 생성.
        우선순위: char_alignment (정밀) > sentence_timings > 글자 수 비례 (레거시)
        """
        from utils.ffmpeg_utils import FFmpegComposer

        os.makedirs(output_dir, exist_ok=True)
        composer = FFmpegComposer()

        srt_paths = []

        for scene in scenes:
            srt_path = f"{output_dir}/scene_{scene.scene_id:02d}.srt"
            narration = scene.narration or scene.sentence or ""

            # 1순위: ElevenLabs char_alignment (캐릭터별 정밀 타이밍)
            _alignment = getattr(scene, '_char_alignment', None)
            if _alignment and _alignment.get('characters'):
                self._generate_srt_from_alignment(_alignment, narration, srt_path, composer)
                print(f"  [SRT] Scene {scene.scene_id}: char_alignment 기반 정밀 SRT")
            # 2순위: sentence_timings (문장별 타이밍)
            elif getattr(scene, '_sentence_timings', None):
                self._generate_srt_from_timings(scene._sentence_timings, srt_path, composer)
                print(f"  [SRT] Scene {scene.scene_id}: sentence_timings 기반 SRT ({len(scene._sentence_timings)}문장)")
            else:
                # 3순위: 글자 수 비례 (레거시 fallback)
                actual_duration = scene.tts_duration_sec if scene.tts_duration_sec else scene.duration_sec
                scene_data = [{
                    "narration": narration,
                    "duration_sec": actual_duration
                }]
                composer.generate_srt_from_scenes(scene_data, srt_path)
                print(f"  [SRT] Scene {scene.scene_id}: 글자수 비례 fallback SRT")

            # 디버그 로그: SRT 내용 출력
            try:
                with open(srt_path, 'r', encoding='utf-8') as _f:
                    _srt_content = _f.read()
                print(f"  [SRT DEBUG] Scene {scene.scene_id} narration: {narration[:80]}...")
                print(f"  [SRT DEBUG] Scene {scene.scene_id} SRT content:\n{_srt_content[:500]}")
            except Exception:
                pass

            scene.assets.subtitle_srt_path = srt_path
            srt_paths.append(srt_path)

        return srt_paths

    def _generate_srt_from_timings(self, timings: List[Dict], srt_path: str, composer=None):
        """실제 TTS 발화 타이밍으로 SRT 생성. 각 문장의 start/end 시간을 그대로 사용."""
        if composer is None:
            from utils.ffmpeg_utils import FFmpegComposer
            composer = FFmpegComposer()

        srt_content = []
        sub_index = 1

        for timing in timings:
            text = timing.get("text", "").strip()
            if not text:
                continue

            # 화자 태그 제거
            text = re.sub(r'\[[\w_]+\](?:\([^)]*\))?\s*', '', text).strip()
            if not text:
                continue

            start_ms = int(timing["start"] * 1000)
            end_ms = int(timing["end"] * 1000)

            # 긴 문장은 자막 표시용으로 분할 (같은 시간대 안에서)
            chunks = composer._split_subtitle_text(text, max_chars=25)

            if len(chunks) == 1:
                srt_content.append(f"{sub_index}")
                srt_content.append(f"{composer._ms_to_srt_time(start_ms)} --> {composer._ms_to_srt_time(end_ms)}")
                srt_content.append(chunks[0])
                srt_content.append("")
                sub_index += 1
            else:
                # 문장 내 청크를 글자 수 비례로 시간 분배 (이 범위 안에서만)
                total_chars = sum(len(c) for c in chunks) or 1
                elapsed = 0.0
                chunk_total_ms = end_ms - start_ms
                for chunk in chunks:
                    chunk_ratio = len(chunk) / total_chars
                    chunk_dur = chunk_total_ms * chunk_ratio
                    chunk_start = start_ms + elapsed
                    chunk_end = chunk_start + chunk_dur
                    srt_content.append(f"{sub_index}")
                    srt_content.append(f"{composer._ms_to_srt_time(int(chunk_start))} --> {composer._ms_to_srt_time(int(chunk_end))}")
                    srt_content.append(chunk)
                    srt_content.append("")
                    sub_index += 1
                    elapsed += chunk_dur

        os.makedirs(os.path.dirname(srt_path) or ".", exist_ok=True)
        with open(srt_path, "w", encoding="utf-8") as f:
            f.write("\n".join(srt_content))

    def _generate_srt_from_alignment(
        self,
        alignment: Dict[str, Any],
        narration: str,
        srt_path: str,
        composer=None
    ):
        """
        ElevenLabs char_alignment 데이터로 정밀 SRT 생성.
        각 자막 청크의 시작/끝 시간을 실제 TTS 발화 타이밍에서 가져옴.
        """
        if composer is None:
            from utils.ffmpeg_utils import FFmpegComposer
            composer = FFmpegComposer()

        chars = alignment["characters"]
        starts = alignment["start_times"]
        ends = alignment["end_times"]

        # 화자 태그 제거한 텍스트 — SRT에 표시할 텍스트
        clean_text = re.sub(r'\[[\w_]+\](?:\([^)]*\))?\s*', '', narration).strip()
        if not clean_text:
            clean_text = narration.strip()

        # alignment 캐릭터를 합쳐서 원본 텍스트 재구성
        aligned_text = "".join(chars)
        print(f"     [Alignment] chars={len(chars)}, aligned_text[:60]={aligned_text[:60]}")
        print(f"     [Alignment] clean_text[:60]={clean_text[:60]}")

        # 자막 청크 분할 (최대 20자)
        chunks = composer._split_subtitle_text(clean_text, max_chars=20)
        print(f"     [Alignment] {len(chunks)} chunks: {[c[:15] for c in chunks]}")

        # 각 청크의 시작/끝 글자 위치를 찾고 alignment에서 타이밍 매핑
        srt_content = []
        sub_index = 1
        char_offset = 0  # alignment 캐릭터 배열에서의 현재 위치

        for chunk in chunks:
            chunk_stripped = chunk.strip()
            if not chunk_stripped:
                continue

            # 청크의 첫 글자/마지막 글자를 alignment에서 찾기
            # alignment 텍스트와 clean_text가 다를 수 있으므로 글자 단위 매칭
            chunk_start_time = None
            chunk_end_time = None
            matched_chars = 0

            # 청크의 각 글자를 alignment에서 순차 탐색
            chunk_char_idx = 0
            scan_start = max(0, char_offset - 5)  # 약간의 여유

            for ai in range(scan_start, len(chars)):
                if chunk_char_idx >= len(chunk_stripped):
                    break
                # 공백/특수문자 스킵 (alignment에서)
                if chars[ai].strip() == '' or chars[ai] == ' ':
                    continue
                # 현재 청크 글자와 매칭 시도
                if chunk_stripped[chunk_char_idx] == ' ':
                    chunk_char_idx += 1
                    if chunk_char_idx >= len(chunk_stripped):
                        break
                if chars[ai] == chunk_stripped[chunk_char_idx]:
                    if chunk_start_time is None:
                        chunk_start_time = starts[ai]
                    chunk_end_time = ends[ai]
                    matched_chars += 1
                    chunk_char_idx += 1
                    char_offset = ai + 1

            # 매칭 실패 시 fallback: 이전 끝 시간 ~ 비례 추정
            if chunk_start_time is None:
                if sub_index > 1 and srt_content:
                    # 이전 엔트리의 끝 시간에서 시작
                    chunk_start_time = ends[min(char_offset, len(ends) - 1)] if char_offset < len(ends) else ends[-1]
                else:
                    chunk_start_time = 0.0
            if chunk_end_time is None:
                chunk_end_time = chunk_start_time + 2.0  # fallback 2초

            start_ms = int(chunk_start_time * 1000)
            end_ms = int(chunk_end_time * 1000)

            srt_content.append(f"{sub_index}")
            srt_content.append(f"{composer._ms_to_srt_time(start_ms)} --> {composer._ms_to_srt_time(end_ms)}")
            srt_content.append(chunk_stripped)
            srt_content.append("")
            sub_index += 1

        os.makedirs(os.path.dirname(srt_path) or ".", exist_ok=True)
        with open(srt_path, "w", encoding="utf-8") as f:
            f.write("\n".join(srt_content))

        print(f"     [Alignment SRT] {sub_index - 1} entries written to {srt_path}")

    def get_processing_stats(
        self,
        scenes: List[Scene]
    ) -> Dict[str, Any]:
        """
        처리 통계 반환.

        Args:
            scenes: 처리된 Scene 목록

        Returns:
            통계 딕셔너리
        """
        video_methods = {}
        for scene in scenes:
            method = scene.generation_method or "unknown"
            video_methods[method] = video_methods.get(method, 0) + 1

        return {
            "total_scenes": len(scenes),
            "video_generation_methods": video_methods,
            "context_carry_over_enabled": self.feature_flags.context_carry_over,
            "hook_scene_video_enabled": self.feature_flags.hook_scene1_video,
        }
