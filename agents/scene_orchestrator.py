"""
Scene Orchestrator: Manages scene-by-scene processing with context carry-over.

P1 ?µì‹¬ ê¸°ëŠ¥:
- ?´ì „ ?¥ë©´???µì‹¬ ?¤ì›Œ???¸ë¬¼/?¥ì†Œ/ê°ì •/?‰ë™)ë¥??¤ìŒ ?¥ë©´ ?„ë¡¬?„íŠ¸???ì†
- Scene ê°??¼ê???? ì?
"""

import os
import json
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
    Scene ?¨ìœ„ ì²˜ë¦¬ ?¤ì??¤íŠ¸?ˆì´??

    P1 ?µì‹¬: Context Carry-over (ë§¥ë½ ?ì†)
    - ?´ì „ ?¥ë©´???µì‹¬ ?¤ì›Œ?œë? ?¤ìŒ ?¥ë©´ ?„ë¡¬?„íŠ¸??ê°•ì œ ?¬í•¨
    - ?¸ë¬¼/?¥ì†Œ/ê°ì •/?‰ë™ ?¼ê???? ì?
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

        # LLM ?´ë¼?´ì–¸??(ë§¥ë½ ì¶”ì¶œ??
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
    # P1: Context Carry-over (ë§¥ë½ ?ì†)
    # =========================================================================

    def extract_entities(
        self,
        sentence: str,
        inherited_keywords: List[str] = None
    ) -> SceneEntities:
        """
        ë¬¸ì¥?ì„œ ?”í‹°???¸ë¬¼/?¥ì†Œ/ê°ì •/?‰ë™) ì¶”ì¶œ.

        P1: ë§¥ë½ ?ì†???„í•œ ?”í‹°??ì¶”ì¶œ

        Args:
            sentence: ?¥ë©´ ë¬¸ì¥
            inherited_keywords: ?´ì „ ?¥ë©´?ì„œ ?ì†ë°›ì? ?¤ì›Œ??

        Returns:
            SceneEntities ê°ì²´
        """
        if not self.llm_client:
            # LLM ?†ìœ¼ë©?ê¸°ë³¸ ?”í‹°??ë°˜í™˜
            return SceneEntities(
                characters=[],
                location=None,
                props=[],
                mood=None,
                action=None
            )

        inherited_context = ", ".join(inherited_keywords) if inherited_keywords else "?†ìŒ"

        prompt = f"""
?¤ìŒ ë¬¸ì¥?ì„œ ?µì‹¬ ?”í‹°?°ë? ì¶”ì¶œ?˜ì„¸??

ë¬¸ì¥: {sentence}
?´ì „ ?¥ë©´ ë§¥ë½: {inherited_context}

JSON ?•ì‹?¼ë¡œ ì¶œë ¥:
{{
    "characters": ["?¸ë¬¼1", "?¸ë¬¼2"],
    "location": "?¥ì†Œ",
    "props": ["?Œí’ˆ1", "?Œí’ˆ2"],
    "mood": "ë¶„ìœ„ê¸?ê°ì •",
    "action": "ì£¼ìš” ?‰ë™"
}}

ì£¼ì˜:
- ?´ì „ ?¥ë©´ ë§¥ë½ê³??¼ê??±ì„ ? ì??˜ì„¸??
- ?¬ê¸ˆ?†ëŠ” ?¸ë¬¼/?¥ì†Œ ë³€ê²?ê°ì? ???´ì „ ë§¥ë½ ?°ì„ 
- ëª…ì‹œ???¸ê¸‰???†ìœ¼ë©?null ?¬ìš©
"""

        try:
            system_prompt = "JSONë§?ì¶œë ¥?˜ì„¸?? ?¤ë¥¸ ?¤ëª… ?†ì´."
            full_prompt = f"{system_prompt}\n\n{prompt}"

            response = self.llm_client.generate_content(
                full_prompt,
                generation_config={
                    "temperature": 0.3,
                    "max_output_tokens": 300
                }
            )

            content = response.text.strip()
            # JSON ?Œì‹±
            if content.startswith("```"):
                content = content.split("```")[1]
                if content.startswith("json"):
                    content = content[4:]

            data = json.loads(content)
            return SceneEntities(**data)

        except Exception as e:
            print(f"  Entity extraction failed: {e}")
            return SceneEntities()

    def summarize_prev_scene(self, scene: Scene) -> str:
        """
        ?´ì „ ?¥ë©´ ?”ì•½ ?ì„±.

        Args:
            scene: ?´ì „ Scene ê°ì²´

        Returns:
            ?”ì•½ ë¬¸ì??
        """
        parts = []

        if scene.entities.characters:
            parts.append(f"?¸ë¬¼: {', '.join(scene.entities.characters)}")
        if scene.entities.location:
            parts.append(f"?¥ì†Œ: {scene.entities.location}")
        if scene.entities.mood:
            parts.append(f"ë¶„ìœ„ê¸? {scene.entities.mood}")
        if scene.entities.action:
            parts.append(f"?‰ë™: {scene.entities.action}")

        return " / ".join(parts) if parts else scene.sentence[:50]

    def extract_key_terms(self, scene: Scene) -> List[str]:
        """
        ?´ì „ ?¥ë©´?ì„œ ?µì‹¬ ?¤ì›Œ??ì¶”ì¶œ.

        P1: ?¤ìŒ ?¥ë©´ ?„ë¡¬?„íŠ¸???ì†???¤ì›Œ??

        Args:
            scene: ?´ì „ Scene ê°ì²´

        Returns:
            ?¤ì›Œ??ëª©ë¡
        """
        keywords = []

        # ?”í‹°?°ì—???¤ì›Œ??ì¶”ì¶œ
        if scene.entities.characters:
            keywords.extend(scene.entities.characters[:2])  # ìµœë? 2ëª?
        if scene.entities.location:
            keywords.append(scene.entities.location)
        if scene.entities.mood:
            keywords.append(scene.entities.mood)
        if scene.entities.action:
            keywords.append(scene.entities.action)

        return keywords[:5]  # ìµœë? 5ê°??¤ì›Œ??

    def build_prompt(
        self,
        sentence: str,
        inherited: List[str],
        entities: SceneEntities,
        style: str = None
    ) -> str:
        """
        ?ìƒ ?ì„± ?„ë¡¬?„íŠ¸ êµ¬ì„±.

        P1: inherited ?¤ì›Œ?œëŠ” ë°˜ë“œ???¬í•¨

        Args:
            sentence: ?¥ë©´ ë¬¸ì¥
            inherited: ?´ì „ ?¥ë©´?ì„œ ?ì†ë°›ì? ?¤ì›Œ??
            entities: ?¥ë©´ ?”í‹°??
            style: ?ìƒ ?¤í???

        Returns:
            ?ìƒ ?ì„± ?„ë¡¬?„íŠ¸
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

        # ?”í‹°?°ë? ë¬¸ì?´ë¡œ ë³€??
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
- ?´ì „ ?¥ë©´ê³??™ì¼ ?¸ë¬¼/ê³µê°„/?¤ì„ ? ì??œë‹¤.
- ?¬ê¸ˆ?†ëŠ” ë°°ê²½/?Œí’ˆ ë³€ê²?ê¸ˆì?.
- ê°ì •?€ ê³¼ì¥?˜ë˜ ê°œì—°??? ì?."""

        return prompt

    def build_negative_prompt(self, style: str = None) -> str:
        """
        ?¤ê±°?°ë¸Œ ?„ë¡¬?„íŠ¸ ?ì„±.

        Args:
            style: ?ìƒ ?¤í???

        Returns:
            ?¤ê±°?°ë¸Œ ?„ë¡¬?„íŠ¸
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
    # ë©”ì¸ ì²˜ë¦¬ ë¡œì§
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
        Scene JSON?ì„œ ìµœì¢… ?ìƒê¹Œì? ?„ì²´ ì²˜ë¦¬.

        P1: ë§¥ë½ ?ì† ?ìš©

        Args:
            story_data: Story JSON (scenes ?¬í•¨)
            output_path: ìµœì¢… ?ìƒ ì¶œë ¥ ê²½ë¡œ
            request: ProjectRequest (feature flags ?¬í•¨)
            progress_callback: ì§„í–‰ ì½œë°±
            style_anchor_path: ?¤í????µì»¤ ?´ë?ì§€ ê²½ë¡œ
            environment_anchors: ?¬ë³„ ?˜ê²½ ?µì»¤ ?´ë?ì§€ ?•ì…”?ˆë¦¬

        Returns:
            ìµœì¢… ?ìƒ ?Œì¼ ê²½ë¡œ
        """
        print(f"\n{'='*60}")
        print(f"STORYCUT - Processing Story: {story_data['title']}")
        print(f"{'='*60}\n")

        # Feature flags ?…ë°?´íŠ¸
        if request:
            self.feature_flags = request.feature_flags
            self.video_agent.feature_flags = request.feature_flags

        scenes = story_data["scenes"]
        total_scenes = len(scenes)
        style = story_data.get("style", request.style_preset if request else "cinematic")
        
        # TTS Voice ?¤ì •
        if request and hasattr(request, 'voice_id'):
            self.tts_agent.voice = request.voice_id
            print(f"TTS Voice set to: {self.tts_agent.voice}")

        # v2.0: ê¸€ë¡œë²Œ ?¤í???ê°€?´ë“œ ì¶”ì¶œ
        global_style = story_data.get("global_style")
        character_sheet = story_data.get("character_sheet", {})

        print(f"Total scenes: {total_scenes}")
        print(f"Target duration: {story_data['total_duration_sec']} seconds")
        print(f"Target duration: {story_data['total_duration_sec']} seconds")
        print(f"Context carry-over: {'ON' if self.feature_flags.context_carry_over else 'OFF'}")
        
        # ?„ë¡œ?íŠ¸ ë² ì´???”ë ‰? ë¦¬ ?¤ì • (final_video.mp4 ê²½ë¡œ ê¸°ë°˜)
        # output_path: outputs/<project_id>/final_video.mp4
        project_dir = os.path.dirname(output_path)
        print(f"Project Directory: {project_dir}")

        # v2.0: ê¸€ë¡œë²Œ ?¤í????•ë³´ ì¶œë ¥
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

        # v2.0: ?µì»¤ ?•ë³´ ë¡œê¹…
        if style_anchor_path:
            print(f"\n[StyleAnchor] Path: {style_anchor_path}")
        if environment_anchors:
            print(f"[EnvAnchors] {len(environment_anchors)} scenes: {list(environment_anchors.keys())}")

        # v2.0: ConsistencyValidator ì´ˆê¸°??
        consistency_validator = None
        if self.feature_flags.consistency_validation:
            from agents.consistency_validator import ConsistencyValidator
            consistency_validator = ConsistencyValidator()
            print(f"[ConsistencyValidator] Enabled (max_retries={self.feature_flags.consistency_max_retries})")

        print()

        # Scene ì²˜ë¦¬
        video_clips = []
        narration_clips = []
        processed_scenes = []
        prev_scene = None

        for i, scene_data in enumerate(scenes, 1):
            print(f"\n{'?€'*60}")
            print(f"Processing Scene {i}/{total_scenes} (ID: {scene_data['scene_id']})")
            print(f"{'?€'*60}")
            print(f"  [DEBUG] Starting scene {i} processing...")

            # Scene ê°ì²´ ?ì„±
            scene = Scene(
                index=i,
                scene_id=scene_data["scene_id"],
                sentence=scene_data.get("narration", ""),
                narration=scene_data.get("narration"),
                visual_description=scene_data.get("visual_description"),
                mood=scene_data.get("mood"),
                duration_sec=scene_data.get("duration_sec", 5),
                # v2.0 ?„ë“œ
                narrative=scene_data.get("narrative"),
                image_prompt=scene_data.get("image_prompt"),
                characters_in_scene=scene_data.get("characters_in_scene", []),
            )

            # v2.0: Character reference ë¡œê·¸ ë°??œë“œ ì¶”ì¶œ
            scene_seed = None
            if scene.image_prompt:
                print(f"  [v2.0] Using image_prompt (character reference enabled)")
            if scene.characters_in_scene:
                print(f"  [v2.0] Characters: {', '.join(scene.characters_in_scene)}")

                # ì²?ë²ˆì§¸ ìºë¦­?°ì˜ visual_seed ?¬ìš©
                if character_sheet and scene.characters_in_scene:
                    first_char_token = scene.characters_in_scene[0]
                    if first_char_token in character_sheet:
                        scene_seed = character_sheet[first_char_token].get("visual_seed")
                        print(f"  [v2.0] Using visual_seed: {scene_seed}")

            # v2.0: Scene??ë©”í??°ì´???€??(video_agentê°€ ?œìš©)
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

            # ?”í‹°??ì¶”ì¶œ
            scene.entities = self.extract_entities(
                scene.sentence,
                scene.inherited_keywords
            )

            # ?„ë¡¬?„íŠ¸ ?ì„±
            # v2.0: image_promptê°€ ?ˆìœ¼ë©??°ì„  ?¬ìš©, ?†ìœ¼ë©?ê¸°ì¡´ ë°©ì‹
            if scene.image_prompt:
                # image_prompt??global_style ?•ë³´ ì¶”ê?
                if global_style:
                    style_suffix = f", {global_style.get('art_style', '')}, {global_style.get('color_palette', '')}"
                    scene.prompt = scene.image_prompt + style_suffix
                else:
                    scene.prompt = scene.image_prompt
                print(f"  [v2.0] Using pre-defined image_prompt")
            else:
                # v1.0 ë°©ì‹: build_promptë¡??ì„±
                scene.prompt = self.build_prompt(
                    sentence=scene.sentence,
                    inherited=scene.inherited_keywords,
                    entities=scene.entities,
                    style=style
                )

            scene.negative_prompt = self.build_negative_prompt(style)

            # ì¹´ë©”???Œí¬ ? ë‹¹ (?¤ì–‘??
            camera_works = list(CameraWork)
            scene.camera_work = camera_works[i % len(camera_works)]

            try:
                # Phase 1: TTS ë¨¼ì? ?ì„±?˜ì—¬ ?¤ì œ duration ?•ë³´
                scene.status = SceneStatus.GENERATING_TTS
                tts_result = self.tts_agent.generate_speech(
                    scene_id=scene.scene_id,
                    narration=scene.narration,
                    emotion=scene.mood
                )
                scene.assets.narration_path = tts_result.audio_path
                scene.tts_duration_sec = tts_result.duration_sec
                # narration_clips.append(tts_result.audio_path) -> REMOVED: ?˜ì¤‘???œêº¼ë²ˆì— ?˜ì§‘

                # TTS ê¸°ë°˜?¼ë¡œ duration ?…ë°?´íŠ¸ (ìµœì†Œ 3ì´? ìµœë? 15ì´?
                if tts_result.duration_sec > 0:
                    scene.duration_sec = max(3, min(15, int(tts_result.duration_sec) + 1))
                    print(f"     [Duration] Updated to {scene.duration_sec}s (TTS: {tts_result.duration_sec:.2f}s)")

                # ?ìƒ ?ì„± (?…ë°?´íŠ¸??duration ?¬ìš©)
                scene.status = SceneStatus.GENERATING_VIDEO

                # ?„ë¡œ?íŠ¸ êµ¬ì¡°??ë§ëŠ” ë¹„ë””???´ë?ì§€ ì¶œë ¥ ê²½ë¡œ ?¤ì •
                video_output_dir = f"{os.path.dirname(output_path)}/media/video"

                video_path = self.video_agent.generate_video(
                    scene_id=scene.scene_id,
                    visual_description=scene.visual_description or scene.prompt,
                    style=style,
                    mood=scene.mood,
                    duration_sec=scene.duration_sec,
                    scene=scene,
                    output_dir=video_output_dir
                )
                # video_clips.append(video_path) -> REMOVED: ?˜ì¤‘???œêº¼ë²ˆì— ?˜ì§‘
                scene.assets.video_path = video_path

                # v2.0: ConsistencyValidator ê²€ì¦?(?´ë?ì§€ ?ì„± ?? ë¹„ë””???©ì„± ??
                if consistency_validator and scene.assets.image_path:
                    # ìºë¦­???µì»¤ ê²½ë¡œ ?˜ì§‘
                    char_anchor_paths = []
                    if scene.characters_in_scene and character_sheet:
                        from agents.character_manager import CharacterManager
                        cm = CharacterManager.__new__(CharacterManager)
                        char_anchor_paths = cm.get_active_character_images(
                            scene.characters_in_scene, character_sheet
                        )

                    env_anchor = environment_anchors.get(scene.scene_id) if environment_anchors else None

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

                # ?„ë£Œ
                scene.status = SceneStatus.COMPLETED

            except Exception as e:
                scene.status = SceneStatus.FAILED
                scene.error_message = str(e)
                scene.retry_count += 1
                print(f"     [ERROR] Scene {i} failed: {e}")
                # ê³„ì† ì§„í–‰ (?¤íŒ¨???¬ì? ?˜ì¤‘???¬ìƒ??ê°€??

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
        
        for s in processed_scenes:
            if s.status == SceneStatus.COMPLETED and s.assets.video_path and s.assets.narration_path:
                video_clips.append(s.assets.video_path)
                narration_clips.append(s.assets.narration_path)
                print(f"  + Added Scene {s.scene_id}")
            else:
                print(f"  - Skipped Scene {s.scene_id} (Status: {s.status})")
                
        if not video_clips:
            raise RuntimeError("No scenes were successfully generated. Cannot compose video.")

        # ë°°ê²½ ?Œì•… ? íƒ
        print(f"{'?€'*60}")
        music_path = self.music_agent.select_music(
            genre=story_data["genre"],
            mood=story_data.get("mood", "neutral"),
            duration_sec=story_data["total_duration_sec"]
        )
        print(f"{'?€'*60}\n")

        # ìµœì¢… ?ìƒ ?©ì„±
        final_video = self.composer_agent.compose_video(
            video_clips=video_clips,
            narration_clips=narration_clips,
            music_path=music_path,
            output_path=output_path
        )

        print(f"\n{'='*60}")
        print(f"SUCCESS! Video ready for YouTube upload")
        print(f"{'='*60}")
        print(f"File: {os.path.abspath(final_video)}\n")

        return final_video

    def generate_images_for_scenes(
        self,
        story_data: Dict[str, Any],
        project_dir: str,
        request: ProjectRequest = None,
        style_anchor_path: Optional[str] = None,
        environment_anchors: Optional[Dict[int, str]] = None,
    ) -> List[Dict[str, Any]]:
        """
        ?´ë?ì§€ë§??ì„± (TTS, ë¹„ë””???¤í‚µ).
        
        ?¬ìš©?ê? ?´ë?ì§€ë¥?ê²€? í•œ ???¬ìƒ??I2V ë³€??ê°€??
        
        Args:
            story_data: Story JSON
            project_dir: ?„ë¡œ?íŠ¸ ?”ë ‰? ë¦¬
            request: ProjectRequest
            style_anchor_path: ?¤í????µì»¤ ê²½ë¡œ
            environment_anchors: ?˜ê²½ ?µì»¤ ?•ì…”?ˆë¦¬
            
        Returns:
            Scene ?°ì´??ëª©ë¡ (?´ë?ì§€ ê²½ë¡œ ?¬í•¨)
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
        print(f"Style: {style}\n")
        
        processed_scenes = []
        prev_scene = None
        
        # Image output directory
        image_output_dir = f"{project_dir}/media/images"
        os.makedirs(image_output_dir, exist_ok=True)
        
        for i, scene_data in enumerate(scenes, 1):
            print(f"\n{'?€'*60}")
            print(f"Generating Image for Scene {i}/{total_scenes} (ID: {scene_data['scene_id']})")
            print(f"{'?€'*60}")
            
            # Scene ê°ì²´ ?ì„±
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
            
            # Seed ì¶”ì¶œ
            scene_seed = None
            if scene.characters_in_scene and character_sheet:
                first_char_token = scene.characters_in_scene[0]
                if first_char_token in character_sheet:
                    scene_seed = character_sheet[first_char_token].get("visual_seed")
            
            # ë©”í??°ì´???€??
            scene._seed = scene_seed
            scene._global_style = global_style
            scene._character_sheet = character_sheet
            scene._style_anchor_path = style_anchor_path
            scene._env_anchor_path = environment_anchors.get(scene.scene_id) if environment_anchors else None
            
            # Context Carry-over
            if self.feature_flags.context_carry_over and prev_scene:
                scene.context_summary = self.summarize_prev_scene(prev_scene)
                scene.inherited_keywords = self.extract_key_terms(prev_scene)
            else:
                scene.inherited_keywords = []
            
            # ?”í‹°??ì¶”ì¶œ
            scene.entities = self.extract_entities(scene.sentence, scene.inherited_keywords)
            
            # ?„ë¡¬?„íŠ¸ ?ì„±
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
            
            scene.negative_prompt = self.build_negative_prompt(style)
            
            try:
                # Generate IMAGE ONLY (no TTS, no video)
                from agents.image_agent import ImageAgent
                image_agent = ImageAgent()
                
                # Character references
                char_refs = []
                if scene.characters_in_scene and character_sheet:
                    from agents.character_manager import CharacterManager
                    cm = CharacterManager.__new__(CharacterManager)
                    char_refs = cm.get_active_character_images(
                        scene.characters_in_scene,
                        character_sheet
                    )
                
                # Generate image
                image_path, image_id = image_agent.generate_image(
                    scene_id=scene.scene_id,
                    prompt=scene.prompt,
                    negative_prompt=scene.negative_prompt,
                    style=style,
                    output_dir=image_output_dir,
                    seed=scene_seed,
                    character_reference_paths=char_refs,
                    image_model="standard"
                )
                
                scene.assets.image_path = image_path
                scene.status = SceneStatus.COMPLETED
                
                print(f"  ??Image generated: {image_path}")
                
            except Exception as e:
                scene.status = SceneStatus.FAILED
                scene.error_message = str(e)
                print(f"  ??Image generation failed: {e}")
            
            # Scene ?°ì´?°ë? ?•ì…”?ˆë¦¬ë¡?ë³€?˜í•˜???€??
            scene_dict = scene_data.copy()
            scene_dict["assets"] = {
                "image_path": scene.assets.image_path if scene.assets else None
            }
            scene_dict["status"] = scene.status
            scene_dict["prompt"] = scene.prompt
            
            processed_scenes.append(scene_dict)
            prev_scene = scene
            
            print(f"Scene {i} image complete\n")
        
        print(f"\n[SUCCESS] {len(processed_scenes)} images generated!")
        return processed_scenes


    def process_scenes_from_script(
        self,
        script_text: str,
        request: ProjectRequest
    ) -> List[Scene]:
        """
        ?¤í¬ë¦½íŠ¸ ?ìŠ¤?¸ì—??Scene ëª©ë¡ ?ì„±.

        P1: ë§¥ë½ ?ì† ?ìš©

        Args:
            script_text: ?„ì²´ ?¤í¬ë¦½íŠ¸ ?ìŠ¤??
            request: ProjectRequest (feature flags ?¬í•¨)

        Returns:
            Scene ê°ì²´ ëª©ë¡
        """
        # ë¬¸ì¥ ?¨ìœ„ ë¶„í• 
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

            # ?”í‹°??ì¶”ì¶œ
            scene.entities = self.extract_entities(
                sentence,
                scene.inherited_keywords
            )

            # ?„ë¡¬?„íŠ¸ ?ì„±
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
        ?ìŠ¤?¸ë? ë¬¸ì¥ ?¨ìœ„ë¡?ë¶„í• .

        Args:
            text: ?„ì²´ ?ìŠ¤??

        Returns:
            ë¬¸ì¥ ëª©ë¡
        """
        import re

        # ?œêµ­??ë°??ì–´ ë¬¸ì¥ ë¶„í• 
        # ë§ˆì¹¨?? ë¬¼ìŒ?? ?ë‚Œ??ê¸°ì?
        sentences = re.split(r'(?<=[.!?])\s+', text.strip())

        # ë¹?ë¬¸ì¥ ?œê±° ë°??•ë¦¬
        sentences = [s.strip() for s in sentences if s.strip()]

        # ?ˆë¬´ ê¸?ë¬¸ì¥?€ ë¶„í• 
        result = []
        for s in sentences:
            if len(s) > 100:
                # ?¼í‘œ???°ê²°??ê¸°ì??¼ë¡œ ì¶”ê? ë¶„í• 
                parts = re.split(r'(?<=,)\s+|(?<=ê·¸ë¦¬ê³?\s+|(?<=?˜ì?ë§?\s+', s)
                result.extend([p.strip() for p in parts if p.strip()])
            else:
                result.append(s)

        return result

    def retry_scene(
        self,
        scene: Dict[str, Any],
        story_style: str = "cinematic"
    ) -> tuple[str, str]:
        """
        ?¨ì¼ Scene ?¬ì²˜ë¦?

        Args:
            scene: Scene ?°ì´??
            story_style: ?ìƒ ?¤í???

        Returns:
            (video_path, audio_path) ?œí”Œ
        """
        print(f"Retrying scene {scene['scene_id']}...")

        video_path = self.video_agent.generate_video(
            scene_id=scene["scene_id"],
            visual_description=scene["visual_description"],
            style=story_style,
            mood=scene["mood"],
            duration_sec=scene["duration_sec"]
        )

        tts_result = self.tts_agent.generate_speech(
            scene_id=scene["scene_id"],
            narration=scene["narration"],
            emotion=scene["mood"]
        )
        audio_path = tts_result.audio_path

        return video_path, audio_path

    def generate_subtitle_files(
        self,
        scenes: List[Scene],
        output_dir: str = "media/subtitles"
    ) -> List[str]:
        """
        ê°?Scene???€??SRT ?ë§‰ ?Œì¼ ?ì„±.

        Args:
            scenes: Scene ëª©ë¡
            output_dir: ì¶œë ¥ ?”ë ‰? ë¦¬

        Returns:
            SRT ?Œì¼ ê²½ë¡œ ëª©ë¡
        """
        from utils.ffmpeg_utils import FFmpegComposer

        os.makedirs(output_dir, exist_ok=True)
        composer = FFmpegComposer()

        srt_paths = []

        for scene in scenes:
            srt_path = f"{output_dir}/scene_{scene.scene_id:02d}.srt"

            # CRITICAL FIX: Use tts_duration_sec if available, otherwise fallback to duration_sec
            # This ensures subtitle timing matches actual TTS audio length
            actual_duration = scene.tts_duration_sec if scene.tts_duration_sec else scene.duration_sec

            # ?¨ì¼ Scene??SRT ?ì„±
            scene_data = [{
                "narration": scene.narration or scene.sentence,
                "duration_sec": actual_duration  # Use ACTUAL TTS duration
            }]

            composer.generate_srt_from_scenes(scene_data, srt_path)
            scene.assets.subtitle_srt_path = srt_path
            srt_paths.append(srt_path)

        return srt_paths

    def get_processing_stats(
        self,
        scenes: List[Scene]
    ) -> Dict[str, Any]:
        """
        ì²˜ë¦¬ ?µê³„ ë°˜í™˜.

        Args:
            scenes: ì²˜ë¦¬??Scene ëª©ë¡

        Returns:
            ?µê³„ ?•ì…”?ˆë¦¬
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

