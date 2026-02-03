"""
Scene Orchestrator: Manages scene-by-scene processing with context carry-over.

P1 íµì¬ ê¸°ë¥:
- ì´ì  ì¥ë©´ì íµì¬ í¤ìë(ì¸ë¬¼/ì¥ì/ê°ì /íë)ë¥¼ ë¤ì ì¥ë©´ íë¡¬íí¸ì ìì
- Scene ê° ì¼ê´ì± ì ì§
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
    Scene ë¨ì ì²ë¦¬ ì¤ì¼ì¤í¸ë ì´í°

    P1 íµì¬: Context Carry-over (ë§¥ë½ ìì)
    - ì´ì  ì¥ë©´ì íµì¬ í¤ìëë¥¼ ë¤ì ì¥ë©´ íë¡¬íí¸ì ê°ì  í¬í¨
    - ì¸ë¬¼/ì¥ì/ê°ì /íë ì¼ê´ì± ì ì§
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

        # LLM í´ë¼ì´ì¸í¸ (ë§¥ë½ ì¶ì¶ì©)
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
    # P1: Context Carry-over (ë§¥ë½ ìì)
    # =========================================================================

    def extract_entities(
        self,
        sentence: str,
        inherited_keywords: List[str] = None
    ) -> SceneEntities:
        """
        ë¬¸ì¥ìì ìí°í°(ì¸ë¬¼/ì¥ì/ê°ì /íë) ì¶ì¶.

        P1: ë§¥ë½ ììì ìí ìí°í° ì¶ì¶

        Args:
            sentence: ì¥ë©´ ë¬¸ì¥
            inherited_keywords: ì´ì  ì¥ë©´ìì ììë°ì í¤ìë

        Returns:
            SceneEntities ê°ì²´
        """
        if not self.llm_client:
            # LLM ìì¼ë©´ ê¸°ë³¸ ìí°í° ë°í
            return SceneEntities(
                characters=[],
                location=None,
                props=[],
                mood=None,
                action=None
            )

        inherited_context = ", ".join(inherited_keywords) if inherited_keywords else "ìì"

        prompt = f"""
ë¤ì ë¬¸ì¥ìì íµì¬ ìí°í°ë¥¼ ì¶ì¶íì¸ì:

ë¬¸ì¥: {sentence}
ì´ì  ì¥ë©´ ë§¥ë½: {inherited_context}

JSON íìì¼ë¡ ì¶ë ¥:
{{
    "characters": ["ì¸ë¬¼1", "ì¸ë¬¼2"],
    "location": "ì¥ì",
    "props": ["ìí1", "ìí2"],
    "mood": "ë¶ìê¸°/ê°ì ",
    "action": "ì£¼ì íë"
}}

ì£¼ì:
- ì´ì  ì¥ë©´ ë§¥ë½ê³¼ ì¼ê´ì±ì ì ì§íì¸ì
- ë¬ê¸ìë ì¸ë¬¼/ì¥ì ë³ê²½ ê°ì§ ì ì´ì  ë§¥ë½ ì°ì 
- ëª
ìì  ì¸ê¸ì´ ìì¼ë©´ null ì¬ì©
"""

        try:
            system_prompt = "JSONë§ ì¶ë ¥íì¸ì. ë¤ë¥¸ ì¤ëª
 ìì´."
            full_prompt = f"{system_prompt}\n\n{prompt}"

            response = self.llm_client.generate_content(
                full_prompt,
                generation_config={
                    "temperature": 0.3,
                    "max_output_tokens": 300
                }
            )

            content = response.text.strip()
            # JSON íì±
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
        ì´ì  ì¥ë©´ ìì½ ìì±.

        Args:
            scene: ì´ì  Scene ê°ì²´

        Returns:
            ìì½ ë¬¸ìì´
        """
        parts = []

        if scene.entities.characters:
            parts.append(f"ì¸ë¬¼: {', '.join(scene.entities.characters)}")
        if scene.entities.location:
            parts.append(f"ì¥ì: {scene.entities.location}")
        if scene.entities.mood:
            parts.append(f"ë¶ìê¸°: {scene.entities.mood}")
        if scene.entities.action:
            parts.append(f"íë: {scene.entities.action}")

        return " / ".join(parts) if parts else scene.sentence[:50]

    def extract_key_terms(self, scene: Scene) -> List[str]:
        """
        ì´ì  ì¥ë©´ìì íµì¬ í¤ìë ì¶ì¶.

        P1: ë¤ì ì¥ë©´ íë¡¬íí¸ì ììí  í¤ìë

        Args:
            scene: ì´ì  Scene ê°ì²´

        Returns:
            í¤ìë ëª©ë¡
        """
        keywords = []

        # ìí°í°ìì í¤ìë ì¶ì¶
        if scene.entities.characters:
            keywords.extend(scene.entities.characters[:2])  # ìµë 2ëª

        if scene.entities.location:
            keywords.append(scene.entities.location)
        if scene.entities.mood:
            keywords.append(scene.entities.mood)
        if scene.entities.action:
            keywords.append(scene.entities.action)

        return keywords[:5]  # ìµë 5ê° í¤ìë

    def build_prompt(
        self,
        sentence: str,
        inherited: List[str],
        entities: SceneEntities,
        style: str = None
    ) -> str:
        """
        ìì ìì± íë¡¬íí¸ êµ¬ì±.

        P1: inherited í¤ìëë ë°ëì í¬í¨

        Args:
            sentence: ì¥ë©´ ë¬¸ì¥
            inherited: ì´ì  ì¥ë©´ìì ììë°ì í¤ìë
            entities: ì¥ë©´ ìí°í°
            style: ìì ì¤íì¼

        Returns:
            ìì ìì± íë¡¬íí¸
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

        # ìí°í°ë¥¼ ë¬¸ìì´ë¡ ë³í
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
- ì´ì  ì¥ë©´ê³¼ ëì¼ ì¸ë¬¼/ê³µê°/í¤ì ì ì§íë¤.
- ë¬ê¸ìë ë°°ê²½/ìí ë³ê²½ ê¸ì§.
- ê°ì ì ê³¼ì¥íë ê°ì°ì± ì ì§."""

        return prompt

    def build_negative_prompt(self, style: str = None) -> str:
        """
        ë¤ê±°í°ë¸ íë¡¬íí¸ ìì±.

        Args:
            style: ìì ì¤íì¼

        Returns:
            ë¤ê±°í°ë¸ íë¡¬íí¸
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
    # ë©ì¸ ì²ë¦¬ ë¡ì§
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
        Scene JSONìì ìµì¢
 ììê¹ì§ ì ì²´ ì²ë¦¬.

        P1: ë§¥ë½ ìì ì ì©

        Args:
            story_data: Story JSON (scenes í¬í¨)
            output_path: ìµì¢
 ìì ì¶ë ¥ ê²½ë¡
            request: ProjectRequest (feature flags í¬í¨)
            progress_callback: ì§í ì½ë°±
            style_anchor_path: ì¤íì¼ ìµì»¤ ì´ë¯¸ì§ ê²½ë¡
            environment_anchors: ì¬ë³ íê²½ ìµì»¤ ì´ë¯¸ì§ ëì
ëë¦¬

        Returns:
            ìµì¢
 ìì íì¼ ê²½ë¡
        """
        print(f"\n{'='*60}")
        print(f"STORYCUT - Processing Story: {story_data['title']}")
        print(f"{'='*60}\n")

        # Feature flags ì
ë°ì´í¸
        if request:
            self.feature_flags = request.feature_flags
            self.video_agent.feature_flags = request.feature_flags

        scenes = story_data["scenes"]
        total_scenes = len(scenes)
        style = story_data.get("style", request.style_preset if request else "cinematic")
        
        # TTS Voice ì¤ì 
        if request and hasattr(request, 'voice_id'):
            self.tts_agent.voice = request.voice_id
            print(f"TTS Voice set to: {self.tts_agent.voice}")

        # v2.0: ê¸ë¡ë² ì¤íì¼ ê°ì´ë ì¶ì¶
        global_style = story_data.get("global_style")
        character_sheet = story_data.get("character_sheet", {})

        print(f"Total scenes: {total_scenes}")
        print(f"Target duration: {story_data['total_duration_sec']} seconds")
        print(f"Target duration: {story_data['total_duration_sec']} seconds")
        print(f"Context carry-over: {'ON' if self.feature_flags.context_carry_over else 'OFF'}")
        
        # íë¡ì í¸ ë² ì´ì¤ ëë í ë¦¬ ì¤ì  (final_video.mp4 ê²½ë¡ ê¸°ë°)
        # output_path: outputs/<project_id>/final_video.mp4
        project_dir = os.path.dirname(output_path)
        print(f"Project Directory: {project_dir}")

        # v2.0: ê¸ë¡ë² ì¤íì¼ ì ë³´ ì¶ë ¥
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

        # v2.0: ìµì»¤ ì ë³´ ë¡ê¹

        if style_anchor_path:
            print(f"\n[StyleAnchor] Path: {style_anchor_path}")
        if environment_anchors:
            print(f"[EnvAnchors] {len(environment_anchors)} scenes: {list(environment_anchors.keys())}")

        # v2.0: ConsistencyValidator ì´ê¸°í
        consistency_validator = None
        if self.feature_flags.consistency_validation:
            from agents.consistency_validator import ConsistencyValidator
            consistency_validator = ConsistencyValidator()
            print(f"[ConsistencyValidator] Enabled (max_retries={self.feature_flags.consistency_max_retries})")

        print()

        # Scene ì²ë¦¬
        video_clips = []
        narration_clips = []
        processed_scenes = []
        prev_scene = None

        for i, scene_data in enumerate(scenes, 1):
            print(f"\n{'â'*60}")
            print(f"Processing Scene {i}/{total_scenes} (ID: {scene_data['scene_id']})")
            print(f"{'â'*60}")
            print(f"  [DEBUG] Starting scene {i} processing...")

            # Scene ê°ì²´ ìì±
            scene = Scene(
                index=i,
                scene_id=scene_data["scene_id"],
                sentence=scene_data.get("narration", ""),
                narration=scene_data.get("narration"),
                visual_description=scene_data.get("visual_description"),
                mood=scene_data.get("mood"),
                duration_sec=scene_data.get("duration_sec", 5),
                # v2.0 íë
                narrative=scene_data.get("narrative"),
                image_prompt=scene_data.get("image_prompt"),
                characters_in_scene=scene_data.get("characters_in_scene", []),
            )

            # v2.0: Character reference ë¡ê·¸ ë° ìë ì¶ì¶
            scene_seed = None
            if scene.image_prompt:
                print(f"  [v2.0] Using image_prompt (character reference enabled)")
            if scene.characters_in_scene:
                print(f"  [v2.0] Characters: {', '.join(scene.characters_in_scene)}")

                # ì²« ë²ì§¸ ìºë¦­í°ì visual_seed ì¬ì©
                if character_sheet and scene.characters_in_scene:
                    first_char_token = scene.characters_in_scene[0]
                    if first_char_token in character_sheet:
                        scene_seed = character_sheet[first_char_token].get("visual_seed")
                        print(f"  [v2.0] Using visual_seed: {scene_seed}")

            # v2.0: Sceneì ë©íë°ì´í° ì ì¥ (video_agentê° íì©)
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

            # ìí°í° ì¶ì¶
            scene.entities = self.extract_entities(
                scene.sentence,
                scene.inherited_keywords
            )

            # íë¡¬íí¸ ìì±
            # v2.0: image_promptê° ìì¼ë©´ ì°ì  ì¬ì©, ìì¼ë©´ ê¸°ì¡´ ë°©ì
            if scene.image_prompt:
                # image_promptì global_style ì ë³´ ì¶ê°
                if global_style:
                    style_suffix = f", {global_style.get('art_style', '')}, {global_style.get('color_palette', '')}"
                    scene.prompt = scene.image_prompt + style_suffix
                else:
                    scene.prompt = scene.image_prompt
                print(f"  [v2.0] Using pre-defined image_prompt")
            else:
                # v1.0 ë°©ì: build_promptë¡ ìì±
                scene.prompt = self.build_prompt(
                    sentence=scene.sentence,
                    inherited=scene.inherited_keywords,
                    entities=scene.entities,
                    style=style
                )

            scene.negative_prompt = self.build_negative_prompt(style)

            # ì¹´ë©ë¼ ìí¬ í ë¹ (ë¤ìí)
            camera_works = list(CameraWork)
            scene.camera_work = camera_works[i % len(camera_works)]

            try:
                # Phase 1: TTS ë¨¼ì  ìì±íì¬ ì¤ì  duration íë³´
                scene.status = SceneStatus.GENERATING_TTS
                tts_result = self.tts_agent.generate_speech(
                    scene_id=scene.scene_id,
                    narration=scene.narration,
                    emotion=scene.mood
                )
                scene.assets.narration_path = tts_result.audio_path
                scene.tts_duration_sec = tts_result.duration_sec
                # narration_clips.append(tts_result.audio_path) -> REMOVED: ëì¤ì íêº¼ë²ì ìì§

                # TTS ê¸°ë°ì¼ë¡ duration ì
ë°ì´í¸ (ìµì 3ì´, ìµë 15ì´)
                if tts_result.duration_sec > 0:
                    scene.duration_sec = max(3, min(15, int(tts_result.duration_sec) + 1))
                    print(f"     [Duration] Updated to {scene.duration_sec}s (TTS: {tts_result.duration_sec:.2f}s)")

                # ìì ìì± (ì
ë°ì´í¸ë duration ì¬ì©)
                scene.status = SceneStatus.GENERATING_VIDEO

                # íë¡ì í¸ êµ¬ì¡°ì ë§ë ë¹ëì¤/ì´ë¯¸ì§ ì¶ë ¥ ê²½ë¡ ì¤ì 
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
                # video_clips.append(video_path) -> REMOVED: ëì¤ì íêº¼ë²ì ìì§
                scene.assets.video_path = video_path

                # v2.0: ConsistencyValidator ê²ì¦ (ì´ë¯¸ì§ ìì± í, ë¹ëì¤ í©ì± ì )
                if consistency_validator and scene.assets.image_path:
                    # ìºë¦­í° ìµì»¤ ê²½ë¡ ìì§
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

                # ìë£
                scene.status = SceneStatus.COMPLETED

            except Exception as e:
                scene.status = SceneStatus.FAILED
                scene.error_message = str(e)
                scene.retry_count += 1
                print(f"     [ERROR] Scene {i} failed: {e}")
                # ê³ì ì§í (ì¤í¨í ì¬ì ëì¤ì ì¬ìì± ê°ë¥)

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

        # ë°°ê²½ ìì
 ì í
        print(f"{'â'*60}")
        music_path = self.music_agent.select_music(
            genre=story_data["genre"],
            mood=story_data.get("mood", "neutral"),
            duration_sec=story_data["total_duration_sec"]
        )
        print(f"{'â'*60}\n")

        # ìµì¢
 ìì í©ì±
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
        ì´ë¯¸ì§ë§ ìì± (TTS, ë¹ëì¤ ì¤íµ).
        
        ì¬ì©ìê° ì´ë¯¸ì§ë¥¼ ê²í í í ì¬ìì±/I2V ë³í ê°ë¥.
        
        Args:
            story_data: Story JSON
            project_dir: íë¡ì í¸ ëë í ë¦¬
            request: ProjectRequest
            style_anchor_path: ì¤íì¼ ìµì»¤ ê²½ë¡
            environment_anchors: íê²½ ìµì»¤ ëì
ëë¦¬
            
        Returns:
            Scene ë°ì´í° ëª©ë¡ (ì´ë¯¸ì§ ê²½ë¡ í¬í¨)
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
            print(f"\n{'â'*60}")
            print(f"Generating Image for Scene {i}/{total_scenes} (ID: {scene_data['scene_id']})")
            print(f"{'â'*60}")
            
            # Scene ê°ì²´ ìì±
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
            
            # Seed ì¶ì¶
            scene_seed = None
            if scene.characters_in_scene and character_sheet:
                first_char_token = scene.characters_in_scene[0]
                if first_char_token in character_sheet:
                    scene_seed = character_sheet[first_char_token].get("visual_seed")
            
            # ë©íë°ì´í° ì ì¥
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
            
            # ìí°í° ì¶ì¶
            scene.entities = self.extract_entities(scene.sentence, scene.inherited_keywords)
            
            # íë¡¬íí¸ ìì±
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
                
                print(f"  â
 Image generated: {image_path}")
                
            except Exception as e:
                scene.status = SceneStatus.FAILED
                scene.error_message = str(e)
                print(f"  â Image generation failed: {e}")
            
            # Scene ë°ì´í°ë¥¼ ëì
ëë¦¬ë¡ ë³ííì¬ ì ì¥
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
        ì¤í¬ë¦½í¸ í
ì¤í¸ìì Scene ëª©ë¡ ìì±.

        P1: ë§¥ë½ ìì ì ì©

        Args:
            script_text: ì ì²´ ì¤í¬ë¦½í¸ í
ì¤í¸
            request: ProjectRequest (feature flags í¬í¨)

        Returns:
            Scene ê°ì²´ ëª©ë¡
        """
        # ë¬¸ì¥ ë¨ì ë¶í 
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

            # ìí°í° ì¶ì¶
            scene.entities = self.extract_entities(
                sentence,
                scene.inherited_keywords
            )

            # íë¡¬íí¸ ìì±
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
        í
ì¤í¸ë¥¼ ë¬¸ì¥ ë¨ìë¡ ë¶í .

        Args:
            text: ì ì²´ í
ì¤í¸

        Returns:
            ë¬¸ì¥ ëª©ë¡
        """
        import re

        # íêµ­ì´ ë° ìì´ ë¬¸ì¥ ë¶í 
        # ë§ì¹¨í, ë¬¼ìí, ëëí ê¸°ì¤
        sentences = re.split(r'(?<=[.!?])\s+', text.strip())

        # ë¹ ë¬¸ì¥ ì ê±° ë° ì ë¦¬
        sentences = [s.strip() for s in sentences if s.strip()]

        # ëë¬´ ê¸´ ë¬¸ì¥ì ë¶í 
        result = []
        for s in sentences:
            if len(s) > 100:
                # ì¼íë ì°ê²°ì´ ê¸°ì¤ì¼ë¡ ì¶ê° ë¶í 
                parts = re.split(r'(?<=,)\s+|(?<=ê·¸ë¦¬ê³ )\s+|(?<=íì§ë§)\s+', s)
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
        ë¨ì¼ Scene ì¬ì²ë¦¬.

        Args:
            scene: Scene ë°ì´í°
            story_style: ìì ì¤íì¼

        Returns:
            (video_path, audio_path) íí
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
        ê° Sceneì ëí SRT ìë§ íì¼ ìì±.

        Args:
            scenes: Scene ëª©ë¡
            output_dir: ì¶ë ¥ ëë í ë¦¬

        Returns:
            SRT íì¼ ê²½ë¡ ëª©ë¡
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

            # ë¨ì¼ Sceneì© SRT ìì±
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
        ì²ë¦¬ íµê³ ë°í.

        Args:
            scenes: ì²ë¦¬ë Scene ëª©ë¡

        Returns:
            íµê³ ëì
ëë¦¬
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