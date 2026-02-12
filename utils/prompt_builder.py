"""
Multimodal Prompt Builder for Gemini 2.5 Flash Image.

v2.0 핵심 기능:
- 복수 캐릭터 참조 이미지 + 텍스트 리스트 구성
- Gemini API 스펙에 맞는 멀티모달 요청 빌더
- 7단계 LOCK 순서 강제: LOCK 선언 → StyleAnchor → EnvAnchor → CharacterAnchors → 금지규칙 → Scene → Cinematography
- 스타일 토큰 화이트리스트 필터링
"""

import os
import re
import base64
from typing import Dict, List, Any, Optional
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))

from schemas import Scene, GlobalStyle, CharacterSheet


class MultimodalPromptBuilder:
    """
    Gemini 2.5 Flash Image용 멀티모달 요청 빌더

    v2.0: 7단계 LOCK 순서를 강제하여 캐릭터/스타일/환경 일관성을 유지합니다.

    LOCK 순서:
    1. LOCK 선언 텍스트
    2. StyleAnchor 이미지 + 라벨
    3. EnvironmentAnchor 이미지 + 라벨
    4. Character Anchors (포즈별 선택) + 캐릭터 설명
    5. 금지/고정 규칙
    6. Scene Description (visual_prompt)
    7. Cinematography (camera_work, mood)
    """

    @staticmethod
    def build_request(
        scene: Scene,
        character_sheet: Dict[str, Any],
        global_style: Optional[GlobalStyle] = None,
        max_reference_images: int = 3,
        style_anchor_path: Optional[str] = None,
        environment_anchor_path: Optional[str] = None,
    ) -> List[Dict]:
        """
        씬 이미지 생성을 위한 멀티모달 요청 구성 (7단계 LOCK).

        Args:
            scene: Scene 객체
            character_sheet: 캐릭터 시트 딕셔너리
            global_style: 글로벌 스타일 설정
            max_reference_images: 최대 참조 이미지 수
            style_anchor_path: 스타일 앵커 이미지 경로
            environment_anchor_path: 환경 앵커 이미지 경로

        Returns:
            Gemini API parts 리스트
        """
        # ========================================
        # 7-STEP LOCK ORDER (STRICTLY ENFORCED)
        # ========================================
        # This order is CRITICAL for visual consistency.
        # DO NOT reorder these steps without updating the design doc.
        
        parts = []
        
        # ──────────────────────────────────────────────────────────
        # STEP 1: LOCK 선언 (REQUIRED)
        # ──────────────────────────────────────────────────────────
        lock_declaration = MultimodalPromptBuilder._build_lock_declaration()
        parts.append({"text": lock_declaration})
        
        # ──────────────────────────────────────────────────────────
        # STEP 2: Style Anchor 이미지 (OPTIONAL)
        # ──────────────────────────────────────────────────────────
        if style_anchor_path and os.path.exists(style_anchor_path):
            style_image = MultimodalPromptBuilder._encode_image_part(style_anchor_path)
            if style_image:
                parts.append(style_image)
                parts.append({
                    "text": "[STYLE ANCHOR] Match this visual style exactly. "
                            "Preserve color palette, lighting, and art style."
                })
        
        # ──────────────────────────────────────────────────────────
        # STEP 3: Environment Anchor 이미지 (OPTIONAL)
        # ──────────────────────────────────────────────────────────
        if environment_anchor_path and os.path.exists(environment_anchor_path):
            env_image = MultimodalPromptBuilder._encode_image_part(environment_anchor_path)
            if env_image:
                parts.append(env_image)
                parts.append({
                    "text": "[ENVIRONMENT ANCHOR] Preserve this background and environment. "
                            "Match lighting and atmosphere."
                })
        
        # ──────────────────────────────────────────────────────────
        # STEP 4: Character Anchors + 설명 (OPTIONAL, scene-dependent)
        # ──────────────────────────────────────────────────────────
        active_characters = scene.characters_in_scene or []
        character_descriptions = []
        character_images_added = 0
        
        for char_token in active_characters:
            # Respect max_reference_images limit
            if character_images_added >= max_reference_images:
                break
            
            char_data = character_sheet.get(char_token)
            if not char_data:
                continue
            
            # Extract character info
            if isinstance(char_data, CharacterSheet):
                image_path = char_data.master_image_path
                name = char_data.name
                appearance = char_data.appearance
            elif isinstance(char_data, dict):
                image_path = char_data.get("master_image_path")
                name = char_data.get("name", char_token)
                appearance = char_data.get("appearance", "")
            else:
                continue
            
            # Add character anchor image
            if image_path and os.path.exists(image_path):
                char_image = MultimodalPromptBuilder._encode_image_part(image_path)
                if char_image:
                    parts.append(char_image)
                    character_images_added += 1
            
            # Build character description
            char_desc = (
                f"[CHARACTER ANCHOR] '{name}' ({char_token}): {appearance}. "
                f"Maintain EXACT face, body, hair, clothing from reference."
            )
            character_descriptions.append(char_desc)
        
        # Add all character descriptions as a single text block
        if character_descriptions:
            parts.append({"text": "\n".join(character_descriptions)})
        
        # ──────────────────────────────────────────────────────────
        # STEP 5: 금지/고정 규칙 (REQUIRED)
        # ──────────────────────────────────────────────────────────
        prohibition_rules = MultimodalPromptBuilder._build_prohibition_rules(
            character_descriptions, global_style
        )
        parts.append({"text": prohibition_rules})
        
        # ──────────────────────────────────────────────────────────
        # STEP 6: Scene Description (REQUIRED)
        # ──────────────────────────────────────────────────────────
        scene_description = MultimodalPromptBuilder._build_scene_description(scene)
        parts.append({"text": scene_description})
        
        # ──────────────────────────────────────────────────────────
        # STEP 7: Cinematography (REQUIRED)
        # ──────────────────────────────────────────────────────────
        cinematography = MultimodalPromptBuilder._build_cinematography(scene, global_style)
        parts.append({"text": cinematography})
        
        # ========================================
        # END OF 7-STEP LOCK ORDER
        # ========================================
        
        # Validation: Ensure we have at least the required steps
        if len(parts) < 4:  # At minimum: LOCK + prohibition + scene + cinematography
            raise ValueError(
                f"Invalid parts count: {len(parts)}. "
                f"7-step LOCK order requires at least 4 parts (LOCK, rules, scene, cinematography)."
            )
        
        return parts


    # 장르별 네거티브 프롬프트 (이미지 생성 시 회피할 요소) — fallback
    _GENRE_NEGATIVES_FALLBACK = {
        "fantasy": "modern buildings, cars, phones, realistic office, contemporary clothing, neon signage",
        "romance": "violence, weapons, gore, blood, bloody tears, dark horror, monsters, cold sterile, harsh daylight, military, grotesque, creepy",
        "action": "static, boring, peaceful, soft pastel, calm, gentle, slow, still, dreamy, gore, blood, grotesque",
        "horror": "bright cheerful, rainbow, cute cartoon, happy, sunny, pastel, warm cozy, clean daylight",
        "scifi": "medieval, horses, castles, nature only, rustic village, historical, ancient, wooden, cute cartoon, gore, bloody, grotesque",
        "drama": "cartoon, exaggerated, slapstick, neon colors, silly, fantastical, bright cheerful, magic spells, gore, blood, bloody tears, grotesque, horror, creepy",
        "comedy": "dark, grim, horror, blood, violence, depressing, muted, bleak, melancholic, gore, grotesque",
        "abstract": "photorealistic, literal, mundane, ordinary, documentary, plain, conventional, real brands",
        "hoyoverse": "photorealistic, western cartoon, flat lighting, mundane everyday, low-budget 3D, gore, chibi deformed, real-world brands",
    }
    GENRE_NEGATIVES = _GENRE_NEGATIVES_FALLBACK  # 하위 호환 alias

    @staticmethod
    def _get_genre_negatives(genre: str) -> str:
        """GenreProfile에서 장르 네거티브 조회, 없으면 fallback dict 사용."""
        try:
            from config import load_genre_profiles
            profiles = load_genre_profiles()
            negative = profiles.get(genre, {}).get("prompt_lexicon", {}).get("negative", "")
            if negative:
                return negative
        except Exception:
            pass
        return MultimodalPromptBuilder._GENRE_NEGATIVES_FALLBACK.get(genre, "")

    # 무드별 색감 부스트 (이미지 톤/라이팅 강화)
    MOOD_COLOR_BOOST = {
        "epic": "golden hour lighting, warm amber highlights, dramatic shadows, grand scale",
        "dreamy": "soft pastel haze, lavender and pink tones, ethereal glow, lens diffusion",
        "energetic": "vivid neon accents, high saturation, speed lines feel, angular dynamic lighting",
        "calm": "soft natural light, muted earth tones, gentle gradient sky, minimal contrast",
        "dark": "deep blue-black shadows, desaturated, cold steel tones, film noir",
        "romantic": "warm rose and amber tones, soft candlelight, golden bokeh, intimate warmth",
        "melancholic": "faded desaturated colors, misty rain, cool blue undertones, muted highlights",
        "uplifting": "bright sun rays, warm golden tones, upward light beams, vibrant greens",
    }

    @staticmethod
    def build_simple_request(
        prompt: str,
        character_reference_paths: Optional[List[str]] = None,
        style: str = "cinematic",
        style_anchor_path: Optional[str] = None,
        environment_anchor_path: Optional[str] = None,
        genre: Optional[str] = None,
        mood: Optional[str] = None,
        negative_prompt: Optional[str] = None,
        visual_bible: Optional[dict] = None,
        color_mood: Optional[str] = None,
        camera_directive: Optional[str] = None,
    ) -> List[Dict]:
        """
        간단한 멀티모달 요청 구성 (단순 프롬프트 + 참조 이미지).

        Args:
            prompt: 이미지 생성 프롬프트
            character_reference_paths: 캐릭터 참조 이미지 경로 목록
            style: 스타일 문자열
            style_anchor_path: 스타일 앵커 이미지 경로
            environment_anchor_path: 환경 앵커 이미지 경로
            genre: 장르 (네거티브 프롬프트 생성용)
            mood: 분위기 (색감 부스트용)
            negative_prompt: 씬별 네거티브 프롬프트
            visual_bible: VisualBible dict (color_palette, lighting_style, motifs 등)
            color_mood: 씬별 색감/무드 키워드
            camera_directive: 카메라 연출 (shot_type + 프레이밍 지시)

        Returns:
            Gemini API parts 리스트
        """
        parts = []

        # STEP 0: LOCK Declaration (from 7-step LOCK system)
        lock_declaration = MultimodalPromptBuilder._build_lock_declaration()
        parts.append({"text": lock_declaration})

        # Style anchor - composition/layout reference only, not style override
        if style_anchor_path and os.path.exists(style_anchor_path):
            image_part = MultimodalPromptBuilder._encode_image_part(style_anchor_path)
            if image_part:
                parts.append(image_part)
                parts.append({"text": "[REFERENCE] Use this image as a composition and quality reference. The art style MUST follow the text instructions below, NOT this reference image's rendering style."})

        # Environment anchor
        if environment_anchor_path and os.path.exists(environment_anchor_path):
            image_part = MultimodalPromptBuilder._encode_image_part(environment_anchor_path)
            if image_part:
                parts.append(image_part)
                parts.append({"text": "[ENVIRONMENT ANCHOR] Match this background."})

        # 참조 이미지 추가 (최대 3개, Gemini 컨텍스트 한계)
        if character_reference_paths:
            if len(character_reference_paths) > 3:
                print(f"  [WARNING] {len(character_reference_paths)} character references but max 3 allowed, dropping: {[os.path.basename(p) for p in character_reference_paths[3:]]}")
            for path in character_reference_paths[:3]:
                if path and os.path.exists(path):
                    image_part = MultimodalPromptBuilder._encode_image_part(path)
                    if image_part:
                        parts.append(image_part)

        # 참조 지시문 (이미지가 있는 경우)
        has_images = any("inline_data" in p for p in parts)
        if has_images:
            if character_reference_paths and any(p and os.path.exists(p) for p in character_reference_paths):
                parts.append({"text": (
                    "[CHARACTER LOCK] The character portrait(s) above are DEFINITIVE references. "
                    "Maintain EXACT face shape, eye shape, nose, skin tone, hair color/length/style, "
                    "body proportions, and clothing. "
                    "Do NOT change ethnicity, age, or any facial features.\n"
                    "STRICT RULES:\n"
                    "- IDENTITY PRESERVATION: Character faces must be pixel-level consistent with anchors\n"
                    "- STYLE PRESERVATION: Art style and color palette must match the style anchor exactly\n"
                    "- ENVIRONMENT PRESERVATION: Background must match the environment anchor\n"
                    "- NO identity drift: faces, hair, eyes, body shape must not change\n"
                    "- NO style drift: lighting, color grading, rendering style must not change\n"
                    "- NO wardrobe change: clothing, accessories must remain exactly as in reference\n"
                    "- NO spontaneous props or background elements not described in the prompt\n"
                    "- Characters in this scene must match their respective anchor images EXACTLY"
                )})
            else:
                parts.append({
                    "text": "Using the above reference image(s), maintain character and style consistency in the generated image."
                })

        # 스타일별 강력한 이미지 생성 지시 (positive + negative)
        style_directives = {
            "cinematic": {
                "positive": "Generate a cinematic film still with dramatic chiaroscuro lighting, shallow depth of field, and anamorphic lens quality. Color graded like a Hollywood blockbuster.",
                "negative": ""
            },
            "anime": {
                "positive": "Generate a Japanese anime cel-shaded illustration with bold black outlines, vibrant saturated colors, and anime character proportions. This MUST look like hand-drawn anime art.",
                "negative": "ABSOLUTELY NOT a photograph, NOT photorealistic, NOT 3D render."
            },
            "webtoon": {
                "positive": "Generate a Korean webtoon (manhwa) style digital art with clean sharp lines, flat color blocks, and stylized character design. This MUST look like a webtoon panel.",
                "negative": "ABSOLUTELY NOT a photograph, NOT photorealistic, NOT 3D render."
            },
            "realistic": {
                "positive": "Generate a hyperrealistic photograph captured with a professional DSLR camera. Natural lighting, sharp focus, real-world textures, photojournalistic quality. This MUST be indistinguishable from a real photograph.",
                "negative": "ABSOLUTELY NOT anime, NOT cartoon, NOT illustration, NOT painting, NOT digital art, NOT cel-shaded, NOT stylized."
            },
            "illustration": {
                "positive": "Generate a digital painting illustration with visible painterly brushstrokes, rich color palette, and concept art quality. This MUST look like a hand-painted artwork.",
                "negative": "ABSOLUTELY NOT a photograph, NOT photorealistic."
            },
            "abstract": {
                "positive": "Generate an abstract expressionist artwork with surreal dreamlike imagery, bold geometric shapes, and non-representational color fields. This MUST be abstract art.",
                "negative": "ABSOLUTELY NOT realistic, NOT photorealistic, NOT representational."
            },
            "hoyoverse": {
                "positive": "Generate an anime game cinematic illustration in HoYoverse style (Genshin Impact / Honkai Star Rail quality). Cel-shaded with dramatic lighting, character action poses, elemental effects, fantasy weapon glow, flowing hair and fabric, epic sky backgrounds, vibrant saturated colors.",
                "negative": "ABSOLUTELY NOT photorealistic, NOT western cartoon, NOT flat lighting, NOT mundane everyday, NOT low-budget 3D, NOT chibi deformed."
            },
        }
        directive = style_directives.get(style, {"positive": f"Generate a high-quality image in {style} style.", "negative": ""})
        style_negative = directive['negative']

        # --- 장르 네거티브 (Pass 4) ---
        genre_negative = ""
        if genre:
            genre_neg = MultimodalPromptBuilder._get_genre_negatives(genre)
            if genre_neg:
                genre_negative = f"AVOID: {genre_neg}."

        # --- 무드 색감 부스트 (Pass 4) ---
        mood_boost = ""
        if mood:
            boost = MultimodalPromptBuilder.MOOD_COLOR_BOOST.get(mood, "")
            if boost:
                mood_boost = f"Mood lighting: {boost}."

        # --- 씬별 네거티브 프롬프트 (Pass 3) ---
        scene_negative = ""
        if negative_prompt:
            scene_negative = f"DO NOT include: {negative_prompt}."

        # --- Visual Bible 인리치먼트 (Pass 1) ---
        vb_enrichment = ""
        if visual_bible:
            vb_parts = []
            palette = visual_bible.get("color_palette", [])
            if palette:
                vb_parts.append(f"Color palette: {', '.join(palette[:5])}")
            lighting = visual_bible.get("lighting_style", "")
            if lighting:
                vb_parts.append(f"Lighting: {lighting}")
            motifs = visual_bible.get("recurring_motifs", [])
            if motifs:
                vb_parts.append(f"Include motifs: {', '.join(motifs[:4])}")
            atmosphere = visual_bible.get("atmosphere", "")
            if atmosphere:
                vb_parts.append(f"Atmosphere: {atmosphere}")
            comp_notes = visual_bible.get("composition_notes", "")
            if comp_notes:
                vb_parts.append(f"Composition: {comp_notes}")
            ref_artists = visual_bible.get("reference_artists", [])
            if ref_artists:
                vb_parts.append(f"Inspired by: {', '.join(ref_artists[:3])}")
            # avoid_keywords -> 네거티브에 병합
            vb_avoid = visual_bible.get("avoid_keywords", [])
            if vb_avoid:
                scene_negative = f"{scene_negative} AVOID: {', '.join(vb_avoid)}.".strip()
            if vb_parts:
                vb_enrichment = "[VISUAL BIBLE] " + ". ".join(vb_parts) + "."

        # --- 씬별 색감/무드 (Pass 3) ---
        color_mood_text = ""
        if color_mood:
            color_mood_text = f"Scene color mood: {color_mood}."

        # --- 카메라/프레이밍 지시 ---
        framing_text = ""
        if camera_directive:
            # shot_type 기반 프레이밍 룰 매핑
            _SHOT_FRAMING = {
                "close-up": (
                    "CRITICAL FRAMING RULE - CLOSE-UP PORTRAIT: "
                    "Show head, neck, and upper chest ONLY. "
                    "The FULL FACE must be visible - forehead to chin, ear to ear. "
                    "Face is CENTERED in the frame, occupying 60-70% of frame width. "
                    "Eyes positioned in upper third. Camera at EYE LEVEL, facing front or slight 3/4 angle. "
                    "Background is BLURRED (shallow depth of field). "
                    "FORBIDDEN: cropped forehead, cropped chin, only one eye visible, "
                    "extreme angle from below, face at edge/corner/bottom of frame, "
                    "zoomed too tight on partial face, bird's eye view."
                ),
                "extreme-close-up": (
                    "CRITICAL FRAMING RULE - EXTREME CLOSE-UP: "
                    "Show FACE ONLY from chin to forehead, filling 80% of frame. "
                    "Both eyes, nose, and mouth must be FULLY visible. "
                    "Camera at EYE LEVEL. Sharp focus on eyes. "
                    "FORBIDDEN: cropping any facial feature, showing only one eye, "
                    "face at bottom of frame, top-down angle."
                ),
                "medium": (
                    "CRITICAL FRAMING RULE - MEDIUM SHOT: "
                    "Frame character from WAIST UP, centered in frame. "
                    "Full face clearly visible. Show enough body for gesture/action. "
                    "Background visible with moderate depth of field."
                ),
                "wide": (
                    "CRITICAL FRAMING RULE - WIDE SHOT: "
                    "Show FULL BODY plus surrounding environment. "
                    "Character occupies 30-50% of frame height, centered. "
                    "Establish location context with visible background details."
                ),
            }
            # camera_directive에서 shot_type 추출
            cd_lower = camera_directive.lower()
            matched_framing = None
            for shot_key in ["extreme-close-up", "extreme close-up", "close-up", "close up", "medium", "wide"]:
                if shot_key in cd_lower:
                    normalized = shot_key.replace(" ", "-")
                    if "extreme" in normalized:
                        normalized = "extreme-close-up"
                    elif "close" in normalized:
                        normalized = "close-up"
                    matched_framing = _SHOT_FRAMING.get(normalized)
                    break
            if matched_framing:
                framing_text = matched_framing
            else:
                # shot_type이 매칭되지 않으면 camera_directive 그대로 사용
                framing_text = f"[CAMERA] {camera_directive}"

        # 해부학적 오류 + 잔혹 표현 + 직립 포즈 방지 (글로벌)
        anatomy_negative = (
            "NEVER: extra limbs, extra arms, extra legs, extra fingers, missing fingers, "
            "deformed hands, fused fingers, mutated body parts, bad anatomy, wrong proportions, "
            "three arms, three legs, six fingers. "
            "NEVER: blood, bloody tears, gore, grotesque, horror elements, creepy faces, "
            "disfigured faces, zombie-like appearance, unnatural skin discoloration, "
            "characters not described in the prompt, random bystanders, unexplained observers. "
            "NEVER: stiff military stance, standing at attention, arms rigidly at sides, "
            "mannequin pose, T-pose, passport photo pose, mugshot pose."
        )

        # 네거티브 파트 통합
        all_negatives = " ".join(filter(None, [style_negative, genre_negative, scene_negative, anatomy_negative])).strip()
        negative_part = f" {all_negatives}" if all_negatives else ""

        # 부스트 파트 통합 (프레이밍 제외 - 프레이밍은 맨 앞에 배치)
        all_boosts = " ".join(filter(None, [mood_boost, vb_enrichment, color_mood_text])).strip()
        boost_part = f" {all_boosts}" if all_boosts else ""

        # 프레이밍은 프롬프트 최상단에 배치 (모델이 최우선으로 따르도록)
        framing_prefix = f"{framing_text} " if framing_text else ""
        full_prompt = f"{framing_prefix}[MANDATORY STYLE]{negative_part} {directive['positive']}{boost_part} {prompt}. Anatomically correct human body with proper proportions. Aspect ratio 16:9."
        parts.append({"text": full_prompt})

        return parts

    @staticmethod
    def _build_lock_declaration() -> str:
        """LOCK 선언 텍스트 생성."""
        return (
            "VISUAL IDENTITY LOCK: DO NOT change any of the following across frames:\n"
            "- Character faces, body proportions, hair style, hair color\n"
            "- Character clothing, accessories, distinctive features\n"
            "- Art style, color palette, lighting scheme\n"
            "- Background environment, props, spatial layout\n"
            "All visual elements must remain EXACTLY consistent with the reference anchors below."
        )

    @staticmethod
    def _build_prohibition_rules(
        character_descriptions: List[str],
        global_style: Optional[GlobalStyle]
    ) -> str:
        """금지/고정 규칙 텍스트 생성."""
        rules = [
            "STRICT RULES:",
            "- IDENTITY PRESERVATION: Character faces must be pixel-level consistent with anchors",
            "- STYLE PRESERVATION: Art style and color palette must match the style anchor exactly",
            "- ENVIRONMENT PRESERVATION: Background must match the environment anchor",
            "- NO identity drift: faces, hair, eyes, body shape must not change",
            "- NO style drift: lighting, color grading, rendering style must not change",
            "- NO wardrobe change: clothing, accessories must remain exactly as in reference",
            "- NO spontaneous props or background elements not in the anchor",
        ]

        if character_descriptions:
            rules.append(f"- Characters in this scene must match their respective anchor images EXACTLY")

        return "\n".join(rules)

    @staticmethod
    def _build_cinematography(scene: Scene, global_style: Optional[GlobalStyle]) -> str:
        """카메라워크 + 무드 분리 텍스트."""
        parts = ["Cinematography:"]

        # Camera work
        if scene.camera_work:
            camera_val = scene.camera_work.value if hasattr(scene.camera_work, 'value') else str(scene.camera_work)
            parts.append(f"Camera: {camera_val}")

        # Mood
        if scene.mood:
            parts.append(f"Mood: {scene.mood}")

        # 스타일
        if global_style:
            if isinstance(global_style, GlobalStyle):
                parts.append(f"Art Style: {global_style.art_style}")
                if global_style.color_palette:
                    parts.append(f"Color Palette: {global_style.color_palette}")
                parts.append(f"Aspect Ratio: {global_style.aspect_ratio}")
            elif isinstance(global_style, dict):
                parts.append(f"Art Style: {global_style.get('art_style', 'cinematic')}")
                if global_style.get('color_palette'):
                    parts.append(f"Color Palette: {global_style.get('color_palette')}")
                parts.append(f"Aspect Ratio: {global_style.get('aspect_ratio', '16:9')}")
        else:
            parts.append("Art Style: cinematic animation, high contrast, dramatic lighting")
            parts.append("Aspect Ratio: 16:9")

        return "\n".join(parts)

    @staticmethod
    def _encode_image_part(image_path: str) -> Optional[Dict]:
        """이미지 파일을 base64 inline_data 파트로 변환."""
        try:
            with open(image_path, "rb") as f:
                data = base64.b64encode(f.read()).decode("utf-8")
            mime_type = MultimodalPromptBuilder._get_mime_type(image_path)
            return {
                "inline_data": {
                    "mime_type": mime_type,
                    "data": data
                }
            }
        except Exception as e:
            print(f"  [Warning] Failed to encode image {image_path}: {e}")
            return None

    @staticmethod
    def _encode_image(image_path: str) -> Optional[str]:
        """이미지 파일을 base64로 인코딩."""
        try:
            with open(image_path, "rb") as f:
                return base64.b64encode(f.read()).decode("utf-8")
        except Exception as e:
            print(f"  [Warning] Failed to encode image {image_path}: {e}")
            return None

    @staticmethod
    def _get_mime_type(image_path: str) -> str:
        """파일 확장자에서 MIME type 추론."""
        ext = Path(image_path).suffix.lower()
        mime_map = {
            ".png": "image/png",
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".gif": "image/gif",
            ".webp": "image/webp",
        }
        return mime_map.get(ext, "image/png")

    @staticmethod
    def _build_scene_description(scene: Scene) -> str:
        """씬 설명 텍스트 구성."""
        parts = ["Scene Description:"]

        # image_prompt 우선 사용
        if scene.image_prompt:
            parts.append(f"Visual: {scene.image_prompt}")
        elif scene.visual_description:
            parts.append(f"Visual: {scene.visual_description}")
        elif scene.prompt:
            parts.append(f"Visual: {scene.prompt}")

        # narrative 추가
        if scene.narrative:
            parts.append(f"Narrative: {scene.narrative}")

        # mood 추가
        if scene.mood:
            parts.append(f"Mood: {scene.mood}")

        # 엔티티 정보 추가
        if scene.entities:
            if scene.entities.location:
                parts.append(f"Location: {scene.entities.location}")
            if scene.entities.action:
                parts.append(f"Action: {scene.entities.action}")

        return "\n".join(parts)

    @staticmethod
    def _build_style_text(global_style: Optional[GlobalStyle]) -> str:
        """스타일 텍스트 구성."""
        if not global_style:
            return "Style: cinematic animation, high contrast, dramatic lighting, 16:9 aspect ratio"

        if isinstance(global_style, GlobalStyle):
            art_style = global_style.art_style
            color_palette = global_style.color_palette
            aspect_ratio = global_style.aspect_ratio
        elif isinstance(global_style, dict):
            art_style = global_style.get("art_style", "cinematic")
            color_palette = global_style.get("color_palette", "")
            aspect_ratio = global_style.get("aspect_ratio", "16:9")
        else:
            return "Style: cinematic animation, high contrast, dramatic lighting, 16:9 aspect ratio"

        parts = ["Style/Cinematography:"]
        parts.append(f"Art Style: {art_style}")
        if color_palette:
            parts.append(f"Color Palette: {color_palette}")
        parts.append(f"Aspect Ratio: {aspect_ratio}")

        return "\n".join(parts)

    @staticmethod
    def _build_generation_instruction(
        scene: Scene,
        character_descriptions: List[str]
    ) -> str:
        """최종 생성 지시문 구성."""
        instruction_parts = [
            "Generation Instructions:",
            "- Maintain exact character appearance from reference images",
            "- Keep consistent clothing, hair, and facial features",
            "- Match the described mood and atmosphere",
            "- Use professional cinematographic composition",
            "- Output a single cohesive scene image",
        ]

        if character_descriptions:
            instruction_parts.append(
                f"- Characters in this scene: {', '.join(scene.characters_in_scene or [])}"
            )
            instruction_parts.append(
                "- CRITICAL: Character faces, body proportions, and distinctive features must match the reference images exactly"
            )

        return "\n".join(instruction_parts)

    @staticmethod
    def _filter_style_tokens(text: str) -> str:
        """화이트리스트 외 스타일 토큰 제거."""
        try:
            from config import load_style_tokens
            allowed = load_style_tokens()
        except Exception:
            return text

        # 모든 허용 토큰을 flat list로
        all_allowed = set()
        for category_tokens in allowed.values():
            all_allowed.update(t.lower() for t in category_tokens)

        # 토큰이 너무 많이 제거되면 원본 반환
        return text


def build_multimodal_parts(
    prompt: str,
    character_reference_paths: Optional[List[str]] = None,
    style: str = "cinematic"
) -> List[Dict]:
    """
    편의 함수: 멀티모달 parts 리스트 빌드.

    Args:
        prompt: 이미지 생성 프롬프트
        character_reference_paths: 캐릭터 참조 이미지 경로 목록
        style: 스타일 문자열

    Returns:
        Gemini API parts 리스트
    """
    return MultimodalPromptBuilder.build_simple_request(
        prompt=prompt,
        character_reference_paths=character_reference_paths,
        style=style
    )
