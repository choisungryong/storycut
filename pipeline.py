"""
STORYCUT 통합 파이프라인

전체 실행 플로우를 관리하는 메인 파이프라인.
outputs/<project_id>/ 구조로 모든 산출물 관리.

실행 플로우:
1. (optional) TopicFindingAgent - 주제 후보 생성
2. StoryAgent - 스토리 생성
3. SceneOrchestrator - Scene 처리 (맥락 상속, 영상/음성 생성)
4. FFmpegComposer - 최종 합성
5. OptimizationAgent - 제목/썸네일/AB테스트 패키지
"""

import os
import json
import time
import uuid
from dotenv import load_dotenv

# .env 파일 로드
load_dotenv()

from datetime import datetime
from typing import Dict, Any, Optional, List
from pathlib import Path

from schemas import (
    FeatureFlags,
    ProjectRequest,
    Scene,
    Manifest,
    ManifestOutputs,
    CostEstimate,
)
from agents import (
    StoryAgent,
    SceneOrchestrator,
    OptimizationAgent,
    CharacterManager,
    StyleAnchorAgent,
    ConsistencyValidator,
)
from utils.ffmpeg_utils import FFmpegComposer


class StorycutPipeline:
    """
    STORYCUT 통합 파이프라인

    모든 에이전트를 조율하고 Manifest를 관리합니다.
    """

    def __init__(self, output_base_dir: str = "outputs"):
        """
        Initialize pipeline.

        Args:
            output_base_dir: 출력 기본 디렉토리
        """
        self.output_base_dir = output_base_dir
        self.story_agent = StoryAgent()
        self.optimization_agent = OptimizationAgent()

    def run(self, request: ProjectRequest) -> Manifest:
        """
        [Legacy] 전체 파이프라인 한 번에 실행.
        """
        # 1. 스토리 생성
        story_data = self.generate_story_only(request)
        
        # 2. 영상 생성 (스토리 기반)
        return self.generate_video_from_story(story_data, request)

    _ETH_RULES = {
        "korean": "All characters MUST be described as 'Korean' (e.g., 'Korean man', 'Korean woman').",
        "japanese": "All characters MUST be described as 'Japanese'.",
        "chinese": "All characters MUST be described as 'Chinese'.",
        "southeast_asian": "All characters MUST be described as 'Southeast Asian'.",
        "european": "All characters MUST be described as 'European/Caucasian'.",
        "black": "All characters MUST be described as 'Black/African'.",
        "hispanic": "All characters MUST be described as 'Hispanic/Latino'.",
        "mixed": "Each character's specific ethnicity MUST be stated explicitly.",
    }

    def _get_ethnicity_rule(self, ethnicity: str) -> str:
        rule = self._ETH_RULES.get(ethnicity, "")
        return f"\n- *** {rule} ***" if rule else ""

    def generate_story_only(self, request: ProjectRequest) -> Dict[str, Any]:
        """Step 1: 스토리만 생성"""
        print(f"\n[STEP 1] Generating story for topic: {request.topic}")
        return self._generate_story(request)

    def generate_story_from_script(self, script_text: str, request: ProjectRequest) -> Dict[str, Any]:
        """
        사용자 스크립트 텍스트를 씬 분할하고 Gemini로 이미지 프롬프트를 생성.

        기존 generate_story_only()와 동일한 story_data 포맷을 반환하여
        리뷰 화면 및 영상 파이프라인을 그대로 재활용할 수 있음.

        Args:
            script_text: 전체 내레이션 스크립트 (빈 줄로 씬 구분)
            request: ProjectRequest (genre, mood, style 등)

        Returns:
            story_data dict (기존 StoryAgent 출력과 동일 형식)
        """
        import re

        print(f"\n[SCRIPT MODE] Generating story_data from user script...")

        # 1. 스크립트를 빈 줄 기준으로 씬 분할
        paragraphs = [p.strip() for p in re.split(r'\n\s*\n', script_text.strip()) if p.strip()]

        # 빈 줄이 없어 1개 단락만 나오면 → 줄 단위로 분할 후 3~4줄씩 그룹핑
        if len(paragraphs) <= 1:
            lines = [l.strip() for l in script_text.strip().split('\n') if l.strip()]
            if len(lines) > 1:
                group_size = 3 if len(lines) <= 12 else 4
                paragraphs = []
                for i in range(0, len(lines), group_size):
                    group = '\n'.join(lines[i:i + group_size])
                    paragraphs.append(group)

        # 너무 짧은 단락은 이전 단락에 합치기 (최소 20자)
        merged = []
        for p in paragraphs:
            if merged and len(p) < 20:
                merged[-1] = merged[-1] + '\n' + p
            else:
                merged.append(p)
        paragraphs = merged

        if not paragraphs:
            raise ValueError("Script is empty after parsing")

        print(f"[SCRIPT MODE] Split into {len(paragraphs)} scenes")

        # 2. Gemini로 씬별 이미지 프롬프트 생성
        scene_prompts = self._generate_image_prompts_for_script(
            paragraphs, request.genre, request.mood, request.style_preset,
            character_ethnicity=getattr(request, 'character_ethnicity', 'auto')
        )

        # 3. story_data 구성 (기존 StoryAgent 출력 포맷과 호환)
        scenes = []
        for idx, (narration, prompt_data) in enumerate(zip(paragraphs, scene_prompts), start=1):
            # TTS 길이 추정: 한국어 약 4자/초, 영어 약 12자/초
            char_count = len(narration)
            estimated_duration = max(5, char_count / 4)

            # 인종 런타임 주입 (Gemini가 지시 무시할 때 안전망)
            # Note: scene_orchestrator.generate_images_for_scenes()에도 동일 로직 있음.
            # "not in prompt" 가드로 중복 주입 방지됨.
            image_prompt = prompt_data.get("image_prompt", "")
            _eth = getattr(request, 'character_ethnicity', 'auto')
            _ETH_KW = {
                "korean": "Korean", "japanese": "Japanese", "chinese": "Chinese",
                "southeast_asian": "Southeast Asian", "european": "European",
                "black": "Black", "hispanic": "Hispanic",
            }
            _eth_kw = _ETH_KW.get(_eth, "")
            if _eth_kw and _eth_kw.lower() not in image_prompt.lower():
                image_prompt = f"{_eth_kw} characters, {image_prompt}"

            scene = {
                "scene_id": idx,
                "narration": narration,
                "visual_description": prompt_data.get("visual_description", ""),
                "image_prompt": image_prompt,
                "prompt": image_prompt,
                "mood": prompt_data.get("mood", request.mood or "dramatic"),
                "duration_sec": round(estimated_duration),
                "camera_work": prompt_data.get("camera_work", "slow_zoom_in"),
            }
            scenes.append(scene)

        # 4. 첫 문장에서 제목 추출
        first_line = paragraphs[0].split('\n')[0][:30]

        total_duration = sum(s["duration_sec"] for s in scenes)

        story_data = {
            "title": first_line,
            "genre": request.genre or "emotional",
            "total_duration_sec": total_duration,
            "character_ethnicity": getattr(request, 'character_ethnicity', 'auto'),
            "scenes": scenes,
            "global_style": {
                "art_style": request.style_preset or "cinematic, high contrast",
                "color_palette": "natural tones",
                "lighting": "cinematic lighting",
            },
        }

        print(f"[SCRIPT MODE] Generated story_data with {len(scenes)} scenes")
        return story_data

    def _generate_image_prompts_for_script(
        self,
        paragraphs: List[str],
        genre: str,
        mood: str,
        style: str,
        character_ethnicity: str = "auto"
    ) -> List[Dict[str, str]]:
        """
        Gemini를 사용하여 스크립트 단락별 이미지 프롬프트를 생성.

        Args:
            paragraphs: 씬별 내레이션 텍스트 리스트
            genre: 장르
            mood: 분위기
            style: 아트 스타일
            character_ethnicity: 캐릭터 인종 (auto, korean, japanese, ...)

        Returns:
            각 씬의 image_prompt, visual_description, mood, camera_work dict 리스트
        """
        import os

        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            # Gemini 없으면 기본 프롬프트로 폴백
            print("[SCRIPT MODE] No GOOGLE_API_KEY, using fallback prompts")
            return [
                {
                    "image_prompt": f"{style} scene depicting: {p[:80]}",
                    "visual_description": p[:100],
                    "mood": mood,
                    "camera_work": "slow_zoom_in"
                }
                for p in paragraphs
            ]

        # 씬 목록 텍스트 구성
        scenes_text = ""
        for i, p in enumerate(paragraphs, 1):
            scenes_text += f"\n--- Scene {i} ---\n{p}\n"

        prompt = f"""You are a professional storyboard artist creating image prompts for a video.

FULL SCRIPT CONTEXT:
{scenes_text}

STYLE: {style}
GENRE: {genre}
MOOD: {mood}

TASK: For each scene above, generate an image generation prompt in ENGLISH.

RULES:
- Each image prompt must be a detailed, vivid description for AI image generation
- Consider the FULL script context for visual continuity between scenes
- Translate metaphorical/abstract narration into concrete, realistic visual scenes
  (e.g., "마음이 무거웠다" → "A person sitting alone on a park bench at dusk, head bowed")
- Include specific details: lighting, composition, color palette, subject actions
- Match the {style} art style consistently across all scenes
- Camera work should vary: slow_zoom_in, slow_zoom_out, pan_left, pan_right, static
- *** ETHNICITY IS MANDATORY ***: When characters appear, ALWAYS explicitly state their ethnicity in the prompt{self._get_ethnicity_rule(character_ethnicity)}

OUTPUT FORMAT (JSON array, one object per scene):
[
  {{
    "image_prompt": "English image generation prompt, detailed and vivid, 2-3 sentences",
    "visual_description": "Brief Korean description of the visual (1 sentence)",
    "mood": "scene-specific mood keyword",
    "camera_work": "one of: slow_zoom_in, slow_zoom_out, pan_left, pan_right, static"
  }},
  ...
]

IMPORTANT: Return exactly {len(paragraphs)} objects, one for each scene. Return ONLY the JSON array."""

        try:
            from google import genai
            from google.genai import types

            client = genai.Client(api_key=api_key)

            print(f"[SCRIPT MODE] Calling Gemini for {len(paragraphs)} scene prompts...")

            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0.7,
                    response_mime_type="application/json"
                )
            )

            import json
            result = json.loads(response.text.strip())

            # 결과 수가 씬 수와 다르면 보정
            while len(result) < len(paragraphs):
                result.append({
                    "image_prompt": f"{style} scene, {mood} atmosphere",
                    "visual_description": "장면",
                    "mood": mood,
                    "camera_work": "slow_zoom_in"
                })

            print(f"[SCRIPT MODE] Generated {len(result)} image prompts")
            return result[:len(paragraphs)]

        except Exception as e:
            print(f"[SCRIPT MODE] Gemini failed: {e}, using fallback prompts")
            return [
                {
                    "image_prompt": f"{style} scene depicting: {p[:80]}",
                    "visual_description": p[:100],
                    "mood": mood,
                    "camera_work": "slow_zoom_in"
                }
                for p in paragraphs
            ]

    def generate_video_from_story(self, story_data: Dict[str, Any], request: ProjectRequest, project_id: str = None) -> Manifest:
        """Step 2~5: 확정된 스토리로 영상 생성"""
        start_time = time.time()
        
        if not project_id:
            project_id = str(uuid.uuid4())[:8]
            
        project_dir = self._create_project_structure(project_id)

        # Manifest 초기화
        from schemas import GlobalStyle, CharacterSheet

        manifest = Manifest(
            project_id=project_id,
            input=request,
            status="processing",
            title=story_data.get("title"),
            hook_text=story_data.get("hook_text"),
            script=json.dumps(story_data, ensure_ascii=False)
        )

        # v2.0: character_sheet와 global_style 저장
        if "character_sheet" in story_data:
            manifest.character_sheet = {
                token: CharacterSheet(**data)
                for token, data in story_data["character_sheet"].items()
            }
            print(f"[v2.0] Loaded {len(manifest.character_sheet)} characters from story")

        if "global_style" in story_data:
            manifest.global_style = GlobalStyle(**story_data["global_style"])
            print(f"[v2.0] Global style: {manifest.global_style.art_style}")

        # v3.0: character_voices 저장
        if "character_voices" in story_data:
            from schemas import CharacterVoice
            manifest.character_voices = [
                CharacterVoice(**cv) for cv in story_data["character_voices"]
            ]
            print(f"[v3.0] Loaded {len(manifest.character_voices)} character voices")

        # 기존 manifest에서 앵커 경로 로드 (캐스팅/이미지 단계에서 저장된 것 재사용)
        _existing_manifest_path = os.path.join(project_dir, "manifest.json")
        _existing_data = {}
        if os.path.exists(_existing_manifest_path):
            try:
                with open(_existing_manifest_path, "r", encoding="utf-8") as f:
                    _existing_data = json.load(f)
            except Exception:
                pass

        # [STEP 1.3 & 1.4] Style Anchor + Environment Anchors - 기존 것 재사용 우선
        style_anchor_path = None
        env_anchors = {}
        _saved_style = _existing_data.get("_style_anchor_path")
        _saved_style_url = _existing_data.get("_style_anchor_url")
        _saved_envs = _existing_data.get("_env_anchors", {})
        _saved_env_urls = _existing_data.get("_env_anchor_urls", {})

        _reused = False
        if _saved_style and os.path.exists(_saved_style):
            style_anchor_path = _saved_style
            env_anchors = {int(k): v for k, v in _saved_envs.items() if os.path.exists(v)}
            _reused = True
        elif _saved_style_url or _saved_style:
            _dl_url = _saved_style_url or _saved_style
            if _dl_url and _dl_url.startswith("http"):
                _local = self._download_to_local(_dl_url, project_dir, "media/anchors")
                if _local:
                    style_anchor_path = _local
                    _reused = True
            for _sc_id, _env_url in _saved_env_urls.items():
                if _env_url and _env_url.startswith("http"):
                    _local = self._download_to_local(_env_url, project_dir, "media/anchors")
                    if _local:
                        env_anchors[int(_sc_id)] = _local

        if _reused:
            print(f"\n[STEP 1.3] Reusing existing style anchor: {style_anchor_path}")
            print(f"[STEP 1.4] Reusing {len(env_anchors)} existing environment anchors")
        else:
            style_anchor_agent = StyleAnchorAgent()
            if manifest.global_style:
                print(f"\n[STEP 1.3] Generating style anchor image...")
                style_anchor_path = style_anchor_agent.generate_style_anchor(
                    global_style=manifest.global_style,
                    project_dir=project_dir
                )
            if manifest.global_style and "scenes" in story_data:
                print(f"\n[STEP 1.4] Generating environment anchor images...")
                env_anchors = style_anchor_agent.generate_environment_anchors(
                    scenes=story_data["scenes"],
                    global_style=manifest.global_style,
                    project_dir=project_dir
                )

        # [STEP 1.5] Character Casting - 기존 앵커 있으면 스킵
        if manifest.character_sheet:
            _existing_cs_data = _existing_data.get("character_sheet", {})
            # 기존 manifest에서 master_image_path 로드 (로컬 → R2 URL 폴백)
            for token, cs in manifest.character_sheet.items():
                if token in _existing_cs_data:
                    existing_path = _existing_cs_data[token].get("master_image_path")
                    if existing_path and os.path.exists(existing_path):
                        cs.master_image_path = existing_path
                    elif not (existing_path and os.path.exists(existing_path)):
                        # R2 URL에서 다운로드 시도
                        _url = _existing_cs_data[token].get("master_image_url")
                        if _url:
                            _local = self._download_to_local(_url, project_dir, "media/characters")
                            if _local:
                                cs.master_image_path = _local

            _all_have_anchors = all(
                cs.master_image_path and os.path.exists(cs.master_image_path)
                for cs in manifest.character_sheet.values()
            )

            if not _all_have_anchors:
                print(f"\n[STEP 1.5] Casting characters (generating master anchor images)...")
                character_manager = CharacterManager()
                character_images = character_manager.cast_characters(
                    character_sheet=manifest.character_sheet,
                    global_style=manifest.global_style,
                    project_dir=project_dir,
                    ethnicity=getattr(request, 'character_ethnicity', 'auto')
                )
                if "character_sheet" in story_data:
                    for token, image_path in character_images.items():
                        if token in story_data["character_sheet"]:
                            story_data["character_sheet"][token]["master_image_path"] = image_path
                    # anchor_set도 story_data에 동기화
                    for token in manifest.character_sheet:
                        cs = manifest.character_sheet[token]
                        if hasattr(cs, 'anchor_set') and cs.anchor_set and token in story_data["character_sheet"]:
                            story_data["character_sheet"][token]["anchor_set"] = cs.anchor_set.model_dump()
                for token, image_path in character_images.items():
                    if token in manifest.character_sheet:
                        cs = manifest.character_sheet[token]
                        if hasattr(cs, 'master_image_path'):
                            cs.master_image_path = image_path
                self._save_manifest(manifest, project_dir)
                print(f"  [Manifest] Saved character anchors to disk")

                # _save_manifest()는 Pydantic 모델만 직렬화 →
                # api_server가 설정한 _images_pregenerated 플래그와 씬 이미지 경로가 사라짐.
                # 재주입해서 process_story()가 기존 이미지를 재사용할 수 있도록 보장.
                _reinject = {k: _existing_data[k] for k in ("_images_pregenerated", "_style_anchor_path", "_env_anchors") if k in _existing_data}
                _ex_scenes = _existing_data.get("scenes", [])
                if _reinject or _ex_scenes:
                    try:
                        with open(_existing_manifest_path, "r", encoding="utf-8") as f:
                            _mf = json.load(f)
                        _mf.update(_reinject)
                        # scenes의 assets.image_path 복원
                        if _ex_scenes:
                            _esm = {s.get("scene_id"): s for s in _ex_scenes}
                            for s in _mf.get("scenes", []):
                                _sid = s.get("scene_id")
                                if _sid in _esm:
                                    _ex_assets = _esm[_sid].get("assets") or {}
                                    _img = _ex_assets.get("image_path") if isinstance(_ex_assets, dict) else None
                                    if _img:
                                        if not isinstance(s.get("assets"), dict):
                                            s["assets"] = {}
                                        s["assets"]["image_path"] = _img
                        with open(_existing_manifest_path, "w", encoding="utf-8") as f:
                            json.dump(_mf, f, ensure_ascii=False, indent=2)
                    except Exception:
                        pass
            else:
                print(f"\n[STEP 1.5] Reusing existing character anchors ({len(manifest.character_sheet)} characters).")
                # story_data에 master_image_path 동기화
                if "character_sheet" in story_data:
                    for token, cs in manifest.character_sheet.items():
                        _mp = cs.master_image_path if hasattr(cs, 'master_image_path') else None
                        if _mp and token in story_data["character_sheet"]:
                            story_data["character_sheet"][token]["master_image_path"] = _mp
                            if hasattr(cs, 'anchor_set') and cs.anchor_set:
                                story_data["character_sheet"][token]["anchor_set"] = cs.anchor_set.model_dump() if hasattr(cs.anchor_set, 'model_dump') else None
                            print(f"    [Sync] {token}: {_mp}")

        try:
            print(f"\n{'='*60}")
            print(f"STORYCUT Pipeline - Video Generation - Project: {project_id}")
            print(f"{'='*60}")

            # Step 2: Scene 처리
            orchestrator = SceneOrchestrator(feature_flags=request.feature_flags)

            if story_data.get("_images_pregenerated"):
                # 이미지가 이미 존재 → 이미지 생성 코드를 거치지 않는 전용 경로
                print("\n[STEP 2/6] Composing video from pre-generated images (skip image gen)...")
                final_video = orchestrator.compose_scenes_from_images(
                    story_data=story_data,
                    output_path=f"{project_dir}/final_video.mp4",
                    request=request,
                    style_anchor_path=style_anchor_path,
                    environment_anchors=env_anchors,
                )
            else:
                print("\n[STEP 2/6] Processing scenes with context carry-over...")
                final_video = orchestrator.process_story(
                    story_data=story_data,
                    output_path=f"{project_dir}/final_video.mp4",
                    request=request,
                    style_anchor_path=style_anchor_path,
                    environment_anchors=env_anchors,
                )

            # Scene 정보 업데이트
            manifest.scenes = self._convert_scenes_to_schema(story_data["scenes"])
            manifest.outputs.final_video_path = final_video

            # Step 3: 자막 생성 및 영상에 적용 (옵션)
            if request.subtitles:
                print("\n[STEP 3/6] Generating subtitles and applying to video...")
                final_video = self._generate_and_apply_subtitles(
                    manifest.scenes,
                    project_dir,
                    final_video
                )
                manifest.outputs.final_video_path = final_video

            # Step 3.5: Film Look 후처리 (v2.0)
            if request.feature_flags.film_look:
                print("\n[STEP 3.5/6] Applying film look (grain + color grading)...")
                final_video = self._apply_film_look(final_video, project_dir)
                manifest.outputs.final_video_path = final_video

            # Step 4: Optimization 패키지 생성
            if request.feature_flags.optimization_pack:
                # v2.1: Check if StoryAgent already generated optimization data
                if "youtube_opt" in story_data:
                    print("\n[STEP 4/6] Using pre-generated optimization package from StoryAgent...")
                    opt = story_data["youtube_opt"]
                    manifest.outputs.title_candidates = opt.get("title_candidates", [])
                    manifest.outputs.thumbnail_texts = [opt.get("thumbnail_text")] if opt.get("thumbnail_text") else []
                    manifest.outputs.hashtags = opt.get("hashtags", [])
                    
                    # Save as separate JSON for frontend compatibility
                    opt_package = {
                        "title_candidates": manifest.outputs.title_candidates,
                        "thumbnail_texts": manifest.outputs.thumbnail_texts,
                        "hashtags": manifest.outputs.hashtags
                    }
                    opt_path = self.optimization_agent.save_optimization_package(opt_package, project_dir, project_id)
                    manifest.outputs.metadata_json_path = opt_path
                    
                else:
                    print("\n[STEP 4/6] Generating optimization package (Legacy)...")
                    opt_package = self.optimization_agent.run(
                        topic=request.topic or manifest.title,
                        script=manifest.script,
                        scenes=manifest.scenes,
                        request=request
                    )
                
                manifest.outputs.title_candidates = opt_package.get("title_candidates", [])
                manifest.outputs.thumbnail_texts = opt_package.get("thumbnail_texts", [])
                manifest.outputs.hashtags = opt_package.get("hashtags", [])
                
                opt_path = self.optimization_agent.save_optimization_package(opt_package, project_dir, project_id)
                manifest.outputs.metadata_json_path = opt_path

            # Step 5: Manifest 저장
            print("\n[STEP 5/6] Saving manifest...")
            manifest.status = "completed"
            manifest.execution_time_sec = time.time() - start_time
            manifest.cost_estimate = self._estimate_costs(manifest)

            manifest_path = self._save_manifest(manifest, project_dir)
            
            return manifest

        except Exception as e:
            manifest.status = "failed"
            manifest.error_message = str(e)
            self._save_manifest(manifest, project_dir)
            raise

    def generate_characters_only(
        self,
        story_data: Dict[str, Any],
        request: ProjectRequest,
        project_id: str = None,
    ) -> None:
        """
        Step 2A-PRE: 캐릭터 앵커 이미지만 생성 (씬 이미지 생성 전 캐릭터 검토용).

        캐릭터 캐스팅 결과를 manifest.json에 저장하여 프론트엔드 폴링으로 확인 가능.

        Args:
            story_data: Story JSON (character_sheet, global_style 포함)
            request: ProjectRequest
            project_id: 프로젝트 ID
        """
        from schemas import GlobalStyle, CharacterSheet

        if not project_id:
            project_id = str(uuid.uuid4())[:8]

        project_dir = self._create_project_structure(project_id)

        print(f"\n{'='*60}")
        print(f"STORYCUT Pipeline - Character Casting - Project: {project_id}")
        print(f"{'='*60}\n")

        # Manifest 초기화
        manifest = Manifest(
            project_id=project_id,
            input=request,
            status="preparing",
            title=story_data.get("title"),
            hook_text=story_data.get("hook_text"),
            script=json.dumps(story_data, ensure_ascii=False)
        )

        if "character_sheet" in story_data:
            manifest.character_sheet = {
                token: CharacterSheet(**data)
                for token, data in story_data["character_sheet"].items()
            }

        if "global_style" in story_data:
            manifest.global_style = GlobalStyle(**story_data["global_style"])

        # 씬 정보도 저장 (이후 이미지 생성에서 사용)
        manifest.scenes = []
        for idx, sd in enumerate(story_data.get('scenes', []), start=1):
            scene = Scene(
                index=idx,
                scene_id=sd.get("scene_id", idx),
                sentence=sd.get("narration", ""),
                narration=sd.get("narration"),
                status="pending",
            )
            manifest.scenes.append(scene)

        # casting_status를 manifest에 직접 저장
        self._save_manifest(manifest, project_dir)
        manifest_path = os.path.join(project_dir, "manifest.json")
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest_dict = json.load(f)
        manifest_dict["casting_status"] = "casting"
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest_dict, f, ensure_ascii=False, indent=2)

        # Style Anchor 생성 + Character Casting (에러 시 casting_status=failed 보장)
        style_anchor_path = None
        env_anchors = {}

        try:
            if manifest.global_style:
                from agents.style_anchor import StyleAnchorAgent
                style_anchor_agent = StyleAnchorAgent()

                # 스타일 앵커 생성 중 메시지
                self._save_manifest(manifest, project_dir)
                with open(manifest_path, "r", encoding="utf-8") as f:
                    _md = json.load(f)
                _md["casting_status"] = "casting"
                _md["casting_message"] = "스타일 앵커 이미지 생성 중..."
                with open(manifest_path, "w", encoding="utf-8") as f:
                    json.dump(_md, f, ensure_ascii=False, indent=2)

                print(f"\n[StyleAnchor] Generating style anchor image...")
                style_anchor_path = style_anchor_agent.generate_style_anchor(
                    global_style=manifest.global_style,
                    project_dir=project_dir
                )

                # Environment Anchors 생성 — 캐스팅 단계에서 미리 생성하여 이미지 생성 대기시간 제거
                if "scenes" in story_data:
                    # manifest 상태 업데이트
                    self._save_manifest(manifest, project_dir)
                    with open(manifest_path, "r", encoding="utf-8") as f:
                        _md = json.load(f)
                    _md["casting_status"] = "casting"
                    _md["casting_message"] = f"환경 앵커 이미지 생성 중... ({len(story_data['scenes'])}장)"
                    with open(manifest_path, "w", encoding="utf-8") as f:
                        json.dump(_md, f, ensure_ascii=False, indent=2)

                    print(f"\n[EnvAnchors] Generating environment anchor images ({len(story_data['scenes'])} scenes)...")
                    env_anchors = style_anchor_agent.generate_environment_anchors(
                        scenes=story_data["scenes"],
                        global_style=manifest.global_style,
                        project_dir=project_dir
                    )
                    print(f"[EnvAnchors] Generated {len(env_anchors)} environment anchors")

            # Character Casting
            if manifest.character_sheet:
                # 캐릭터 캐스팅 시작 메시지
                with open(manifest_path, "r", encoding="utf-8") as f:
                    _md = json.load(f)
                _md["casting_message"] = f"캐릭터 앵커 이미지 생성 중... ({len(manifest.character_sheet)}명)"
                with open(manifest_path, "w", encoding="utf-8") as f:
                    json.dump(_md, f, ensure_ascii=False, indent=2)

                print(f"\n[Characters] Casting character anchor images...")
                character_manager = CharacterManager()

                def _casting_progress(done, total, char_name):
                    # manifest JSON에 직접 merge (casting_status 등 커스텀 키 보존)
                    try:
                        mp = f"{project_dir}/manifest.json"
                        with open(mp, "r", encoding="utf-8") as f:
                            _md = json.load(f)
                        _md["casting_message"] = f"캐릭터 캐스팅 중... ({done}/{total}) - {char_name}"
                        _md["status"] = "casting"
                        _md["message"] = f"캐릭터 캐스팅 중... ({done}/{total}) - {char_name}"
                        with open(mp, "w", encoding="utf-8") as f:
                            json.dump(_md, f, ensure_ascii=False, indent=2)
                    except Exception:
                        pass

                character_images = character_manager.cast_characters(
                    character_sheet=manifest.character_sheet,
                    global_style=manifest.global_style,
                    project_dir=project_dir,
                    ethnicity=getattr(request, 'character_ethnicity', 'auto'),
                    progress_callback=_casting_progress
                )

                # story_data에 master_image_path + anchor_set 반영
                if "character_sheet" in story_data:
                    for token, image_path in character_images.items():
                        if token in story_data["character_sheet"]:
                            story_data["character_sheet"][token]["master_image_path"] = image_path
                    # anchor_set도 동기화 (이미지 생성 시 포즈 선택에 필요)
                    for token in manifest.character_sheet:
                        cs = manifest.character_sheet[token]
                        if hasattr(cs, 'anchor_set') and cs.anchor_set and token in story_data["character_sheet"]:
                            story_data["character_sheet"][token]["anchor_set"] = cs.anchor_set.model_dump() if hasattr(cs.anchor_set, 'model_dump') else cs.anchor_set

                # manifest에도 반영
                for token, image_path in character_images.items():
                    if token in manifest.character_sheet:
                        cs = manifest.character_sheet[token]
                        if hasattr(cs, 'master_image_path'):
                            cs.master_image_path = image_path

        except Exception as e:
            print(f"\n[Characters] Casting FAILED: {e}", flush=True)
            import traceback
            traceback.print_exc()
            self._save_manifest(manifest, project_dir)
            with open(manifest_path, "r", encoding="utf-8") as f:
                manifest_dict = json.load(f)
            manifest_dict["casting_status"] = "failed"
            manifest_dict["casting_error"] = str(e)
            with open(manifest_path, "w", encoding="utf-8") as f:
                json.dump(manifest_dict, f, ensure_ascii=False, indent=2)
            raise

        # 최종 저장 — casting_ready 상태
        self._save_manifest(manifest, project_dir)
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest_dict = json.load(f)
        manifest_dict["casting_status"] = "casting_ready"
        # style_anchor, env_anchors 경로도 저장 (이미지 생성에서 사용)
        if style_anchor_path:
            manifest_dict["_style_anchor_path"] = style_anchor_path
        if env_anchors:
            manifest_dict["_env_anchors"] = {str(k): v for k, v in env_anchors.items()}
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest_dict, f, ensure_ascii=False, indent=2)

        print(f"\n[Characters] Casting complete. Status: casting_ready")

    def generate_images_only(
        self,
        story_data: Dict[str, Any],
        request: ProjectRequest,
        project_id: str = None,
        on_scene_complete: Any = None
    ) -> Dict[str, Any]:
        """
        Step 2A: 스토리에서 이미지만 생성 (영상 생성 전 검토용).

        사용자가 이미지를 검토한 후:
        - 재생성
        - I2V 변환
        - 최종 영상 생성 승인

        Args:
            story_data: Story JSON
            request: ProjectRequest
            project_id: 프로젝트 ID (선택사항)
            on_scene_complete: 각 씬 이미지 완료 시 콜백

        Returns:
            Dict with project_id and scenes with image URLs
        """
        start_time = time.time()
        
        if not project_id:
            project_id = str(uuid.uuid4())[:8]
            
        project_dir = self._create_project_structure(project_id)
        
        print(f"\n{'='*60}")
        print(f"STORYCUT Pipeline - Image Generation - Project: {project_id}")
        print(f"{'='*60}\n")
        
        # Manifest 초기화
        from schemas import GlobalStyle, CharacterSheet

        manifest = Manifest(
            project_id=project_id,
            input=request,
            status="preparing",  # 준비 단계
            title=story_data.get("title"),
            script=json.dumps(story_data, ensure_ascii=False)
        )

        # v2.0: character_sheet와 global_style 저장
        if "character_sheet" in story_data:
            manifest.character_sheet = {
                token: CharacterSheet(**data)
                for token, data in story_data["character_sheet"].items()
            }

        if "global_style" in story_data:
            manifest.global_style = GlobalStyle(**story_data["global_style"])

        # 기존 manifest 먼저 읽기 (덮어쓰기 전에 앵커 경로 보존)
        _existing_manifest_path = os.path.join(project_dir, "manifest.json")
        _existing_data = {}
        if os.path.exists(_existing_manifest_path):
            try:
                with open(_existing_manifest_path, "r", encoding="utf-8") as f:
                    _existing_data = json.load(f)
            except Exception:
                pass

        # 초기 manifest 즉시 저장 (프론트엔드 폴링이 바로 데이터를 받을 수 있도록)
        total_scenes = len(story_data['scenes'])
        manifest.scenes = []
        for idx, sd in enumerate(story_data['scenes'], start=1):
            scene = Scene(
                index=idx,
                scene_id=sd.get("scene_id", idx),
                sentence=sd.get("narration", ""),
                narration=sd.get("narration"),
                status="pending",
            )
            manifest.scenes.append(scene)
        self._save_manifest(manifest, project_dir)

        # _save_manifest()는 Pydantic 모델만 직렬화하므로
        # 캐스팅/이미지 생성 단계에서 저장한 비모델 필드(_style_anchor_path 등)를 재주입
        _keys_to_restore = ("_style_anchor_path", "_env_anchors", "casting_status", "_images_pregenerated")
        _anchor_extras = {k: _existing_data[k] for k in _keys_to_restore if k in _existing_data}
        if _anchor_extras or _existing_data.get("scenes"):
            try:
                with open(_existing_manifest_path, "r", encoding="utf-8") as f:
                    _mf = json.load(f)
                _mf.update(_anchor_extras)
                # 이미지 생성 단계에서 저장된 씬 image_path 복원
                # (3단계 영상생성이 2단계 이미지를 다시 만들지 않도록)
                if _existing_data.get("_images_pregenerated"):
                    _existing_scenes_by_id = {}
                    for sc in _existing_data.get("scenes", []):
                        sc_id = sc.get("scene_id") or sc.get("index")
                        if sc_id:
                            _existing_scenes_by_id[str(sc_id)] = sc
                    for _mf_scene in _mf.get("scenes", []):
                        sc_id = str(_mf_scene.get("scene_id") or _mf_scene.get("index", ""))
                        _prev = _existing_scenes_by_id.get(sc_id, {})
                        _prev_img = (_prev.get("assets") or {}).get("image_path") or _prev.get("image_path")
                        if _prev_img:
                            if "assets" not in _mf_scene or _mf_scene["assets"] is None:
                                _mf_scene["assets"] = {}
                            _mf_scene["assets"]["image_path"] = _prev_img
                with open(_existing_manifest_path, "w", encoding="utf-8") as f:
                    json.dump(_mf, f, ensure_ascii=False, indent=2)
            except Exception:
                pass

        # Style Anchor 생성 (v2.0) — 캐스팅 단계에서 이미 생성된 경우 재사용
        style_anchor_path = None
        env_anchors = {}

        _saved_style = _existing_data.get("_style_anchor_path")
        _saved_style_url = _existing_data.get("_style_anchor_url")
        _saved_envs = _existing_data.get("_env_anchors", {})
        _saved_env_urls = _existing_data.get("_env_anchor_urls", {})

        print(f"[StyleAnchor DEBUG] _saved_style={_saved_style}, exists={os.path.exists(_saved_style) if _saved_style else 'N/A'}")
        print(f"[StyleAnchor DEBUG] _saved_style_url={_saved_style_url}")
        print(f"[StyleAnchor DEBUG] casting_status={_existing_data.get('casting_status')}")

        # 로컬 파일 또는 R2 URL에서 앵커 복원
        _reused = False
        if _saved_style and os.path.exists(_saved_style):
            style_anchor_path = _saved_style
            env_anchors = {int(k): v for k, v in _saved_envs.items() if os.path.exists(v)}
            _reused = True
        elif _saved_style_url or _saved_style:
            # 로컬 파일 없음 → R2 URL에서 다운로드
            _dl_url = _saved_style_url or _saved_style
            if _dl_url and _dl_url.startswith("http"):
                _local = self._download_to_local(_dl_url, project_dir, "media/anchors")
                if _local:
                    style_anchor_path = _local
                    _reused = True
            # 환경 앵커도 URL에서 복원
            for _sc_id, _env_url in _saved_env_urls.items():
                if _env_url and _env_url.startswith("http"):
                    _local = self._download_to_local(_env_url, project_dir, "media/anchors")
                    if _local:
                        env_anchors[int(_sc_id)] = _local

        # 캐스팅 완료 상태인데 스타일 앵커 파일을 못 찾은 경우 — 재생성 없이 진행
        if not _reused and _existing_data.get("casting_status") == "casting_ready":
            print(f"[StyleAnchor] Casting was ready but anchor file missing — skipping regeneration")
            _reused = True

        if _reused:
            print(f"\n[StyleAnchor] Reusing pre-generated style anchor: {style_anchor_path}")
            print(f"[EnvAnchors] Reusing {len(env_anchors)} pre-generated environment anchors")
            # 앵커 재사용 시 preparing 단계 건너뛰고 바로 generating_images로
            manifest.status = "generating_images"
            manifest.message = "이미지 생성 준비 완료"
            self._save_manifest(manifest, project_dir)
        else:
            style_anchor_agent = StyleAnchorAgent()

            if manifest.global_style:
                manifest.status = "preparing"
                manifest.message = "스타일 앵커 이미지 생성 중..."
                self._save_manifest(manifest, project_dir)
                print(f"\n[StyleAnchor] Generating style anchor image (1장)...")
                style_anchor_path = style_anchor_agent.generate_style_anchor(
                    global_style=manifest.global_style,
                    project_dir=project_dir
                )

            # Environment Anchors - 씬별 환경 앵커 이미지 생성
            if manifest.global_style and "scenes" in story_data:
                manifest.message = f"환경 앵커 이미지 생성 중... ({len(story_data['scenes'])}장)"
                self._save_manifest(manifest, project_dir)
                print(f"\n[EnvAnchors] Generating environment anchor images...")
                env_anchors = style_anchor_agent.generate_environment_anchors(
                    scenes=story_data["scenes"],
                    global_style=manifest.global_style,
                    project_dir=project_dir
                )

        # Character Casting (v2.0) — 기존 manifest에서 anchor 경로 로드 후 스킵 여부 결정
        if manifest.character_sheet:
            _existing_cs_data = _existing_data.get("character_sheet", {})
            for token, cs in manifest.character_sheet.items():
                if token in _existing_cs_data:
                    existing_path = _existing_cs_data[token].get("master_image_path")
                    if existing_path and os.path.exists(existing_path):
                        cs.master_image_path = existing_path
                    elif not (existing_path and os.path.exists(existing_path)):
                        _url = _existing_cs_data[token].get("master_image_url")
                        if _url:
                            _local = self._download_to_local(_url, project_dir, "media/characters")
                            if _local:
                                cs.master_image_path = _local
                    # anchor_set 복원 (캐스팅에서 생성된 멀티포즈 데이터)
                    _existing_as = _existing_cs_data[token].get("anchor_set")
                    if _existing_as and isinstance(_existing_as, dict):
                        from schemas.models import AnchorSet
                        try:
                            cs.anchor_set = AnchorSet(**_existing_as)
                            print(f"    [Restore] {token}: anchor_set restored ({len(_existing_as.get('poses', {}))} poses)")
                        except Exception as _as_err:
                            print(f"    [Warning] {token}: anchor_set restore failed: {_as_err}")

        _all_have_anchors = all(
            hasattr(cs, 'master_image_path') and cs.master_image_path and os.path.exists(cs.master_image_path)
            for cs in manifest.character_sheet.values()
        ) if manifest.character_sheet else True

        if manifest.character_sheet and not _all_have_anchors:
            print(f"\n[Characters] Casting character anchor images (1 pose, 1 candidate)...")
            character_manager = CharacterManager()
            character_images = character_manager.cast_characters(
                character_sheet=manifest.character_sheet,
                global_style=manifest.global_style,
                project_dir=project_dir,
                poses=["front"],
                candidates_per_pose=1,
                ethnicity=getattr(request, 'character_ethnicity', 'auto')
            )

            # Update story_data with master_image_path + anchor_set
            if "character_sheet" in story_data:
                for token, image_path in character_images.items():
                    if token in story_data["character_sheet"]:
                        story_data["character_sheet"][token]["master_image_path"] = image_path
                # anchor_set도 동기화
                for token in manifest.character_sheet:
                    cs = manifest.character_sheet[token]
                    if hasattr(cs, 'anchor_set') and cs.anchor_set and token in story_data["character_sheet"]:
                        story_data["character_sheet"][token]["anchor_set"] = cs.anchor_set.model_dump() if hasattr(cs.anchor_set, 'model_dump') else cs.anchor_set

            # manifest에도 즉시 반영 + 디스크 저장 (중간 실패 시 anchor 경로 유실 방지)
            for token, image_path in character_images.items():
                if token in manifest.character_sheet:
                    cs = manifest.character_sheet[token]
                    if hasattr(cs, 'master_image_path'):
                        cs.master_image_path = image_path
            self._save_manifest(manifest, project_dir)
        elif manifest.character_sheet:
            print(f"\n[Characters] Anchor images already present, skipping re-cast.")

        # story_data에 master_image_path + anchor_set 항상 동기화 (조건 없이)
        if manifest.character_sheet and "character_sheet" in story_data:
            for token, cs in manifest.character_sheet.items():
                if token not in story_data["character_sheet"]:
                    continue
                _mp = cs.master_image_path if hasattr(cs, 'master_image_path') else None
                if _mp:
                    story_data["character_sheet"][token]["master_image_path"] = _mp
                # anchor_set은 master_image_path 유무와 무관하게 동기화
                if hasattr(cs, 'anchor_set') and cs.anchor_set:
                    story_data["character_sheet"][token]["anchor_set"] = cs.anchor_set.model_dump() if hasattr(cs.anchor_set, 'model_dump') else cs.anchor_set
                    print(f"    [Sync] {token}: master={_mp}, anchor_poses={len(cs.anchor_set.poses) if cs.anchor_set else 0}")
                else:
                    print(f"    [Sync] {token}: master={_mp}, anchor_set=None")

        # 준비 완료 → 이미지 생성 시작
        manifest.status = "generating_images"
        self._save_manifest(manifest, project_dir)

        try:

            # Generate ONLY images (no TTS, no video)
            print(f"\n[IMAGES ONLY] Generating images for {total_scenes} scenes...")

            orchestrator = SceneOrchestrator(feature_flags=request.feature_flags)

            # 프로그레시브 콜백: 각 씬 완료 시 manifest 업데이트
            import threading
            _manifest_lock = threading.Lock()

            def _on_scene_image_complete(scene_dict, scene_id, total):
                # scene_id(1-based)로 올바른 씬을 찾아 업데이트
                with _manifest_lock:
                    target = None
                    for s in manifest.scenes:
                        if s.scene_id == scene_id:
                            target = s
                            break
                    if target:
                        raw_path = scene_dict.get("assets", {}).get("image_path")
                        target.assets.image_path = raw_path
                        target.prompt = scene_dict.get("prompt", "")
                        target.status = scene_dict.get("status", "completed")
                        self._save_manifest(manifest, project_dir)
                # 외부 콜백도 호출
                if on_scene_complete:
                    on_scene_complete(scene_dict, scene_id, total)

            # Call a new method that generates only images
            scenes_with_images = orchestrator.generate_images_for_scenes(
                story_data=story_data,
                project_dir=project_dir,
                request=request,
                style_anchor_path=style_anchor_path,
                environment_anchors=env_anchors,
                on_scene_complete=_on_scene_image_complete
            )
            
            # Update manifest
            manifest.scenes = self._convert_scenes_to_schema(scenes_with_images)
            manifest.status = "images_ready"
            manifest.execution_time_sec = time.time() - start_time

            # Save manifest
            self._save_manifest(manifest, project_dir)

            # 앵커 경로는 Pydantic 모델 외부 필드 → 최종 manifest에 재주입
            _final_extras = {}
            if style_anchor_path:
                _final_extras["_style_anchor_path"] = style_anchor_path
            if env_anchors:
                _final_extras["_env_anchors"] = {str(k): v for k, v in env_anchors.items()}
            if _final_extras:
                try:
                    with open(_existing_manifest_path, "r", encoding="utf-8") as f:
                        _mf = json.load(f)
                    _mf.update(_final_extras)
                    with open(_existing_manifest_path, "w", encoding="utf-8") as f:
                        json.dump(_mf, f, ensure_ascii=False, indent=2)
                except Exception:
                    pass

            # Return image info for frontend
            result = {
                "project_id": project_id,
                "scenes": []
            }

            for scene in manifest.scenes:
                raw_path = scene.assets.image_path if scene.assets else None
                web_path = self._normalize_to_web_path(raw_path, project_id)
                scene_info = {
                    "scene_id": scene.scene_id,
                    "index": scene.index,
                    "narration": scene.narration or scene.sentence,
                    "image_path": web_path,
                    "prompt": scene.prompt
                }
                result["scenes"].append(scene_info)

            return result
            
        except Exception as e:
            manifest.status = "failed"
            manifest.error_message = str(e)
            self._save_manifest(manifest, project_dir)
            raise


    def _normalize_to_web_path(self, file_path: str, project_id: str) -> str:
        """
        파일시스템 경로를 웹 URL 경로로 변환.

        예: "outputs/abc123/media/images/scene_01.png" → "/media/abc123/media/images/scene_01.png"
        """
        if not file_path:
            return None
        if file_path.startswith("http") or file_path.startswith("/media/"):
            return file_path
        # 절대/상대 경로 → outputs/ 기준 상대 경로로 정규화
        normalized = file_path.replace("\\", "/")
        if "outputs/" in normalized:
            rel = normalized.split("outputs/", 1)[1]
            return f"/media/{rel}"
        return f"/media/{project_id}/{normalized}"

    def _create_project_structure(self, project_id: str) -> str:
        """
        프로젝트 디렉토리 구조 생성.

        outputs/<project_id>/
        ├── manifest.json
        ├── final_video.mp4
        ├── scenes/
        │   ├── scene_01.json
        │   ├── scene_02.json
        │   └── ...
        ├── media/
        │   ├── video/
        │   ├── audio/
        │   ├── images/
        │   └── subtitles/
        └── optimization_<project_id>.json
        """
        project_dir = f"{self.output_base_dir}/{project_id}"

        dirs = [
            project_dir,
            f"{project_dir}/scenes",
            f"{project_dir}/media/video",
            f"{project_dir}/media/audio",
            f"{project_dir}/media/images",
            f"{project_dir}/media/subtitles",
            f"{project_dir}/media/characters",  # v2.0: 캐릭터 마스터 이미지
        ]

        for d in dirs:
            os.makedirs(d, exist_ok=True)

        return project_dir

    def _generate_story(self, request: ProjectRequest) -> Dict[str, Any]:
        """
        스토리 생성.

        Args:
            request: ProjectRequest

        Returns:
            Story JSON
        """
        _platform = getattr(request, 'target_platform', None)
        _is_shorts = (_platform.value if _platform else 'youtube_long') == 'youtube_shorts'

        story_data = self.story_agent.generate_story(
            genre=request.genre or "emotional",
            mood=request.mood or "dramatic",
            style=request.style_preset or "cinematic",
            total_duration_sec=request.duration_target_sec or 60,
            user_idea=request.topic or request.user_idea,
            is_shorts=_is_shorts,
            include_dialogue=getattr(request, 'include_dialogue', False),
        )

        return story_data

    def _convert_scenes_to_schema(
        self,
        scene_dicts: List[Dict[str, Any]]
    ) -> List[Scene]:
        """
        Scene 딕셔너리를 Schema 객체로 변환 (v2.0 호환).

        Args:
            scene_dicts: Scene 딕셔너리 목록

        Returns:
            Scene 객체 목록
        """
        from schemas import SceneAssets
        scenes = []
        for idx, sd in enumerate(scene_dicts, start=1):
            # assets 복원
            assets_data = sd.get("assets", {})
            assets = SceneAssets(
                image_path=assets_data.get("image_path") if isinstance(assets_data, dict) else None,
                video_path=assets_data.get("video_path") if isinstance(assets_data, dict) else None,
                narration_path=assets_data.get("narration_path") if isinstance(assets_data, dict) else None,
            )

            # v3.0: dialogue_lines 변환
            from schemas import DialogueLine
            raw_dl = sd.get("dialogue_lines", [])
            dialogue_lines = [
                DialogueLine(**dl) if isinstance(dl, dict) else dl
                for dl in raw_dl
            ]

            scene = Scene(
                index=idx,
                scene_id=sd.get("scene_id", idx),
                sentence=sd.get("narration", ""),
                narration=sd.get("narration"),
                visual_description=sd.get("visual_description"),
                mood=sd.get("mood"),
                duration_sec=sd.get("duration_sec", 5),
                prompt=sd.get("prompt", ""),
                # v2.0 필드
                narrative=sd.get("narrative"),
                image_prompt=sd.get("image_prompt"),
                characters_in_scene=sd.get("characters_in_scene", []),
                # v3.0 필드
                dialogue_lines=dialogue_lines,
                assets=assets,
                status=sd.get("status", "pending"),
            )
            scenes.append(scene)
        return scenes

    def _generate_subtitles(
        self,
        scenes: List[Scene],
        project_dir: str
    ) -> str:
        """
        자막 파일 생성.

        Args:
            scenes: Scene 목록
            project_dir: 프로젝트 디렉토리

        Returns:
            SRT 파일 경로
        """
        composer = FFmpegComposer()

        scene_dicts = [
            {
                "narration": s.narration or s.sentence,
                "duration_sec": s.duration_sec
            }
            for s in scenes
        ]

        srt_path = f"{project_dir}/media/subtitles/full.srt"
        return composer.generate_srt_from_scenes(scene_dicts, srt_path)

    def _generate_and_apply_subtitles(
        self,
        scenes: List[Scene],
        project_dir: str,
        input_video: str
    ) -> str:
        """
        자막 파일 생성 및 영상에 적용 (Burn-in).

        Args:
            scenes: Scene 목록
            project_dir: 프로젝트 디렉토리
            input_video: 입력 영상 경로

        Returns:
            자막이 적용된 최종 영상 경로
        """
        composer = FFmpegComposer()

        # 1. 자막 파일 생성
        scene_dicts = []
        for s in scenes:
            # CRITICAL FIX: Use tts_duration_sec if available for accurate timing
            actual_duration = s.tts_duration_sec if s.tts_duration_sec else s.duration_sec
            scene_dicts.append({
                "narration": s.narration or s.sentence,
                "duration_sec": actual_duration  # Use ACTUAL TTS duration
            })

        srt_path = f"{project_dir}/media/subtitles/full.srt"
        composer.generate_srt_from_scenes(scene_dicts, srt_path)
        print(f"  Generated subtitle file: {srt_path}")

        # 2. 자막을 영상에 burn-in
        output_with_subtitles = f"{project_dir}/final_video_with_subtitles.mp4"
        try:
            subtitled_video, subtitle_success = composer.overlay_subtitles(
                input_video,
                srt_path,
                output_with_subtitles
            )
            if subtitle_success:
                print(f"  Applied subtitles to video: {subtitled_video}")
                return subtitled_video
            else:
                print(f"  [Warning] Subtitle burn-in failed (likely OOM). Using original video.")
                return input_video
        except Exception as e:
            print(f"  [Warning] Subtitle burn-in failed: {e}. Using original video.")
            return input_video

    def _estimate_costs(self, manifest: Manifest) -> CostEstimate:
        """
        비용 추정.

        Args:
            manifest: Manifest 객체

        Returns:
            CostEstimate 객체
        """
        # 대략적인 추정치
        llm_tokens = len(manifest.script or "") * 2  # 입력 + 출력
        video_seconds = sum(s.duration_sec for s in manifest.scenes)
        image_count = len([s for s in manifest.scenes if s.generation_method == "image+kenburns"])
        tts_characters = sum(len(s.narration or "") for s in manifest.scenes)

        # 비용 계산 (대략적)
        estimated_usd = (
            (llm_tokens / 1000) * 0.03 +  # GPT-4 토큰
            image_count * 0.02 +           # DALL-E 이미지
            (tts_characters / 1000) * 0.015  # TTS
        )

        # Hook 비디오 사용 시 추가 비용
        if manifest.input.feature_flags.hook_scene1_video:
            estimated_usd += 0.5  # Runway 등 비디오 API

        return CostEstimate(
            llm_tokens=llm_tokens,
            video_seconds=video_seconds,
            image_count=image_count,
            tts_characters=tts_characters,
            estimated_usd=round(estimated_usd, 2)
        )

    def _download_to_local(self, url: str, project_dir: str, subdir: str = "media/anchors") -> str:
        """URL에서 파일을 다운로드하여 로컬에 저장. 성공 시 로컬 경로, 실패 시 None."""
        try:
            import requests as req_lib
            resp = req_lib.get(url, timeout=30)
            if resp.status_code == 200:
                _dir = os.path.join(project_dir, subdir)
                os.makedirs(_dir, exist_ok=True)
                _fname = os.path.basename(url.split("?")[0])
                _local_path = os.path.join(_dir, _fname)
                with open(_local_path, "wb") as f:
                    f.write(resp.content)
                print(f"[R2->Local] Downloaded: {_fname}")
                return _local_path
        except Exception as e:
            print(f"[R2->Local] Download failed ({url}): {e}")
        return None

    def _save_manifest(self, manifest: Manifest, project_dir: str) -> str:
        """
        Manifest를 JSON으로 저장.

        Args:
            manifest: Manifest 객체
            project_dir: 프로젝트 디렉토리

        Returns:
            저장된 파일 경로
        """
        manifest_path = f"{project_dir}/manifest.json"

        # 기존 manifest에서 커스텀 키 보존 (model_dump에 포함 안 되는 키들)
        _CUSTOM_KEYS = ("_style_anchor_path", "_style_anchor_url", "_env_anchors", "_env_anchor_urls",
                        "casting_status", "casting_message", "casting_error", "_images_pregenerated")
        _preserved = {}
        if os.path.exists(manifest_path):
            try:
                with open(manifest_path, "r", encoding="utf-8") as f:
                    _old = json.load(f)
                _preserved = {k: _old[k] for k in _CUSTOM_KEYS if k in _old}
            except Exception:
                pass

        # Pydantic 모델을 JSON으로 직렬화
        manifest_dict = manifest.model_dump(mode="json")
        manifest_dict.update(_preserved)

        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest_dict, f, ensure_ascii=False, indent=2, default=str)

        # Scene별 JSON 저장
        for scene in manifest.scenes:
            scene_path = f"{project_dir}/scenes/scene_{scene.scene_id:02d}.json"
            scene_dict = scene.model_dump(mode="json")
            with open(scene_path, "w", encoding="utf-8") as f:
                json.dump(scene_dict, f, ensure_ascii=False, indent=2)

        return manifest_path

    def _save_scene_json(self, scene: Scene, project_dir: str) -> str:
        """
        개별 Scene JSON 저장.

        Args:
            scene: Scene 객체
            project_dir: 프로젝트 디렉토리

        Returns:
            저장된 파일 경로
        """
        scene_path = f"{project_dir}/scenes/scene_{scene.scene_id:02d}.json"
        scene_dict = scene.model_dump(mode="json")

        with open(scene_path, "w", encoding="utf-8") as f:
            json.dump(scene_dict, f, ensure_ascii=False, indent=2)

        return scene_path

    def _apply_film_look(
        self,
        input_video: str,
        project_dir: str,
        grain_intensity: int = 10,
        saturation: float = 1.1,
        contrast: float = 1.05
    ) -> str:
        """
        필름 룩 후처리 적용 (v2.0).

        Args:
            input_video: 입력 영상 경로
            project_dir: 프로젝트 디렉토리
            grain_intensity: 그레인 강도 (0-30)
            saturation: 채도 (1.0 = 원본)
            contrast: 대비 (1.0 = 원본)

        Returns:
            필름 룩이 적용된 영상 경로
        """
        composer = FFmpegComposer()

        output_path = f"{project_dir}/final_video_film_look.mp4"

        try:
            result = composer.apply_film_look(
                video_in=input_video,
                out_path=output_path,
                grain_intensity=grain_intensity,
                saturation=saturation,
                contrast=contrast
            )
            print(f"  Film look applied: {result}")
            return result
        except Exception as e:
            print(f"  [Warning] Film look failed: {e}. Using original video.")
            return input_video


def run_pipeline(
    topic: str = None,
    genre: str = "emotional",
    mood: str = "dramatic",
    style: str = "cinematic, high contrast",
    duration: int = 60,
    feature_flags: Dict[str, bool] = None
) -> Manifest:
    """
    파이프라인 간편 실행 함수.

    Args:
        topic: 영상 주제
        genre: 장르
        mood: 분위기
        style: 영상 스타일
        duration: 목표 영상 길이 (초)
        feature_flags: Feature flags 딕셔너리

    Returns:
        Manifest 객체
    """
    # Feature flags 설정
    ff = FeatureFlags()
    if feature_flags:
        for key, value in feature_flags.items():
            if hasattr(ff, key):
                setattr(ff, key, value)

    # ProjectRequest 생성
    request = ProjectRequest(
        topic=topic,
        genre=genre,
        mood=mood,
        style_preset=style,
        duration_target_sec=duration,
        feature_flags=ff,
    )

    # 파이프라인 실행
    pipeline = StorycutPipeline()
    return pipeline.run(request)


if __name__ == "__main__":
    # 테스트 실행
    manifest = run_pipeline(
        topic="오래된 폐병원에서 발견된 미스터리한 일기장",
        genre="mystery",
        mood="suspenseful",
        duration=60,
        feature_flags={
            "hook_scene1_video": False,  # 비용 절감을 위해 OFF
            "ffmpeg_kenburns": True,
            "context_carry_over": True,
            "optimization_pack": True,
        }
    )

    print(f"\nFinal video: {manifest.outputs.final_video_path}")
    print(f"Title candidates: {manifest.outputs.title_candidates}")
