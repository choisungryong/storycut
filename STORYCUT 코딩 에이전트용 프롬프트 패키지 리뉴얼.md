
# STORYCUT 코딩 에이전트용 프롬프트 패키지 (아키텍처 + Agent별 구현 스펙/의사코드)

본 문서는 “어제 구현해놓은 STORYCUT 서비스”를 기준으로, **조회수/수익형 유튜브 제작 머신** 관점에서 필요한 보완 사항을 반영한 **코딩 에이전트 지시서**입니다.  
목표는 다음 3가지를 **최소 변경으로 빠르게** 달성하는 것입니다.

- (P0) **Scene 1 Hook 전용 고품질 비디오 생성 강제**
- (P0) **FFmpeg: Audio Ducking + Subtitle Overlay + Ken Burns(이미지 모션)**
- (P1) **Scene Orchestrator: Context Carry-over(맥락 상속)**
- (P2) **Optimization: 제목/썸네일/AB 테스트 출력**
- (P2) **Topic Finding(선택): 트렌드 기반 주제 후보 생성(추가 에이전트)**

---

## 0) 코딩 에이전트에게 주는 최상위 지시 (Architecture Prompt)

### 역할
너는 STORYCUT의 “기존 코드베이스”를 수정하여 **에이전트 파이프라인을 고도화**하는 소프트웨어 엔지니어다.  
추측 금지. 불확실한 부분은 `[확인 필요]`로 남기고, 해당 파일/함수/라인을 찾아 근거를 확보한 뒤 수정하라.

### 핵심 원칙
1. **기존 플로우를 깨지 말 것**: 기존 API/CLI/엔드포인트 입력/출력 호환성을 유지한다.
2. **기능은 Feature Flag로 도입**: 기본값 OFF → 안정화 후 ON 전환 가능하게 한다.
3. **비용/시간 최적화**: 고비용 모델 사용은 “Scene 1 Hook”에만 강제한다.
4. **아웃풋 품질은 FFmpeg에서 결정**: 이미지 기반 장면은 Ken Burns로 영상처럼 보이게 만든다.
5. **재현 가능성**: 생성 결과물(중간 산출물 포함) 경로와 메타데이터(JSON)를 남긴다.

### 구현 범위(필수)
- Video Agent: `scene_index==1`이면 **video 모델 강제**, 그 외 **image+motion**
- Composer/FFmpeg:  
  - Audio Ducking (내레이션 존재 시 BGM 자동 감쇠/복구)  
  - Subtitle Overlay(자막 burn-in)  
  - Ken Burns Effect(이미지 줌/팬)  
- Scene Orchestrator:
  - 이전 장면 핵심 키워드(인물/장소/감정/행동)를 다음 장면 프롬프트에 상속
- Optimization Agent:
  - 유튜브 제목 3종, 썸네일 프롬프트 2종, 설명문/태그 후보, AB 테스트 메타데이터 JSON 출력

### 구현 범위(선택)
- Topic Finding Agent:
  - 트렌드/댓글/검색 기반 “주제 후보” 생성 (외부 API는 키/쿼터 고려)
  - 일단은 “플러그형 인터페이스 + 더미 구현”으로 시작 가능

### 산출물/파일
- `outputs/<project_id>/manifest.json` (필수)
  - 입력/모델/비용추정/씬별 산출물 경로/실행 시간/에러 로그 경로
- `outputs/<project_id>/scenes/<n>/scene.json` (필수)
  - scene index, script sentence, prompt, negative prompt, style tokens, assets paths

### 비기능 요구사항
- 에러는 “중단”이 아니라 “degrade”로 처리(가능하면 이미지 기반으로 폴백).
- 모든 외부 호출(모델/검색 API)은 timeout + retry + circuit breaker(간단형) 적용.
- 로깅: 단계별 시작/종료/소요시간/결과물 경로/비용추정치를 남긴다.

### 완료 기준(Definition of Done)
- (D1) Scene 1은 비디오 생성 경로를 타고, Scene 2~N은 이미지+KenBurns 경로를 탄다.
- (D2) 내레이션이 있으면 BGM이 자동 감쇠되고, 없으면 원복된다.
- (D3) 자막이 영상 위에 burn-in 된다(옵션으로 on/off).
- (D4) 씬 프롬프트에 이전 씬 핵심 키워드가 상속된다.
- (D5) Optimization Agent가 제목/썸네일/메타 JSON을 출력한다.
- (D6) 기존 실행 방식(API/CLI)이 깨지지 않는다.

---

## 1) 추천 모듈/디렉터리 구조(가이드)

> 실제 코드 구조는 먼저 검색 후 맞춰라. 아래는 “권장 배치”다.

- `agents/`
  - `topic_finding_agent.py` (optional)
  - `story_agent.py`
  - `scene_orchestrator.py`
  - `video_agent.py`
  - `image_agent.py` (있다면)
  - `optimization_agent.py`
- `composer/`
  - `ffmpeg_composer.py`
  - `ffmpeg_utils.py`
- `models/`
  - `video_providers/` (runway, pika 등)
  - `image_providers/` (sd, dalle 등)
  - `llm_client.py`
- `schemas/`
  - `manifest.py` or `manifest_schema.json`
- `config/`
  - `feature_flags.yaml`
  - `providers.yaml`

---

## 2) 공통 데이터 모델(의사코드)

```pseudo
type ProjectRequest {
  topic: string?
  style_preset: string?
  target_platform: enum(YOUTUBE_LONG, YOUTUBE_SHORTS)
  language: string = "ko"
  duration_target_sec: int?
  voice_over: bool
  bgm: bool
  subtitles: bool
  feature_flags: FeatureFlags
}

type FeatureFlags {
  hook_scene1_video: bool = false
  ffmpeg_kenburns: bool = true
  ffmpeg_audio_ducking: bool = false
  subtitle_burn_in: bool = true
  context_carry_over: bool = true
  optimization_pack: bool = true
  topic_finding: bool = false
}

type Scene {
  index: int
  sentence: string
  context_summary: string?      // 이전 씬 요약(상속용)
  entities: {characters[], location?, props[], mood?, action?}
  prompt: string
  negative_prompt: string?
  assets: {
    image_path?: string
    video_path?: string
    narration_path?: string
    subtitle_srt_path?: string
    bgm_path?: string
  }
  timing: {
    start_ms: int
    end_ms: int
    duration_ms: int
  }
}

type Manifest {
  project_id: string
  created_at: datetime
  input: ProjectRequest
  scenes: list<Scene>
  outputs: {
    final_video_path: string
    shorts_video_path?: string
    title_candidates?: string[]
    thumbnail_prompts?: string[]
    metadata_json_path?: string
  }
  cost_estimate: {
    llm_tokens: int
    video_seconds: int
    image_count: int
    estimated_usd: float
  }
  logs: {path: string}
}
```

---

## 3) Agent별 상세 구현 스펙 (의사코드)

### 3.1 (선택) TopicFindingAgent — “데이터 기반 주제 후보”

**목적**: 사용자가 주제를 안 줬거나, “조회수 터질 소재 추천” 옵션이 켜진 경우 주제 후보를 산출.

```pseudo
class TopicFindingAgent:
  def run(request: ProjectRequest) -> list<string>:
    if not request.feature_flags.topic_finding:
      return []

    # 단계 1: 후보 소스 수집 (초기에는 더미/로컬 리스트로 시작 가능)
    trends = fetch_trends_sources()           # [확인 필요] 실제 연동 여부
    comments = fetch_youtube_comments_seed()  # [확인 필요]
    queries = merge(trends, comments)

    # 단계 2: 후보 정제 (LLM)
    prompt = """
    다음 키워드/이슈 목록을 바탕으로 유튜브 조회수 잠재력이 높은 '질문형 주제' 10개를 제안하라.
    조건: 자극적이되 허위/명예훼손/불법 유도 금지. 클릭 유도형 제목으로.
    """
    candidates = llm.generate_list(prompt, queries)

    # 단계 3: 필터링(금칙어/리스크) + 중복 제거
    return sanitize_and_rank(candidates)
```

출력은 `topic_candidates`로 Manifest에 기록. 최종 선택은 사용자 또는 후속 Agent가 한다.

---

### 3.2 StoryAgent — “이탈 방지형 대본 생성”

**목적**: 문학적 완성도보다 **Retention(보유율)** 중심.
**필수**: Hook을 Scene 1에 집중시키기 위한 서두 구조를 명시.

```pseudo
class StoryAgent:
  def run(topic: string, request: ProjectRequest) -> string:
    prompt = f"""
    너는 유튜브 스크립트 작가다.
    주제: {topic}

    요구사항:
    - 첫 5~10초에 강한 Hook(반전/질문/긴장) 2문장 배치
    - 전체 길이 목표: {request.duration_target_sec or 60}초 내외
    - 문장은 장면 분할이 쉽도록 1~2문장 단위로 짧게 작성
    - 사실 단정이 어려우면 '가능성이 있다/추정된다'로 표현
    - 한국어로 작성
    출력: 순수 스크립트 텍스트
    """
    return llm.generate_text(prompt)
```

---

### 3.3 SceneOrchestrator — “문장 단위 장면화 + 맥락 상속”

**목적**: 문장→장면 매핑 + 장면 간 일관성 유지.

핵심은 **Context Carry-over**: 이전 장면의 핵심 키워드를 다음 장면 프롬프트에 “상속 필드”로 강제.

```pseudo
class SceneOrchestrator:
  def run(script_text: string, request: ProjectRequest) -> list<Scene>:
    sentences = split_into_sentences(script_text)   # 1~2문장 단위 유지
    scenes = []
    prev = null

    for idx, s in enumerate(sentences, start=1):
      scene = Scene(index=idx, sentence=s)

      if request.feature_flags.context_carry_over and prev != null:
        scene.context_summary = summarize_prev(prev)   # 인물/장소/감정/행동 요약
        inherited = extract_key_terms(prev)            # 예: ["남성 주인공", "긴장", "밤거리"]
      else:
        inherited = []

      scene.entities = llm.extract_entities(s, inherited)

      # Prompt 빌드 규칙: inherited는 반드시 포함
      scene.prompt = build_prompt(sentence=s, inherited=inherited, entities=scene.entities, style=request.style_preset)
      scene.negative_prompt = build_negative_prompt(style=request.style_preset)

      scenes.append(scene)
      prev = scene

    return scenes

def build_prompt(sentence, inherited, entities, style) -> string:
  return f"""
  [STYLE] {style or "cinematic, high contrast, webtoon-like"}
  [INHERITED CONTEXT] {", ".join(inherited) if inherited else "none"}
  [SCENE SENTENCE] {sentence}
  [ENTITIES] {entities_to_string(entities)}
  [RULES]
  - 이전 장면과 동일 인물/공간/톤을 유지한다.
  - 뜬금없는 배경/소품 변경 금지.
  - 감정은 과장하되 개연성 유지.
  """
```

---

### 3.4 VideoAgent — “Scene 1 Hook 전용 비디오 강제 + 폴백”

**목적**: 비용 폭발 방지. Scene 1만 고퀄 비디오.

```pseudo
class VideoAgent:
  def run(scene: Scene, request: ProjectRequest) -> Scene:
    if request.feature_flags.hook_scene1_video and scene.index == 1:
      # 고비용 비디오 생성 강제
      try:
        scene.assets.video_path = high_quality_video_provider.generate(
          prompt=scene.prompt,
          duration_sec=clamp(5..10),
          aspect_ratio=pick_aspect(request.target_platform),
          seed=stable_seed(scene)
        )
        return scene
      except Exception as e:
        log_warn("Scene1 video failed; fallback to image+kenburns", e)
        # 폴백: 이미지 생성 후 Ken Burns로 처리
        scene.assets.image_path = image_provider.generate(prompt=scene.prompt, negative=scene.negative_prompt)
        return scene
    else:
      # 나머지는 이미지 기반
      scene.assets.image_path = image_provider.generate(prompt=scene.prompt, negative=scene.negative_prompt)
      return scene
```

---

### 3.5 ComposerAgent(FFmpeg) — “Ken Burns + Ducking + Subtitle Overlay”

**목적**: 체감 퀄리티를 만드는 핵심 엔진.

#### 3.5.1 Ken Burns (이미지 → 영상 클립)

```pseudo
def ken_burns_clip(image_path, duration_sec, out_path):
  # scale 크게 → crop 이동/줌
  # 예: zoompan = 'z=...:x=...:y=...:d=...:s=...'
  ffmpeg(
    input=image_path,
    filter_complex=f"zoompan=...:d={duration_sec*fps}:s={width}x{height},fps={fps}",
    out=out_path
  )
```

#### 3.5.2 Subtitle Overlay (burn-in)

```pseudo
def overlay_subtitles(video_in, srt_path, out_path, style="FontName=...,FontSize=..."):
  ffmpeg(
    input=video_in,
    filter_complex=f"subtitles='{srt_path}':force_style='{style}'",
    out=out_path
  )
```

#### 3.5.3 Audio Ducking (내레이션 있을 때 BGM 감쇠)

```pseudo
def mix_with_ducking(video_in, narration_wav, bgm_wav, out_path):
  # 핵심: sidechaincompress 또는 sidechaingate 기반
  # 단순 구현: narration이 있을 때만 bgm gain을 낮추는 필터 체인
  # (정교화는 P1~P2에서)
  ffmpeg(
    inputs=[video_in, narration_wav, bgm_wav],
    filter_complex="""
      [2:a]volume=1.0[bgm];
      [1:a]volume=1.0[narr];

      # bgm이 narr에 의해 눌리도록 sidechaincompress 적용
      [bgm][narr]sidechaincompress=threshold=0.02:ratio=10:attack=20:release=200[bgm_ducked];

      # 최종 믹스
      [bgm_ducked][narr]amix=inputs=2:normalize=0[aout]
    """,
    map_video_from=video_in,
    map_audio="[aout]",
    out=out_path
  )
```

#### 3.5.4 Scene 단위 조립 → 전체 concat

```pseudo
class FFmpegComposer:
  def render_scene(scene: Scene, request: ProjectRequest) -> string:
    if scene.assets.video_path exists:
      base_video = scene.assets.video_path
    else:
      base_video = ken_burns_clip(scene.assets.image_path, duration_for(scene), temp_path())

    if request.subtitles and request.feature_flags.subtitle_burn_in:
      base_video = overlay_subtitles(base_video, scene.assets.subtitle_srt_path, temp_path())

    if request.voice_over or request.bgm:
      base_video = mix_audio(base_video, scene.assets.narration_path?, scene.assets.bgm_path?, ducking=request.feature_flags.ffmpeg_audio_ducking)

    return base_video

  def compose_all(scenes, out_path):
    clips = [render_scene(s) for s in scenes]
    ffmpeg_concat(clips, out_path)
    return out_path
```

---

### 3.6 OptimizationAgent — “제목/썸네일/AB 테스트 패키지”

**목적**: 생성 후 끝이 아니라, 게시/실험까지 지원.

```pseudo
class OptimizationAgent:
  def run(topic: string, script: string, scenes: list<Scene>, request: ProjectRequest) -> dict:
    if not request.feature_flags.optimization_pack:
      return {}

    prompt = f"""
    너는 유튜브 그로스 매니저다.
    주제: {topic}
    스크립트 요약: {summarize(script)}
    타깃: {request.target_platform}

    출력:
    1) 제목 후보 3개 (서로 스타일 다르게: 충격/질문/요약)
    2) 썸네일 문구 3개 (짧게 2~5단어)
    3) 썸네일 이미지 프롬프트 2개 (과장된 감정, 높은 대비)
    4) 해시태그 10개
    5) AB 테스트 메타(JSON): {{"titleA":..., "titleB":..., "thumbnailA":..., "thumbnailB":...}}
    """
    result = llm.generate_structured(prompt)

    return result
```

---

## 4) 실행 플로우(통합 의사코드)

```pseudo
def pipeline(request: ProjectRequest):
  project_id = new_id()
  manifest = init_manifest(project_id, request)

  # (optional) topic finding
  if request.feature_flags.topic_finding and not request.topic:
    candidates = TopicFindingAgent().run(request)
    request.topic = pick_best(candidates)  # 기본 자동 선택 or 사용자 선택 [확인 필요]
    manifest.outputs.topic_candidates = candidates

  script = StoryAgent().run(request.topic, request)
  scenes = SceneOrchestrator().run(script, request)

  for scene in scenes:
    scene = VideoAgent().run(scene, request)
    scene.assets.subtitle_srt_path = SubtitleAgent().run(scene.sentence) if request.subtitles else null
    scene.assets.narration_path = TTSAgent().run(scene.sentence) if request.voice_over else null
    scene.assets.bgm_path = BgmAgent().pick(request.topic) if request.bgm else null
    persist_scene_json(scene)

  final_video = FFmpegComposer().compose_all(scenes, out_path=f"outputs/{project_id}/final.mp4")
  manifest.outputs.final_video_path = final_video

  opt = OptimizationAgent().run(request.topic, script, scenes, request)
  persist_opt_outputs(opt, project_id)
  update_manifest(manifest, scenes, opt)

  return manifest
```

---

## 5) P0 우선순위 작업 목록(코딩 에이전트용 체크리스트)

### P0-1. Feature Flag 추가

* `hook_scene1_video`
* `ffmpeg_audio_ducking`
* `subtitle_burn_in`
* `context_carry_over`
* `optimization_pack`

### P0-2. VideoAgent 수정

* Scene 1일 때 고퀄 비디오 생성 강제
* 실패 시 이미지+KenBurns 폴백

### P0-3. FFmpeg Composer 강화

* Ken Burns: 이미지 기반 씬은 무조건 모션 처리
* Ducking: 내레이션 존재 시 자동 감쇠
* Subtitles: burn-in 처리

### P1. SceneOrchestrator 맥락 상속

* prev scene 요약/키워드 추출 → prompt에 강제 포함

### P2. OptimizationAgent 출력

* titles/thumbnail prompts/AB meta JSON

---

## 6) “검토/추가 지시” 템플릿 (코딩 에이전트가 PR에 남길 내용)

* 변경 파일 목록:
* 변경 이유:
* 기존 호환성 영향:
* Feature Flag 기본값:
* 실패 시 폴백 동작:
* 비용 영향(추정):
* 로깅/메타데이터 저장 위치:
* 남은 [확인 필요] 항목:

---

