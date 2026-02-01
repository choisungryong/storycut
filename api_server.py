"""
STORYCUT FastAPI Server

웹 UI와 Python 파이프라인을 연결하는 API 서버
WebSocket으로 실시간 진행상황 전달
"""

import os
import sys
import json
import asyncio

from dotenv import load_dotenv
from pathlib import Path

# .env 파일 로드 (절대 경로 강제)
env_path = Path(__file__).parent / ".env"
print(f"DEBUG: Loading .env from {env_path}")
load_dotenv(dotenv_path=env_path)

# [Fallback] R2 키가 로드되지 않았으면 수동 파싱 시도 (Encoding/Format 문제 대응)
if not os.getenv("R2_ACCOUNT_ID") and env_path.exists():
    print("DEBUG: load_dotenv failed for R2 keys. Attempting manual parsing...")
    try:
        # utf-8-sig로 BOM 처리
        content = env_path.read_text(encoding="utf-8-sig")
        for line in content.splitlines():
            line = line.strip()
            if not line or line.startswith("#"): continue
            if "=" in line:
                key, val = line.split("=", 1)
                key = key.strip()
                val = val.strip()
                # 따옴표 제거
                if (val.startswith('"') and val.endswith('"')) or (val.startswith("'") and val.endswith("'")):
                    val = val[1:-1]
                
                if key.startswith("R2_"):
                    os.environ[key] = val
                    print(f"DEBUG: Manually loaded {key}")
    except Exception as e:
        print(f"DEBUG: Manual parsing failed: {e}")

# 최종 확인
print(f"DEBUG: R2_ACCOUNT_ID Check: {'Set' if os.getenv('R2_ACCOUNT_ID') else 'Unset'}")

api_key = os.getenv("OPENAI_API_KEY")
print(f"DEBUG: Loaded OPENAI_API_KEY: {api_key[:10]}..." if api_key else "DEBUG: OPENAI_API_KEY is None")

google_api_key = os.getenv("GOOGLE_API_KEY")
print(f"DEBUG: Loaded GOOGLE_API_KEY: {google_api_key[:10]}..." if google_api_key else "DEBUG: GOOGLE_API_KEY is None")

from typing import Optional, Dict, Any, List
from datetime import datetime
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, BackgroundTasks
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# STORYCUT 모듈 import
sys.path.append(str(Path(__file__).parent))
from schemas import FeatureFlags, ProjectRequest, TargetPlatform, GenerateVideoRequest
from pipeline import StorycutPipeline
from utils.storage import StorageManager

# R2 Storage Manager
storage_manager = StorageManager()

# FastAPI 앱 생성
app = FastAPI(title="STORYCUT API", version="2.0")

# CORS 설정
# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:8000",
        "http://127.0.0.1:8000",
        "http://localhost:3000",
        "https://storycut.pages.dev",
        "https://storycut-web.pages.dev",
        "https://storycut-worker.twinspa0713.workers.dev",
        "https://web-production-bb6bf.up.railway.app",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Optional: Add a broad CORS handler for non-credentialed requests if needed
# But for now, let's stick to the specific list which is safer with allow_credentials=True

# Global Exception Handler to ensure CORS headers on 500 errors
from fastapi import Request
from fastapi.responses import JSONResponse
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    print(f"[CRITICAL ERROR] Global exception caught: {exc}")
    import traceback
    traceback.print_exc()
    return JSONResponse(
        status_code=500,
        content={"detail": str(exc), "traceback": traceback.format_exc()},
        headers={
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "*",
            "Access-Control-Allow-Headers": "*",
        }
    )

# 정적 파일 서빙
app.mount("/static", StaticFiles(directory="web/templates/static"), name="static")

# outputs 디렉토리 없으면 생성 (서버 시작 시 에러 방지)
os.makedirs("outputs", exist_ok=True)
app.mount("/media", StaticFiles(directory="outputs"), name="media")  # 이미지/영상 서빙용

# 활성 WebSocket 연결 관리
active_connections: Dict[str, WebSocket] = {}
# 프로젝트별 메시지 히스토리 (재접속/새로고침 시 상태 복구용)
project_event_history: Dict[str, List[Dict[str, Any]]] = {}


def run_pipeline_wrapper(pipeline: 'TrackedPipeline', request: 'ProjectRequest'):
    """
    BackgroundTasks용 wrapper 함수.

    새로운 스레드에서 새로운 이벤트 루프를 생성하여 실행합니다.
    """
    import threading
    import requests
    import sys

    print(f"[DEBUG] run_pipeline_wrapper called", flush=True)
    sys.stdout.flush()

    def run_in_thread():
        print(f"[DEBUG] Thread started", flush=True)
        sys.stdout.flush()
        try:
            # 새 스레드에서는 새 이벤트 루프 생성 가능
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            print(f"[DEBUG] Event loop created", flush=True)
            sys.stdout.flush()
            try:
                print(f"[DEBUG] Starting pipeline.run_async()", flush=True)
                sys.stdout.flush()
                manifest = loop.run_until_complete(pipeline.run_async(request))

                # Webhook 호출 (requests 사용 - threading 환경에서 안정적)
                if pipeline.webhook_url:
                    try:
                        print(f"Calling webhook: {pipeline.webhook_url}")
                        response = requests.post(
                            pipeline.webhook_url,
                            json={
                                "status": "completed",
                                "output_url": manifest.outputs.final_video_path,
                            },
                            timeout=10
                        )
                        print(f"Webhook response: {response.status_code}")
                    except Exception as webhook_err:
                        print(f"Webhook call failed: {webhook_err}")
            finally:
                loop.close()
        except Exception as e:
            # 실패 시 webhook 호출
            if pipeline.webhook_url:
                try:
                    print(f"Calling webhook (failure): {pipeline.webhook_url}")
                    requests.post(
                        pipeline.webhook_url,
                        json={
                            "status": "failed",
                            "error": str(e)
                        },
                        timeout=10
                    )
                except Exception as webhook_err:
                    print(f"Webhook call failed: {webhook_err}")

            print(f"Pipeline execution error: {e}")
            import traceback
            traceback.print_exc()

    thread = threading.Thread(target=run_in_thread, daemon=True)
    print(f"[DEBUG] Starting thread", flush=True)
    sys.stdout.flush()
    thread.start()
    print(f"[DEBUG] Thread started: {thread.is_alive()}", flush=True)
    sys.stdout.flush()


def run_video_pipeline_wrapper(pipeline: 'TrackedPipeline', story_data: Dict[str, Any], request: 'ProjectRequest'):
    """
    Step 2용 wrapper (스토리 확정 후 영상 생성).
    """
    import threading
    import requests
    import sys
    import asyncio

    print(f"[DEBUG] run_video_pipeline_wrapper called", flush=True)
    sys.stdout.flush()

    def run_in_thread():
        print(f"[DEBUG] Video Thread started", flush=True)
        sys.stdout.flush()
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                loop.run_until_complete(pipeline.run_video_only_async(story_data, request))
                
                # Webhook logic if needed (can be added here)
                
            finally:
                loop.close()
        except Exception as e:
            print(f"Video Pipeline execution error: {e}")
            import traceback
            traceback.print_exc()

    thread = threading.Thread(target=run_in_thread, daemon=True)
    thread.start()


# ============================================================================
# Pydantic 모델
# ============================================================================

class GenerateRequest(BaseModel):
    """영상 생성 요청"""
    topic: Optional[str] = None
    genre: str = "emotional"
    mood: str = "dramatic"
    style: str = "cinematic, high contrast"
    voice: str = "voice_brian"  # Default voice
    duration: int = 60
    platform: str = "youtube_long"

    # Feature Flags
    hook_scene1_video: bool = False
    ffmpeg_kenburns: bool = True
    ffmpeg_audio_ducking: bool = False
    subtitle_burn_in: bool = True
    context_carry_over: bool = True
    optimization_pack: bool = True

    # Worker에서 전달받는 필드
    project_id: Optional[str] = None
    webhook_url: Optional[str] = None


class LoginRequest(BaseModel):
    email: str
    password: str


class RegisterRequest(BaseModel):
    username: str
    email: str
    password: str


# ============================================================================
# WebSocket 진행상황 전달
# ============================================================================

async def send_progress(project_id: str, step: str, progress: int, message: str, data: Dict = None):
    """
    WebSocket으로 진행상황 전송

    Args:
        project_id: 프로젝트 ID
        step: 현재 단계 (story, scene_1, scene_2, ..., compose, optimize)
        progress: 진행률 (0-100)
        message: 상태 메시지
        data: 추가 데이터
    """
    print(f"[DEBUG] send_progress called: {project_id} - {step} - {progress}% - {message}", flush=True)

    payload = {
        "type": "progress",
        "step": step,
        "progress": progress,
        "message": message,
        "data": data or {},
        "timestamp": datetime.now().isoformat()
    }

    # 히스토리 저장
    if project_id not in project_event_history:
        project_event_history[project_id] = []
    project_event_history[project_id].append(payload)
    print(f"[DEBUG] History saved for {project_id}", flush=True)

    if project_id in active_connections:
        print(f"[DEBUG] Found active WS connection for {project_id}", flush=True)
        ws = active_connections[project_id]
        try:
            await ws.send_json(payload)
            print(f"[DEBUG] WS message sent", flush=True)
        except Exception as e:
            print(f"[DEBUG] WS send failed: {e}", flush=True)
            print(f"WebSocket send error: {e}")
            del active_connections[project_id]


class ProgressTracker:
    """진행상황 추적 헬퍼"""

    def __init__(self, project_id: str, total_scenes: int = 10):
        self.project_id = project_id
        self.total_scenes = total_scenes
        self.current_step = ""
        self.current_progress = 0

    async def update(self, step: str, progress: int, message: str, data: Dict = None):
        """진행상황 업데이트"""
        self.current_step = step
        self.current_progress = progress
        await send_progress(self.project_id, step, progress, message, data)

    async def story_start(self):
        await self.update("story", 5, "스토리 생성 중...")

    async def story_complete(self, title: str, scene_count: int, story_data: Dict = None):
        self.total_scenes = scene_count
        await self.update("story", 15, f"스토리 생성 완료: {title}", {
            "title": title,
            "scene_count": scene_count,
            "story_data": story_data
        })

    async def scene_start(self, scene_id: int):
        progress = 15 + int((scene_id / self.total_scenes) * 60)
        await self.update(f"scene_{scene_id}", progress, f"Scene {scene_id}/{self.total_scenes} 처리 중...")

    async def scene_complete(self, scene_id: int, method: str, image_url: str = None):
        progress = 15 + int((scene_id / self.total_scenes) * 60)
        await self.update(f"scene_{scene_id}", progress, f"Scene {scene_id} 완료", {
            "scene_id": scene_id,
            "method": method,
            "image_url": image_url
        })

    async def compose_start(self):
        await self.update("compose", 80, "영상 합성 중...")

    async def compose_complete(self):
        await self.update("compose", 90, "영상 합성 완료")

    async def optimize_start(self):
        await self.update("optimize", 92, "최적화 패키지 생성 중...")

    async def optimize_complete(self):
        await self.update("optimize", 95, "최적화 완료")

    async def complete(self, manifest):
        await self.update("complete", 100, "영상 생성 완료!", {
            "project_id": manifest.project_id,
            "video_path": manifest.outputs.final_video_path,
            "title_candidates": manifest.outputs.title_candidates,
            "thumbnail_texts": manifest.outputs.thumbnail_texts,
            "hashtags": manifest.outputs.hashtags[:5],
        })


# ============================================================================
# 커스텀 Pipeline (진행상황 추적 버전)
# ============================================================================

class TrackedPipeline(StorycutPipeline):
    """진행상황 추적이 가능한 Pipeline"""

    def __init__(self, tracker: ProgressTracker, webhook_url: Optional[str] = None):
        super().__init__()
        self.tracker = tracker
        self.webhook_url = webhook_url

    async def run_async(self, request: ProjectRequest):
        """비동기 실행 (진행상황 전송)"""
        import time

        start_time = time.time()

        # 프로젝트 ID 및 디렉토리 생성
        project_id = self.tracker.project_id
        project_dir = self._create_project_structure(project_id)

        from schemas import Manifest
        manifest = Manifest(
            project_id=project_id,
            input=request,
            status="processing",
        )

        try:
            # Step 1: 스토리 생성
            print(f"[DEBUG] Calling tracker.story_start()", flush=True)
            await self.tracker.story_start()
            print(f"[DEBUG] tracker.story_start() completed", flush=True)
            print(f"[DEBUG] Calling _generate_story()", flush=True)
            story_data = self._generate_story(request)
            print(f"[DEBUG] _generate_story() completed", flush=True)
            manifest.title = story_data.get("title")
            manifest.script = json.dumps(story_data, ensure_ascii=False)
            await self.tracker.story_complete(story_data["title"], len(story_data["scenes"]), story_data)

            # Step 2: Scene 처리
            from agents import SceneOrchestrator
            
            # Inject image_model into feature_flags for VideoAgent/ImageAgent
            if hasattr(request, 'image_model') and request.image_model:
                setattr(request.feature_flags, 'image_model', request.image_model)
                
            orchestrator = SceneOrchestrator(feature_flags=request.feature_flags)

            # Scene별 처리 (진행상황 전송) - Orchestrator 내부에서 콜백으로 처리되므로 제거
            # loop.run_in_executor로 대체됨
            pass

            # Orchestrator 실행 (별도 스레드에서 실행하여 이벤트 루프 차단 방지)
            loop = asyncio.get_running_loop()
            
            def on_scene_progress(scene, index):
                # Scene 완료 콜백 (동기 함수 -> 비동기 스케줄링)
                method = scene.generation_method or "unknown"
                # 이미지 경로를 URL로 변환
                # 예: "outputs/1234/media/images/scene_01.png" -> "/media/1234/media/images/scene_01.png"
                image_url = None
                if scene.assets.image_path:
                    # 절대 경로일 수도 있고 상대 경로일 수도 있음
                    # outputs_base_dir가 "outputs"라고 가정
                    rel_path = os.path.relpath(scene.assets.image_path, start=os.getcwd())
                    # rel_path가 "outputs\1234\..." 형태라면
                    rel_path = rel_path.replace("\\", "/")
                    if rel_path.startswith("outputs/"):
                        image_url = f"/media/{rel_path[len('outputs/'):]}"
                
                asyncio.run_coroutine_threadsafe(
                    self.tracker.scene_complete(scene.scene_id, method, image_url),
                    loop
                )

            final_video = await loop.run_in_executor(
                None,
                lambda: orchestrator.process_story(
                    story_data=story_data,
                    output_path=f"{project_dir}/final_video.mp4",
                    request=request,
                    progress_callback=on_scene_progress
                )
            )

            # [R2 Upload]
            if final_video and os.path.exists(final_video) and storage_manager.s3_client:
                r2_path = f"videos/{project_id}/final_video.mp4"
                print(f"[R2] Uploading final video to {r2_path}...", flush=True)
                success = await loop.run_in_executor(
                    None, 
                    lambda: storage_manager.upload_file(final_video, r2_path)
                )
                if success:
                    print(f"[R2] Upload complete.", flush=True)
                else:
                    print(f"[R2] Upload failed.", flush=True)

            manifest.scenes = self._convert_scenes_to_schema(story_data["scenes"])
            manifest.outputs.final_video_path = final_video

            await self.tracker.compose_complete()

            # Step 3: Optimization
            if request.feature_flags.optimization_pack:
                await self.tracker.optimize_start()
                opt_package = self.optimization_agent.run(
                    topic=request.topic or manifest.title,
                    script=manifest.script,
                    scenes=manifest.scenes,
                    request=request
                )

                manifest.outputs.title_candidates = opt_package.get("title_candidates", [])
                manifest.outputs.thumbnail_prompts = opt_package.get("thumbnail_prompts", [])
                manifest.outputs.thumbnail_texts = opt_package.get("thumbnail_texts", [])
                manifest.outputs.hashtags = opt_package.get("hashtags", [])
                manifest.outputs.description = opt_package.get("description")
                manifest.outputs.ab_test_meta = opt_package.get("ab_test_meta")

                opt_path = self.optimization_agent.save_optimization_package(
                    opt_package, project_dir, project_id
                )
                manifest.outputs.metadata_json_path = opt_path

                await self.tracker.optimize_complete()

            # 완료
            manifest.status = "completed"
            manifest.execution_time_sec = time.time() - start_time
            manifest.cost_estimate = self._estimate_costs(manifest)

            manifest_path = self._save_manifest(manifest, project_dir)

            await self.tracker.complete(manifest)

            # Webhook은 run_pipeline_wrapper에서 requests로 호출 (threading 안전)
            print(f"[DEBUG] Pipeline completed, returning manifest", flush=True)
            return manifest

        except Exception as e:
            manifest.status = "failed"
            manifest.error_message = str(e)
            await self.tracker.update("error", 0, f"오류 발생: {str(e)}")

            # Webhook은 run_pipeline_wrapper에서 처리
            print(f"[DEBUG] Pipeline failed: {e}", flush=True)
            # Webhook은 run_pipeline_wrapper에서 처리
            print(f"[DEBUG] Pipeline failed: {e}", flush=True)
            raise

    async def run_video_only_async(self, story_data: Dict[str, Any], request: ProjectRequest):
        """Step 2 Only: 스토리 기반 영상 생성 (Async)"""
        import time
        start_time = time.time()
        project_id = self.tracker.project_id
        project_dir = self._create_project_structure(project_id)
        
        from schemas import Manifest
        manifest = Manifest(
            project_id=project_id,
            input=request,
            status="processing",
            title=story_data.get("title"),
            script=json.dumps(story_data, ensure_ascii=False)
        )
        
        try:
            # 진행상황 초기화 (Story complete = 20% of total)
            await self.tracker.update("story", 20, "스토리 확정됨 - 영상 생성 시작", {
                "title": story_data["title"],
                "scene_count": len(story_data["scenes"])
            })
            
            # Step 2: Scene 처리
            from agents import SceneOrchestrator
            orchestrator = SceneOrchestrator(feature_flags=request.feature_flags)

            loop = asyncio.get_running_loop()
            
            def on_scene_progress(scene, index):
                method = scene.generation_method or "unknown"
                image_url = None
                if scene.assets.image_path:
                    rel_path = os.path.relpath(scene.assets.image_path, start=os.getcwd())
                    rel_path = rel_path.replace("\\", "/")
                    if rel_path.startswith("outputs/"):
                        image_url = f"/media/{rel_path[len('outputs/'):]}"
                
                asyncio.run_coroutine_threadsafe(
                    self.tracker.scene_complete(scene.scene_id, method, image_url),
                    loop
                )

            final_video = await loop.run_in_executor(
                None,
                lambda: orchestrator.process_story(
                    story_data=story_data,
                    output_path=f"{project_dir}/final_video.mp4",
                    request=request,
                    progress_callback=on_scene_progress
                )
            )

            manifest.scenes = self._convert_scenes_to_schema(story_data["scenes"])
            manifest.outputs.final_video_path = final_video

            await self.tracker.compose_complete()

            # Step 3: Optimization
            if request.feature_flags.optimization_pack:
                await self.tracker.optimize_start()
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
                
                await self.tracker.optimize_complete()

            # 완료
            manifest.status = "completed"
            manifest.execution_time_sec = time.time() - start_time
            manifest.cost_estimate = self._estimate_costs(manifest)
            manifest_path = self._save_manifest(manifest, project_dir)
            
            await self.tracker.complete(manifest)
            return manifest

        except Exception as e:
            manifest.status = "failed"
            manifest.error_message = str(e)
            await self.tracker.update("error", 0, f"오류 발생: {str(e)}")
            print(f"[DEBUG] Video Pipeline failed: {e}", flush=True)
            raise

# ============================================================================
# API 엔드포인트
# ============================================================================

@app.get("/", response_class=HTMLResponse)
async def read_root():
    """메인 페이지"""
    html_path = Path("web/templates/index.html")
    if html_path.exists():
        return HTMLResponse(content=html_path.read_text(encoding="utf-8"))
    else:
        return HTMLResponse(content="""
        <!DOCTYPE html>
        <html>
        <head><title>STORYCUT</title></head>
        <body>
            <h1>STORYCUT v2.0</h1>
            <p>웹 UI를 로드할 수 없습니다. web/templates/index.html 파일을 확인하세요.</p>
        </body>
        </html>
        """)


@app.get("/login.html", response_class=HTMLResponse)
async def login_page():
    """로그인 페이지"""
    return HTMLResponse(content=Path("web/templates/login.html").read_text(encoding="utf-8"))


@app.get("/signup.html", response_class=HTMLResponse)
async def signup_page():
    """회원가입 페이지"""
    return HTMLResponse(content=Path("web/templates/signup.html").read_text(encoding="utf-8"))


@app.websocket("/ws/{project_id}")
async def websocket_endpoint(websocket: WebSocket, project_id: str):
    """WebSocket 연결 (실시간 진행상황)"""
    await websocket.accept()
    active_connections[project_id] = websocket

    # 1. 접속 시 지난 히스토리 모두 전송 (상태 복구)
    if project_id in project_event_history:
        print(f"DEBUG: Replaying {len(project_event_history[project_id])} events for {project_id}")
        for event in project_event_history[project_id]:
            try:
                await websocket.send_json(event)
            except Exception as e:
                print(f"Error replaying event: {e}")
                break

    try:
        while True:
            # 클라이언트로부터 메시지 수신 (연결 유지용)
            data = await websocket.receive_text()
            if data == "ping":
                await websocket.send_json({"type": "pong"})
    except WebSocketDisconnect:
        # 연결이 끊겨도 히스토리는 유지 (재접속 가능성)
        if project_id in active_connections:
            del active_connections[project_id]


@app.get("/api/manifest/{project_id}")
async def get_manifest(project_id: str):
    """프로젝트 Manifest 조회"""
    manifest_path = Path(f"outputs/{project_id}/manifest.json")
    if not manifest_path.exists():
        raise HTTPException(status_code=404, detail="Manifest not found")
    
    try:
        content = manifest_path.read_text(encoding="utf-8")
        return json.loads(content)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to read manifest: {str(e)}")


@app.post("/api/generate/story")
async def generate_story(req: GenerateRequest):
    """Step 1: 스토리만 생성"""
    import uuid
    # 1. Request 변환
    project_id = req.project_id or str(uuid.uuid4())[:8]
    
    feature_flags = FeatureFlags(
        hook_scene1_video=req.hook_scene1_video,
        ffmpeg_kenburns=req.ffmpeg_kenburns,
        ffmpeg_audio_ducking=req.ffmpeg_audio_ducking,
        subtitle_burn_in=req.subtitle_burn_in,
        context_carry_over=req.context_carry_over,
        optimization_pack=req.optimization_pack,
    )
    
    platform = TargetPlatform.YOUTUBE_SHORTS if req.platform == "youtube_shorts" else TargetPlatform.YOUTUBE_LONG
    
    project_request = ProjectRequest(
        topic=req.topic,
        genre=req.genre,
        mood=req.mood,
        style_preset=req.style,
        duration_target_sec=req.duration,
        target_platform=platform,
        voice_id=req.voice,  # Pass voice selection
        voice_over=True,
        bgm=True,
        subtitles=req.subtitle_burn_in,
        feature_flags=feature_flags,
    )
    
    # 2. Pipeline으로 스토리 생성 (Synchronous for Step 1)
    # Step 1은 비교적 빠르므로 동기 실행 하거나, 길어지면 async로 변경 고려
    # 여기서는 동기로 처리 (사용자 피드백 즉시 필요)
    
    print(f"Generating story for project {project_id}...")
    # 임시 Tracker (No websocket needed just yet, or maybe yes?)
    # Story generation relies on LLM, might take 10-20s.
    
    # TrackedPipeline needs a tracker. 
    # We can use a dummy tracker or the real one but we won't broadcast nicely in Sync call unless we use background tasks.
    # But checking app.js, it waits for response. So synchronous is OK if timeout is long enough.
    
    # We will use TrackedPipeline but call internal method directly.
    # TrackedPipeline init requires tracker.
    tracker = ProgressTracker(project_id)
    pipeline = TrackedPipeline(tracker)
    
    try:
        from starlette.concurrency import run_in_threadpool
        story_data = await run_in_threadpool(pipeline.generate_story_only, project_request)
        return {
            "story_data": story_data,
            "request_params": project_request.dict(), # Pydantic v1/v2 compatibility
            "project_id": project_id
        }
    except Exception as e:
        print(f"Story generation failed: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/generate/video")
async def generate_video_from_story(req: GenerateVideoRequest, background_tasks: BackgroundTasks):
    """Step 2: 확정된 스토리로 영상 생성 시작"""
    import uuid
    from pathlib import Path
    import threading

    project_id = req.project_id or str(uuid.uuid4())[:8]

    # Project 디렉토리 생성
    project_dir = f"outputs/{project_id}"
    Path(project_dir).mkdir(parents=True, exist_ok=True)

    # 초기 manifest 생성 (processing 상태) - 매우 중요!
    initial_manifest = {
        "project_id": project_id,
        "status": "processing",
        "progress": 5,
        "message": "영상 생성 준비 중...",
        "created_at": datetime.now().isoformat(),
        "title": req.story_data.get("title", "제목 없음"),
        "input": {},
        "outputs": {},
        "error_message": None
    }

    manifest_path = f"{project_dir}/manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(initial_manifest, f, ensure_ascii=False, indent=2)

    print(f"\n[API] ========== PROJECT INITIALIZED ==========", flush=True)
    print(f"[API] Project ID: {project_id}", flush=True)
    print(f"[API] Manifest path: {manifest_path}", flush=True)
    print(f"[API] Initial manifest status: {initial_manifest['status']}", flush=True)
    print(f"[API] ==========================================\n", flush=True)

    # 비동기 작업 시작
    tracker = ProgressTracker(project_id, total_scenes=len(req.story_data.get("scenes", [])))
    pipeline = TrackedPipeline(tracker)

    # 별도 스레드에서 실행 (BackgroundTask 대신 Threading 사용)
    def run_pipeline_thread():
        print(f"[THREAD] Starting pipeline thread for project: {project_id}", flush=True)
        try:
            run_video_pipeline_wrapper(pipeline, req.story_data, req.request_params)
            print(f"[THREAD] Pipeline thread completed for project: {project_id}", flush=True)
        except Exception as e:
            print(f"[THREAD] Pipeline thread error: {str(e)}", flush=True)
            import traceback
            print(f"[THREAD] {traceback.format_exc()}", flush=True)

    # 스레드 실행
    thread = threading.Thread(target=run_pipeline_thread, daemon=False)
    thread.start()

    print(f"[API] Background thread started, returning response", flush=True)

    return {
        "project_id": project_id,
        "status": "started",
        "message": "영상 생성이 시작되었습니다.",
        "ws_url": f"/ws/{project_id}",
        "manifest_path": manifest_path
    }

def run_video_pipeline_wrapper(pipeline: 'TrackedPipeline', story_data: Dict, request_params: Dict):
    """영상 생성 전용 Wrapper - 별도 스레드에서 실행"""
    import asyncio
    import traceback

    project_id = pipeline.tracker.project_id

    print(f"\n[WRAPPER] ========== PIPELINE STARTED ==========", flush=True)
    print(f"[WRAPPER] Project ID: {project_id}", flush=True)
    print(f"[WRAPPER] Story title: {story_data.get('title')}", flush=True)
    print(f"[WRAPPER] Scene count: {len(story_data.get('scenes', []))}", flush=True)
    print(f"[WRAPPER] Request params type: {type(request_params)}", flush=True)

    # 새 이벤트 루프에서 실행
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    try:
        # request_params가 딕셔너리일 수 있으므로 ProjectRequest로 변환
        if isinstance(request_params, dict):
            print(f"[WRAPPER] Converting request_params dict to ProjectRequest", flush=True)
            from schemas import ProjectRequest, FeatureFlags

            # Feature flags 추출
            feature_flags = FeatureFlags(
                hook_scene1_video=request_params.get('hook_scene1_video', False),
                ffmpeg_kenburns=request_params.get('ffmpeg_kenburns', True),
                ffmpeg_audio_ducking=request_params.get('ffmpeg_audio_ducking', False),
                subtitle_burn_in=request_params.get('subtitle_burn_in', True),
                context_carry_over=request_params.get('context_carry_over', True),
                optimization_pack=request_params.get('optimization_pack', True),
            )

            # ProjectRequest 생성
            request = ProjectRequest(
                topic=request_params.get('topic'),
                genre=request_params.get('genre', 'emotional'),
                mood=request_params.get('mood', 'dramatic'),
                style_preset=request_params.get('style_preset') or request_params.get('style', 'cinematic'),
                duration_target_sec=request_params.get('duration_target_sec') or request_params.get('duration', 60),
                voice_id=request_params.get('voice_id') or request_params.get('voice', 'onyx'),
                voice_over=request_params.get('voice_over', True),
                bgm=request_params.get('bgm', True),
                # subtitle_burn_in is what frontend sends. Map it to subtitles.
                subtitles=request_params.get('subtitle_burn_in', True),
                feature_flags=feature_flags
            )
            print(f"[WRAPPER] ProjectRequest created successfully", flush=True)
        else:
            print(f"[WRAPPER] Request is already ProjectRequest", flush=True)
            request = request_params

        print(f"[WRAPPER] Calling run_video_only_async...", flush=True)
        print(f"[WRAPPER] =========================================\n", flush=True)

        # 비동기 실행
        manifest = loop.run_until_complete(pipeline.run_video_only_async(story_data, request))

        print(f"\n[WRAPPER] =========================================", flush=True)
        print(f"[WRAPPER] ✅ VIDEO GENERATION COMPLETED", flush=True)
        print(f"[WRAPPER] Project ID: {project_id}", flush=True)
        print(f"[WRAPPER] Final video path: {manifest.outputs.final_video_path}", flush=True)
        print(f"[WRAPPER] =========================================\n", flush=True)

        # [Production] R2 업로드 (배포 환경 지원)
        try:
            print(f"[WRAPPER] Starting R2 Upload...", flush=True)
            from utils.storage import StorageManager
            import os
            import json
            
            storage = StorageManager()
            local_video_path = manifest.outputs.final_video_path
            
            if local_video_path and os.path.exists(local_video_path):
                r2_key = f"videos/{project_id}/final_video.mp4"
                
                if storage.upload_file(local_video_path, r2_key):
                    # Backend URL (Worker 대신 백엔드가 직접 처리)
                    backend_url = "https://web-production-bb6bf.up.railway.app"
                    public_url = f"{backend_url}/api/video/{project_id}"
                    
                    print(f"[WRAPPER] R2 Upload Success! Public URL: {public_url}", flush=True)
                    
                    # Manifest 업데이트 (frontend가 이 URL을 보게 됨)
                    manifest_path = f"outputs/{project_id}/manifest.json"
                    
                    # [비동기] 씬(Scene)별 에셋 업로드 (이미지, 오디오, 조각 영상)
                    try:
                        print(f"[WRAPPER] Uploading scene assets...", flush=True)
                        if manifest.scenes:
                            for idx, scene in enumerate(manifest.scenes):
                                # 1. Image Upload
                                if hasattr(scene, "image_path") and scene.image_path and os.path.exists(scene.image_path):
                                    img_filename = os.path.basename(scene.image_path)
                                    r2_key = f"images/{project_id}/{img_filename}"
                                    if storage.upload_file(scene.image_path, r2_key):
                                        scene.image_path = f"{backend_url}/api/asset/{project_id}/image/{img_filename}"

                                # 2. Audio Upload
                                if hasattr(scene, "audio_path") and scene.audio_path and os.path.exists(scene.audio_path):
                                    audio_filename = os.path.basename(scene.audio_path)
                                    r2_key = f"audio/{project_id}/{audio_filename}"
                                    if storage.upload_file(scene.audio_path, r2_key):
                                        scene.audio_path = f"{backend_url}/api/asset/{project_id}/audio/{audio_filename}"

                                # 3. Scene Video Upload
                                if hasattr(scene, "video_path") and scene.video_path and os.path.exists(scene.video_path):
                                    vid_filename = os.path.basename(scene.video_path)
                                    r2_key = f"videos/{project_id}/{vid_filename}"
                                    if storage.upload_file(scene.video_path, r2_key):
                                        scene.video_path = f"{backend_url}/api/asset/{project_id}/video/{vid_filename}"
                        
                        print(f"[WRAPPER] Scene assets uploaded.", flush=True)
                    except Exception as e:
                        print(f"[WRAPPER] Asset upload error: {e}")

                    # Manifest 저장
                    manifest.outputs.final_video_path = public_url # Main video
                    
                    with open(manifest_path, "w", encoding="utf-8") as f:
                        f.write(json.dumps(manifest.model_dump(mode='json'), ensure_ascii=False, indent=2))
                    
                    print(f"[WRAPPER] Manifest updated with video_url and asset URLs", flush=True)
                else:
                    print(f"[WRAPPER] R2 Upload Failed (Check credentials)", flush=True)
            else:
                print(f"[WRAPPER] Local video file not found, skipping upload", flush=True)
                
        except Exception as upload_err:
            print(f"[WRAPPER] R2 Upload Error: {upload_err}", flush=True)
            import traceback
            traceback.print_exc()

    except Exception as e:
        print(f"\n[WRAPPER] =========================================", flush=True)
        print(f"[WRAPPER] ❌ ERROR IN VIDEO GENERATION", flush=True)
        print(f"[WRAPPER] Error: {str(e)}", flush=True)
        print(f"[WRAPPER] Traceback:\n{traceback.format_exc()}", flush=True)
        print(f"[WRAPPER] =========================================\n", flush=True)

        # 에러를 manifest에 기록
        try:
            manifest_path = f"outputs/{project_id}/manifest.json"
            if os.path.exists(manifest_path):
                with open(manifest_path, 'r', encoding='utf-8') as f:
                    manifest_data = json.load(f)
                manifest_data['status'] = 'failed'
                manifest_data['error_message'] = str(e)
                with open(manifest_path, 'w', encoding='utf-8') as f:
                    json.dump(manifest_data, f, ensure_ascii=False, indent=2)
                print(f"[WRAPPER] Manifest updated with error status", flush=True)
        except Exception as e2:
            print(f"[WRAPPER] Failed to update manifest with error: {str(e2)}", flush=True)

    finally:
        loop.close()
        print(f"[WRAPPER] Event loop closed for project: {project_id}", flush=True)



@app.get("/api/sample-voice/{voice_id}")
async def get_voice_sample(voice_id: str):
    """TTS 목소리 샘플 반환 (없으면 생성)"""
    from agents.tts_agent import TTSAgent
    
    # 샘플 디렉토리
    sample_dir = "media/samples"
    os.makedirs(sample_dir, exist_ok=True)
    
    file_path = f"{sample_dir}/{voice_id}.mp3"
    
    # 캐시된 파일이 없으면 생성
    if not os.path.exists(file_path):
        print(f"[API] Generating sample for voice: {voice_id}")
        agent = TTSAgent()
        agent.voice = voice_id
        
        # 샘플 멘트
        text = "안녕하세요? 저는 당신의 이야기를 들려줄 AI 목소리입니다."
        
        # TTS 생성 (Run synchronous agent method in threadpool)
        from starlette.concurrency import run_in_threadpool
        
        try:
            def generate_sample():
                try:
                    print(f"[DEBUG] Processing sample for voice_id: '{voice_id}'")
                    
                    # 1. Try Google Neural2 / Gemini (PRIMARY)
                    if voice_id.startswith("neural2") or voice_id.startswith("gemini"):
                        print(f"[API] Generating Google/Gemini sample with voice: {voice_id}")
                        if "gemini" in voice_id:
                             # Gemini Options - Neural2와 확실히 다른 목소리 사용
                             if "flash" in voice_id:
                                 # Flash -> Wavenet A (여성, Neural2-A와 다른 톤)
                                 voice_name = "ko-KR-Wavenet-A"
                             else:
                                 # Pro -> Wavenet D (남성, 깊은 목소리)
                                 voice_name = "ko-KR-Wavenet-D"
                             agent._call_google_neural2(text, voice_name, file_path)
                        else:
                             # Neural2 Options A, B, C
                             voice_name = "ko-KR-Neural2-A" # Default
                             if "_b" in voice_id:
                                 voice_name = "ko-KR-Neural2-B"
                             elif "_c" in voice_id or "male" in voice_id:
                                 voice_name = "ko-KR-Neural2-C"
                             
                             agent._call_google_neural2(text, voice_name, file_path)
                        return

                    # 2. Try ElevenLabs
                    elif hasattr(agent, '_call_elevenlabs_api') and agent.elevenlabs_key:
                        # OpenAI voices -> ElevenLabs IDs mapping
                        voice_map = {
                            # --- TOP 3 SELECTED VOICES ---
                            "voice_brian": "nPczCjzI2devNBz1zQrb",
                            "voice_sarah": "EXAVITQu4vr4xnSDxMaL",
                            "voice_laura": "FGY2WhTYpPnrIDTdsKH5",
                            
                            # Fallbacks
                            "onyx": "nPczCjzI2devNBz1zQrb",
                            "alloy": "EXAVITQu4vr4xnSDxMaL",
                        }
                        
                        # Use mapped ID or fallback
                        target_voice = voice_map.get(voice_id, "pNInz6obpgDQGcFmaJgB")
                        print(f"[API] Generating ElevenLabs sample with voice: {target_voice} (mapped from {voice_id})")
                        agent._call_elevenlabs_api(text, target_voice, file_path)
                        
                    # 3. Try OpenAI
                    elif hasattr(agent, '_call_tts_api') and agent.api_key:
                        agent._call_tts_api(text, file_path)
                        
                    # 4. Fallback: pyttsx3 (Local)
                    elif hasattr(agent, '_call_pyttsx3_local'):
                        print("[API] No Cloud API keys found. Using Local TTS.")
                        agent._call_pyttsx3_local(0, text, file_path)
                    
                    else:
                        raise Exception("No available TTS method found.")

                except Exception as e:
                    print(f"[API] Primary TTS failed ({e}). Trying fallback to Local...")
                    # Critical Fallback: pyttsx3
                    if hasattr(agent, '_call_pyttsx3_local'):
                        agent._call_pyttsx3_local(0, text, file_path)
                    else:
                        raise e

            await run_in_threadpool(generate_sample)
            
        except Exception as e:
            print(f"[API] All TTS methods failed: {e}")
            raise HTTPException(status_code=500, detail=f"TTS 생성 실패: {str(e)}")
            
    return FileResponse(file_path, media_type="audio/mpeg")


@app.get("/api/download/{project_id}")
async def download_video(project_id: str):
    """생성된 영상 다운로드"""
    # 가능한 경로들 시도
    possible_paths = [
        f"outputs/{project_id}/final_video_with_subtitles.mp4",  # 자막 적용된 버전 우선
        f"outputs/{project_id}/final_video.mp4",                  # 원본
    ]

    video_path = None
    for path in possible_paths:
        if os.path.exists(path):
            video_path = path
            break

    if not video_path:
        print(f"[DEBUG] Video not found for project {project_id}")
        print(f"[DEBUG] Checked paths: {possible_paths}")
        raise HTTPException(status_code=404, detail="영상을 찾을 수 없습니다.")

    print(f"[DEBUG] Downloading video from: {video_path}")

    return FileResponse(
        video_path,
        media_type="video/mp4",
        filename=f"storycut_{project_id}.mp4"
    )




# ============================================================================
# Auth Endpoints (Mock for Local)
# ============================================================================

@app.post("/api/auth/register")
async def register(req: RegisterRequest):
    """회원가입 (Local Mock)"""
    # 실제 구현에서는 DB 저장 필요
    return {
        "message": "User created successfully",
        "user": {
            "id": "user_local_123",
            "username": req.username,
            "email": req.email
        }
    }


@app.post("/api/auth/login")
async def login(req: LoginRequest):
    """로그인 (Local Mock)"""
    # 실제 구현에서는 PW 검증 필요
    if not req.email or not req.password:
        raise HTTPException(status_code=400, detail="이메일과 비밀번호를 입력하세요.")
    
    return {
        "token": "local_mock_token_12345",
        "user": {
            "id": "user_local_123",
            "username": req.email.split("@")[0],
            "email": req.email,
            "credits": 100
        }
    }


@app.get("/api/status/{project_id}")
async def get_video_status(project_id: str):
    """영상 생성 진행 상태 조회"""
    manifest_path = f"outputs/{project_id}/manifest.json"

    print(f"[STATUS] Checking status for project: {project_id}", flush=True)
    print(f"[STATUS] Manifest path: {manifest_path}", flush=True)
    print(f"[STATUS] Manifest exists: {os.path.exists(manifest_path)}", flush=True)

    # Manifest가 없으면 아직 처리 중
    if not os.path.exists(manifest_path):
        print(f"[STATUS] Manifest not found, returning processing status", flush=True)
        return {
            "project_id": project_id,
            "status": "processing",
            "progress": 15,
            "message": "영상 생성 준비 중...",
            "title": None,
            "video_url": None,
            "output_url": None,
            "error_message": None
        }

    # Manifest가 있으면 최종 상태 반환
    try:
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)
            status = manifest.get("status", "unknown")
            progress = manifest.get("progress", 50)

            print(f"[STATUS] Manifest status: {status}, progress: {progress}", flush=True)

            return {
                "project_id": project_id,
                "status": status,
                "progress": progress,
                "message": manifest.get("message", "처리 중"),
                "title": manifest.get("title"),
                "video_url": manifest.get("outputs", {}).get("final_video_path"),
                "output_url": manifest.get("outputs", {}).get("final_video_path"),
                "error_message": manifest.get("error_message"),
                "execution_time_sec": manifest.get("execution_time_sec")
            }
    except Exception as e:
        print(f"[STATUS] Error reading manifest: {str(e)}", flush=True)
        return {
            "project_id": project_id,
            "status": "error",
            "progress": 0,
            "message": f"상태 조회 실패: {str(e)}",
            "error_message": str(e)
        }


@app.get("/api/manifest/{project_id}")
async def get_manifest(project_id: str):
    """Manifest 조회"""
    manifest_path = f"outputs/{project_id}/manifest.json"

    if not os.path.exists(manifest_path):
        raise HTTPException(status_code=404, detail="프로젝트를 찾을 수 없습니다.")

    with open(manifest_path, "r", encoding="utf-8") as f:
        return json.load(f)


@app.get("/api/stream/{project_id}")
async def stream_video(project_id: str):
    """영상 스트리밍 (Inline Playback)"""
    possible_paths = [
        f"outputs/{project_id}/final_video_with_subtitles.mp4",
        f"outputs/{project_id}/final_video.mp4",
    ]
    video_path = None
    for path in possible_paths:
        if os.path.exists(path):
            video_path = path
            break
            
    if not video_path:
        raise HTTPException(status_code=404, detail="영상을 찾을 수 없습니다.")
        
    return FileResponse(video_path, media_type="video/mp4")  # filename 생략 -> Inline 재생


@app.get("/api/asset/{project_id}/{asset_type}/{filename}")
async def get_asset(project_id: str, asset_type: str, filename: str):
    """
    R2 또는 로컬에서 에셋 파일 제공 (이미지, 오디오, 비디오)
    asset_type: image, audio, video
    """
    # 1. 로컬 파일 먼저 확인
    type_to_dir = {
        "image": "scenes",
        "audio": "audio",
        "video": ""
    }
    local_dir = type_to_dir.get(asset_type, "")
    local_path = f"outputs/{project_id}/{local_dir}/{filename}" if local_dir else f"outputs/{project_id}/{filename}"

    if os.path.exists(local_path):
        media_types = {
            "image": "image/png",
            "audio": "audio/mpeg",
            "video": "video/mp4"
        }
        return FileResponse(local_path, media_type=media_types.get(asset_type, "application/octet-stream"))

    # 2. R2에서 가져오기
    r2_type_map = {
        "image": "images",
        "audio": "audio",
        "video": "videos"
    }
    r2_prefix = r2_type_map.get(asset_type, asset_type)
    r2_path = f"{r2_prefix}/{project_id}/{filename}"

    data = storage_manager.get_object(r2_path)
    if data:
        from fastapi.responses import Response
        media_types = {
            "image": "image/png",
            "audio": "audio/mpeg",
            "video": "video/mp4"
        }
        return Response(content=data, media_type=media_types.get(asset_type, "application/octet-stream"))

    raise HTTPException(status_code=404, detail=f"에셋을 찾을 수 없습니다: {filename}")


@app.get("/api/video/{project_id}")
async def get_video_from_r2(project_id: str):
    """
    R2에서 최종 비디오 제공 (Worker 대신 백엔드가 처리)
    """
    # 1. 로컬 파일 먼저 확인
    possible_paths = [
        f"outputs/{project_id}/final_video_with_subtitles.mp4",
        f"outputs/{project_id}/final_video.mp4",
    ]
    for path in possible_paths:
        if os.path.exists(path):
            return FileResponse(path, media_type="video/mp4", filename=f"{project_id}_video.mp4")

    # 2. R2에서 가져오기
    r2_path = f"videos/{project_id}/final_video.mp4"
    data = storage_manager.get_object(r2_path)
    if data:
        from fastapi.responses import Response
        return Response(
            content=data,
            media_type="video/mp4",
            headers={"Content-Disposition": f"attachment; filename={project_id}_video.mp4"}
        )

    raise HTTPException(status_code=404, detail="비디오를 찾을 수 없습니다.")


@app.get("/api/history")
async def get_history_list():
    """완료된 프로젝트 목록 조회 (R2 기반)"""
    from utils.storage import StorageManager
    
    storage = StorageManager()
    
    # R2에서 프로젝트 목록 가져오기
    projects = storage.list_projects()
    
    # R2 사용 불가능하면 로컬 폴더 폴백
    if not projects:
        print("[API] R2 unavailable, falling back to local outputs folder")
        outputs_dir = "outputs"
        projects = []

        if not os.path.exists(outputs_dir):
            return {"projects": []}

        # 디렉토리 순회 (최신순 정렬)
        try:
            dirs = [d for d in os.listdir(outputs_dir) if os.path.isdir(os.path.join(outputs_dir, d))]
            # 수정 시간 기준 내림차순 정렬
            dirs.sort(key=lambda x: os.path.getmtime(os.path.join(outputs_dir, x)), reverse=True)

            for pid in dirs:
                manifest_path = os.path.join(outputs_dir, pid, "manifest.json")
                if os.path.exists(manifest_path):
                    try:
                        with open(manifest_path, "r", encoding="utf-8") as f:
                            data = json.load(f)
                            # 필수 정보만 추출
                            projects.append({
                                "project_id": data.get("project_id"),
                                "title": data.get("title", "제목 없음"),
                                "status": data.get("status"),
                                "created_at": data.get("created_at"),
                                "thumbnail_url": f"/media/{pid}/thumbnail.png" if os.path.exists(os.path.join(outputs_dir, pid, "thumbnail.png")) else None,
                                "video_url": f"/api/stream/{pid}" if data.get("status") == "completed" else None,
                                "download_url": f"/api/download/{pid}" if data.get("status") == "completed" else None
                            })
                    except Exception:
                        continue
        except Exception as e:
            print(f"Error scanning history: {e}")
            return {"projects": []}

    return {"projects": projects}


# ============================================================================
# 마스터 캐릭터 이미지 생성 API (캐릭터 참조 시스템)
# ============================================================================

class GenerateCharacterRequest(BaseModel):
    """마스터 캐릭터 이미지 생성 요청"""
    character_token: str  # STORYCUT_HERO_A
    name: str
    gender: str = "unknown"
    age: str = "unknown"
    appearance: str  # "shoulder-length black hair, soft brown eyes..."
    clothing_default: str = ""
    visual_seed: int = 42
    style: str = "cinematic portrait, high quality, detailed"


@app.post("/api/projects/{project_id}/characters/generate")
async def generate_master_character(
    project_id: str,
    req: GenerateCharacterRequest
):
    """
    마스터 캐릭터 이미지 생성 (캐릭터 참조 시스템)

    스토리 생성 전에 주요 캐릭터의 마스터 이미지를 먼저 생성합니다.
    이후 모든 씬에서 이 이미지를 참조하여 캐릭터 일관성을 유지합니다.
    """
    from agents.image_agent import ImageAgent
    from schemas import CharacterSheet

    print(f"\n[Character Generation] Project: {project_id}, Token: {req.character_token}")

    # 프로젝트 디렉토리 확인/생성
    project_dir = f"outputs/{project_id}"
    characters_dir = f"{project_dir}/characters"
    os.makedirs(characters_dir, exist_ok=True)

    # 마스터 이미지 생성
    try:
        image_agent = ImageAgent()

        # 상세한 프롬프트 생성 (마스터 이미지용)
        master_prompt = f"""
Character portrait reference sheet for {req.name}:
- Gender: {req.gender}
- Age: {req.age}
- Appearance: {req.appearance}
- Clothing: {req.clothing_default}

Style: {req.style}, front view, neutral expression, clean background, masterpiece quality,
perfect for character reference, consistent lighting, sharp details
""".strip()

        print(f"[Character] Generating master image...")
        print(f"[Character] Prompt: {master_prompt[:100]}...")

        # 이미지 생성 (v2.0: tuple 반환 처리)
        image_path, image_id = image_agent.generate_image(
            scene_id=0,  # 마스터 이미지는 scene_id=0
            prompt=master_prompt,
            style="portrait",
            output_dir=f"{characters_dir}/{req.character_token}.png",  # 전체 경로 전달
            seed=req.visual_seed
        )

        print(f"[Character] Master image generated: {image_path}")
        if image_id:
            print(f"[Character] Image ID (for reference): {image_id}")

        # CharacterSheet 생성
        character_sheet = CharacterSheet(
            name=req.name,
            gender=req.gender,
            age=req.age,
            appearance=req.appearance,
            clothing_default=req.clothing_default,
            visual_seed=req.visual_seed,
            master_image_path=image_path,
            master_image_url=f"/media/{project_id}/characters/{req.character_token}.png",
            master_image_id=image_id  # v2.0: NanoBanana API가 반환한 ID 저장
        )

        # Manifest 업데이트
        manifest_path = f"{project_dir}/manifest.json"
        if os.path.exists(manifest_path):
            with open(manifest_path, "r", encoding="utf-8") as f:
                manifest_data = json.load(f)

            if "character_sheet" not in manifest_data:
                manifest_data["character_sheet"] = {}

            manifest_data["character_sheet"][req.character_token] = character_sheet.dict()

            with open(manifest_path, "w", encoding="utf-8") as f:
                json.dump(manifest_data, f, ensure_ascii=False, indent=2)

            print(f"[Character] Updated manifest with character: {req.character_token}")

        return {
            "success": True,
            "project_id": project_id,
            "character_token": req.character_token,
            "character_sheet": character_sheet.dict(),
            "message": f"Master character image generated: {req.name}"
        }

    except Exception as e:
        print(f"[Character] Error: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Character generation failed: {str(e)}")


@app.get("/api/projects/{project_id}/characters")
async def get_project_characters(project_id: str):
    """프로젝트의 모든 캐릭터 조회"""
    manifest_path = f"outputs/{project_id}/manifest.json"

    if not os.path.exists(manifest_path):
        raise HTTPException(status_code=404, detail="프로젝트를 찾을 수 없습니다.")

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest_data = json.load(f)

    character_sheet = manifest_data.get("character_sheet", {})

    return {
        "project_id": project_id,
        "total_characters": len(character_sheet),
        "characters": character_sheet
    }


# ============================================================================
# 씬별 재생성 API (Phase 1)
# ============================================================================

class RegenerateSceneRequest(BaseModel):
    """씬 재생성 요청"""
    style: Optional[str] = None  # 새로운 스타일 (옵션)
    regenerate_image: bool = True
    regenerate_tts: bool = True
    regenerate_video: bool = True


@app.post("/api/projects/{project_id}/scenes/{scene_id}/regenerate")
async def regenerate_scene(
    project_id: str,
    scene_id: int,
    req: RegenerateSceneRequest = None,
    background_tasks: BackgroundTasks = None
):
    """
    특정 씬만 재생성

    - 이미지/TTS/비디오를 선택적으로 재생성
    - 기존 에셋은 백업 후 교체
    - 완료 후 manifest 업데이트
    """
    from agents import SceneOrchestrator
    from schemas import SceneStatus

    if req is None:
        req = RegenerateSceneRequest()

    manifest_path = f"outputs/{project_id}/manifest.json"

    if not os.path.exists(manifest_path):
        raise HTTPException(status_code=404, detail="프로젝트를 찾을 수 없습니다.")

    # Manifest 로드
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest_data = json.load(f)

    scenes = manifest_data.get("scenes", [])

    # 해당 씬 찾기
    target_scene = None
    target_idx = None
    for idx, scene in enumerate(scenes):
        if scene.get("scene_id") == scene_id or scene.get("index") == scene_id:
            target_scene = scene
            target_idx = idx
            break

    if target_scene is None:
        raise HTTPException(status_code=404, detail=f"Scene {scene_id}를 찾을 수 없습니다.")

    # 재생성 실행
    try:
        orchestrator = SceneOrchestrator()
        style = req.style or manifest_data.get("input", {}).get("style_preset", "cinematic")

        # 씬 상태 업데이트
        target_scene["status"] = "regenerating"
        target_scene["retry_count"] = target_scene.get("retry_count", 0) + 1

        # 재생성
        video_path, audio_path = orchestrator.retry_scene(
            scene=target_scene,
            story_style=style
        )

        # 결과 업데이트
        if target_scene.get("assets") is None:
            target_scene["assets"] = {}

        target_scene["assets"]["video_path"] = video_path
        target_scene["assets"]["narration_path"] = audio_path
        target_scene["status"] = "completed"
        target_scene["error_message"] = None

        # Manifest 저장
        scenes[target_idx] = target_scene
        manifest_data["scenes"] = scenes

        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest_data, f, ensure_ascii=False, indent=2)

        return {
            "success": True,
            "project_id": project_id,
            "scene_id": scene_id,
            "message": f"Scene {scene_id} 재생성 완료",
            "video_path": video_path,
            "audio_path": audio_path
        }

    except Exception as e:
        # 실패 시 상태 업데이트
        target_scene["status"] = "failed"
        target_scene["error_message"] = str(e)
        scenes[target_idx] = target_scene
        manifest_data["scenes"] = scenes

        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest_data, f, ensure_ascii=False, indent=2)

        raise HTTPException(status_code=500, detail=f"씬 재생성 실패: {str(e)}")


@app.get("/api/projects/{project_id}/scenes")
async def get_project_scenes(project_id: str):
    """프로젝트의 모든 씬 상태 조회"""
    manifest_path = f"outputs/{project_id}/manifest.json"

    if not os.path.exists(manifest_path):
        raise HTTPException(status_code=404, detail="프로젝트를 찾을 수 없습니다.")

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest_data = json.load(f)

    scenes = manifest_data.get("scenes", [])

    return {
        "project_id": project_id,
        "total_scenes": len(scenes),
        "scenes": [
            {
                "scene_id": s.get("scene_id", s.get("index")),
                "status": s.get("status", "unknown"),
                "narration": s.get("narration", s.get("sentence", ""))[:50] + "...",
                "duration_sec": s.get("duration_sec"),
                "tts_duration_sec": s.get("tts_duration_sec"),
                "generation_method": s.get("generation_method"),
                "error_message": s.get("error_message"),
                "retry_count": s.get("retry_count", 0),
                "assets": s.get("assets", {})
            }
            for s in scenes
        ]
    }


@app.post("/api/projects/{project_id}/recompose")
async def recompose_video(project_id: str):
    """모든 씬을 다시 합성하여 최종 영상 생성"""
    from agents import ComposerAgent

    manifest_path = f"outputs/{project_id}/manifest.json"

    if not os.path.exists(manifest_path):
        raise HTTPException(status_code=404, detail="프로젝트를 찾을 수 없습니다.")

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest_data = json.load(f)

    scenes = manifest_data.get("scenes", [])

    # 에셋 경로 수집
    video_clips = []
    narration_clips = []

    for scene in scenes:
        assets = scene.get("assets", {})
        if assets.get("video_path"):
            video_clips.append(assets["video_path"])
        if assets.get("narration_path"):
            narration_clips.append(assets["narration_path"])

    if not video_clips:
        raise HTTPException(status_code=400, detail="합성할 비디오 클립이 없습니다.")

    # 재합성
    composer = ComposerAgent()
    output_path = f"outputs/{project_id}/final_video.mp4"

    try:
        final_video = composer.compose_video(
            video_clips=video_clips,
            narration_clips=narration_clips,
            music_path=None,  # 기존 BGM 사용 또는 None
            output_path=output_path
        )

        # Manifest 업데이트
        if "outputs" not in manifest_data:
            manifest_data["outputs"] = {}
        manifest_data["outputs"]["final_video_path"] = final_video
        manifest_data["status"] = "completed"

        # [Production] R2 업로드 (배포 환경 지원)
        try:
            from utils.storage import StorageManager
            storage = StorageManager()
            r2_key = f"videos/{project_id}/final_video.mp4"
            
            if storage.upload_file(final_video, r2_key):
                backend_url = "https://web-production-bb6bf.up.railway.app"
                public_url = f"{backend_url}/api/video/{project_id}"
                
                manifest_data["outputs"]["video_url"] = public_url
                print(f"[RECOMPOSE] R2 Upload Success: {public_url}")
            else:
                print(f"[RECOMPOSE] R2 Upload Failed")
        except Exception as upload_err:
             print(f"[RECOMPOSE] Upload Error: {upload_err}")

        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest_data, f, ensure_ascii=False, indent=2)

        return {
            "success": True,
            "project_id": project_id,
            "video_path": final_video,
            "message": "영상 재합성 완료"
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"영상 합성 실패: {str(e)}")


@app.get("/health")
async def health_check():
    """헬스 체크 및 진단"""
    import os
    from pathlib import Path
    
    # 필수 파일 체크
    prompt_path = Path("prompts/story_prompt.md")
    prompt_exists = prompt_path.exists()
    
    # API 키 체크 (값 노출 안함)
    google_key = os.getenv("GOOGLE_API_KEY")
    openai_key = os.getenv("OPENAI_API_KEY")
    
    return {
        "status": "ok",
        "version": "2.5",
        "diagnostics": {
            "google_api_key_set": google_key is not None and len(google_key) > 0,
            "openai_api_key_set": openai_key is not None and len(openai_key) > 0,
            "prompt_file_exists": prompt_exists,
            "cwd": os.getcwd(),
            "port": os.getenv("PORT", "8000")
        },
        "active_connections": len(active_connections)
    }


@app.get("/api/system/errors")
async def get_system_errors(limit: int = 50):
    """최근 시스템 에러 로그 조회"""
    try:
        from utils.error_manager import ErrorManager
        return {
            "success": True, 
            "errors": ErrorManager.get_recent_errors(limit)
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


if __name__ == "__main__":
    import uvicorn

    print("""
============================================================
              STORYCUT API Server v2.5 (More Voices)
============================================================
  Server: http://localhost:8000
  API Docs: http://localhost:8000/docs
============================================================
    """)

    port = int(os.getenv("PORT", 8000))
    uvicorn.run(
        "api_server:app",
        host="0.0.0.0",
        port=port,
        reload=False,
        log_level="info"
    )
