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

# [보안] API 키 마스킹 함수
def mask_api_key(key: str) -> str:
    """API 키를 안전하게 마스킹하여 로그에 출력."""
    if not key or len(key) < 10:
        return "(not set)"
    return f"{key[:4]}...{key[-4:]}"

# [보안] 프로덕션 모드 확인
IS_PRODUCTION = os.getenv("PRODUCTION", "").lower() == "true" or os.getenv("RAILWAY_ENVIRONMENT") is not None

api_key = os.getenv("OPENAI_API_KEY")
print(f"DEBUG: OPENAI_API_KEY: {mask_api_key(api_key)}")

google_api_key = os.getenv("GOOGLE_API_KEY")
print(f"DEBUG: GOOGLE_API_KEY: {mask_api_key(google_api_key)}")

from typing import Optional, Dict, Any, List
from datetime import datetime

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, BackgroundTasks, File, UploadFile, Form
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

# ============================================================
# Security: Input Validation Helpers
# ============================================================
import re as _re
import urllib.parse as _urlparse

_SAFE_PROJECT_ID = _re.compile(r'^[a-zA-Z0-9_\-]+$')
_SAFE_SIMPLE_ID = _re.compile(r'^[a-zA-Z0-9_\-\.]+$')

def validate_project_id(project_id: str) -> str:
    """project_id Path Traversal 방어"""
    if not project_id or not _SAFE_PROJECT_ID.match(project_id) or '..' in project_id:
        raise HTTPException(status_code=400, detail="Invalid project_id")
    return project_id

def validate_webhook_url(url: str) -> str:
    """Webhook URL SSRF 방어 - 내부 네트워크 주소 차단"""
    parsed = _urlparse.urlparse(url)
    if parsed.scheme not in ("https", "http"):
        raise HTTPException(status_code=400, detail="Webhook URL must use http(s)")
    hostname = (parsed.hostname or "").lower()
    # 내부 네트워크 / 메타데이터 차단
    blocked = (
        hostname in ("localhost", "127.0.0.1", "0.0.0.0", "[::1]", "metadata.google.internal")
        or hostname.startswith("169.254.")
        or hostname.startswith("10.")
        or hostname.startswith("192.168.")
        or hostname.startswith("172.16.") or hostname.startswith("172.17.")
        or hostname.startswith("172.18.") or hostname.startswith("172.19.")
        or hostname.startswith("172.2") or hostname.startswith("172.30.")
        or hostname.startswith("172.31.")
    )
    if blocked:
        raise HTTPException(status_code=400, detail="Webhook URL cannot point to internal network")
    return url

def load_manifest(project_id: str) -> dict:
    """프로젝트 매니페스트 로딩 헬퍼 (404 자동 처리)"""
    manifest_path = f"outputs/{project_id}/manifest.json"
    if not os.path.exists(manifest_path):
        raise HTTPException(status_code=404, detail="Project not found")
    with open(manifest_path, "r", encoding="utf-8") as f:
        return json.load(f)

# Global Exception Handler to ensure CORS headers on 500 errors
from fastapi import Request
from fastapi.responses import JSONResponse
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    print(f"[CRITICAL ERROR] Global exception caught: {exc}")
    import traceback
    traceback.print_exc()
    
    # [보안] 프로덕션에서는 트레이스백 숨김
    if IS_PRODUCTION:
        return JSONResponse(
            status_code=500,
            content={"detail": "Internal Server Error"},
            headers={
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Methods": "*",
                "Access-Control-Allow-Headers": "*",
            }
        )
    else:
        # 개발 모드에서는 상세 정보 표시
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
    voice: str = "uyVNoMrnUku1dZyVEXwD"  # Default voice (ElevenLabs Anna Kim)
    duration: int = 60
    platform: str = "youtube_long"
    character_ethnicity: str = "auto"

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


class ScriptRequest(BaseModel):
    """스크립트 직접 입력으로 영상 생성 요청"""
    script: str              # 전체 내레이션 스크립트 텍스트
    genre: str = "emotional"
    mood: str = "dramatic"
    style: str = "cinematic, high contrast"
    voice: str = "uyVNoMrnUku1dZyVEXwD"
    duration: int = 60
    platform: str = "youtube_long"
    character_ethnicity: str = "auto"

    # Feature Flags
    hook_scene1_video: bool = False
    ffmpeg_kenburns: bool = True
    ffmpeg_audio_ducking: bool = False
    subtitle_burn_in: bool = True
    context_carry_over: bool = True
    optimization_pack: bool = True

    project_id: Optional[str] = None


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

    # 히스토리 저장 (최대 100개 이벤트, 초과 시 오래된 것부터 제거)
    if project_id not in project_event_history:
        project_event_history[project_id] = []
    project_event_history[project_id].append(payload)
    if len(project_event_history[project_id]) > 100:
        project_event_history[project_id] = project_event_history[project_id][-50:]
    print(f"[DEBUG] History saved for {project_id}", flush=True)

    # 완료/실패 시 5분 후 히스토리 정리 예약
    if progress >= 100 or step == "error":
        async def _cleanup_history(pid: str):
            await asyncio.sleep(300)  # 5분
            project_event_history.pop(pid, None)
            active_connections.pop(pid, None)
        asyncio.ensure_future(_cleanup_history(project_id))

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
        progress = max(progress, self.current_progress)  # Never decrease
        await self.update(f"scene_{scene_id}", progress, f"Scene {scene_id}/{self.total_scenes} 처리 중...")

    async def scene_complete(self, scene_id: int, method: str, image_url: str = None):
        progress = 15 + int((scene_id / self.total_scenes) * 60)
        progress = max(progress, self.current_progress)  # Never decrease
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
        # SSRF 방어: webhook URL 검증
        if webhook_url:
            validate_webhook_url(webhook_url)
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
            # 진행상황 초기화 (Story complete = 25% of total, matching initial manifest)
            await self.tracker.update("story", 25, "스토리 확정됨 - 영상 생성 시작", {
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


## manifest 엔드포인트는 아래(line ~1646)에 통합 정의됨


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
        character_ethnicity=req.character_ethnicity,
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


@app.post("/api/generate/from-script")
async def generate_from_script(req: ScriptRequest):
    """스크립트 직접 입력 → 씬 분할 + 이미지 프롬프트 생성"""
    import uuid

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
        topic=None,
        genre=req.genre,
        mood=req.mood,
        style_preset=req.style,
        duration_target_sec=req.duration,
        target_platform=platform,
        voice_id=req.voice,
        voice_over=True,
        bgm=True,
        subtitles=req.subtitle_burn_in,
        character_ethnicity=req.character_ethnicity,
        feature_flags=feature_flags,
    )

    tracker = ProgressTracker(project_id)
    pipeline = TrackedPipeline(tracker)

    try:
        from starlette.concurrency import run_in_threadpool
        story_data = await run_in_threadpool(
            pipeline.generate_story_from_script, req.script, project_request
        )
        return {
            "story_data": story_data,
            "request_params": project_request.dict(),
            "project_id": project_id
        }
    except Exception as e:
        print(f"Script generation failed: {e}")
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
        "progress": 25,
        "message": "장면 처리 준비 중...",
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
                voice_id=request_params.get('voice_id') or request_params.get('voice', 'uyVNoMrnUku1dZyVEXwD'),
                voice_over=request_params.get('voice_over', True),
                bgm=request_params.get('bgm', True),
                # subtitle_burn_in is what frontend sends. Map it to subtitles.
                subtitles=request_params.get('subtitle_burn_in', True),
                character_ethnicity=request_params.get('character_ethnicity', 'auto'),
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
        print(f"[WRAPPER] VIDEO GENERATION COMPLETED", flush=True)
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
                    
                    # manifest.json도 R2에 업로드 (history 조회용)
                    manifest_r2_key = f"videos/{project_id}/manifest.json"
                    if storage.upload_file(manifest_path, manifest_r2_key):
                        print(f"[WRAPPER] Manifest uploaded to R2: {manifest_r2_key}", flush=True)
                    
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
        print(f"[WRAPPER] ERROR IN VIDEO GENERATION", flush=True)
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


@app.post("/api/generate/images")
async def generate_images_only(req: GenerateVideoRequest, background_tasks: BackgroundTasks):
    """
    Step 2A: 스토리 확정 후 이미지만 생성 (비동기, 프로그레시브).

    즉시 응답 반환 후 백그라운드에서 이미지 생성.
    프론트엔드는 GET /api/status/images/{project_id}로 진행 상황 폴링.

    Returns:
        project_id, status, total_scenes
    """
    import uuid
    import threading
    from pathlib import Path

    project_id = req.project_id or str(uuid.uuid4())[:8]
    total_scenes = len(req.story_data.get('scenes', []))

    print(f"\n[API] ========== IMAGES ONLY GENERATION (ASYNC) =========")
    print(f"[API] Project ID: {project_id}")
    print(f"[API] Scene count: {total_scenes}")

    # req.request_params는 이미 ProjectRequest 객체
    request_obj = req.request_params
    story_data = req.story_data

    def run_image_generation():
        try:
            pipeline = StorycutPipeline()
            pipeline.generate_images_only(
                story_data,
                request_obj,
                project_id
            )
            print(f"[API] Images generated successfully!")
        except Exception as e:
            print(f"[API] Image generation failed: {e}")
            import traceback
            traceback.print_exc()
            # manifest에 에러 기록
            manifest_path = f"outputs/{project_id}/manifest.json"
            try:
                if os.path.exists(manifest_path):
                    with open(manifest_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    data['status'] = 'failed'
                    data['error_message'] = str(e)
                    with open(manifest_path, 'w', encoding='utf-8') as f:
                        json.dump(data, f, ensure_ascii=False, indent=2)
            except Exception:
                pass

    thread = threading.Thread(target=run_image_generation, daemon=True)
    thread.start()

    return {
        "project_id": project_id,
        "status": "started",
        "total_scenes": total_scenes,
        "message": "이미지 생성이 시작되었습니다."
    }


@app.get("/api/status/images/{project_id}")
async def get_image_generation_status(project_id: str):
    """
    이미지 생성 진행 상황 조회 (프로그레시브 폴링용).

    manifest.json을 읽어서 각 씬의 이미지 생성 상태 반환.
    """
    validate_project_id(project_id)
    manifest_path = f"outputs/{project_id}/manifest.json"

    if not os.path.exists(manifest_path):
        return {
            "project_id": project_id,
            "status": "not_found",
            "completed": 0,
            "total": 0,
            "scenes": []
        }

    try:
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest_data = json.load(f)

        scenes = manifest_data.get("scenes", [])
        total = len(scenes)
        completed = 0
        scene_list = []

        for s in scenes:
            status = s.get("status", "pending")
            assets = s.get("assets", {})
            image_path = assets.get("image_path") if isinstance(assets, dict) else None

            # 이미지 경로를 웹 URL로 변환
            web_path = None
            if image_path:
                normalized = image_path.replace("\\", "/")
                if normalized.startswith("/media/"):
                    web_path = normalized
                elif "outputs/" in normalized:
                    rel = normalized.split("outputs/", 1)[1]
                    web_path = f"/media/{rel}"
                else:
                    web_path = f"/media/{project_id}/{normalized}"

            if status == "completed":
                completed += 1

            scene_list.append({
                "scene_id": s.get("scene_id", s.get("index")),
                "status": status,
                "image_path": web_path,
                "narration": s.get("narration", s.get("sentence", "")),
                "prompt": s.get("prompt", ""),
            })

        overall_status = manifest_data.get("status", "generating_images")

        return {
            "project_id": project_id,
            "status": overall_status,
            "completed": completed,
            "total": total,
            "scenes": scene_list,
            "error_message": manifest_data.get("error_message"),
        }

    except Exception as e:
        return {
            "project_id": project_id,
            "status": "error",
            "error_message": str(e),
            "completed": 0,
            "total": 0,
            "scenes": []
        }


@app.post("/api/regenerate/image/{project_id}/{scene_id}")
async def regenerate_scene_image(project_id: str, scene_id: int, req: Optional[dict] = None):
    """
    특정 씬의 이미지를 재생성합니다.

    Args:
        project_id: 프로젝트 ID
        scene_id: 씬 ID
        req: {"prompt": "새 프롬프트"} (선택사항)

    Returns:
        새로 생성된 이미지 URL
    """
    validate_project_id(project_id)
    from starlette.concurrency import run_in_threadpool
    from agents.image_agent import ImageAgent
    import json

    print(f"\n[API] Regenerating image for scene {scene_id} in project {project_id}")

    manifest_data = load_manifest(project_id)
    manifest_path = f"outputs/{project_id}/manifest.json"

    # 해당 씬 찾기
    scenes = manifest_data.get("scenes", [])
    target_scene = None
    for scene in scenes:
        if scene.get("scene_id") == scene_id:
            target_scene = scene
            break

    if not target_scene:
        raise HTTPException(status_code=404, detail=f"Scene {scene_id} not found")

    # 이미지 재생성
    image_agent = ImageAgent()
    if req is None:
        req = {}
    new_prompt = req.get("prompt")

    # 인종 런타임 주입
    final_prompt = new_prompt or target_scene.get("prompt", "")
    _eth = manifest_data.get("character_ethnicity", "auto")
    _ETH_KW = {
        "korean": "Korean", "japanese": "Japanese", "chinese": "Chinese",
        "southeast_asian": "Southeast Asian", "european": "European",
        "black": "Black", "hispanic": "Hispanic",
    }
    _eth_kw = _ETH_KW.get(_eth, "")
    if _eth_kw and _eth_kw.lower() not in final_prompt.lower():
        final_prompt = f"{_eth_kw} characters, {final_prompt}"

    try:
        image_path, image_id = await run_in_threadpool(
            image_agent.generate_image,
            scene_id=scene_id,
            prompt=final_prompt,
            style=target_scene.get("style", "cinematic"),
            output_dir=f"outputs/{project_id}/media/images",
            seed=None  # New random seed for regeneration
        )
        
        # image_path를 웹 URL로 정규화
        web_path = image_path
        if image_path:
            normalized = image_path.replace("\\", "/")
            if not normalized.startswith("/media/") and not normalized.startswith("http"):
                if "outputs/" in normalized:
                    rel = normalized.split("outputs/", 1)[1]
                    web_path = f"/media/{rel}"
                else:
                    web_path = f"/media/{project_id}/{normalized}"

        # Manifest 업데이트 (원본 경로 유지)
        for scene in scenes:
            if scene.get("scene_id") == scene_id:
                if "assets" not in scene:
                    scene["assets"] = {}
                scene["assets"]["image_path"] = image_path
                break

        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest_data, f, ensure_ascii=False, indent=2)

        return {
            "scene_id": scene_id,
            "image_path": web_path,
            "prompt": new_prompt
        }
        
    except Exception as e:
        print(f"[API] Image regeneration failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/test/image")
async def test_image_generation():
    """
    이미지 생성 테스트 — gemini-2.5-flash-image (Nano Banana).
    """
    import requests as req_lib
    import base64

    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        return {"status": "error", "detail": "GOOGLE_API_KEY not set"}

    model = "gemini-2.5-flash-image"
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
    payload = {
        "contents": [{"parts": [{"text": "Generate an image of a cute cat sitting on a chair, digital art style, cinematic."}]}],
        "generationConfig": {"responseModalities": ["TEXT", "IMAGE"]}
    }

    try:
        resp = req_lib.post(url, json=payload, headers={"Content-Type": "application/json"}, timeout=60)
        has_image = False
        text_content = ""
        finish_reason = ""
        response_keys = []

        if resp.status_code == 200:
            data = resp.json()
            if "candidates" in data and len(data["candidates"]) > 0:
                cand = data["candidates"][0]
                finish_reason = cand.get("finishReason", "")
                parts = cand.get("content", {}).get("parts", [])
                response_keys = [list(p.keys()) for p in parts]
                for part in parts:
                    # REST API는 camelCase (inlineData), SDK는 snake_case (inline_data)
                    image_part = part.get("inlineData") or part.get("inline_data")
                    if image_part:
                        has_image = True
                        os.makedirs("outputs/_test", exist_ok=True)
                        img_data = base64.b64decode(image_part["data"])
                        with open("outputs/_test/test_image.png", "wb") as f:
                            f.write(img_data)
                    if "text" in part:
                        text_content += part["text"][:100]

        return {
            "status": "ok" if has_image else "failed",
            "model": model,
            "status_code": resp.status_code,
            "has_image": has_image,
            "text": text_content[:200] if text_content else None,
            "finish_reason": finish_reason,
            "response_part_keys": response_keys,
            "test_image_url": "/media/_test/test_image.png" if has_image else None,
            "error": resp.text[:200] if resp.status_code != 200 else None,
        }
    except Exception as e:
        return {"status": "error", "model": model, "error": str(e)}


@app.post("/api/convert/i2v/{project_id}/{scene_id}")
async def convert_image_to_video(project_id: str, scene_id: int, req: dict = {"motion_prompt": None}):
    """
    특정 씬의 이미지를 Veo I2V로 영상 변환합니다.
    
    Args:
        project_id: 프로젝트 ID
        scene_id: 씬 ID
        req: {"motion_prompt": "camera slowly zooms in"} (선택사항)
    
    Returns:
        변환된 비디오 URL
    """
    from starlette.concurrency import run_in_threadpool
    from agents.video_agent import VideoAgent
    import json
    
    print(f"\n[API] Converting image to video for scene {scene_id} in project {project_id}")

    manifest_path = f"outputs/{project_id}/manifest.json"
    manifest_data = load_manifest(project_id)

    # 해당 씬 찾기
    scenes = manifest_data.get("scenes", [])
    target_scene = None
    for scene in scenes:
        if scene.get("scene_id") == scene_id:
            target_scene = scene
            break
    
    if not target_scene:
        raise HTTPException(status_code=404, detail=f"Scene {scene_id} not found")
    
    # 이미지 경로 확인 (웹 URL → 파일시스템 경로 변환)
    image_path = target_scene.get("assets", {}).get("image_path")
    if image_path:
        # /media/xxx → outputs/xxx 변환
        normalized = image_path.replace("\\", "/")
        if normalized.startswith("/media/"):
            image_path = f"outputs/{normalized[len('/media/'):]}"
        elif not os.path.isabs(normalized) and not normalized.startswith("outputs/"):
            image_path = f"outputs/{project_id}/{normalized}"
    if not image_path or not os.path.exists(image_path):
        raise HTTPException(status_code=404, detail=f"Image not found for scene {scene_id}: {image_path}")
    
    # I2V 변환
    video_agent = VideoAgent()
    motion_prompt = req.get("motion_prompt") if req else "camera slowly pans across the scene"
    
    try:
        video_path = await run_in_threadpool(
            video_agent.generate_from_image,
            image_path=image_path,
            prompt=target_scene.get("prompt", ""),
            duration_sec=target_scene.get("duration_sec", 5),
            output_dir=f"outputs/{project_id}/media/video",
            scene_id=scene_id,
            motion_prompt=motion_prompt
        )
        
        # Manifest 업데이트
        for scene in scenes:
            if scene.get("scene_id") == scene_id:
                if "assets" not in scene:
                    scene["assets"] = {}
                scene["assets"]["video_path"] = video_path
                scene["i2v_converted"] = True
                break
        
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest_data, f, ensure_ascii=False, indent=2)
        
        return {
            "scene_id": scene_id,
            "video_path": video_path,
            "motion_prompt": motion_prompt
        }
        
    except Exception as e:
        print(f"[API] I2V conversion failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/toggle/hook-video/{project_id}/{scene_id}")
async def toggle_hook_video(project_id: str, scene_id: int, req: dict):
    """
    특정 씬을 Hook Video로 설정/해제합니다.
    
    Args:
        project_id: 프로젝트 ID
        scene_id: 씬 ID
        req: {"enable": true/false}
    
    Returns:
        Hook video 설정 상태
    """
    import json
    
    enable = req.get("enable", False)
    
    print(f"\n[API] Toggle hook video for scene {scene_id}: {enable}")

    manifest_data = load_manifest(project_id)

    # 해당 씬 찾기
    scenes = manifest_data.get("scenes", [])
    found = False
    for scene in scenes:
        if scene.get("scene_id") == scene_id:
            scene["hook_video_enabled"] = enable
            found = True
            break
    
    if not found:
        raise HTTPException(status_code=404, detail=f"Scene {scene_id} not found")
    
    #  Manifest 저장
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest_data, f, ensure_ascii=False, indent=2)
    
    return {
        "scene_id": scene_id,
        "hook_video_enabled": enable
    }



@app.get("/api/sample-voice/{voice_id}")
async def get_voice_sample(voice_id: str):
    """TTS 목소리 샘플 반환 (없으면 생성)"""
    if not _SAFE_SIMPLE_ID.match(voice_id) or '..' in voice_id:
        raise HTTPException(status_code=400, detail="Invalid voice_id")
    from agents.tts_agent import TTSAgent

    # 샘플 디렉토리
    sample_dir = "media/samples"
    os.makedirs(sample_dir, exist_ok=True)

    # 서버 시작 후 최초 1회: 이전 프로바이더 캐시 파일 정리
    if not getattr(get_voice_sample, '_cache_cleaned', False):
        for old_file in os.listdir(sample_dir):
            old_path = os.path.join(sample_dir, old_file)
            if os.path.isfile(old_path):
                os.remove(old_path)
                print(f"[API] Cleaned old sample cache: {old_file}")
        get_voice_sample._cache_cleaned = True

    file_path = f"{sample_dir}/{voice_id}.mp3"

    # 캐시된 파일이 없거나 빈 파일이면 (재)생성
    if not os.path.exists(file_path) or os.path.getsize(file_path) == 0:
        # 빈 파일 정리
        if os.path.exists(file_path):
            os.remove(file_path)

        print(f"[API] Generating sample for voice: {voice_id}")
        agent = TTSAgent(voice=voice_id)

        # 샘플 멘트
        text = "안녕하세요? 저는 당신의 이야기를 들려줄 AI 목소리입니다."

        # TTS 생성 (Run synchronous agent method in threadpool)
        from starlette.concurrency import run_in_threadpool

        try:
            def generate_sample():
                print(f"[DEBUG] Processing sample for voice_id: '{voice_id}'")
                if agent.elevenlabs_key:
                    agent._call_elevenlabs_api(text, voice_id, file_path)
                else:
                    raise Exception("ELEVENLABS_API_KEY not set")

            await run_in_threadpool(generate_sample)

        except Exception as e:
            # 실패 시 빈/불완전 파일 정리
            if os.path.exists(file_path):
                os.remove(file_path)
            print(f"[API] TTS sample generation failed: {e}")
            raise HTTPException(status_code=500, detail=f"TTS 생성 실패: {str(e)}")

    return FileResponse(file_path, media_type="audio/mpeg")


@app.get("/api/download/{project_id}")
async def download_video(project_id: str):
    """생성된 영상 다운로드"""
    validate_project_id(project_id)

    # 가능한 경로들 시도
    possible_paths = [
        f"outputs/{project_id}/final_video_with_subtitles.mp4",  # 자막 적용된 버전 우선
        f"outputs/{project_id}/final_video.mp4",                  # 원본
    ]

    outputs_base = os.path.realpath("outputs")
    video_path = None
    
    for path in possible_paths:
        resolved = os.path.realpath(path)
        if not resolved.startswith(outputs_base):
            continue
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
# Auth Endpoints (Mock for Local Development ONLY)
# ============================================================================
# ⚠️ 보안 경고: 이 엔드포인트들은 로컬 개발용입니다!
# 실제 인증은 Cloudflare Worker (worker.js)에서 D1 DB와 bcrypt를 사용합니다.
# Railway 배포 환경에서는 이 엔드포인트들을 사용하지 마세요.
# ============================================================================

@app.post("/api/auth/register")
async def register(req: RegisterRequest):
    """
    회원가입 (Local Mock ONLY)
    
    ⚠️ 경고: 이것은 개발용 Mock입니다. 실제 회원가입은 Cloudflare Worker가 처리합니다.
    """
    print("[SECURITY WARNING] Using LOCAL MOCK auth - NOT for production!")
    return {
        "message": "User created successfully (LOCAL MOCK)",
        "user": {
            "id": "user_local_mock",
            "username": req.username,
            "email": req.email
        },
        "_warning": "This is a mock response for local development only"
    }


@app.post("/api/auth/login")
async def login(req: LoginRequest):
    """
    로그인 (Local Mock ONLY)
    
    ⚠️ 경고: 이것은 개발용 Mock입니다. 실제 인증은 Cloudflare Worker가 처리합니다.
    """
    print("[SECURITY WARNING] Using LOCAL MOCK auth - NOT for production!")
    
    if not req.email or not req.password:
        raise HTTPException(status_code=400, detail="이메일과 비밀번호를 입력하세요.")
    
    return {
        "token": "local_mock_token_DO_NOT_USE_IN_PRODUCTION",
        "user": {
            "id": "user_local_mock",
            "username": req.email.split("@")[0],
            "email": req.email,
            "credits": 100
        },
        "_warning": "This is a mock response for local development only"
    }


@app.get("/api/status/{project_id}")
async def get_video_status(project_id: str):
    """영상 생성 진행 상태 조회"""
    validate_project_id(project_id)
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
    """Manifest 조회 (로컬 우선, R2 fallback)"""
    validate_project_id(project_id)
    manifest_path = f"outputs/{project_id}/manifest.json"

    if os.path.exists(manifest_path):
        with open(manifest_path, "r", encoding="utf-8") as f:
            return json.load(f)

    # R2 fallback (videos/{project_id}/manifest.json 경로로 저장됨)
    try:
        from utils.storage import StorageManager
        storage = StorageManager()
        # 먼저 videos/ 경로 시도 (표준 업로드 경로)
        raw = storage.get_object(f"videos/{project_id}/manifest.json")
        if not raw:
            # 레거시 경로 시도
            raw = storage.get_object(f"{project_id}/manifest.json")
        if raw:
            return json.loads(raw.decode("utf-8"))
    except Exception:
        pass

    raise HTTPException(status_code=404, detail="프로젝트를 찾을 수 없습니다.")


@app.get("/api/stream/{project_id}")
async def stream_video(project_id: str):
    """영상 스트리밍 (Inline Playback)"""
    validate_project_id(project_id)

    possible_paths = [
        f"outputs/{project_id}/final_video_with_subtitles.mp4",
        f"outputs/{project_id}/final_video.mp4",
    ]

    outputs_base = os.path.realpath("outputs")
    video_path = None
    
    for path in possible_paths:
        resolved = os.path.realpath(path)
        if not resolved.startswith(outputs_base):
            continue
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
    validate_project_id(project_id)

    # filename 검증
    if not filename or not _SAFE_SIMPLE_ID.match(filename) or '..' in filename:
        raise HTTPException(status_code=400, detail="Invalid filename")

    # asset_type 화이트리스트 검증
    if asset_type not in ["image", "audio", "video"]:
        raise HTTPException(status_code=400, detail="Invalid asset_type")
    
    # 4. 로컬 파일 경로 생성
    type_to_dir = {
        "image": "scenes",
        "audio": "audio",
        "video": ""
    }
    local_dir = type_to_dir.get(asset_type, "")
    local_path = f"outputs/{project_id}/{local_dir}/{filename}" if local_dir else f"outputs/{project_id}/{filename}"
    
    # 5. [보안] 최종 경로가 outputs 디렉토리 내부인지 확인
    outputs_base = os.path.realpath("outputs")
    resolved_path = os.path.realpath(local_path)
    
    if not resolved_path.startswith(outputs_base):
        raise HTTPException(status_code=403, detail="접근이 거부되었습니다.")

    # MV 프로젝트: media/images/ 경로도 확인 (fallback)
    if not os.path.exists(local_path) and asset_type == "image":
        alt_path = f"outputs/{project_id}/media/images/{filename}"
        alt_resolved = os.path.realpath(alt_path)
        if alt_resolved.startswith(outputs_base) and os.path.exists(alt_path):
            local_path = alt_path

    if os.path.exists(local_path):
        media_types = {
            "image": "image/png",
            "audio": "audio/mpeg",
            "video": "video/mp4"
        }
        return FileResponse(local_path, media_type=media_types.get(asset_type, "application/octet-stream"))

    # 6. R2에서 가져오기
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
    validate_project_id(project_id)

    # 1. 로컬 파일 먼저 확인
    possible_paths = [
        f"outputs/{project_id}/final_video_with_subtitles.mp4",
        f"outputs/{project_id}/final_video.mp4",
    ]
    
    outputs_base = os.path.realpath("outputs")
    
    for path in possible_paths:
        resolved = os.path.realpath(path)
        if not resolved.startswith(outputs_base):
            continue  # 보안 위반 시 스킵
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
    """완료된 프로젝트 목록 조회 (R2 + 로컬 병합)"""
    from utils.storage import StorageManager

    storage = StorageManager()
    outputs_dir = "outputs"

    # R2에서 프로젝트 목록 가져오기
    r2_projects = storage.list_projects() or []

    # R2 프로젝트에 type 필드 보강
    seen_ids = set()
    for p in r2_projects:
        pid = p.get("project_id", "")
        seen_ids.add(pid)
        if "type" not in p:
            p["type"] = "mv" if pid.startswith("mv_") else "video"
        # MV인데 URL이 일반 경로면 수정
        if p["type"] == "mv":
            if p.get("video_url") and "/api/stream/" in p["video_url"] and "/api/mv/" not in p["video_url"]:
                p["video_url"] = f"/api/mv/stream/{pid}"
            if p.get("download_url") and "/api/download/" in p["download_url"] and "/api/mv/" not in p["download_url"]:
                p["download_url"] = f"/api/mv/download/{pid}"

    # 로컬 폴더에서 R2에 없는 프로젝트 보충 (특히 MV)
    local_projects = []
    if os.path.exists(outputs_dir):
        try:
            dirs = [d for d in os.listdir(outputs_dir) if os.path.isdir(os.path.join(outputs_dir, d)) and d not in seen_ids]
            dirs.sort(key=lambda x: os.path.getmtime(os.path.join(outputs_dir, x)), reverse=True)

            for pid in dirs:
                manifest_path = os.path.join(outputs_dir, pid, "manifest.json")
                if not os.path.exists(manifest_path):
                    continue
                try:
                    with open(manifest_path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                        is_mv = pid.startswith("mv_")
                        is_completed = data.get("status") == "completed"

                        if is_mv:
                            mv_title = data.get("concept") or ""
                            if not mv_title:
                                music_path = data.get("music_analysis", {}).get("file_path", "")
                                if music_path:
                                    mv_title = os.path.splitext(os.path.basename(music_path))[0]
                            if not mv_title:
                                mv_title = f"MV {pid}"

                            music_analysis = data.get("music_analysis", {})
                            scenes = data.get("scenes", [])

                            local_projects.append({
                                "project_id": pid,
                                "title": mv_title,
                                "type": "mv",
                                "status": data.get("status"),
                                "created_at": data.get("created_at"),
                                "thumbnail_url": f"/media/{pid}/thumbnail.png" if os.path.exists(os.path.join(outputs_dir, pid, "thumbnail.png")) else None,
                                "video_url": f"/api/mv/stream/{pid}" if is_completed else None,
                                "download_url": f"/api/mv/download/{pid}" if is_completed else None,
                                "duration_sec": music_analysis.get("duration_sec"),
                                "genre": music_analysis.get("genre"),
                                "style": data.get("style"),
                                "scene_count": len(scenes),
                            })
                        else:
                            scenes = data.get("scenes", [])
                            local_projects.append({
                                "project_id": data.get("project_id", pid),
                                "title": data.get("title", "제목 없음"),
                                "type": "video",
                                "status": data.get("status"),
                                "created_at": data.get("created_at"),
                                "thumbnail_url": f"/media/{pid}/thumbnail.png" if os.path.exists(os.path.join(outputs_dir, pid, "thumbnail.png")) else None,
                                "video_url": f"/api/stream/{pid}" if is_completed else None,
                                "download_url": f"/api/download/{pid}" if is_completed else None,
                                "scene_count": len(scenes),
                            })
                except Exception:
                    continue
        except Exception as e:
            print(f"Error scanning local history: {e}")

    # R2 + 로컬 병합 후 시간순 정렬 (최신 먼저)
    all_projects = r2_projects + local_projects
    all_projects.sort(key=lambda p: p.get("created_at") or "", reverse=True)
    return {"projects": all_projects}


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
    validate_project_id(project_id)
    manifest_data = load_manifest(project_id)
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
    validate_project_id(project_id)
    from agents import SceneOrchestrator
    from schemas import SceneStatus

    if req is None:
        req = RegenerateSceneRequest()

    manifest_data = load_manifest(project_id)
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
    validate_project_id(project_id)
    manifest_data = load_manifest(project_id)
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


class UpdateNarrationRequest(BaseModel):
    narration: str

@app.put("/api/projects/{project_id}/scenes/{scene_id}/narration")
async def update_scene_narration(project_id: str, scene_id: int, req: UpdateNarrationRequest):
    """씬 내레이션 텍스트 수정 + SRT 자막 재생성"""
    validate_project_id(project_id)
    manifest_data = load_manifest(project_id)
    scenes = manifest_data.get("scenes", [])

    # 해당 씬 찾기
    target = None
    for s in scenes:
        if s.get("scene_id") == scene_id:
            target = s
            break
    if not target:
        raise HTTPException(status_code=404, detail=f"Scene {scene_id} not found")

    # 1) 내레이션 텍스트 업데이트
    target["narration"] = req.narration

    # 2) 해당 씬의 SRT 재생성
    from utils.ffmpeg_utils import FFmpegComposer
    composer = FFmpegComposer()
    srt_path = f"outputs/{project_id}/media/subtitles/scene_{scene_id:02d}.srt"
    duration_sec = target.get("tts_duration_sec") or target.get("duration_sec", 5)
    composer.generate_srt_from_scenes(
        [{"narration": req.narration, "duration_sec": duration_sec}],
        srt_path
    )
    target.setdefault("assets", {})["subtitle_srt_path"] = srt_path

    # 3) 수정됨 플래그 (재합성 시 씬 비디오 재렌더링 필요)
    target["_narration_modified"] = True

    # 4) 매니페스트 저장
    manifest_path = f"outputs/{project_id}/manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest_data, f, ensure_ascii=False, indent=2)

    return {"success": True, "scene_id": scene_id, "narration": req.narration}


@app.post("/api/projects/{project_id}/recompose")
async def recompose_video(project_id: str):
    """모든 씬을 다시 합성하여 최종 영상 생성 (내레이션 수정된 씬은 자막 재렌더링)"""
    validate_project_id(project_id)
    from agents import ComposerAgent
    from utils.ffmpeg_utils import FFmpegComposer

    manifest_data = load_manifest(project_id)
    manifest_path = f"outputs/{project_id}/manifest.json"
    scenes = manifest_data.get("scenes", [])

    ffmpeg = FFmpegComposer()
    render_dir = f"outputs/{project_id}/media/rendered"
    os.makedirs(render_dir, exist_ok=True)

    # 에셋 경로 수집 (수정된 씬은 재렌더링)
    video_clips = []
    narration_clips = []

    for scene in scenes:
        assets = scene.get("assets", {})

        if scene.get("_narration_modified"):
            # 내레이션 수정된 씬: 이미지부터 재렌더링 (Ken Burns + 자막 burn-in)
            print(f"[RECOMPOSE] Scene {scene.get('scene_id')} narration modified - re-rendering with new subtitles")
            rendered = ffmpeg.render_scene(
                scene,
                {"ffmpeg_kenburns": True, "subtitle_burn_in": True},
                render_dir
            )
            video_clips.append(rendered)
            # 재렌더링된 비디오 경로를 assets에 업데이트
            assets["video_path"] = rendered
            scene.pop("_narration_modified", None)
        elif assets.get("video_path"):
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
            music_path=None,
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

# ============================================================
# Music Video Mode API Endpoints
# ============================================================

from schemas.mv_models import (
    MVProject, MVProjectRequest, MVUploadResponse,
    MVGenerateResponse, MVStatusResponse, MVResultResponse,
    MVProjectStatus
)

@app.post("/api/mv/upload", response_model=MVUploadResponse)
async def mv_upload_music(music_file: UploadFile = File(...), lyrics: str = Form(""), character_setup: str = Form("auto"), character_ethnicity: str = Form("auto")):
    """
    Step 1: 음악 파일 업로드 및 분석

    - 지원 포맷: mp3, wav, m4a, ogg, flac
    - 최대 길이: 10분
    - lyrics: 사용자 가사 (있으면 Gemini 추출 대신 타이밍만 싱크)
    """
    from agents.mv_pipeline import MVPipeline
    import uuid
    import shutil

    # 지원 포맷 확인 + 파일명 sanitize
    raw_filename = music_file.filename or "unknown.mp3"
    filename = os.path.basename(raw_filename)
    filename = _re.sub(r'[^\w\-.]', '_', filename)
    if not filename or filename.startswith('.'):
        filename = "upload.mp3"

    ext = os.path.splitext(filename)[1].lower()
    supported = ['.mp3', '.wav', '.m4a', '.ogg', '.flac']

    if ext not in supported:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported format: {ext}. Supported: {supported}"
        )

    # 파일 크기 제한 (50MB)
    content = await music_file.read()
    if len(content) > 50 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="File too large (max 50MB)")

    # 프로젝트 ID 생성
    project_id = f"mv_{uuid.uuid4().hex[:8]}"
    project_dir = f"outputs/{project_id}"
    os.makedirs(f"{project_dir}/music", exist_ok=True)

    # 파일 저장
    music_path = f"{project_dir}/music/{filename}"
    with open(music_path, "wb") as f:
        f.write(content)

    print(f"[MV API] Music uploaded: {music_path}".encode('ascii', 'replace').decode())

    # 분석
    try:
        pipeline = MVPipeline()
        user_lyrics = lyrics.strip() if lyrics else None
        if user_lyrics:
            print(f"[MV API] User provided lyrics: {len(user_lyrics)} chars")
        project = pipeline.upload_and_analyze(music_path, project_id, user_lyrics=user_lyrics)

        if project.status == MVProjectStatus.FAILED:
            raise HTTPException(status_code=500, detail=project.error_message)

        extracted_lyrics = project.music_analysis.extracted_lyrics if project.music_analysis else None

        return MVUploadResponse(
            project_id=project.project_id,
            status=project.status.value,
            music_analysis=project.music_analysis,
            extracted_lyrics=extracted_lyrics,
            message="음악 업로드 및 분석 완료"
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/mv/generate", response_model=MVGenerateResponse)
async def mv_generate(request: MVProjectRequest, background_tasks: BackgroundTasks):
    """
    Step 2: 뮤직비디오 생성 시작 (비동기)

    업로드된 음악을 기반으로 MV 생성을 시작합니다.
    """
    from agents.mv_pipeline import MVPipeline
    import threading

    if not request.project_id:
        raise HTTPException(status_code=400, detail="project_id is required")

    pipeline = MVPipeline()
    project = pipeline.load_project(request.project_id)

    if not project:
        raise HTTPException(status_code=404, detail=f"Project not found: {request.project_id}")

    if project.status == MVProjectStatus.GENERATING:
        raise HTTPException(status_code=400, detail="Generation already in progress")

    total_scenes = len(project.music_analysis.segments) if project.music_analysis else 6
    estimated_time = total_scenes * 30  # 씬당 약 30초

    def run_mv_generation():
        try:
            print(f"[MV Thread] Starting generation for {request.project_id}")
            cancel_path = f"outputs/{request.project_id}/.cancel"

            # Step 2: 씬 생성
            project_updated = pipeline.generate_scenes(project, request)

            if os.path.exists(cancel_path):
                print(f"[MV Thread] Cancelled before visual bible")
                return

            # Step 2.5: Visual Bible 생성 (Pass 1)
            project_updated = pipeline.generate_visual_bible(project_updated)

            if os.path.exists(cancel_path):
                print(f"[MV Thread] Cancelled before style anchor")
                return

            # Step 2.6: 전용 스타일 앵커 생성 (Pass 2)
            project_updated = pipeline.generate_style_anchor(project_updated)

            # Step 2.7: 캐릭터 앵커 생성 (Pass 2.5)
            project_updated = pipeline.generate_character_anchors(project_updated)

            if os.path.exists(cancel_path):
                print(f"[MV Thread] Cancelled before image generation")
                return

            # Step 3: 이미지 생성 (IMAGES_READY에서 멈춤)
            project_updated = pipeline.generate_images(project_updated)

            # compose_video()는 사용자 리뷰 후 /api/mv/compose에서 별도 호출
            print(f"[MV Thread] Images ready, waiting for user review: {project_updated.status}")

        except Exception as e:
            print(f"[MV Thread] Error: {e}")
            import traceback
            traceback.print_exc()

            # 프로젝트 상태를 FAILED로 설정하여 프론트엔드에 에러 전달
            try:
                project.status = MVProjectStatus.FAILED
                project.error_message = str(e)[:500]
                project.progress = 0
                project_dir = f"outputs/{request.project_id}"
                pipeline._save_manifest(project, project_dir)
            except Exception:
                pass

    # 백그라운드 스레드에서 실행
    thread = threading.Thread(target=run_mv_generation, daemon=True)
    thread.start()

    return MVGenerateResponse(
        project_id=request.project_id,
        status="generating",
        total_scenes=total_scenes,
        estimated_time_sec=estimated_time,
        message="뮤직비디오 생성이 시작되었습니다"
    )


@app.post("/api/mv/cancel/{project_id}")
async def mv_cancel(project_id: str):
    """MV 생성 중단"""
    validate_project_id(project_id)
    project_dir = f"outputs/{project_id}"
    if not os.path.exists(project_dir):
        raise HTTPException(status_code=404, detail=f"Project not found: {project_id}")

    cancel_path = os.path.join(project_dir, ".cancel")
    with open(cancel_path, "w") as f:
        f.write("cancelled")

    print(f"[MV] Cancel requested for {project_id}")
    return {"status": "cancel_requested", "project_id": project_id}


@app.get("/api/mv/status/{project_id}", response_model=MVStatusResponse)
async def mv_status(project_id: str):
    """
    MV 생성 상태 조회
    """
    validate_project_id(project_id)
    from agents.mv_pipeline import MVPipeline

    pipeline = MVPipeline()
    project = pipeline.load_project(project_id)

    if not project:
        raise HTTPException(status_code=404, detail=f"Project not found: {project_id}")

    return MVStatusResponse(
        project_id=project.project_id,
        status=project.status,
        progress=project.progress,
        current_step=project.current_step,
        scenes=project.scenes,
        error_message=project.error_message
    )


@app.get("/api/mv/result/{project_id}", response_model=MVResultResponse)
async def mv_result(project_id: str):
    """
    MV 결과 조회
    """
    validate_project_id(project_id)
    from agents.mv_pipeline import MVPipeline

    pipeline = MVPipeline()
    project = pipeline.load_project(project_id)

    if not project:
        raise HTTPException(status_code=404, detail=f"Project not found: {project_id}")

    video_url = None
    download_url = None

    if project.final_video_path:
        video_url = f"/api/mv/stream/{project_id}"
        download_url = f"/api/mv/download/{project_id}"

    return MVResultResponse(
        project_id=project.project_id,
        status=project.status,
        video_url=video_url,
        thumbnail_url=None,  # TODO: 썸네일 생성
        duration_sec=project.music_analysis.duration_sec if project.music_analysis else 0,
        scenes=project.scenes,
        download_url=download_url
    )


@app.post("/api/mv/scenes/{project_id}/{scene_id}/regenerate")
async def mv_regenerate_scene(project_id: str, scene_id: int):
    """
    MV 씬 이미지 재생성
    """
    validate_project_id(project_id)
    from agents.mv_pipeline import MVPipeline

    pipeline = MVPipeline()
    project = pipeline.load_project(project_id)

    if not project:
        raise HTTPException(status_code=404, detail=f"Project not found: {project_id}")

    if scene_id < 1 or scene_id > len(project.scenes):
        raise HTTPException(status_code=400, detail=f"Invalid scene_id: {scene_id}")

    try:
        scene = pipeline.regenerate_scene_image(project, scene_id)
        image_url = None
        if scene.image_path:
            # outputs/mv_xxx/media/images/scene_01.png -> /media/mv_xxx/media/images/scene_01.png
            if scene.image_path.startswith("outputs/"):
                image_url = f"/media/{scene.image_path[len('outputs/'):]}"
            else:
                image_url = scene.image_path
        return {"success": True, "image_path": scene.image_path, "image_url": image_url}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/mv/scenes/{project_id}/{scene_id}/i2v")
async def mv_scene_i2v(project_id: str, scene_id: int):
    """
    MV 씬 I2V (Veo 3.1 Image-to-Video) 변환
    """
    validate_project_id(project_id)
    from agents.mv_pipeline import MVPipeline

    pipeline = MVPipeline()
    project = pipeline.load_project(project_id)

    if not project:
        raise HTTPException(status_code=404, detail=f"Project not found: {project_id}")

    if scene_id < 1 or scene_id > len(project.scenes):
        raise HTTPException(status_code=400, detail=f"Invalid scene_id: {scene_id}")

    try:
        scene = pipeline.generate_scene_i2v(project, scene_id)
        return {"success": True, "video_path": scene.video_path}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/mv/compose/{project_id}")
async def mv_compose(project_id: str):
    """
    이미지 리뷰 승인 후 최종 뮤직비디오 합성
    """
    validate_project_id(project_id)
    from agents.mv_pipeline import MVPipeline
    import threading

    pipeline = MVPipeline()
    project = pipeline.load_project(project_id)

    if not project:
        raise HTTPException(status_code=404, detail=f"Project not found: {project_id}")

    if project.status == MVProjectStatus.COMPOSING:
        raise HTTPException(status_code=400, detail="Composition already in progress")

    def run_compose():
        try:
            print(f"[MV Compose Thread] Starting composition for {project_id}")
            pipeline.compose_video(project)
            print(f"[MV Compose Thread] Composition complete: {project.status}")

            # [Production] MV R2 업로드 (일반 파이프라인과 동일 패턴)
            try:
                from utils.storage import StorageManager
                import json

                storage = StorageManager()
                backend_url = "https://web-production-bb6bf.up.railway.app"
                project_dir = f"outputs/{project_id}"

                # 1. 최종 비디오 업로드
                if project.final_video_path and os.path.exists(project.final_video_path):
                    r2_key = f"videos/{project_id}/final_video.mp4"
                    if storage.upload_file(project.final_video_path, r2_key):
                        print(f"[MV R2] Video uploaded: {r2_key}")

                # 2. 씬 이미지 업로드 + 경로를 HTTP URL로 업데이트
                for scene in project.scenes:
                    if scene.image_path and os.path.exists(scene.image_path):
                        img_filename = os.path.basename(scene.image_path)
                        r2_key = f"images/{project_id}/{img_filename}"
                        if storage.upload_file(scene.image_path, r2_key):
                            scene.image_path = f"{backend_url}/api/asset/{project_id}/image/{img_filename}"

                # 3. 음악 파일 업로드 (리컴포즈 시 복원용)
                if project.music_file_path and os.path.exists(project.music_file_path):
                    music_filename = os.path.basename(project.music_file_path)
                    r2_key = f"music/{project_id}/{music_filename}"
                    if storage.upload_file(project.music_file_path, r2_key):
                        print(f"[MV R2] Music uploaded: {r2_key}")

                # 4. 매니페스트 저장 + R2 업로드
                pipeline._save_manifest(project, project_dir)
                manifest_path = f"{project_dir}/manifest.json"
                manifest_r2_key = f"videos/{project_id}/manifest.json"
                if storage.upload_file(manifest_path, manifest_r2_key):
                    print(f"[MV R2] Manifest uploaded: {manifest_r2_key}")

                print(f"[MV R2] All assets uploaded for {project_id}")
            except Exception as e:
                print(f"[MV R2] Upload error (non-fatal): {e}")

        except Exception as e:
            print(f"[MV Compose Thread] Error: {e}")
            import traceback
            traceback.print_exc()
            try:
                project.status = MVProjectStatus.FAILED
                project.error_message = str(e)[:500]
                project_dir = f"outputs/{project_id}"
                pipeline._save_manifest(project, project_dir)
            except Exception:
                pass

    thread = threading.Thread(target=run_compose, daemon=True)
    thread.start()

    return {"status": "composing", "project_id": project_id}


class UpdateLyricsTimelineRequest(BaseModel):
    timed_lyrics: List[Dict[str, Any]]  # [{t: float, text: str}, ...]

@app.get("/api/mv/{project_id}/lyrics-timeline")
async def get_mv_lyrics_timeline(project_id: str):
    """MV 가사 타이밍 에디터용 데이터 반환 (STT 원문 + 정렬 가사)"""
    validate_project_id(project_id)
    from agents.mv_pipeline import MVPipeline

    pipeline = MVPipeline()
    project = pipeline.load_project(project_id)

    if not project:
        raise HTTPException(status_code=404, detail=f"Project not found: {project_id}")

    stt_sentences = None
    timed_lyrics = None
    duration_sec = 0.0

    if project.music_analysis:
        stt_sentences = project.music_analysis.stt_sentences
        timed_lyrics = project.music_analysis.timed_lyrics
        duration_sec = project.music_analysis.duration_sec

    # edited_timed_lyrics가 있으면 has_edits=True
    has_edits = bool(project.edited_timed_lyrics)

    return {
        "stt_sentences": stt_sentences or [],
        "timed_lyrics": project.edited_timed_lyrics or timed_lyrics or [],
        "duration_sec": duration_sec,
        "has_edits": has_edits,
    }

@app.put("/api/mv/{project_id}/lyrics-timeline")
async def update_mv_lyrics_timeline(project_id: str, req: UpdateLyricsTimelineRequest):
    """MV 가사 타이밍 에디터 저장 → edited_timed_lyrics에 저장 + SRT 재생성"""
    validate_project_id(project_id)
    from agents.mv_pipeline import MVPipeline

    pipeline = MVPipeline()
    project = pipeline.load_project(project_id)

    if not project:
        raise HTTPException(status_code=404, detail=f"Project not found: {project_id}")

    # 편집 결과 저장
    project.edited_timed_lyrics = req.timed_lyrics
    print(f"[Lyrics Timeline] Saved {len(req.timed_lyrics)} edited entries for {project_id}")

    # SRT 재생성
    project_dir = f"outputs/{project_id}"
    srt_path = f"{project_dir}/media/subtitles/lyrics.srt"
    os.makedirs(os.path.dirname(srt_path), exist_ok=True)

    pipeline._generate_lyrics_srt(project.scenes, srt_path, timed_lyrics=req.timed_lyrics)

    # 매니페스트 저장
    pipeline._save_manifest(project, project_dir)
    try:
        from utils.storage import StorageManager
        storage = StorageManager()
        manifest_file = f"{project_dir}/manifest.json"
        if os.path.exists(manifest_file):
            storage.upload_file(manifest_file, f"videos/{project_id}/manifest.json")
    except Exception as e:
        print(f"[Lyrics Timeline] R2 manifest upload error (non-fatal): {e}")

    return {"success": True, "entries": len(req.timed_lyrics)}


class UpdateLyricsRequest(BaseModel):
    lyrics: str

@app.put("/api/mv/{project_id}/scenes/{scene_id}/lyrics")
async def update_mv_scene_lyrics(project_id: str, scene_id: int, req: UpdateLyricsRequest):
    """MV 씬 가사 텍스트 수정 + SRT 자막 재생성"""
    validate_project_id(project_id)
    from agents.mv_pipeline import MVPipeline

    pipeline = MVPipeline()
    project = pipeline.load_project(project_id)

    if not project:
        raise HTTPException(status_code=404, detail=f"Project not found: {project_id}")

    if scene_id < 1 or scene_id > len(project.scenes):
        raise HTTPException(status_code=400, detail=f"Invalid scene_id: {scene_id}")

    # 1) 가사 텍스트 업데이트 + lyrics_modified 플래그 설정
    scene = project.scenes[scene_id - 1]
    scene.lyrics_text = req.lyrics
    scene.lyrics_modified = True

    # 2) SRT 재생성 (하이브리드: 수정된 씬은 균등 분배, 나머지는 timed_lyrics 유지)
    timed_lyrics = None
    if project.music_analysis and project.music_analysis.timed_lyrics:
        timed_lyrics = project.music_analysis.timed_lyrics

    project_dir = f"outputs/{project_id}"
    srt_path = f"{project_dir}/media/subtitles/lyrics.srt"
    os.makedirs(os.path.dirname(srt_path), exist_ok=True)

    pipeline._generate_lyrics_srt(project.scenes, srt_path, timed_lyrics=timed_lyrics)

    # 3) 매니페스트 저장 (로컬 + R2)
    pipeline._save_manifest(project, project_dir)
    try:
        from utils.storage import StorageManager
        storage = StorageManager()
        manifest_file = f"{project_dir}/manifest.json"
        if os.path.exists(manifest_file):
            storage.upload_file(manifest_file, f"videos/{project_id}/manifest.json")
    except Exception as e:
        print(f"[MV Lyrics] R2 manifest upload error (non-fatal): {e}")

    return {"success": True, "scene_id": scene_id, "lyrics": req.lyrics}


@app.post("/api/mv/{project_id}/upload-music")
async def mv_upload_music_for_recompose(project_id: str, music_file: UploadFile = File(...)):
    """기존 MV 프로젝트에 음악 파일 재업로드 (리컴포즈용)"""
    validate_project_id(project_id)
    from agents.mv_pipeline import MVPipeline

    pipeline = MVPipeline()
    project = pipeline.load_project(project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    # 파일명 sanitize
    raw_filename = music_file.filename or "music.mp3"
    filename = os.path.basename(raw_filename)
    filename = _re.sub(r'[^\w\-.]', '_', filename)
    if not filename or filename.startswith('.'):
        filename = "music.mp3"

    content = await music_file.read()
    if len(content) > 50 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="File too large (max 50MB)")

    # 로컬 저장
    project_dir = f"outputs/{project_id}"
    music_dir = f"{project_dir}/music"
    os.makedirs(music_dir, exist_ok=True)
    music_path = f"{music_dir}/{filename}"
    with open(music_path, "wb") as f:
        f.write(content)

    # 매니페스트 업데이트
    project.music_file_path = music_path
    pipeline._save_manifest(project, project_dir)

    # R2에도 업로드
    try:
        from utils.storage import StorageManager
        storage = StorageManager()
        r2_key = f"music/{project_id}/{filename}"
        storage.upload_file(music_path, r2_key)
        # 매니페스트도 R2 동기화
        manifest_file = f"{project_dir}/manifest.json"
        storage.upload_file(manifest_file, f"videos/{project_id}/manifest.json")
    except Exception as e:
        print(f"[MV Music Upload] R2 error (non-fatal): {e}")

    return {"success": True, "music_path": music_path, "filename": filename}


@app.get("/api/mv/{project_id}/debug")
async def mv_debug(project_id: str):
    """MV 프로젝트 디버그 정보 (R2 에셋 존재 여부 확인)"""
    validate_project_id(project_id)
    from agents.mv_pipeline import MVPipeline
    from utils.storage import StorageManager

    pipeline = MVPipeline()
    project = pipeline.load_project(project_id)
    storage = StorageManager()

    if not project:
        return {"error": "Project not found"}

    info = {
        "project_id": project_id,
        "status": str(project.status),
        "scene_count": len(project.scenes),
        "music_file_path": project.music_file_path,
        "music_exists_local": os.path.exists(project.music_file_path) if project.music_file_path else False,
        "final_video_path": project.final_video_path,
        "scenes": [],
        "r2_checks": {}
    }

    # 씬별 상태
    for s in project.scenes:
        local_exists = os.path.exists(s.image_path) if s.image_path else False
        info["scenes"].append({
            "scene_id": s.scene_id,
            "status": str(s.status),
            "image_path": s.image_path,
            "image_exists_local": local_exists,
            "is_http": bool(s.image_path and s.image_path.startswith("http")),
        })

    # R2 에셋 확인
    for key_name, r2_key in [
        ("manifest", f"videos/{project_id}/manifest.json"),
        ("image_01", f"images/{project_id}/scene_01.png"),
        ("image_02", f"images/{project_id}/scene_02.png"),
    ]:
        data = storage.get_object(r2_key)
        info["r2_checks"][key_name] = {"key": r2_key, "exists": data is not None, "size": len(data) if data else 0}

    if project.music_file_path:
        music_fn = os.path.basename(project.music_file_path)
        for prefix in ["music", "uploads", "videos"]:
            r2_key = f"{prefix}/{project_id}/{music_fn}"
            data = storage.get_object(r2_key)
            info["r2_checks"][f"music_{prefix}"] = {"key": r2_key, "exists": data is not None, "size": len(data) if data else 0}

    return info


@app.post("/api/mv/{project_id}/recompose")
async def mv_recompose(project_id: str):
    """MV 리컴포즈 - 가사/이미지 수정 후 최종 영상 재합성"""
    validate_project_id(project_id)
    from agents.mv_pipeline import MVPipeline
    import threading

    pipeline = MVPipeline()
    project = pipeline.load_project(project_id)

    if not project:
        raise HTTPException(status_code=404, detail=f"Project not found: {project_id}")

    if project.status == MVProjectStatus.COMPOSING:
        raise HTTPException(status_code=400, detail="Composition already in progress")

    def run_recompose():
        try:
            print(f"[MV Recompose Thread] Starting recompose for {project_id}")
            from utils.storage import StorageManager
            storage = StorageManager()

            project_dir = f"outputs/{project_id}"
            os.makedirs(f"{project_dir}/media/images", exist_ok=True)
            os.makedirs(f"{project_dir}/media/video", exist_ok=True)
            os.makedirs(f"{project_dir}/media/subtitles", exist_ok=True)

            # R2에서 씬 이미지 복원
            restored = 0
            for scene in project.scenes:
                local_img = f"{project_dir}/media/images/scene_{scene.scene_id:02d}.png"

                if os.path.exists(local_img):
                    scene.image_path = local_img
                    restored += 1
                    continue

                # R2에서 직접 다운로드
                img_filename = f"scene_{scene.scene_id:02d}.png"
                r2_data = storage.get_object(f"images/{project_id}/{img_filename}")
                if r2_data:
                    with open(local_img, "wb") as f:
                        f.write(r2_data)
                    scene.image_path = local_img
                    restored += 1
                    print(f"[MV Recompose] Restored from R2: {img_filename}")
                elif scene.image_path and scene.image_path.startswith("http"):
                    # R2 직접 실패 시 HTTP fallback
                    try:
                        import requests as http_req
                        resp = http_req.get(scene.image_path, timeout=30)
                        if resp.status_code == 200:
                            with open(local_img, "wb") as f:
                                f.write(resp.content)
                            scene.image_path = local_img
                            restored += 1
                            print(f"[MV Recompose] Restored via HTTP: {img_filename}")
                    except Exception as dl_err:
                        print(f"[MV Recompose] HTTP download also failed: {dl_err}")
                else:
                    print(f"[MV Recompose] Cannot restore scene {scene.scene_id} image")

            print(f"[MV Recompose] Images restored: {restored}/{len(project.scenes)}")

            # 복원 결과 로그
            for scene in project.scenes:
                exists = os.path.exists(scene.image_path) if scene.image_path else False
                print(f"[MV Recompose] Scene {scene.scene_id}: status={scene.status}, image={scene.image_path}, exists={exists}")

            # 음악 파일 복원
            music_found = project.music_file_path and os.path.exists(project.music_file_path)

            # 1) 로컬 music 디렉토리에서 아무 음악 파일 탐색
            if not music_found:
                music_dir = f"{project_dir}/music"
                if os.path.exists(music_dir):
                    for fn in os.listdir(music_dir):
                        if fn.lower().endswith(('.mp3', '.wav', '.m4a', '.ogg', '.flac')):
                            project.music_file_path = f"{music_dir}/{fn}"
                            music_found = True
                            print(f"[MV Recompose] Found local music: {fn}")
                            break

            # 2) R2에서 복원
            if not music_found and project.music_file_path:
                music_filename = os.path.basename(project.music_file_path)
                for r2_key in [
                    f"music/{project_id}/{music_filename}",
                    f"uploads/{project_id}/{music_filename}",
                    f"videos/{project_id}/{music_filename}",
                ]:
                    data = storage.get_object(r2_key)
                    if data:
                        os.makedirs(os.path.dirname(project.music_file_path) or ".", exist_ok=True)
                        with open(project.music_file_path, "wb") as f:
                            f.write(data)
                        print(f"[MV Recompose] Restored music from R2: {r2_key}")
                        music_found = True
                        break

            if not music_found:
                print(f"[MV Recompose] WARNING: No music file available")

            print(f"[MV Recompose] Music file: {project.music_file_path}, exists={os.path.exists(project.music_file_path) if project.music_file_path else False}")

            # 하이브리드 SRT: lyrics_modified=True인 씬만 균등 분배, 나머지는 timed_lyrics 유지
            modified_count = sum(1 for s in project.scenes if getattr(s, 'lyrics_modified', False))
            if modified_count > 0:
                print(f"[MV Recompose] {modified_count} scene(s) have modified lyrics - hybrid SRT will be used")

            pipeline.compose_video(project)
            print(f"[MV Recompose Thread] Recompose complete: {project.status}")

            # R2 업로드 (기존 compose 로직 재사용)
            try:
                from utils.storage import StorageManager

                storage = StorageManager()
                backend_url = "https://web-production-bb6bf.up.railway.app"
                project_dir = f"outputs/{project_id}"

                if project.final_video_path and os.path.exists(project.final_video_path):
                    r2_key = f"videos/{project_id}/final_video.mp4"
                    if storage.upload_file(project.final_video_path, r2_key):
                        print(f"[MV R2] Video uploaded: {r2_key}")

                for scene in project.scenes:
                    if scene.image_path and os.path.exists(scene.image_path):
                        img_filename = os.path.basename(scene.image_path)
                        r2_key = f"images/{project_id}/{img_filename}"
                        if storage.upload_file(scene.image_path, r2_key):
                            scene.image_path = f"{backend_url}/api/asset/{project_id}/image/{img_filename}"

                pipeline._save_manifest(project, project_dir)
                manifest_path = f"{project_dir}/manifest.json"
                manifest_r2_key = f"videos/{project_id}/manifest.json"
                if storage.upload_file(manifest_path, manifest_r2_key):
                    print(f"[MV R2] Manifest uploaded: {manifest_r2_key}")

                print(f"[MV R2] All assets uploaded for {project_id}")
            except Exception as e:
                print(f"[MV R2] Upload error (non-fatal): {e}")

        except Exception as e:
            print(f"[MV Recompose Thread] Error: {e}")
            import traceback
            traceback.print_exc()
            try:
                project.status = MVProjectStatus.FAILED
                project.error_message = str(e)[:500]
                project_dir = f"outputs/{project_id}"
                pipeline._save_manifest(project, project_dir)
            except Exception:
                pass

    thread = threading.Thread(target=run_recompose, daemon=True)
    thread.start()

    return {"status": "composing", "project_id": project_id}


@app.get("/api/mv/stream/{project_id}")
async def mv_stream(project_id: str):
    """MV 스트리밍 (로컬 우선, R2 fallback)"""
    validate_project_id(project_id)
    from fastapi.responses import FileResponse, Response

    # 1. 로컬 파일 확인
    possible_paths = [
        f"outputs/{project_id}/final_mv.mp4",
        f"outputs/{project_id}/final_video.mp4",
    ]
    outputs_base = os.path.realpath("outputs")

    for path in possible_paths:
        resolved = os.path.realpath(path)
        if resolved.startswith(outputs_base) and os.path.exists(path):
            return FileResponse(path, media_type="video/mp4", filename=f"mv_{project_id}.mp4")

    # 2. R2 fallback
    try:
        from utils.storage import StorageManager
        storage = StorageManager()
        data = storage.get_object(f"videos/{project_id}/final_video.mp4")
        if data:
            return Response(content=data, media_type="video/mp4")
    except Exception as e:
        print(f"[MV Stream] R2 fallback error: {e}")

    raise HTTPException(status_code=404, detail="Video not found")


@app.get("/api/mv/download/{project_id}")
async def mv_download(project_id: str):
    """MV 다운로드 (로컬 우선, R2 fallback)"""
    validate_project_id(project_id)
    from fastapi.responses import FileResponse, Response

    # 1. 로컬 파일 확인
    possible_paths = [
        f"outputs/{project_id}/final_mv.mp4",
        f"outputs/{project_id}/final_video.mp4",
    ]
    outputs_base = os.path.realpath("outputs")

    for path in possible_paths:
        resolved = os.path.realpath(path)
        if resolved.startswith(outputs_base) and os.path.exists(path):
            return FileResponse(
                path, media_type="video/mp4",
                filename=f"storycut_mv_{project_id}.mp4",
                headers={"Content-Disposition": f"attachment; filename=storycut_mv_{project_id}.mp4"}
            )

    # 2. R2 fallback
    try:
        from utils.storage import StorageManager
        storage = StorageManager()
        data = storage.get_object(f"videos/{project_id}/final_video.mp4")
        if data:
            return Response(
                content=data, media_type="video/mp4",
                headers={"Content-Disposition": f"attachment; filename=storycut_mv_{project_id}.mp4"}
            )
    except Exception as e:
        print(f"[MV Download] R2 fallback error: {e}")

    raise HTTPException(status_code=404, detail="Video not found")


@app.on_event("startup")
async def on_startup():
    from utils.cleanup import start_cleanup_scheduler
    start_cleanup_scheduler()
    print("[STARTUP] Cleanup scheduler registered (daily 03:00)")


if __name__ == "__main__":
    import uvicorn

    print("""
============================================================
              STORYCUT API Server v2.6 (Music Video Mode)
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
