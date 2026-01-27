"""
STORYCUT FastAPI Server

웹 UI와 Python 파이프라인을 연결하는 API 서버
WebSocket으로 실시간 진행상황 전달
"""

import os
import sys
import json
import asyncio
import aiohttp
from dotenv import load_dotenv

# .env 파일 로드
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
print(f"DEBUG: Loaded OPENAI_API_KEY: {api_key[:10]}..." if api_key else "DEBUG: OPENAI_API_KEY is None")

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
from schemas import FeatureFlags, ProjectRequest, TargetPlatform
from pipeline import StorycutPipeline

# FastAPI 앱 생성
app = FastAPI(title="STORYCUT API", version="2.0")

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 정적 파일 서빙
app.mount("/static", StaticFiles(directory="web/static"), name="static")
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


# ============================================================================
# Pydantic 모델
# ============================================================================

class GenerateRequest(BaseModel):
    """영상 생성 요청"""
    topic: Optional[str] = None
    genre: str = "emotional"
    mood: str = "dramatic"
    style: str = "cinematic, high contrast"
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


@app.post("/api/generate")
async def generate_video(req: GenerateRequest, background_tasks: BackgroundTasks):
    """영상 생성 시작"""
    import uuid

    # Worker에서 project_id를 보냈으면 사용, 없으면 생성 (로컬 테스트용)
    project_id = req.project_id or str(uuid.uuid4())[:8]

    # ProjectRequest 생성
    feature_flags = FeatureFlags(
        hook_scene1_video=req.hook_scene1_video,
        ffmpeg_kenburns=req.ffmpeg_kenburns,
        ffmpeg_audio_ducking=req.ffmpeg_audio_ducking,
        subtitle_burn_in=req.subtitle_burn_in,
        context_carry_over=req.context_carry_over,
        optimization_pack=req.optimization_pack,
    )

    platform = TargetPlatform.YOUTUBE_SHORTS if req.platform == "youtube_shorts" else TargetPlatform.YOUTUBE_LONG

    request = ProjectRequest(
        topic=req.topic,
        genre=req.genre,
        mood=req.mood,
        style_preset=req.style,
        duration_target_sec=req.duration,
        target_platform=platform,
        voice_over=True,
        bgm=True,
        subtitles=req.subtitle_burn_in,
        feature_flags=feature_flags,
    )

    # 비동기 작업 시작
    tracker = ProgressTracker(project_id)
    pipeline = TrackedPipeline(tracker, webhook_url=req.webhook_url)

    # FastAPI BackgroundTasks 사용 (더 안정적)
    background_tasks.add_task(run_pipeline_wrapper, pipeline, request)

    return {
        "project_id": project_id,
        "status": "started",
        "message": "영상 생성이 시작되었습니다. WebSocket으로 진행상황을 확인하세요.",
        "ws_url": f"/ws/{project_id}"
    }


@app.get("/api/download/{project_id}")
async def download_video(project_id: str):
    """생성된 영상 다운로드"""
    video_path = f"outputs/{project_id}/final_video.mp4"

    if not os.path.exists(video_path):
        raise HTTPException(status_code=404, detail="영상을 찾을 수 없습니다.")

    return FileResponse(
        video_path,
        media_type="video/mp4",
        filename=f"storycut_{project_id}.mp4"
    )


@app.get("/api/manifest/{project_id}")
async def get_manifest(project_id: str):
    """Manifest 조회"""
    manifest_path = f"outputs/{project_id}/manifest.json"

    if not os.path.exists(manifest_path):
        raise HTTPException(status_code=404, detail="프로젝트를 찾을 수 없습니다.")

    with open(manifest_path, "r", encoding="utf-8") as f:
        return json.load(f)


@app.get("/health")
async def health_check():
    """헬스 체크"""
    return {
        "status": "ok",
        "version": "2.0",
        "active_connections": len(active_connections)
    }


if __name__ == "__main__":
    import uvicorn

    print("""
============================================================
              STORYCUT API Server v2.0
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
