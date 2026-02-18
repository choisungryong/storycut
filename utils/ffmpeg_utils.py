"""
FFmpeg Utilities for Video Composition

P0 핵심 기능:
- Ken Burns Effect: 이미지 기반 씬을 영상처럼 변환
- Audio Ducking: 내레이션 시 BGM 자동 감쇠
- Subtitle Burn-in: 자막을 영상에 직접 렌더링

v2.0 추가 기능:
- Film Look: 필름 그레인 + 색보정 후처리
"""

import subprocess
import os
import re
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any
import sys

sys.path.append(str(Path(__file__).parent.parent))


# [보안] FFmpeg 필터 파라미터 검증 유틸리티
def _sanitize_ffmpeg_number(value, default=0.0, min_val=None, max_val=None):
    """FFmpeg 필터에 삽입될 숫자 값 검증 — 인젝션 방지"""
    try:
        num = float(value)
        if min_val is not None:
            num = max(num, min_val)
        if max_val is not None:
            num = min(num, max_val)
        return num
    except (TypeError, ValueError):
        return default


def _sanitize_ffmpeg_style_value(value: str) -> str:
    """FFmpeg force_style 내 개별 값 검증 — 세미콜론/따옴표 주입 방지"""
    if not isinstance(value, str):
        return str(value)
    # 세미콜론, 따옴표, 줄바꿈 등 FFmpeg 필터 구분자 제거
    return re.sub(r"[;'\"\\`\n\r]", "", value)


class FFmpegComposer:
    """
    FFmpeg 기반 비디오 합성 엔진

    P0 기능:
    - Ken Burns Effect (이미지 모션)
    - Audio Ducking (BGM 자동 감쇠)
    - Subtitle Burn-in (자막 렌더링)
    """

    def __init__(self, resolution: str = "1920x1080", fps: int = 30):
        self.resolution = resolution
        self.fps = fps
        self.width, self.height = map(int, resolution.split("x"))

    # =========================================================================
    # P0: Ken Burns Effect
    # =========================================================================

    def _dynamic_timeout(self, video_path: str, factor: float = 3.0, minimum: int = 120) -> int:
        """
        영상 길이 기반 동적 timeout 계산.
        factor: 영상 1초당 허용할 인코딩 시간 (초)
        minimum: 최소 timeout (초)
        """
        try:
            duration = self.get_video_duration(video_path)
            return max(minimum, int(duration * factor))
        except Exception:
            return minimum

    def ken_burns_clip(
        self,
        image_path: str,
        duration_sec: float,
        out_path: str,
        effect_type: str = "zoom_in",
        zoom_range: Tuple[float, float] = (1.0, 1.1),  # Minimal zoom for calm vibe
        hflip: bool = False
    ) -> str:
        """
        이미지에 Ken Burns (줌/팬) 효과를 적용하여 영상 클립 생성.

        P0: 이미지 기반 씬은 무조건 모션 처리

        Args:
            image_path: 입력 이미지 경로
            duration_sec: 영상 길이 (초)
            out_path: 출력 영상 경로
            effect_type: 효과 유형 (zoom_in, zoom_out, pan_left, pan_right, diagonal)
            zoom_range: 줌 범위 (최소, 최대)
            hflip: 좌우 반전 여부 (파생 씬용)

        Returns:
            출력 영상 경로
        """
        total_frames = int(duration_sec * self.fps)
        zoom_min, zoom_max = zoom_range

        # 효과별 zoompan 필터 설정
        effects = {
            "zoom_in": (
                f"zoompan=z='min(zoom+{(zoom_max-zoom_min)/total_frames},{zoom_max})':"
                f"x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':"
                f"d={total_frames}:s={self.resolution}:fps={self.fps}"
            ),
            "zoom_out": (
                f"zoompan=z='if(lte(zoom,{zoom_min}),{zoom_max},max({zoom_min},zoom-{(zoom_max-zoom_min)/total_frames}))':"
                f"x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':"
                f"d={total_frames}:s={self.resolution}:fps={self.fps}"
            ),
            "pan_left": (
                f"zoompan=z='{zoom_min + 0.1}':"
                f"x='if(lte(on,1),0,min(iw/zoom-iw,x+{self.width/(total_frames*2)}))':"
                f"y='ih/2-(ih/zoom/2)':"
                f"d={total_frames}:s={self.resolution}:fps={self.fps}"
            ),
            "pan_right": (
                f"zoompan=z='{zoom_min + 0.1}':"
                f"x='if(lte(on,1),iw/zoom-iw,max(0,x-{self.width/(total_frames*2)}))':"
                f"y='ih/2-(ih/zoom/2)':"
                f"d={total_frames}:s={self.resolution}:fps={self.fps}"
            ),
            "diagonal": (
                f"zoompan=z='min(zoom+{(zoom_max-zoom_min)/total_frames},{zoom_max-0.05})':"
                f"x='if(lte(on,1),0,min(iw/zoom-iw,x+0.5))':"
                f"y='if(lte(on,1),0,min(ih/zoom-ih,y+0.3))':"
                f"d={total_frames}:s={self.resolution}:fps={self.fps}"
            ),
        }

        # 파생 효과 해석: crop_left/crop_right는 크롭 위치 + 모션으로 매핑
        scaled_w = self.width * 2
        scaled_h = self.height * 2
        crop_x = 0  # default: center
        if effect_type == "crop_left":
            crop_x = 0
            effect_type = "pan_right"  # 왼쪽 영역에서 오른쪽으로 패닝
        elif effect_type == "crop_right":
            crop_x = scaled_w // 3
            effect_type = "pan_left"  # 오른쪽 영역에서 왼쪽으로 패닝

        filter_str = effects.get(effect_type, effects["zoom_in"])

        # v2.2: Pre-scale image to correct aspect ratio (16:9) at larger size for zoom headroom
        prescale_filter = f"scale={scaled_w}:{scaled_h}:force_original_aspect_ratio=increase,crop={scaled_w}:{scaled_h}:{crop_x}:0"
        if hflip:
            prescale_filter += ",hflip"
        full_filter = f"{prescale_filter},{filter_str}"

        cmd = [
            "ffmpeg",
            "-y",
            "-loop", "1",
            "-i", image_path,
            "-vf", full_filter,
            "-c:v", "libx264",
            "-preset", "ultrafast",
            "-t", str(duration_sec),
            "-pix_fmt", "yuv420p",
            "-r", str(self.fps),
            out_path
        ]

        result = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8', errors='replace', timeout=240)

        if result.returncode != 0:
            raise RuntimeError(f"Ken Burns effect failed: {result.stderr}")

        return out_path

    # =========================================================================
    # Derived Cut Clip (리프레이밍 + 모션 변형)
    # =========================================================================

    # shot_type별 줌 스케일 범위 (Quality_upgrade2.md Rule 2)
    _SHOT_SCALE = {
        "wide":   (1.00, 1.05),
        "medium": (1.05, 1.12),
        "close":  (1.10, 1.18),
        "detail": (1.10, 1.30),
    }
    _MAX_DELTA_SCALE = 0.06  # Rule 5: 한 컷 내 최대 줌 변화량

    def derived_cut_clip(
        self,
        image_path: str,
        duration_sec: float,
        out_path: str,
        reframe: str = "wide",
        crop_anchor: Tuple[float, float] = (0.5, 0.5),
        effect_type: str = "zoom_in",
        zoom_range: Tuple[float, float] = (1.0, 1.08),
    ) -> str:
        """
        이미지에 리프레이밍(crop) + easeInOutSine 모션을 결합한 파생 컷 생성.

        모션은 easeInOutSine 커브로 시작/끝이 완만하고 중간이 빠른 자연스러운 움직임.
        Δscale <= 0.06 강제, 패닝 제한 max_dx=3% max_dy=2%.

        Args:
            image_path: 입력 이미지 경로
            duration_sec: 컷 길이 (초)
            out_path: 출력 영상 경로
            reframe: 리프레이밍 레벨 (wide|medium|close|detail)
            crop_anchor: 크롭 중심점 (0~1 정규화 x, y)
            effect_type: 모션 효과 (zoom_in, zoom_out, pan_left, pan_right, diagonal)
            zoom_range: 줌 범위 (최소, 최대) - Δscale 자동 클램핑

        Returns:
            출력 영상 경로
        """
        # 프리스케일 해상도 (줌 헤드룸 확보)
        prescale_w = 3840
        prescale_h = 2160

        # 리프레이밍별 크롭 영역
        reframe_ratios = {
            "wide": 1.0,
            "medium": 0.75,
            "close": 0.5,
            "detail": 0.33,
        }
        ratio = reframe_ratios.get(reframe, 1.0)
        crop_w = int(prescale_w * ratio)
        crop_h = int(prescale_h * ratio)

        # crop_anchor 기반 크롭 위치 계산 (경계 클램핑)
        anchor_x, anchor_y = crop_anchor
        crop_x = max(0, min(int(anchor_x * prescale_w - crop_w / 2), prescale_w - crop_w))
        crop_y = max(0, min(int(anchor_y * prescale_h - crop_h / 2), prescale_h - crop_h))

        total_frames = int(duration_sec * self.fps)
        if total_frames < 1:
            total_frames = 1

        # Δscale 클램핑 (Rule 5: max 0.06, Rule 5b: max 0.02/sec)
        zoom_min, zoom_max = zoom_range
        max_delta_for_duration = 0.02 * duration_sec
        delta = min(zoom_max - zoom_min, self._MAX_DELTA_SCALE, max_delta_for_duration)
        zoom_max = zoom_min + delta

        # shot_type별 스케일 범위 검증 (Rule 2)
        scale_min, scale_max = self._SHOT_SCALE.get(reframe, (1.0, 1.30))
        zoom_min = max(zoom_min, scale_min)
        zoom_max = min(zoom_max, scale_max)
        if zoom_max <= zoom_min:
            zoom_max = zoom_min + min(delta, 0.04)

        N = total_frames
        # easeInOutSine: (1 - cos(PI * t)) / 2, t = on/N
        ease = f"(1-cos(on/{N}*3.14159265))/2"

        # 패닝 제한 (Rule 4): max_dx=3%, max_dy=2%, hero/closeup은 dy=1%
        max_dx = 0.03
        max_dy = 0.01 if reframe in ("close", "detail") else 0.02
        # zoompan 입력은 upscaled crop (self.width*2 x self.height*2)
        zp_w = self.width * 2
        zp_h = self.height * 2
        pan_px_x = max_dx * zp_w
        pan_px_y = max_dy * zp_h

        # 정적 줌 (패닝 이펙트용): 약한 줌으로 시각적 깊이감
        static_z = zoom_min + min(delta * 0.5, 0.03)

        # 중앙 좌표 표현식
        cx = "iw/2-iw/zoom/2"
        cy = "ih/2-ih/zoom/2"

        # 효과별 zoompan 필터 (easeInOutSine 적용)
        effects = {
            "zoom_in": (
                f"zoompan=z='{zoom_min}+{delta}*{ease}':"
                f"x='{cx}':y='{cy}':"
                f"d={N}:s={self.resolution}:fps={self.fps}"
            ),
            "zoom_out": (
                f"zoompan=z='{zoom_max}-{delta}*{ease}':"
                f"x='{cx}':y='{cy}':"
                f"d={N}:s={self.resolution}:fps={self.fps}"
            ),
            "pan_left": (
                f"zoompan=z='{static_z}':"
                f"x='max(0,min(iw-iw/zoom,{cx}+{pan_px_x:.1f}*{ease}))':"
                f"y='{cy}':"
                f"d={N}:s={self.resolution}:fps={self.fps}"
            ),
            "pan_right": (
                f"zoompan=z='{static_z}':"
                f"x='max(0,min(iw-iw/zoom,{cx}-{pan_px_x:.1f}*{ease}))':"
                f"y='{cy}':"
                f"d={N}:s={self.resolution}:fps={self.fps}"
            ),
            "diagonal": (
                f"zoompan=z='{zoom_min}+{delta}*{ease}':"
                f"x='max(0,min(iw-iw/zoom,{cx}+{pan_px_x * 0.7:.1f}*{ease}))':"
                f"y='max(0,min(ih-ih/zoom,{cy}+{pan_px_y * 0.7:.1f}*{ease}))':"
                f"d={N}:s={self.resolution}:fps={self.fps}"
            ),
        }

        filter_str = effects.get(effect_type, effects["zoom_in"])

        # 필터 체인: prescale → crop → zoompan
        full_filter = (
            f"scale={prescale_w}:{prescale_h}:force_original_aspect_ratio=increase,"
            f"crop={prescale_w}:{prescale_h},"
            f"crop={crop_w}:{crop_h}:{crop_x}:{crop_y},"
            f"scale={self.width * 2}:{self.height * 2},"
            f"{filter_str}"
        )

        cmd = [
            "ffmpeg",
            "-y",
            "-loop", "1",
            "-i", image_path,
            "-vf", full_filter,
            "-c:v", "libx264",
            "-preset", "ultrafast",
            "-t", str(duration_sec),
            "-pix_fmt", "yuv420p",
            "-r", str(self.fps),
            out_path
        ]

        result = subprocess.run(
            cmd, capture_output=True, text=True,
            encoding='utf-8', errors='replace', timeout=60
        )

        if result.returncode != 0:
            raise RuntimeError(f"Derived cut clip failed: {result.stderr[-300:]}")

        return out_path

    def apply_brightness_ramp(
        self,
        clip_path: str,
        out_path: str,
        brightness_offset: float,
        ramp_frames: int = 6,
    ) -> Optional[str]:
        """씬 경계에서 밝기 점프를 완화하기 위해 클립 시작 N프레임에 밝기 램프 적용.

        이전 씬과 다음 씬의 밝기 차이가 클 때, 다음 씬 첫 클립의 시작 부분을
        이전 씬 밝기에 맞추고 점진적으로 원래 밝기로 돌아오게 합니다.

        Args:
            clip_path: 입력 클립 경로
            out_path: 출력 클립 경로
            brightness_offset: 밝기 오프셋 (-1.0~1.0). 양수=밝게, 음수=어둡게
            ramp_frames: 램프 프레임 수 (기본 6 = 0.2초@30fps)

        Returns:
            출력 경로 or None (실패 시)
        """
        # 오프셋이 너무 작으면 스킵
        if abs(brightness_offset) < 0.02:
            return None

        # 클램핑: 과보정 방지
        brightness_offset = max(-0.15, min(0.15, brightness_offset))

        # FFmpeg eq 필터 표현식: 첫 ramp_frames 동안 offset→0 선형 감소
        # if() 안의 쉼표는 FFmpeg 필터 그래프에서 \, 이스케이핑 필요
        expr = (
            f"if(lt(n\\,{ramp_frames})\\,"
            f"{brightness_offset:.4f}*(1-n/{ramp_frames})\\,"
            f"0)"
        )

        cmd = [
            "ffmpeg", "-y",
            "-i", os.path.abspath(clip_path),
            "-vf", f"eq=brightness='{expr}'",
            "-c:v", "libx264", "-preset", "ultrafast", "-crf", "23",
            "-pix_fmt", "yuv420p",
            "-an",
            out_path,
        ]

        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True,
                encoding='utf-8', errors='replace', timeout=30,
            )
            if result.returncode == 0 and os.path.exists(out_path):
                return out_path
            else:
                print(f"  [FFmpeg] brightness ramp failed: {result.stderr[-200:]}")
                return None
        except Exception:
            return None

    # =========================================================================
    # P0: Subtitle Overlay (Burn-in)
    # =========================================================================

    def overlay_subtitles(
        self,
        video_in: str,
        srt_path: str,
        out_path: str,
        style: Optional[Dict[str, Any]] = None
    ) -> tuple[str, bool]:
        """
        자막을 영상에 burn-in 처리.

        P0: 자막이 영상 위에 직접 렌더링됨

        Args:
            video_in: 입력 영상 경로
            srt_path: SRT 자막 파일 경로
            out_path: 출력 영상 경로
            style: 자막 스타일 설정 (FontName, FontSize, PrimaryColour 등)

        Returns:
            출력 영상 경로
        """
        from config import get_subtitle_style

        # 기본 스타일 또는 설정에서 로드
        # 폰트 설정 (Railway Nixpacks 환경 호환)
        # Noto Sans CJK KR은 'noto-fonts-cjk' 패키지 설치 시 사용 가능
        # Linux(Railway)와 Windows(Local) 호환성 고려
        
        import platform
        system = platform.system()

        default_font = "Malgun Gothic" # Windows Default
        if system == "Linux":
            default_font = "Noto Sans CJK KR" # standard name after installing noto-fonts-cjk

        # style이 None이면 빈 딕셔너리로 대체
        if style is None:
            style = {}

        # ASS 스타일 문자열 생성
        # [보안] 스타일 값 검증 — FFmpeg 필터 인젝션 방지
        font_size = int(_sanitize_ffmpeg_number(style.get('font_size', 24), 24, 8, 120))
        primary_color = _sanitize_ffmpeg_style_value(style.get('primary_color', '&HFFFFFF'))
        outline_color = _sanitize_ffmpeg_style_value(style.get('outline_color', '&H000000'))
        outline_width = int(_sanitize_ffmpeg_number(style.get('outline_width', 2), 2, 0, 10))
        margin_v = int(_sanitize_ffmpeg_number(style.get('margin_v', 20), 20, 0, 200))

        force_style = (
            f"FontName={_sanitize_ffmpeg_style_value(default_font)},"
            f"FontSize={font_size},"
            f"PrimaryColour={primary_color},"
            f"OutlineColour={outline_color},"
            f"Outline={outline_width},"
            f"MarginV={margin_v}"
        )

        # SRT 경로를 절대 경로로 변환 (Linux에서 상대 경로 문제 해결)
        srt_path_abs = os.path.abspath(srt_path)

        # Windows 경로 처리: 백슬래시를 슬래시로, 콜론 이스케이프
        # FFmpeg subtitles 필터는 콜론을 옵션 구분자로 사용하므로 경로 내 콜론을 \: 로 이스케이프
        srt_path_escaped = srt_path_abs.replace("\\", "/").replace(":", "\\:")

        # FFmpeg subtitle filter - Windows에서는 콜론 이중 이스케이프 필요
        vf_filter = f"subtitles='{srt_path_escaped}':force_style='{force_style}'"

        # DEBUG: SRT 파일 존재 확인
        if not os.path.exists(srt_path):
            print(f"[ERROR] SRT file does not exist: {srt_path}")
            return video_in, False

        # DEBUG: SRT 파일 내용 일부 출력
        try:
            with open(srt_path, 'r', encoding='utf-8') as f:
                srt_preview = f.read(200)
                print(f"[DEBUG] SRT file exists: {srt_path}")
                print(f"[DEBUG] SRT preview (first 200 chars): {srt_preview}")
        except Exception as e:
            print(f"[ERROR] Cannot read SRT file: {e}")
            return video_in, False


        # 모든 경로를 절대 경로로 변환 (Linux에서 상대 경로 문제 해결)
        video_in_abs = os.path.abspath(video_in)
        out_path_abs = os.path.abspath(out_path)

        # Railway 메모리 제한 대응: ultrafast preset + 스레드 제한
        cmd = [
            "ffmpeg",
            "-y",
            "-loglevel", "warning",
            "-threads", "2",  # 스레드 수 제한 (메모리 절약)
            "-i", video_in_abs,
            "-vf", vf_filter,
            "-c:v", "libx264",
            "-preset", "ultrafast",  # 메모리 사용량 최소화
            "-crf", "28",  # 품질 약간 낮춤 (파일 크기/메모리 절약)
            "-threads", "2",  # 인코딩 스레드도 제한
            out_path_abs
        ]

        print(f"[DEBUG] Running FFmpeg subtitle burn-in:")
        print(f"[DEBUG] Input: {video_in_abs}")
        print(f"[DEBUG] Output: {out_path_abs}")
        print(f"[DEBUG] Filter: {vf_filter}")

        sub_timeout = self._dynamic_timeout(video_in_abs, factor=5.0, minimum=120)
        print(f"[DEBUG] Subtitle timeout: {sub_timeout}s")
        result = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8', errors='replace', timeout=sub_timeout)

        print(f"[DEBUG] FFmpeg returncode: {result.returncode}")
        print(f"[DEBUG] FFmpeg stderr length: {len(result.stderr)}")

        subtitle_applied = False
        
        if result.returncode != 0:
            print(f"[ERROR] Subtitle overlay failed! (returncode={result.returncode})")
            if result.returncode == -9:
                print(f"[ERROR] Process killed by SIGKILL - likely OOM (Out of Memory)")
                print(f"[ERROR] Consider increasing Railway container memory or reducing video quality")
            print(f"[ERROR] FFmpeg command: {' '.join(cmd)}")
            print(f"[ERROR] FFmpeg stderr (last 500 chars):")
            print(result.stderr[-500:] if result.stderr else "(empty)")
            # 자막 실패 시 원본 복사로 폴백
            self._copy_file(video_in_abs, out_path_abs)
            subtitle_applied = False
        else:
            # FFmpeg가 returncode=0이어도 파일이 제대로 생성되었는지 확인
            if not os.path.exists(out_path_abs):
                print(f"[ERROR] Output file not created: {out_path_abs}")
                self._copy_file(video_in_abs, out_path_abs)
                subtitle_applied = False
            elif os.path.getsize(out_path_abs) < 1024:  # 1KB 미만이면 실패로 간주
                print(f"[ERROR] Output file too small ({os.path.getsize(out_path_abs)} bytes): {out_path_abs}")
                print(f"[ERROR] FFmpeg likely failed silently (frame=0)")
                os.remove(out_path_abs)
                self._copy_file(video_in_abs, out_path_abs)
                subtitle_applied = False
            else:
                print(f"[SUCCESS] Subtitle burn-in completed: {out_path_abs} ({os.path.getsize(out_path_abs)} bytes)")
                subtitle_applied = True

        return out_path, subtitle_applied

    def generate_srt_from_scenes(
        self,
        scenes: List[Dict[str, Any]],
        output_path: str
    ) -> str:
        """
        Scene 목록에서 SRT 자막 파일 생성.
        긴 문장을 15-20자 단위로 분할하여 순차 표시.

        Args:
            scenes: Scene 데이터 목록
            output_path: SRT 파일 출력 경로

        Returns:
            SRT 파일 경로
        """
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

        srt_content = []
        current_time_ms = 0
        sub_index = 1

        for scene in scenes:
            duration_sec = scene.get("duration_sec", 5)
            duration_ms = duration_sec * 1000
            text = scene.get("narration") or scene.get("sentence", "")

            # Strip speaker tags: [narrator], [male_1](whisper), [STORYCUT_HERO_A](angry), etc.
            if text:
                text = re.sub(r'\[[\w_]+\](?:\([^)]*\))?\s*', '', text).strip()

            if text:
                # 긴 텍스트를 짧은 청크로 분할
                chunks = self._split_subtitle_text(text, max_chars=20)

                if len(chunks) == 1:
                    # 짧은 텍스트: 그대로 표시
                    start_time = self._ms_to_srt_time(current_time_ms)
                    end_time = self._ms_to_srt_time(current_time_ms + duration_ms)
                    srt_content.append(f"{sub_index}")
                    srt_content.append(f"{start_time} --> {end_time}")
                    srt_content.append(chunks[0])
                    srt_content.append("")
                    sub_index += 1
                else:
                    # 긴 텍스트: 청크별로 균등 시간 분배
                    chunk_duration_ms = duration_ms / len(chunks)
                    for j, chunk in enumerate(chunks):
                        chunk_start = current_time_ms + j * chunk_duration_ms
                        chunk_end = chunk_start + chunk_duration_ms
                        srt_content.append(f"{sub_index}")
                        srt_content.append(f"{self._ms_to_srt_time(chunk_start)} --> {self._ms_to_srt_time(chunk_end)}")
                        srt_content.append(chunk)
                        srt_content.append("")
                        sub_index += 1

            current_time_ms += duration_ms

        with open(output_path, "w", encoding="utf-8") as f:
            f.write("\n".join(srt_content))

        return output_path

    def _split_subtitle_text(self, text: str, max_chars: int = 20) -> List[str]:
        """
        자막 텍스트를 자연스러운 단위로 분할.
        문장 부호(. ? ! ,) 우선, 그 외 띄어쓰기 기준.
        """
        text = text.strip()
        if len(text) <= max_chars:
            return [text]

        chunks = []

        # 1차: 문장 단위 분할 (. ? !)
        import re
        sentences = re.split(r'(?<=[.?!])\s+', text)

        for sentence in sentences:
            if len(sentence) <= max_chars:
                chunks.append(sentence)
            else:
                # 2차: 쉼표 기준 분할
                parts = sentence.split(',')
                buffer = ""
                for part in parts:
                    part = part.strip()
                    if not part:
                        continue
                    test = f"{buffer}, {part}" if buffer else part
                    if len(test) <= max_chars:
                        buffer = test
                    else:
                        if buffer:
                            chunks.append(buffer)
                        # 그래도 길면 띄어쓰기 기준으로 자르기
                        if len(part) > max_chars:
                            words = part.split()
                            buffer = ""
                            for word in words:
                                test_w = f"{buffer} {word}" if buffer else word
                                if len(test_w) <= max_chars:
                                    buffer = test_w
                                else:
                                    if buffer:
                                        chunks.append(buffer)
                                    buffer = word
                        else:
                            buffer = part
                if buffer:
                    chunks.append(buffer)

        return chunks if chunks else [text]

    def _ms_to_srt_time(self, ms) -> str:
        """밀리초를 SRT 시간 형식으로 변환."""
        ms = int(ms)
        hours = ms // 3600000
        minutes = (ms % 3600000) // 60000
        seconds = (ms % 60000) // 1000
        milliseconds = ms % 1000
        return f"{hours:02d}:{minutes:02d}:{seconds:02d},{milliseconds:03d}"

    # =========================================================================
    # P0: Audio Ducking
    # =========================================================================

    def mix_with_ducking(
        self,
        video_in: str,
        narration_path: Optional[str],
        bgm_path: Optional[str],
        out_path: str,
        ducking_config: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        내레이션과 BGM을 믹싱하며 Audio Ducking 적용.

        P0: 내레이션이 있을 때 BGM이 자동 감쇠되고, 없으면 원복

        Args:
            video_in: 입력 영상 경로
            narration_path: 내레이션 오디오 경로 (없으면 None)
            bgm_path: BGM 오디오 경로 (없으면 None)
            out_path: 출력 영상 경로
            ducking_config: 덕킹 설정 (threshold, ratio, attack, release)

        Returns:
            출력 영상 경로
        """
        from config import get_ducking_config

        # 기본 설정 로드
        if ducking_config is None:
            ducking_config = get_ducking_config()

        # 오디오가 모두 없는 경우
        if not narration_path and not bgm_path:
            self._copy_file(video_in, out_path)
            return out_path

        # 내레이션만 있는 경우
        if narration_path and not bgm_path:
            return self._add_audio_to_video(video_in, narration_path, out_path)

        # BGM만 있는 경우
        if bgm_path and not narration_path:
            return self._add_audio_to_video(
                video_in, bgm_path, out_path,
                volume=ducking_config.get("bgm_volume_normal", 0.3)
            )

        # 둘 다 있는 경우 - Ducking 적용
        # [보안] 숫자 값 검증 — FFmpeg 필터 인젝션 방지
        threshold = _sanitize_ffmpeg_number(ducking_config.get("threshold", 0.02), 0.02, 0.001, 1.0)
        ratio = _sanitize_ffmpeg_number(ducking_config.get("ratio", 10), 10, 1, 100)
        attack = _sanitize_ffmpeg_number(ducking_config.get("attack_ms", 20), 20, 1, 5000)
        release = _sanitize_ffmpeg_number(ducking_config.get("release_ms", 200), 200, 1, 10000)

        # sidechaincompress를 사용한 Audio Ducking
        filter_complex = (
            f"[1:a]aformat=fltp:44100:stereo,volume=1.0[narr];"
            f"[2:a]aformat=fltp:44100:stereo,volume={ducking_config.get('bgm_volume_normal', 0.3)}[bgm];"
            f"[bgm][narr]sidechaincompress="
            f"threshold={threshold}:ratio={ratio}:attack={attack}:release={release}[bgm_ducked];"
            f"[bgm_ducked][narr]amix=inputs=2:duration=longest:normalize=0[aout]"
        )

        cmd = [
            "ffmpeg",
            "-y",
            "-i", video_in,
            "-i", narration_path,
            "-i", bgm_path,
            "-filter_complex", filter_complex,
            "-map", "0:v",
            "-map", "[aout]",
            "-c:v", "copy",
            "-c:a", "aac",
            "-b:a", "192k",
            "-shortest",
            out_path
        ]

        result = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8', errors='replace', timeout=120)

        if result.returncode != 0:
            print(f"Audio ducking failed: {result.stderr[:500]}")
            # 덕킹 실패 시 단순 믹스로 폴백
            return self._simple_audio_mix(video_in, narration_path, bgm_path, out_path)

        return out_path

    def _simple_audio_mix(
        self,
        video_in: str,
        narration_path: str,
        bgm_path: str,
        out_path: str
    ) -> str:
        """단순 오디오 믹스 (덕킹 없음)."""
        filter_complex = (
            "[1:a]volume=1.0[narr];"
            "[2:a]volume=0.2[bgm];"
            "[narr][bgm]amix=inputs=2:duration=longest:normalize=0[aout]"
        )

        cmd = [
            "ffmpeg",
            "-y",
            "-i", video_in,
            "-i", narration_path,
            "-i", bgm_path,
            "-filter_complex", filter_complex,
            "-map", "0:v",
            "-map", "[aout]",
            "-c:v", "copy",
            "-c:a", "aac",
            "-shortest",
            out_path
        ]

        result = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8', errors='replace', timeout=120)

        if result.returncode != 0:
            raise RuntimeError(f"Audio mix failed: {result.stderr}")

        return out_path

    def _add_audio_to_video(
        self,
        video_in: str,
        audio_path: str,
        out_path: str,
        volume: float = 1.0
    ) -> str:
        """영상에 단일 오디오 트랙 추가."""
        cmd = [
            "ffmpeg",
            "-y",
            "-i", video_in,
            "-i", audio_path,
            "-filter_complex", f"[1:a]volume={volume}[aout]",
            "-map", "0:v",
            "-map", "[aout]",
            "-c:v", "copy",
            "-c:a", "aac",
            "-shortest",
            out_path
        ]

        audio_timeout = self._dynamic_timeout(video_in, factor=2.0, minimum=120)
        result = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8', errors='replace', timeout=audio_timeout)

        if result.returncode != 0:
            raise RuntimeError(f"Failed to add audio: {result.stderr}")

        return out_path

    # =========================================================================
    # 크로스페이드 전환 (씬 간 부드러운 전환)
    # =========================================================================

    # 전환 타입 -> FFmpeg xfade 전환 이름 매핑
    _XFADE_MAP = {
        "xfade":      "fade",
        "fadeblack":  "fadeblack",
        "whiteflash": "fadewhite",
        "filmburn":   "fadeblack",   # fadeblack + 긴 duration으로 시뮬레이션
        "glitch":     "pixelize",    # FFmpeg 내장 pixelize 전환
        # "cut"은 xfade 없이 concat demuxer 사용
    }

    def concatenate_with_crossfade(
        self,
        scene_groups: List[List[str]],
        output_path: str,
        fade_duration: float = 0.3,
        transition_plan: List[dict] = None,
    ) -> bool:
        """씬 그룹 단위로 전환을 적용하여 연결.

        같은 씬 내 컷들은 하드 컷으로 빠르게 연결하고,
        씬 간 전환에는 transition_plan에 따라 다양한 전환 타입을 적용합니다.

        Args:
            scene_groups: [[cut1_a, cut1_b], [cut2_a], [cut3_a, cut3_b, cut3_c], ...]
                          각 리스트가 하나의 씬, 내부 요소가 해당 씬의 컷 클립들
            output_path: 최종 출력 경로
            fade_duration: 기본 크로스페이드 길이 (초, transition_plan 없을 때 사용)
            transition_plan: TransitionPlanner 출력. 씬 경계마다 전환 타입 지정.

        Returns:
            성공 여부
        """
        if not scene_groups:
            return False

        output_dir = os.path.dirname(os.path.abspath(output_path))
        temp_scene_videos = []

        try:
            # Stage 1: 각 씬 내 컷들을 하드 컷으로 연결 (빠름)
            for si, clips in enumerate(scene_groups):
                if not clips:
                    continue
                if len(clips) == 1:
                    temp_scene_videos.append(clips[0])
                    continue

                scene_concat = os.path.join(output_dir, f"temp_scene_{si:03d}.mp4")
                concat_file = os.path.join(output_dir, f"temp_scene_{si:03d}.txt")
                with open(concat_file, "w", encoding="utf-8") as f:
                    for clip in clips:
                        f.write(f"file '{os.path.abspath(clip)}'\n")
                cmd = [
                    "ffmpeg", "-y", "-f", "concat", "-safe", "0",
                    "-i", concat_file, "-c", "copy", scene_concat
                ]
                result = subprocess.run(cmd, capture_output=True, text=True,
                                        encoding='utf-8', errors='replace', timeout=60)
                if os.path.exists(concat_file):
                    os.remove(concat_file)
                if result.returncode != 0 or not os.path.exists(scene_concat):
                    # copy 실패 시 재인코딩
                    cmd_re = [
                        "ffmpeg", "-y", "-f", "concat", "-safe", "0",
                        "-i", concat_file if os.path.exists(concat_file) else "",
                    ]
                    # fallback: 그냥 첫 번째 클립 사용
                    temp_scene_videos.append(clips[0])
                    continue
                temp_scene_videos.append(scene_concat)

            if not temp_scene_videos:
                return False

            # 씬이 하나면 xfade 불필요
            if len(temp_scene_videos) == 1:
                import shutil
                shutil.copy2(temp_scene_videos[0], output_path)
                return True

            # Stage 2: 씬 간 전환 적용 (transition_plan 기반 또는 기본 fade)
            print(f"[FFmpeg] Applying transitions between {len(temp_scene_videos)} scenes...")

            # 각 씬 영상 duration 확인
            durations = []
            for sv in temp_scene_videos:
                dur = self.get_video_duration(sv)
                durations.append(max(dur, 0.5))  # 최소 0.5초 보장

            # boundary_transitions: 씬 경계별 전환 정보 (len = len(temp_scene_videos) - 1)
            boundary_transitions = None
            if transition_plan and len(transition_plan) == len(temp_scene_videos) - 1:
                boundary_transitions = transition_plan
            elif transition_plan:
                print(f"  [WARNING] transition_plan length mismatch: {len(transition_plan)} vs {len(temp_scene_videos)-1} boundaries, using default fade")

            # "cut" 전환인 씬 경계는 concat demuxer로 pre-merge
            if boundary_transitions:
                temp_scene_videos, durations, boundary_transitions = self._premerge_cut_groups(
                    temp_scene_videos, durations, boundary_transitions, output_dir
                )

            if len(temp_scene_videos) <= 1:
                import shutil
                shutil.copy2(temp_scene_videos[0], output_path)
                return True

            # xfade 필터 체인 구성
            # 씬이 너무 많으면 xfade 체인이 너무 길어지므로 배치 처리
            MAX_XFADE_BATCH = 10
            if len(temp_scene_videos) > MAX_XFADE_BATCH:
                return self._batched_crossfade(
                    temp_scene_videos, durations, output_path,
                    fade_duration, MAX_XFADE_BATCH, output_dir,
                    boundary_transitions=boundary_transitions,
                )

            return self._apply_xfade_chain(
                temp_scene_videos, durations, output_path, fade_duration,
                boundary_transitions=boundary_transitions,
            )

        finally:
            # 임시 씬 파일 정리
            for sv in temp_scene_videos:
                if sv.startswith(os.path.join(output_dir, "temp_scene_")) and os.path.exists(sv):
                    os.remove(sv)

    def _premerge_cut_groups(
        self,
        video_paths: List[str],
        durations: List[float],
        boundary_transitions: List[dict],
        output_dir: str,
    ):
        """'cut' 전환으로 연결된 인접 씬들을 concat demuxer로 pre-merge.

        Returns:
            (merged_paths, merged_durations, merged_transitions)
            - "cut" 경계를 제거하고 인접 씬들을 합친 결과
        """
        # 그룹 빌드: 연속 "cut" 경계로 연결된 씬들을 하나의 그룹으로
        groups = [[0]]  # group of video indices
        for i, t in enumerate(boundary_transitions):
            if t.get("transition_type") == "cut":
                groups[-1].append(i + 1)
            else:
                groups.append([i + 1])

        # 모든 경계가 non-cut이면 그대로 반환
        if all(len(g) == 1 for g in groups):
            return video_paths, durations, boundary_transitions

        merged_paths = []
        merged_durations = []
        merged_transitions = []

        for gi, group in enumerate(groups):
            if len(group) == 1:
                idx = group[0]
                merged_paths.append(video_paths[idx])
                merged_durations.append(durations[idx])
            else:
                # concat demuxer로 그룹 내 씬들 합치기
                group_paths = [video_paths[idx] for idx in group]
                group_dur = sum(durations[idx] for idx in group)
                merge_out = os.path.join(output_dir, f"temp_cutmerge_{gi:03d}.mp4")
                concat_file = os.path.join(output_dir, f"temp_cutmerge_{gi:03d}.txt")
                with open(concat_file, "w", encoding="utf-8") as f:
                    for p in group_paths:
                        f.write(f"file '{os.path.abspath(p)}'\n")
                cmd = [
                    "ffmpeg", "-y", "-f", "concat", "-safe", "0",
                    "-i", concat_file, "-c", "copy", merge_out
                ]
                result = subprocess.run(cmd, capture_output=True, text=True,
                                        encoding='utf-8', errors='replace', timeout=60)
                if os.path.exists(concat_file):
                    os.remove(concat_file)
                if result.returncode == 0 and os.path.exists(merge_out):
                    merged_paths.append(merge_out)
                    merged_durations.append(group_dur)
                else:
                    # fallback: 첫 번째 클립만 사용
                    merged_paths.append(video_paths[group[0]])
                    merged_durations.append(durations[group[0]])

            # 그룹 간 전환 추가 (마지막 그룹 제외)
            if gi < len(groups) - 1:
                # 이 그룹의 마지막 씬과 다음 그룹의 첫 씬 사이의 전환
                last_idx = group[-1]
                if last_idx < len(boundary_transitions):
                    merged_transitions.append(boundary_transitions[last_idx])

        cut_count = sum(1 for t in boundary_transitions if t.get("transition_type") == "cut")
        if cut_count > 0:
            print(f"  [FFmpeg] Pre-merged {cut_count} hard-cut boundaries into {len(merged_paths)} groups")

        return merged_paths, merged_durations, merged_transitions

    def _apply_xfade_chain(
        self,
        video_paths: List[str],
        durations: List[float],
        output_path: str,
        fade_duration: float,
        boundary_transitions: List[dict] = None,
    ) -> bool:
        """xfade 필터 체인으로 전환 적용. 경계별 다른 전환 타입 지원."""
        n = len(video_paths)
        if n <= 1:
            return False

        # 입력 파일
        inputs = []
        for path in video_paths:
            inputs.extend(["-i", os.path.abspath(path)])

        # xfade 필터 체인
        filter_parts = []
        offset = durations[0] - fade_duration

        for i in range(1, n):
            # 경계별 전환 타입과 duration 결정
            if boundary_transitions and i - 1 < len(boundary_transitions):
                bt = boundary_transitions[i - 1]
                transition_type = bt.get("transition_type", "xfade")
                xfade_name = self._XFADE_MAP.get(transition_type, "fade")
                t_frames = bt.get("transition_frames", int(fade_duration * self.fps))
                actual_fade = t_frames / self.fps if t_frames > 0 else fade_duration
            else:
                xfade_name = "fade"
                actual_fade = fade_duration

            # fade_duration이 클립보다 길면 조정
            actual_fade = min(actual_fade, durations[i] * 0.4, durations[i-1] * 0.4 if i == 1 else actual_fade)
            if actual_fade < 0.05:
                actual_fade = 0.05  # 최소 전환 길이

            if i == 1:
                offset = durations[0] - actual_fade

            prev_label = f"[v{i-1}]" if i > 1 else "[0:v]"
            curr_label = f"[{i}:v]"
            out_label = f"[v{i}]"

            filter_parts.append(
                f"{prev_label}{curr_label}xfade=transition={xfade_name}:duration={actual_fade:.3f}:offset={max(0, offset):.3f}{out_label}"
            )

            if i < n - 1:
                offset += durations[i] - actual_fade

        filter_complex = ";".join(filter_parts)
        final_label = f"[v{n-1}]"

        cmd = ["ffmpeg", "-y"] + inputs + [
            "-filter_complex", filter_complex,
            "-map", final_label,
            "-c:v", "libx264",
            "-preset", "ultrafast",
            "-crf", "23",
            "-pix_fmt", "yuv420p",
            "-r", str(self.fps),
            output_path
        ]

        timeout = max(180, n * 30)
        # 전환 타입 요약 로그
        if boundary_transitions:
            type_counts = {}
            for bt in boundary_transitions:
                tt = bt.get("transition_type", "xfade")
                type_counts[tt] = type_counts.get(tt, 0) + 1
            summary = ", ".join(f"{k}:{v}" for k, v in sorted(type_counts.items()))
            print(f"[FFmpeg] xfade chain: {n} clips, transitions: {summary} (timeout={timeout}s)")
        else:
            print(f"[FFmpeg] xfade chain: {n} clips, fade={fade_duration}s (timeout={timeout}s)")

        result = subprocess.run(cmd, capture_output=True, text=True,
                                encoding='utf-8', errors='replace', timeout=timeout)

        if result.returncode != 0:
            print(f"[FFmpeg] xfade failed: {result.stderr[-500:]}")
            # Fallback: 일반 concat
            print(f"[FFmpeg] Falling back to hard-cut concatenation...")
            return self.concatenate_videos(video_paths, output_path)

        print(f"[FFmpeg] Crossfade complete: {output_path}")
        return True

    def _batched_crossfade(
        self,
        video_paths: List[str],
        durations: List[float],
        output_path: str,
        fade_duration: float,
        batch_size: int,
        output_dir: str,
        boundary_transitions: List[dict] = None,
    ) -> bool:
        """씬이 많을 때 배치 처리로 xfade 적용. 배치별 전환 정보 슬라이싱."""
        batch_outputs = []
        try:
            for start in range(0, len(video_paths), batch_size):
                end = min(start + batch_size, len(video_paths))
                batch_paths = video_paths[start:end]
                batch_durs = durations[start:end]

                # 배치별 전환 정보 슬라이싱
                batch_trans = None
                if boundary_transitions:
                    # 전환은 경계 수 = 클립 수 - 1
                    trans_start = start
                    trans_end = min(end - 1, len(boundary_transitions))
                    if trans_start < trans_end:
                        batch_trans = boundary_transitions[trans_start:trans_end]

                if len(batch_paths) == 1:
                    batch_outputs.append(batch_paths[0])
                    continue

                batch_out = os.path.join(output_dir, f"temp_batch_{start:03d}.mp4")
                ok = self._apply_xfade_chain(
                    batch_paths, batch_durs, batch_out, fade_duration,
                    boundary_transitions=batch_trans,
                )
                if ok and os.path.exists(batch_out):
                    batch_outputs.append(batch_out)
                else:
                    batch_outputs.extend(batch_paths)

            # 배치 결과물을 최종 xfade 또는 concat
            if len(batch_outputs) <= 1:
                import shutil
                shutil.copy2(batch_outputs[0], output_path)
                return True

            batch_durs = [self.get_video_duration(p) for p in batch_outputs]
            return self._apply_xfade_chain(batch_outputs, batch_durs, output_path, fade_duration)

        finally:
            for bo in batch_outputs:
                if bo.startswith(os.path.join(output_dir, "temp_batch_")) and os.path.exists(bo):
                    os.remove(bo)

    # =========================================================================
    # 기존 기능 (하위 호환성 유지)
    # =========================================================================

    def concatenate_videos(
        self,
        video_paths: List[str],
        output_path: str,
        transition: str = "cut"
    ) -> bool:
        """
        여러 영상 클립을 하나로 연결.

        Args:
            video_paths: 입력 영상 파일 경로 목록
            output_path: 출력 영상 파일 경로
            transition: "cut" 또는 "fade"

        Returns:
            성공 여부
        """
        if not video_paths:
            raise ValueError("No video paths provided")

        # temp 파일을 출력 디렉토리에 생성 (CWD 의존 제거)
        output_dir = os.path.dirname(os.path.abspath(output_path))
        concat_file = os.path.join(output_dir, "temp_concat_list.txt")

        with open(concat_file, "w", encoding="utf-8") as f:
            for video_path in video_paths:
                f.write(f"file '{os.path.abspath(video_path)}'\n")

        try:
            # 먼저 -c copy 시도 (빠름, 동일 코덱일 때)
            print(f"[FFmpeg] Concatenating {len(video_paths)} clips...")
            cmd = [
                "ffmpeg",
                "-y",
                "-f", "concat",
                "-safe", "0",
                "-i", concat_file,
                "-c", "copy",
                output_path
            ]

            # copy는 빠르므로 클립당 10초 + 최소 60초
            copy_timeout = max(60, len(video_paths) * 10)
            result = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8', errors='replace', timeout=copy_timeout)

            if result.returncode != 0:
                print(f"[FFmpeg] concat -c copy failed, retrying with re-encode...")
                print(f"[FFmpeg] Error: {result.stderr[-300:]}")
                # 재인코딩 fallback (해상도/코덱 불일치 시) - ultrafast로 속도 우선
                cmd_reencode = [
                    "ffmpeg",
                    "-y",
                    "-f", "concat",
                    "-safe", "0",
                    "-i", concat_file,
                    "-c:v", "libx264",
                    "-preset", "ultrafast",
                    "-crf", "23",
                    "-pix_fmt", "yuv420p",
                    "-vf", f"scale={self.width}:{self.height}:force_original_aspect_ratio=decrease,"
                           f"pad={self.width}:{self.height}:(ow-iw)/2:(oh-ih)/2",
                    "-r", str(self.fps),
                    output_path
                ]

                # 재인코딩은 클립당 45초 + 최소 180초 (Railway 등 저사양 서버 대응)
                reencode_timeout = max(180, len(video_paths) * 45)
                print(f"[FFmpeg] Re-encoding with ultrafast preset... (timeout={reencode_timeout}s)")
                result2 = subprocess.run(cmd_reencode, capture_output=True, text=True, encoding='utf-8', errors='replace', timeout=reencode_timeout)
                if result2.returncode != 0:
                    print(f"[FFmpeg] Re-encode concat also failed: {result2.stderr[-300:]}")
                    return False

            print(f"[FFmpeg] Concatenation complete: {output_path}")
            return True

        finally:
            if os.path.exists(concat_file):
                os.remove(concat_file)

    def mix_audio(
        self,
        video_path: str,
        narration_paths: List[str],
        music_path: str = None,
        output_path: str = None,
        music_volume: float = 0.2,
        scene_durations: List[float] = None
    ) -> str:
        """
        내레이션과 배경 음악을 영상에 믹스.

        기존 API 호환성 유지.

        Args:
            video_path: 입력 영상 파일
            narration_paths: 내레이션 오디오 파일 목록 (씬 순서)
            music_path: 배경 음악 파일 (선택)
            output_path: 출력 영상 파일
            music_volume: 배경 음악 볼륨 (0.0-1.0)

        Returns:
            출력 파일 경로
        """
        if output_path is None:
            output_path = video_path.replace(".mp4", "_with_audio.mp4")

        # 내레이션 연결 (output_path 기준 디렉토리에 임시 파일 생성 - 동시 요청 충돌 방지)
        _out_dir = os.path.dirname(output_path) or "."
        narration_concat = os.path.join(_out_dir, "temp_narration.wav")

        # 씬별 비디오 duration으로 TTS 오디오를 패딩하여 싱크 맞춤
        if scene_durations and len(scene_durations) == len(narration_paths):
            padded_paths = []
            try:
                for i, (audio_path, target_dur) in enumerate(zip(narration_paths, scene_durations)):
                    padded_path = os.path.join(_out_dir, f"temp_padded_narr_{i:02d}.wav")
                    self._pad_audio_to_duration(audio_path, float(target_dur), padded_path)
                    padded_paths.append(padded_path)
                self._concatenate_audio(padded_paths, narration_concat)
            finally:
                for p in padded_paths:
                    if os.path.exists(p):
                        os.remove(p)
        else:
            self._concatenate_audio(narration_paths, narration_concat)

        # FFmpeg 명령 구성
        inputs = ["-i", video_path, "-i", narration_concat]

        if music_path and os.path.exists(music_path):
            # 음악이 있을 때: 내레이션과 음악 믹스
            inputs.extend(["-i", music_path])
            filter_complex = (
                f"[1:a]volume=1.0[narr];"
                f"[2:a]volume={music_volume}[mus];"
                f"[narr][mus]amix=inputs=2:duration=longest[aout]"
            )
            cmd = [
                "ffmpeg",
                *inputs,
                "-filter_complex", filter_complex,
                "-map", "0:v",
                "-map", "[aout]",
                "-c:v", "copy",
                "-c:a", "aac",
                "-shortest",
                output_path,
                "-y"
            ]
        else:
            # 음악이 없을 때: 내레이션만 사용 (filter_complex 불필요)
            cmd = [
                "ffmpeg",
                *inputs,
                "-map", "0:v",
                "-map", "1:a",
                "-c:v", "copy",
                "-c:a", "aac",
                "-shortest",
                output_path,
                "-y"
            ]

        print(f"[FFmpeg] Mixing audio...")
        print(f"  Video: {video_path}")
        print(f"  Narration files: {len(narration_paths)}")
        print(f"  Music: {music_path if music_path else 'None'}")

        result = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8', errors='replace', timeout=180)

        # 임시 파일 정리
        if os.path.exists(narration_concat):
            os.remove(narration_concat)

        if result.returncode != 0:
            print(f"\n[ERROR] FFmpeg audio mix failed!")
            print(f"Command: {' '.join(cmd)}")
            print(f"Stderr: {result.stderr}")
            print(f"Stdout: {result.stdout}")
            raise RuntimeError(f"Failed to mix audio: {result.stderr}")

        print(f"[FFmpeg] Audio mixed successfully: {output_path}")
        return output_path

    def _pad_audio_to_duration(self, audio_path: str, target_duration: float, output_path: str):
        """오디오를 target_duration 길이로 패딩 (뒤에 silence 추가)."""
        cmd = [
            "ffmpeg", "-y",
            "-i", audio_path,
            "-af", "apad",
            "-t", str(target_duration),
            "-ar", "44100",
            "-ac", "2",
            "-c:a", "pcm_s16le",
            output_path
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8', errors='replace', timeout=60)
        if result.returncode != 0:
            raise RuntimeError(f"Failed to pad audio: {result.stderr}")

    def _concatenate_audio(self, audio_paths: List[str], output_path: str):
        """여러 오디오 파일을 하나로 연결."""
        # 파일 존재 확인
        for audio_path in audio_paths:
            if not os.path.exists(audio_path):
                raise FileNotFoundError(f"Audio file not found: {audio_path}")

        _out_dir = os.path.dirname(output_path) or "."
        concat_file = os.path.join(_out_dir, "temp_audio_concat.txt")
        with open(concat_file, "w", encoding="utf-8") as f:
            for audio_path in audio_paths:
                # Windows 경로를 Unix 스타일로 변환 (ffmpeg concat에서 더 안정적)
                abs_path = os.path.abspath(audio_path).replace("\\", "/")
                f.write(f"file '{abs_path}'\n")

        cmd = [
            "ffmpeg",
            "-y",
            "-f", "concat",
            "-safe", "0",
            "-i", concat_file,
            # codec copy 대신 재인코딩 (호환성 향상)
            "-ar", "44100",  # 샘플레이트 통일
            "-ac", "2",      # 스테레오
            "-c:a", "pcm_s16le",  # WAV용 코덱
            output_path
        ]

        result = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8', errors='replace', timeout=120)

        if os.path.exists(concat_file):
            os.remove(concat_file)

        if result.returncode != 0:
            print(f"FFmpeg concat command: {' '.join(cmd)}")
            print(f"FFmpeg concat error: {result.stderr}")
            raise RuntimeError(f"Failed to concatenate audio: {result.stderr}")

    def get_video_duration(self, video_path: str) -> float:
        """ffprobe로 영상 길이 확인."""
        cmd = [
            "ffprobe",
            "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            video_path
        ]

        result = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8', errors='replace', timeout=30)

        if result.returncode == 0:
            return float(result.stdout.strip())
        else:
            raise RuntimeError(f"Failed to get video duration: {result.stderr}")

    def get_audio_duration(self, audio_path: str) -> float:
        """ffprobe로 오디오 길이 확인."""
        return self.get_video_duration(audio_path)

    def compose_final_video(
        self,
        video_clips: List[str],
        narration_clips: List[str],
        music_path: str,
        output_path: str,
        use_ducking: bool = False,
        scene_durations: List[float] = None
    ) -> str:
        """
        전체 합성 파이프라인: 영상 연결, 오디오 믹스, 최종 MP4 출력.

        Args:
            video_clips: 씬 영상 파일 목록 (순서대로)
            narration_clips: 내레이션 오디오 파일 목록 (순서대로)
            music_path: 배경 음악 파일
            output_path: 최종 출력 영상 경로
            use_ducking: 오디오 덕킹 사용 여부

        Returns:
            최종 영상 파일 경로
        """
        print("Starting video composition...")

        # Step 1: 영상 클립 연결 (output_path 기준 디렉토리에 임시 파일 - 동시 요청 충돌 방지)
        _out_dir = os.path.dirname(output_path) or "."
        print("  -> Concatenating video clips...")
        temp_video = os.path.join(_out_dir, "temp_concatenated.mp4")
        self.concatenate_videos(video_clips, temp_video)

        # Step 2: 오디오 믹스 (내레이션 + BGM)
        if use_ducking and music_path and os.path.exists(music_path):
            print("  -> Mixing audio with DUCKING (narration + BGM)...")
            # 내레이션 연결 (씬별 duration 패딩 적용)
            narration_concat = os.path.join(_out_dir, "temp_narration_ducking.wav")
            if scene_durations and len(scene_durations) == len(narration_clips):
                padded_paths = []
                try:
                    for i, (audio_path, target_dur) in enumerate(zip(narration_clips, scene_durations)):
                        padded_path = os.path.join(_out_dir, f"temp_padded_narr_{i:02d}.wav")
                        self._pad_audio_to_duration(audio_path, float(target_dur), padded_path)
                        padded_paths.append(padded_path)
                    self._concatenate_audio(padded_paths, narration_concat)
                finally:
                    for p in padded_paths:
                        if os.path.exists(p):
                            os.remove(p)
            else:
                self._concatenate_audio(narration_clips, narration_concat)
            # 덕킹 적용 믹싱
            final_video = self.mix_with_ducking(
                temp_video,
                narration_concat,
                music_path,
                output_path
            )
            # 임시 파일 정리
            if os.path.exists(narration_concat):
                os.remove(narration_concat)
        else:
            print("  -> Mixing audio (narration + background music)...")
            final_video = self.mix_audio(
                temp_video,
                narration_clips,
                music_path,
                output_path,
                scene_durations=scene_durations
            )

        # 임시 파일 정리
        if os.path.exists(temp_video):
            os.remove(temp_video)

        print(f"Final video created: {final_video}")

        return final_video

    # =========================================================================
    # P0: 통합 Scene 렌더링
    # =========================================================================

    def render_scene(
        self,
        scene: Dict[str, Any],
        feature_flags: Dict[str, bool],
        output_dir: str = "media/rendered"
    ) -> str:
        """
        단일 Scene을 완전히 렌더링 (영상 + 자막 + 오디오).

        P0: Scene 단위 조립

        Args:
            scene: Scene 데이터
            feature_flags: Feature flags 설정
            output_dir: 출력 디렉토리

        Returns:
            렌더링된 영상 경로
        """
        os.makedirs(output_dir, exist_ok=True)
        scene_id = scene.get("scene_id", scene.get("index", 1))
        output_path = f"{output_dir}/rendered_scene_{scene_id:02d}.mp4"

        # 기본 영상 가져오기 (이미지+Ken Burns 또는 비디오)
        if scene.get("assets", {}).get("video_path"):
            base_video = scene["assets"]["video_path"]
        elif scene.get("assets", {}).get("image_path"):
            # 이미지가 있으면 Ken Burns 적용
            if feature_flags.get("ffmpeg_kenburns", True):
                base_video = f"{output_dir}/temp_kb_{scene_id:02d}.mp4"
                self.ken_burns_clip(
                    scene["assets"]["image_path"],
                    scene.get("duration_sec", 5),
                    base_video,
                    effect_type=["zoom_in", "zoom_out", "pan_left", "diagonal"][scene_id % 4]
                )
            else:
                # 정적 이미지 → 영상 변환
                base_video = f"{output_dir}/temp_static_{scene_id:02d}.mp4"
                self._image_to_static_video(
                    scene["assets"]["image_path"],
                    scene.get("duration_sec", 5),
                    base_video
                )
        else:
            raise ValueError(f"Scene {scene_id} has no video or image asset")

        current_video = base_video

        # 자막 burn-in
        if (feature_flags.get("subtitle_burn_in", True) and
            scene.get("assets", {}).get("subtitle_srt_path")):
            subtitled_video = f"{output_dir}/temp_sub_{scene_id:02d}.mp4"
            current_video, _ = self.overlay_subtitles(
                current_video,
                scene["assets"]["subtitle_srt_path"],
                subtitled_video
            )

        # 오디오 믹싱 (내레이션 + BGM)
        narration_path = scene.get("assets", {}).get("narration_path")
        bgm_path = scene.get("assets", {}).get("bgm_path")

        if narration_path or bgm_path:
            use_ducking = feature_flags.get("ffmpeg_audio_ducking", False)

            if use_ducking:
                current_video = self.mix_with_ducking(
                    current_video,
                    narration_path,
                    bgm_path,
                    output_path
                )
            else:
                # 단순 믹스
                if narration_path and bgm_path:
                    current_video = self._simple_audio_mix(
                        current_video, narration_path, bgm_path, output_path
                    )
                elif narration_path:
                    current_video = self._add_audio_to_video(
                        current_video, narration_path, output_path
                    )
                elif bgm_path:
                    current_video = self._add_audio_to_video(
                        current_video, bgm_path, output_path, volume=0.3
                    )
        else:
            # 오디오 없으면 영상만 복사
            self._copy_file(current_video, output_path)
            current_video = output_path

        return output_path

    def _image_to_static_video(
        self,
        image_path: str,
        duration_sec: float,
        output_path: str
    ) -> str:
        """정적 이미지를 영상으로 변환 (Ken Burns 없음)."""
        cmd = [
            "ffmpeg",
            "-y",
            "-loop", "1",
            "-i", image_path,
            "-c:v", "libx264",
            "-preset", "ultrafast",
            "-t", str(duration_sec),
            "-pix_fmt", "yuv420p",
            "-vf", f"scale={self.width}:{self.height}:force_original_aspect_ratio=decrease,"
                   f"pad={self.width}:{self.height}:(ow-iw)/2:(oh-ih)/2",
            "-r", str(self.fps),
            output_path
        ]

        result = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8', errors='replace', timeout=240)

        if result.returncode != 0:
            raise RuntimeError(f"Image to video conversion failed: {result.stderr}")

        return output_path

    def _copy_file(self, src: str, dst: str):
        """파일 복사."""
        import shutil
        shutil.copy2(src, dst)

    # =========================================================================
    # v2.0: Film Look Post-Processing
    # =========================================================================

    def apply_film_look(
        self,
        video_in: str,
        out_path: str,
        grain_intensity: int = 10,
        saturation: float = 1.1,
        contrast: float = 1.05,
        brightness: float = 0.0,
        gamma: float = 1.0,
        vignette: bool = False
    ) -> str:
        """
        FFmpeg 필름 그레인 + 색보정 적용.

        v2.0: 시네마틱 필름 룩 후처리
        - 필름 그레인 노이즈 추가
        - 색상 보정 (saturation, contrast, brightness, gamma)
        - 선택적 비네팅 효과

        Args:
            video_in: 입력 영상 경로
            out_path: 출력 영상 경로
            grain_intensity: 노이즈 강도 (0-30, 기본값 10)
            saturation: 채도 (1.0 = 원본, 기본값 1.1 = 약간 증가)
            contrast: 대비 (1.0 = 원본, 기본값 1.05 = 약간 증가)
            brightness: 밝기 조정 (-1.0 ~ 1.0, 기본값 0.0)
            gamma: 감마 보정 (0.1 ~ 10.0, 기본값 1.0)
            vignette: 비네팅 효과 적용 여부

        Returns:
            출력 영상 경로
        """
        print(f"[FFmpeg] Applying film look to video...")
        print(f"  Input: {video_in}")
        print(f"  Grain: {grain_intensity}, Saturation: {saturation}, Contrast: {contrast}")

        # 필터 체인 구성
        filters = []

        # 1. Film grain (noise filter)
        # noise=alls=<intensity>:allf=t+u
        # t = temporal noise, u = uniform noise
        if grain_intensity > 0:
            filters.append(f"noise=alls={grain_intensity}:allf=t+u")

        # 2. Color correction (eq filter)
        # eq=saturation=<sat>:contrast=<con>:brightness=<br>:gamma=<gam>
        eq_params = []
        if saturation != 1.0:
            eq_params.append(f"saturation={saturation}")
        if contrast != 1.0:
            eq_params.append(f"contrast={contrast}")
        if brightness != 0.0:
            eq_params.append(f"brightness={brightness}")
        if gamma != 1.0:
            eq_params.append(f"gamma={gamma}")

        if eq_params:
            filters.append(f"eq={':'.join(eq_params)}")

        # 3. Vignette effect (optional)
        if vignette:
            # vignette=a=PI/4:mode=backward
            filters.append("vignette=a=PI/6:mode=backward")

        # 필터 체인 조합
        if not filters:
            print("  No filters to apply, copying file...")
            self._copy_file(video_in, out_path)
            return out_path

        vf_filter = ",".join(filters)
        print(f"  Filter chain: {vf_filter}")

        cmd = [
            "ffmpeg",
            "-y",
            "-i", video_in,
            "-vf", vf_filter,
            "-c:v", "libx264",
            "-preset", "fast",
            "-crf", "23",
            "-c:a", "copy",  # 오디오는 그대로 복사
            out_path
        ]

        result = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8', errors='replace', timeout=300)

        if result.returncode != 0:
            print(f"[ERROR] Film look failed: {result.stderr[-500:]}")
            # 실패 시 원본 복사
            self._copy_file(video_in, out_path)
        else:
            print(f"[SUCCESS] Film look applied: {out_path}")

        return out_path

    # =========================================================================
    # Final Encode (최종 인코딩 + 워터마크)
    # =========================================================================

    def final_encode(
        self,
        video_in: str,
        audio_path: str,
        out_path: str,
        watermark_text: Optional[str] = None,
        watermark_opacity: float = 0.3,
        audio_bitrate: str = "192k",
    ) -> str:
        """
        최종 영상 인코딩: H.264 High Profile + AAC + 조건부 워터마크.

        Args:
            video_in: 입력 영상 (자막 포함)
            audio_path: 오디오 파일 (음악)
            out_path: 최종 출력 영상 경로
            watermark_text: 워터마크 텍스트 (None이면 워터마크 없음)
            watermark_opacity: 워터마크 투명도 (0~1)
            audio_bitrate: 오디오 비트레이트

        Returns:
            출력 영상 경로
        """
        print(f"[FFmpeg] Final encode: {video_in}")

        video_in_abs = os.path.abspath(video_in)
        audio_abs = os.path.abspath(audio_path)
        out_abs = os.path.abspath(out_path)

        if watermark_text:
            # 워터마크 포함 인코딩
            opacity_hex = f"{watermark_opacity:.2f}"
            filter_complex = (
                f"[0:v]drawtext=text='{watermark_text}':"
                f"fontsize=24:fontcolor=white@{opacity_hex}:"
                f"x=w-tw-20:y=20[vout]"
            )
            cmd = [
                "ffmpeg", "-y",
                "-i", video_in_abs,
                "-i", audio_abs,
                "-filter_complex", filter_complex,
                "-map", "[vout]", "-map", "1:a",
                "-c:v", "libx264", "-profile:v", "high",
                "-preset", "medium", "-crf", "20",
                "-pix_fmt", "yuv420p", "-r", str(self.fps),
                "-c:a", "aac", "-b:a", audio_bitrate, "-ar", "48000",
                "-shortest",
                out_abs
            ]
        else:
            # 워터마크 없이 인코딩
            cmd = [
                "ffmpeg", "-y",
                "-i", video_in_abs,
                "-i", audio_abs,
                "-map", "0:v", "-map", "1:a",
                "-c:v", "libx264", "-profile:v", "high",
                "-preset", "medium", "-crf", "20",
                "-pix_fmt", "yuv420p", "-r", str(self.fps),
                "-c:a", "aac", "-b:a", audio_bitrate, "-ar", "48000",
                "-shortest",
                out_abs
            ]

        # 동적 timeout: 영상 길이 기반
        try:
            duration = self.get_video_duration(video_in_abs)
            timeout = max(300, int(duration * 4))
        except Exception:
            timeout = 600

        print(f"  Preset: fast, CRF: 22, Audio: AAC {audio_bitrate}")
        print(f"  Watermark: {'Yes - ' + watermark_text if watermark_text else 'None'}")
        print(f"  Timeout: {timeout}s")

        result = subprocess.run(
            cmd, capture_output=True, text=True,
            encoding='utf-8', errors='replace', timeout=timeout
        )

        if result.returncode != 0:
            print(f"[ERROR] Final encode failed: {result.stderr[-500:]}")
            # 폴백: 단순 오디오 합성
            print(f"[FALLBACK] Trying simple audio merge...")
            self._add_audio_to_video(video_in_abs, audio_abs, out_abs)
        else:
            print(f"[SUCCESS] Final encode complete: {out_abs}")

        return out_path
