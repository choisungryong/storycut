"""
FFmpeg Utilities for Video Composition

P0 핵심 기능:
- Ken Burns Effect: 이미지 기반 씬을 영상처럼 변환
- Audio Ducking: 내레이션 시 BGM 자동 감쇠
- Subtitle Burn-in: 자막을 영상에 직접 렌더링
"""

import subprocess
import os
import re
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any
import sys

sys.path.append(str(Path(__file__).parent.parent))


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

    def ken_burns_clip(
        self,
        image_path: str,
        duration_sec: float,
        out_path: str,
        effect_type: str = "zoom_in",
        zoom_range: Tuple[float, float] = (1.0, 1.1)  # Minimal zoom for calm vibe
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

        filter_str = effects.get(effect_type, effects["zoom_in"])

        cmd = [
            "ffmpeg",
            "-y",
            "-loop", "1",
            "-i", image_path,
            "-vf", filter_str,
            "-c:v", "libx264",
            "-t", str(duration_sec),
            "-pix_fmt", "yuv420p",
            "-r", str(self.fps),
            out_path
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            raise RuntimeError(f"Ken Burns effect failed: {result.stderr}")

        return out_path

    # =========================================================================
    # P0: Subtitle Overlay (Burn-in)
    # =========================================================================

    def overlay_subtitles(
        self,
        video_in: str,
        srt_path: str,
        out_path: str,
        style: Optional[Dict[str, Any]] = None
    ) -> str:
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
        
        # ASS 스타일 문자열 생성
        force_style = (
            f"FontName={style.get('font_name', default_font)},"
            f"FontSize={style.get('font_size', 24)}," # Adjusted size
            f"PrimaryColour={style.get('primary_color', '&HFFFFFF')},"
            f"OutlineColour={style.get('outline_color', '&H000000')},"
            f"Outline={style.get('outline_width', 2)},"
            f"MarginV={style.get('margin_v', 20)},"
            f"Bold=1" 
        )

        # Windows 경로 처리 (백슬래시 이스케이프)
        srt_path_escaped = srt_path.replace("\\", "/").replace(":", "\\:")
        
        # fontsdir 옵션은 로컬 파일 사용시에만 필요하므로 제거 (Nixpacks로 시스템 설치됨)
        # vf_filter = f"subtitles='{srt_path_escaped}':fontsdir='{fonts_dir}':force_style='{force_style}'"
        vf_filter = f"subtitles='{srt_path_escaped}':force_style='{force_style}'"

        # DEBUG: SRT 파일 존재 확인
        if not os.path.exists(srt_path):
            print(f"[ERROR] SRT file does not exist: {srt_path}")
            return video_in
        
        # DEBUG: SRT 파일 내용 일부 출력
        try:
            with open(srt_path, 'r', encoding='utf-8') as f:
                srt_preview = f.read(200)
                print(f"[DEBUG] SRT file exists: {srt_path}")
                print(f"[DEBUG] SRT preview (first 200 chars): {srt_preview}")
        except Exception as e:
            print(f"[ERROR] Cannot read SRT file: {e}")
            return video_in

        cmd = [
            "ffmpeg",
            "-y",
            "-i", video_in,
            "-vf", vf_filter,
            out_path
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            print(f"[ERROR] Subtitle overlay failed!")
            print(f"[ERROR] FFmpeg command: {' '.join(cmd)}")
            print(f"[ERROR] FFmpeg stderr (full):")
            print(result.stderr)
            print(f"[ERROR] FFmpeg stdout:")
            print(result.stdout)
            # 자막 실패 시 원본 복사로 폴백
            self._copy_file(video_in, out_path)

        return out_path

    def generate_srt_from_scenes(
        self,
        scenes: List[Dict[str, Any]],
        output_path: str
    ) -> str:
        """
        Scene 목록에서 SRT 자막 파일 생성.

        Args:
            scenes: Scene 데이터 목록
            output_path: SRT 파일 출력 경로

        Returns:
            SRT 파일 경로
        """
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

        srt_content = []
        current_time_ms = 0

        for i, scene in enumerate(scenes, start=1):
            # Scene에서 타이밍 정보 추출
            duration_sec = scene.get("duration_sec", 5)
            duration_ms = duration_sec * 1000

            start_time = self._ms_to_srt_time(current_time_ms)
            end_time = self._ms_to_srt_time(current_time_ms + duration_ms)

            # 자막 텍스트 (narration 또는 sentence)
            text = scene.get("narration") or scene.get("sentence", "")

            if text:
                srt_content.append(f"{i}")
                srt_content.append(f"{start_time} --> {end_time}")
                srt_content.append(text)
                srt_content.append("")

            current_time_ms += duration_ms

        with open(output_path, "w", encoding="utf-8") as f:
            f.write("\n".join(srt_content))

        return output_path

    def _ms_to_srt_time(self, ms: int) -> str:
        """밀리초를 SRT 시간 형식으로 변환."""
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
        threshold = ducking_config.get("threshold", 0.02)
        ratio = ducking_config.get("ratio", 10)
        attack = ducking_config.get("attack_ms", 20)
        release = ducking_config.get("release_ms", 200)

        # sidechaincompress를 사용한 Audio Ducking
        filter_complex = (
            f"[1:a]aformat=fltp:44100:stereo,volume=1.0[narr];"
            f"[2:a]aformat=fltp:44100:stereo,volume={ducking_config.get('bgm_volume_normal', 0.3)}[bgm];"
            f"[bgm][narr]sidechaincompress="
            f"threshold={threshold}:ratio={ratio}:attack={attack}:release={release}[bgm_ducked];"
            f"[bgm_ducked][narr]amix=inputs=2:duration=first:normalize=0[aout]"
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

        result = subprocess.run(cmd, capture_output=True, text=True)

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
            "[narr][bgm]amix=inputs=2:duration=first:normalize=0[aout]"
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

        result = subprocess.run(cmd, capture_output=True, text=True)

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

        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            raise RuntimeError(f"Failed to add audio: {result.stderr}")

        return out_path

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

        # FFmpeg concat 리스트 파일 생성
        concat_file = "temp_concat_list.txt"
        with open(concat_file, "w", encoding="utf-8") as f:
            for video_path in video_paths:
                f.write(f"file '{os.path.abspath(video_path)}'\n")

        try:
            cmd = [
                "ffmpeg",
                "-y",
                "-f", "concat",
                "-safe", "0",
                "-i", concat_file,
                "-c", "copy",
                output_path
            ]

            result = subprocess.run(cmd, capture_output=True, text=True)

            if result.returncode != 0:
                print(f"FFmpeg error: {result.stderr}")
                return False

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
        music_volume: float = 0.2
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

        # 내레이션 연결
        narration_concat = "temp_narration.wav"
        self._concatenate_audio(narration_paths, narration_concat)

        # FFmpeg 명령 구성
        inputs = ["-i", video_path, "-i", narration_concat]

        if music_path and os.path.exists(music_path):
            # 음악이 있을 때: 내레이션과 음악 믹스
            inputs.extend(["-i", music_path])
            filter_complex = (
                f"[1:a]volume=1.0[narr];"
                f"[2:a]volume={music_volume}[mus];"
                f"[narr][mus]amix=inputs=2:duration=first[aout]"
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

        result = subprocess.run(cmd, capture_output=True, text=True)

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

    def _concatenate_audio(self, audio_paths: List[str], output_path: str):
        """여러 오디오 파일을 하나로 연결."""
        # 파일 존재 확인
        for audio_path in audio_paths:
            if not os.path.exists(audio_path):
                raise FileNotFoundError(f"Audio file not found: {audio_path}")

        concat_file = "temp_audio_concat.txt"
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

        result = subprocess.run(cmd, capture_output=True, text=True)

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

        result = subprocess.run(cmd, capture_output=True, text=True)

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
        output_path: str
    ) -> str:
        """
        전체 합성 파이프라인: 영상 연결, 오디오 믹스, 최종 MP4 출력.

        Args:
            video_clips: 씬 영상 파일 목록 (순서대로)
            narration_clips: 내레이션 오디오 파일 목록 (순서대로)
            music_path: 배경 음악 파일
            output_path: 최종 출력 영상 경로

        Returns:
            최종 영상 파일 경로
        """
        print("Starting video composition...")

        # Step 1: 영상 클립 연결
        print("  -> Concatenating video clips...")
        temp_video = "temp_concatenated.mp4"
        self.concatenate_videos(video_clips, temp_video)

        # Step 2: 오디오 믹스 (내레이션 + BGM)
        print("  -> Mixing audio (narration + background music)...")
        final_video = self.mix_audio(
            temp_video,
            narration_clips,
            music_path,
            output_path
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
            current_video = self.overlay_subtitles(
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
            "-t", str(duration_sec),
            "-pix_fmt", "yuv420p",
            "-vf", f"scale={self.width}:{self.height}:force_original_aspect_ratio=decrease,"
                   f"pad={self.width}:{self.height}:(ow-iw)/2:(oh-ih)/2",
            "-r", str(self.fps),
            output_path
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            raise RuntimeError(f"Image to video conversion failed: {result.stderr}")

        return output_path

    def _copy_file(self, src: str, dst: str):
        """파일 복사."""
        import shutil
        shutil.copy2(src, dst)
