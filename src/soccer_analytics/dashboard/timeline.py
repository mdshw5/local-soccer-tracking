"""A scrubbable video timeline for choosing the frame to place landmarks on.

Step 2 originally offered a plain ``st.slider`` over the analysed frame index, and it felt wrong for two reasons:

* every notch of the slider is a full Streamlit rerun, so dragging re-ran the whole page - and re-read a frame from
  the 4K source with ffmpeg - many times a second, and
* the slider showed no picture, so you could not see what you were scrubbing towards.

A real timeline needs the picture to live in the browser, because a round trip per pixel is far too slow. So this
module builds a small, fast-to-seek H.264 proxy of *just the analysed segment* (cached beside the segment results);
the landmark viewport plays it, so scrubbing and aiming happen on the same picture and the drag never leaves the
browser. Scrubbing reports nothing to Python at all - the magnified crop is only re-read when the user aims at a
point, which is what keeps the drag smooth.

The proxy is only a navigator; the frame that is actually clicked is still read at full resolution from the source
by :func:`~soccer_analytics.ingest.ffmpeg_reader.grab_frame`. Nothing here ever moves a click.

Keep this module free of Streamlit: the ffmpeg plumbing and the time <-> frame mapping are the parts worth testing,
and they cannot be tested inside a Streamlit script.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from soccer_analytics.ingest.ffmpeg_reader import FFmpegError, probe_video, run_ffmpeg_with_progress

PROXY_WIDTH = 960  # wide enough to pick out pitch markings while scrubbing, small enough to seek instantly
PROXY_FPS = 30.0  # twice the 15 fps analysis rate, so a scrub never skips an analysed frame
PROXY_FILE = "proxy.mp4"
PROXY_DIR = "scrubber"  # sits under the segment directory, so it invalidates when the segment is rebuilt
BUILD_STATE_FILE = "scrubber_build.json"

# Long windows are built from keyframes alone. The proxy is only a navigator - whatever frame is landed on is
# re-read from the source at full resolution - and on this camera's footage (one keyframe per second) skipping the
# rest of the decode is 30-50x faster: measured on the real 4K game file, decoding every frame ran at 3.4x realtime
# and keyframes-only at 34-51x, and the build was never NVENC-bound (the encoder sat at 1%, the decoder at 100%).
# Below the threshold the window is short enough that every-frame decoding is cheap, and the smoother scrub it gives
# is worth having for clicking landmarks.
KEYFRAME_ONLY_AFTER_S = 900.0
KEYFRAME_SKIP = "nokey"


def proxy_skip_frame(duration_s: float) -> str | None:
    """``"nokey"`` when a window is long enough to build from keyframes alone, else ``None`` (decode everything)."""
    return KEYFRAME_SKIP if duration_s >= KEYFRAME_ONLY_AFTER_S else None
REPO_ROOT = Path(__file__).resolve().parents[3]
BUILD_SCRIPT = REPO_ROOT / "scripts" / "run_timeline_proxy.py"

# Live builds, so a later rerun can reap a finished child (and never mistake a zombie for a running build).
_BUILD_PROCESSES: dict[str, subprocess.Popen] = {}


def _even(value: float) -> int:
    """H.264 wants even dimensions; round to the nearest, never below two."""
    return max(2, int(round(value / 2.0)) * 2)


def scrubber_dir(segment_dir: str | Path) -> Path:
    """Where the segment's timeline proxy is cached."""
    return Path(segment_dir) / PROXY_DIR


def proxy_path(segment_dir: str | Path) -> Path:
    return scrubber_dir(segment_dir) / PROXY_FILE


def proxy_is_ready(segment_dir: str | Path) -> bool:
    """Whether the segment's timeline proxy exists and can be played."""
    return proxy_path(segment_dir).exists()


def frame_time(times: np.ndarray | list[float], index: int) -> float:
    """Source seconds of an analysed frame index, clamped to the segment."""
    if len(times) == 0:
        raise ValueError("no frames in segment")
    return float(np.asarray(times)[int(np.clip(index, 0, len(times) - 1))])


def nearest_frame_index(times: np.ndarray | list[float], t_source: float) -> int:
    """Index of the analysed frame closest in source time to ``t_source``, clamped into range."""
    if len(times) == 0:
        return 0
    return int(np.clip(int(np.abs(np.asarray(times) - float(t_source)).argmin()), 0, len(times) - 1))


def proxy_command(
    video_path: str | Path,
    tmp_path: str | Path,
    *,
    start_s: float,
    duration_s: float,
    width: int,
    height: int,
    fps: float,
    gop: int,
    skip_frame: str | None,
    use_gpu: bool,
) -> list[str]:
    """The ffmpeg command for one proxy attempt. Separate from the running of it so the flags can be tested."""
    command = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin", "-y"]
    if skip_frame:
        # An input option: the decoder is told not to decode frames nobody asked for. On a 4K60 camera that is the
        # difference between ~24 minutes and ~2 for a full game, and the cost is temporal: one picture per second
        # instead of a smooth video. Scrub position and timestamps stay exact.
        command += ["-skip_frame", skip_frame]
    if use_gpu:
        command += ["-hwaccel", "cuda", "-hwaccel_output_format", "cuda"]
    command += ["-ss", f"{start_s:.3f}", "-i", str(video_path), "-an", "-t", f"{duration_s:.3f}"]
    if use_gpu:
        # Decode and scale on the GPU, then hand NVENC system-memory nv12. Forcing `-pix_fmt yuv420p` here makes
        # ffmpeg try to swscale straight out of the CUDA frames, which it cannot do; the explicit `hwdownload`
        # is what bridges the two, and NVENC takes nv12 directly.
        video_filter = f"fps={fps},scale_cuda={width}:{height},hwdownload,format=nv12"
        encoder = ["-c:v", "h264_nvenc", "-preset", "p4", "-cq", "38"]
        pixel_format: list[str] = []
    else:
        video_filter = f"fps={fps},scale={width}:{height}"
        encoder = ["-c:v", "libx264", "-preset", "veryfast", "-crf", "31"]
        pixel_format = ["-pix_fmt", "yuv420p"]
    return command + [
        "-vf", video_filter,
        *encoder,
        *pixel_format,
        "-g", str(gop),
        "-movflags", "+faststart",
        "-progress", "pipe:1", "-nostats",
        str(tmp_path),
    ]


def build_proxy(
    video_path: str | Path,
    segment_dir: str | Path,
    *,
    start_s: float,
    duration_s: float,
    width: int = PROXY_WIDTH,
    fps: float = PROXY_FPS,
    prefer_gpu: bool = True,
    skip_frame: str | None = None,
    on_progress=None,
) -> Path:
    """Encode a small, seek-friendly H.264 proxy of ``[start_s, start_s + duration_s)`` and cache it.

    The picture is scaled down and the keyframe interval kept at one second, so the browser can jump anywhere in it
    promptly. Encoding writes to a temporary file and renames into place, so a half-written proxy is never mistaken
    for a finished one. ``on_progress(fraction)`` follows the encode. Raises :class:`FFmpegError` if neither the GPU
    nor the software encoder works.

    ``skip_frame`` is passed to the decoder; :func:`proxy_skip_frame` decides it from the window length. It trades
    temporal smoothness (one picture per second on this footage) for speed, and nothing else: the frame that is
    finally clicked is still read from the source at full resolution.
    """
    video_path = Path(video_path)
    probe = probe_video(video_path)
    out_width = _even(width)
    out_height = _even(out_width * probe.height / probe.width)
    out_path = proxy_path(segment_dir)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.with_name(out_path.name + ".tmp.mp4")
    gop = max(1, int(round(fps)))

    last_error = ""
    for use_gpu in ([True, False] if prefer_gpu else [False]):
        if tmp_path.exists():
            tmp_path.unlink()
        command = proxy_command(
            video_path,
            tmp_path,
            start_s=start_s,
            duration_s=duration_s,
            width=out_width,
            height=out_height,
            fps=fps,
            gop=gop,
            skip_frame=skip_frame,
            use_gpu=use_gpu,
        )
        returncode, output = run_ffmpeg_with_progress(command, duration_s, on_progress)
        if returncode == 0 and tmp_path.exists() and tmp_path.stat().st_size > 0:
            os.replace(tmp_path, out_path)
            if on_progress is not None:
                on_progress(1.0)
            return out_path
        last_error = output or f"ffmpeg exited with {returncode}"
    if tmp_path.exists():
        tmp_path.unlink()
    raise FFmpegError(f"could not build a timeline proxy for {video_path}: {last_error}")


# --------------------------------------------------------------------------------------------------------------
# Running the build in the background
# --------------------------------------------------------------------------------------------------------------
# A proxy is a full pass over the segment, which for 4K/60 footage is minutes - far too long to hold the page on.
# So it runs as its own process, exactly like Stage A, and the dashboard shows a slider until it is ready.


def build_state_path(segment_dir: str | Path) -> Path:
    return Path(segment_dir) / BUILD_STATE_FILE


def read_build_state(segment_dir: str | Path) -> dict:
    path = build_state_path(segment_dir)
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return {}


def write_build_state(segment_dir: str | Path, **changes) -> dict:
    """Merge ``changes`` into the segment's build state, written atomically so a reader never sees a half file."""
    segment_dir = Path(segment_dir)
    segment_dir.mkdir(parents=True, exist_ok=True)
    state = read_build_state(segment_dir)
    state.update(changes)
    path = build_state_path(segment_dir)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(state, indent=2))
    os.replace(tmp, path)
    return state


def process_alive(pid: object) -> bool:
    """Whether a pid is running. Used only to decide if a build that never wrote a finish state died."""
    if not isinstance(pid, int) or pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:  # exists but owned by someone else - still alive
        return True
    return True


def build_in_progress(segment_dir: str | Path) -> bool:
    """True while a build is running, whether it was started in this server process or an earlier one."""
    process = _BUILD_PROCESSES.get(str(segment_dir))
    if process is not None and process.poll() is None:
        return True
    state = read_build_state(segment_dir)
    return state.get("state") == "running" and process_alive(state.get("pid"))


def start_background_build(
    segment_dir: str | Path,
    video_path: str | Path,
    *,
    start_s: float,
    duration_s: float,
    prefer_gpu: bool = True,
) -> bool:
    """Start the proxy build as its own process. Returns ``True`` when a new build was started."""
    if proxy_is_ready(segment_dir) or build_in_progress(segment_dir):
        return False
    command = [
        sys.executable,
        str(BUILD_SCRIPT),
        "--video", str(video_path),
        "--out", str(segment_dir),
        "--start", f"{start_s:.3f}",
        "--duration", f"{duration_s:.3f}",
    ]
    if not prefer_gpu:
        command.append("--cpu")
    process = subprocess.Popen(
        command, cwd=str(REPO_ROOT), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
    )
    _BUILD_PROCESSES[str(segment_dir)] = process
    write_build_state(segment_dir, state="running", pid=process.pid, started=time.time(), error=None)
    return True
