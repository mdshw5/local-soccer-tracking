"""ffmpeg-backed frame and audio I/O for long, high-resolution gimbal footage.

OpenCV's `VideoCapture` decodes 4K/60 HEVC on the CPU at roughly real-time, which makes a 20+
minute segment unusable. This reader pipes frames out of ffmpeg instead, decoding on NVDEC and
scaling on the GPU (`scale_cuda`) when available (~3x real-time on a Quadro P2200), with an automatic
software fallback. Frames are resampled to a constant analysis rate so downstream code never has to
deal with the camera's native 60 fps.
"""

from __future__ import annotations

import json
import subprocess
import tempfile
import wave
from collections import deque
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class VideoProbe:
    width: int
    height: int
    fps: float
    duration_s: float
    codec: str
    has_audio: bool


class FFmpegError(RuntimeError):
    pass


def run_ffmpeg_with_progress(
    command: list[str], expected_s: float, progress=None
) -> tuple[int, str]:
    """Run an ffmpeg command that carries ``-progress pipe:1``, reporting a real fraction to ``progress``.

    The encode's wall-clock progress is the only honest number for a cut: ffmpeg knows how far through the window it
    is, and ``out_time_us`` on the progress stream is it. stderr is folded into stdout so neither pipe can fill and
    stall the encode, and the tail of the output is returned for error messages (``-loglevel error`` keeps normal
    runs quiet). Returns ``(returncode, output_tail)``.
    """
    process = subprocess.Popen(
        command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, errors="replace"
    )
    tail: deque[str] = deque(maxlen=30)
    total_us = max(1.0, float(expected_s) * 1e6)
    assert process.stdout is not None
    for raw in process.stdout:
        line = raw.strip()
        if line.startswith("out_time_us="):
            value = line.partition("=")[2]
            if progress is not None and value.isdigit():
                progress(min(1.0, int(value) / total_us))
        elif line:
            tail.append(line)
    return process.wait(), "\n".join(tail)


def probe_video(path: str | Path) -> VideoProbe:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Video not found: {path}")
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-print_format", "json", "-show_streams", "-show_format", str(path)],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise FFmpegError(f"ffprobe failed for {path}: {result.stderr.strip()}")
    info = json.loads(result.stdout)
    video = next((s for s in info.get("streams", []) if s.get("codec_type") == "video"), None)
    if video is None:
        raise FFmpegError(f"No video stream in {path}")
    numerator, _, denominator = str(video.get("r_frame_rate", "0/1")).partition("/")
    fps = float(numerator) / float(denominator or 1) if float(denominator or 1) else 0.0
    duration = float(info.get("format", {}).get("duration") or video.get("duration") or 0.0)
    return VideoProbe(
        width=int(video["width"]),
        height=int(video["height"]),
        fps=fps,
        duration_s=duration,
        codec=str(video.get("codec_name", "")),
        has_audio=any(s.get("codec_type") == "audio" for s in info.get("streams", [])),
    )


def _even(value: float) -> int:
    return max(2, int(round(value / 2.0)) * 2)


class _StartupError(Exception):
    """ffmpeg exited before producing any frame (e.g. no CUDA); safe to retry another way."""


class FFmpegFrameReader:
    """Yields `(source_time_s, BGR frame)` at a constant `fps`, scaled to `width`."""

    def __init__(
        self,
        path: str | Path,
        *,
        fps: float = 10.0,
        width: int = 1920,
        start_s: float = 0.0,
        duration_s: float | None = None,
        prefer_gpu: bool = True,
    ):
        if fps <= 0:
            raise ValueError("fps must be > 0")
        if start_s < 0:
            raise ValueError("start_s must be >= 0")
        self.path = Path(path)
        self.probe = probe_video(self.path)
        self.fps = fps
        self.start_s = start_s
        self.duration_s = duration_s
        self.prefer_gpu = prefer_gpu
        self.out_width = _even(width)
        self.out_height = _even(self.out_width * self.probe.height / self.probe.width)
        self.used_gpu: bool | None = None

    def _command(self, use_gpu: bool) -> list[str]:
        command = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin"]
        if use_gpu:
            command += ["-hwaccel", "cuda", "-hwaccel_output_format", "cuda"]
        command += ["-ss", f"{self.start_s:.3f}", "-i", str(self.path), "-an"]
        if self.duration_s is not None:
            command += ["-t", f"{self.duration_s:.3f}"]
        if use_gpu:
            video_filter = (
                f"fps={self.fps},scale_cuda={self.out_width}:{self.out_height},"
                "hwdownload,format=nv12,format=bgr24"
            )
        else:
            video_filter = f"fps={self.fps},scale={self.out_width}:{self.out_height}"
        return command + ["-vf", video_filter, "-f", "rawvideo", "-pix_fmt", "bgr24", "-"]

    def _run(self, use_gpu: bool) -> Iterator[tuple[float, np.ndarray]]:
        frame_bytes = self.out_width * self.out_height * 3
        with tempfile.TemporaryFile() as stderr_file:
            process = subprocess.Popen(
                self._command(use_gpu), stdout=subprocess.PIPE, stderr=stderr_file
            )
            produced = 0
            try:
                while True:
                    buffer = bytearray(frame_bytes)
                    view = memoryview(buffer)
                    filled = 0
                    while filled < frame_bytes:
                        read = process.stdout.readinto(view[filled:])
                        if not read:
                            break
                        filled += read
                    if filled < frame_bytes:
                        break
                    frame = np.frombuffer(buffer, dtype=np.uint8).reshape(
                        self.out_height, self.out_width, 3
                    )
                    timestamp = self.start_s + produced / self.fps
                    # Count before yielding: an early-exiting consumer closes us at the yield, and the
                    # finally block must not mistake our own kill for a startup failure.
                    produced += 1
                    yield timestamp, frame
            finally:
                # SIGTERM alone deadlocks: ffmpeg's muxer thread is blocked writing into the pipe we
                # stopped reading, and ffmpeg waits for it. Close the pipe, then kill outright.
                if process.poll() is None:
                    process.stdout.close()
                    process.kill()
                process.wait()
                if produced == 0 and process.returncode not in (0, None):
                    stderr_file.seek(0)
                    raise _StartupError(stderr_file.read().decode(errors="replace").strip())

    def frames(self) -> Iterator[tuple[float, np.ndarray]]:
        modes = [True, False] if self.prefer_gpu else [False]
        last_error = ""
        for use_gpu in modes:
            try:
                self.used_gpu = use_gpu
                yield from self._run(use_gpu)
                return
            except _StartupError as exc:
                last_error = str(exc)
        raise FFmpegError(f"ffmpeg could not decode {self.path}: {last_error}")


def grab_frame(path: str | Path, time_s: float, *, width: int = 1600) -> np.ndarray | None:
    """Decodes a single frame near `time_s` (for browsing/calibration, not for analysis)."""
    reader = FFmpegFrameReader(path, fps=2.0, width=width, start_s=max(0.0, time_s), duration_s=1.2)
    for _t, frame in reader.frames():
        return frame
    return None


def extract_audio(
    path: str | Path,
    wav_path: str | Path,
    *,
    sample_rate: int = 16000,
    start_s: float = 0.0,
    duration_s: float | None = None,
    on_progress=None,
) -> Path:
    """Writes a mono 16-bit wav of the source's audio track.

    ``on_progress(fraction)`` follows the decode through :func:`run_ffmpeg_with_progress`, so a background task can
    show how far through a long recording it is - a full game takes long enough that a spinner tells the user
    nothing. The fraction is against ``duration_s`` when given, and against the rest of the file when it is not.
    The wav is written straight to ``wav_path``; callers that must not mistake a half-written file for a finished
    one use their own temporary name around this call.
    """
    wav_path = Path(wav_path)
    wav_path.parent.mkdir(parents=True, exist_ok=True)
    command = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin", "-y"]
    command += ["-ss", f"{start_s:.3f}", "-i", str(path), "-vn", "-ac", "1", "-ar", str(sample_rate)]
    if duration_s is not None:
        command += ["-t", f"{duration_s:.3f}"]
    command += ["-c:a", "pcm_s16le"]
    if on_progress is None:
        result = subprocess.run(command + [str(wav_path)], capture_output=True, text=True)
        if result.returncode != 0:
            raise FFmpegError(f"audio extraction failed for {path}: {result.stderr.strip()}")
        return wav_path
    if duration_s is not None:
        expected_s = float(duration_s)
    else:
        expected_s = max(1.0, float(probe_video(path).duration_s) - float(start_s))
    returncode, output = run_ffmpeg_with_progress(
        command + ["-progress", "pipe:1", "-nostats", str(wav_path)], expected_s, on_progress
    )
    if returncode != 0:
        raise FFmpegError(f"audio extraction failed for {path}: {output.strip()}")
    return wav_path


def read_wav_mono(path: str | Path) -> tuple[np.ndarray, int]:
    """Returns float32 samples in [-1, 1] and the sample rate."""
    with wave.open(str(path), "rb") as wav:
        sample_rate = wav.getframerate()
        samples = np.frombuffer(wav.readframes(wav.getnframes()), dtype=np.int16)
    return samples.astype(np.float32) / 32768.0, sample_rate
