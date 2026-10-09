"""H.264 annotated match video: the same overlays as the MJPEG stream, at the source's own frame rate.

The MJPEG endpoint pushes one JPEG per *analysis* frame - five a second - which is honest for a live overlay
but leaves the source's detail (and its 60 fps) on the table. This module drives the same renderer at whatever
rate the caller asks for, interpolating between analysis samples (``AnnotatedMatch.render_at``), and hands the
frames to an H.264 encoder over a pipe:

* :func:`encode_clip` - a bounded window encoded into an MP4 with its index at the front, so a player can seek
  it (the server caches these and serves them with range requests);
* :func:`iter_live_chunks` - the endless variant: a fragmented MP4 streamed out as the frames are made, paced
  to real time, which browsers play from a plain ``<video>`` (no MSE, no HLS).

What the box can *sustain* is the caller's to know: this machine decodes full 4K at ~28 fps, so a live 4K60
stream cannot keep up, while a bounded 4K clip simply takes longer to encode. NVENC is preferred, x264 is the
fallback; the frames never touch disk - only a finished clip does.
"""

from __future__ import annotations

import hashlib
import os
import select
import subprocess
import tempfile
import threading
import time
from collections import deque
from contextlib import contextmanager, suppress
from functools import lru_cache
from pathlib import Path

import numpy as np

from soccer_analytics.ingest.ffmpeg_reader import FFmpegFrameReader

# Widths the encoded routes will serve. Above the MJPEG cap on purpose: this is the endpoint for the full
# picture. 3840 is what the source cameras record; anything wider is a mistake, not a request.
MAX_VIDEO_WIDTH = 3840
# A live stream defaults to 1080p: the decode has to run ahead of the encoder for a stream to be live at all,
# and this box decodes 4K at ~28 fps - fine for a clip, not enough for 60 fps of anything.
DEFAULT_LIVE_WIDTH = 1920
DEFAULT_CLIP_SECONDS = 10.0
MAX_CLIP_SECONDS = 300.0
CLIP_CACHE_TTL_S = 900.0
CLIP_CACHE_MAX = 8
_CHUNK_BYTES = 64 * 1024


class VideoError(RuntimeError):
    """An encode that could not start, keep up, or finish - the route turns it into an error response."""


class _NeverStops:
    stopped = False


_NEVER_STOPPED = _NeverStops()


@lru_cache(maxsize=1)
def pick_encoder() -> str:
    """The H.264 encoder this ffmpeg can actually open: NVENC when the driver takes it, else CPU x264.

    Probed with two tiny frames rather than assumed: a machine with an NVIDIA card but no encode permission
    (or a filled session table) must fall back to a working encoder, not fail every request. Cached: the answer
    cannot change while the process lives.
    """
    for encoder in ("h264_nvenc", "libx264"):
        command = [
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
            "-f", "lavfi", "-i", "nullsrc=size=64x64:rate=10",
            "-frames:v", "2", "-c:v", encoder, "-f", "null", "-",
        ]
        if subprocess.run(command, capture_output=True).returncode == 0:
            return encoder
    raise VideoError("no usable H.264 encoder (tried h264_nvenc and libx264)")


def encoder_command(
    *, width: int, height: int, fps: float, output: str, live: bool, encoder: str, keyframe_s: float
) -> list[str]:
    """The ffmpeg invocation both encoders share: raw BGR frames in, H.264 MP4 out.

    ``live`` picks the fragmented-MP4 flags - a stream a player can join mid-flight - and a low-latency tune;
    a clip gets the index at the front instead (``faststart``), which is what makes it seekable, and a slower,
    denser tune. The keyframe interval doubles as the fragment interval for a live stream: about a second of
    video per fragment is what keeps a viewer near the live edge without chopping the stream into confetti.
    """
    if encoder.endswith("nvenc"):
        quality = ["-preset", "p4", "-tune", "ll" if live else "hq", "-rc", "vbr", "-cq", "23"]
    else:
        quality = ["-preset", "veryfast" if live else "medium", "-crf", "20"]
    gop = max(1, int(round(fps * keyframe_s)))
    finishing = (
        ["-movflags", "+frag_keyframe+empty_moov+default_base_moof", "-f", "mp4", output]
        if live
        else ["-movflags", "+faststart", "-f", "mp4", output]
    )
    return [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin", "-y",
        "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{width}x{height}", "-framerate", f"{fps:g}", "-i", "pipe:0",
        "-an", "-c:v", encoder, *quality, "-pix_fmt", "yuv420p", "-g", str(gop), "-keyint_min", str(gop),
        *finishing,
    ]


class _StderrTail:
    """The last lines an encoder printed, drained on a thread so a chatty process cannot block on a full pipe."""

    def __init__(self, stream, limit: int = 20):
        self._stream = stream
        self._lines: deque[str] = deque(maxlen=limit)
        self._thread = threading.Thread(target=self._drain, name="match-video-stderr", daemon=True)
        self._thread.start()

    def _drain(self) -> None:
        try:
            for raw in self._stream:
                line = raw.decode(errors="replace").strip()
                if line:
                    self._lines.append(line)
        except (ValueError, OSError):
            pass

    def text(self) -> str:
        return " | ".join(self._lines)

    def close(self) -> None:
        with suppress(OSError, ValueError):
            self._stream.close()


def _chain(first, rest):
    """The first frame back in front of the rest of the reader."""
    yield first
    yield from rest


def _start(encoder: str, *, fps: float, width: int, height: int, output: str, live: bool, keyframe_s: float):
    command = encoder_command(
        width=width, height=height, fps=fps, output=output, live=live, encoder=encoder, keyframe_s=keyframe_s
    )
    return subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)


def _first_frame(reader):
    frames = reader.frames()
    first = next(frames, None)
    if first is None:
        raise VideoError("no frames decoded for the requested window")
    return first, frames


def encode_clip(
    match,
    *,
    start_s: float,
    duration_s: float,
    fps: float,
    width: int,
    overlays: dict,
    output: Path,
    encoder: str | None = None,
    reader_factory=FFmpegFrameReader,
    on_progress=None,
) -> int:
    """Encode ``[start_s, start_s + duration_s]`` of the annotated match into ``output``; returns frame count.

    The frame size is taken from the first decoded frame rather than computed: the height follows the source's
    aspect, and the reader owns that decision - a raw-video pipe that disagrees with the frames it receives
    corrupts the encode. Encoding to a file means nothing has to drain the encoder's output, so this is a
    straight loop: decode, draw, write.
    """
    encoder = encoder or pick_encoder()
    reader = reader_factory(match.video, fps=fps, width=width, start_s=start_s, duration_s=duration_s)
    first, rest = _first_frame(reader)
    _timestamp, frame = first
    height, actual_width = frame.shape[:2]
    process = _start(
        encoder, fps=fps, width=actual_width, height=height, output=str(output), live=False, keyframe_s=2.0
    )
    stderr = _StderrTail(process.stderr)
    written = 0
    try:
        for timestamp, frame in _chain(first, rest):
            match.render_at(frame, timestamp, **overlays)
            process.stdin.write(np.ascontiguousarray(frame).tobytes())
            written += 1
            if on_progress is not None:
                on_progress(float(np.clip((timestamp - start_s) / max(duration_s, 1e-6), 0.0, 1.0)))
        process.stdin.close()
        code = process.wait()
    except (BrokenPipeError, OSError) as error:
        process.kill()
        process.wait()
        output.unlink(missing_ok=True)
        raise VideoError(f"the encoder stopped early: {stderr.text() or error}") from error
    finally:
        stderr.close()
    if code != 0:
        output.unlink(missing_ok=True)
        raise VideoError(f"ffmpeg exited with code {code}: {stderr.text()}")
    return written


def iter_live_chunks(
    match,
    *,
    start_s: float,
    fps: float,
    width: int,
    overlays: dict,
    control=None,
    encoder: str | None = None,
    reader_factory=FFmpegFrameReader,
    pace: bool = True,
):
    """Yield the annotated match as a fragmented MP4, chunk by chunk, until the window ends or ``control`` stops.

    A writer thread renders and feeds the encoder; this generator drains the encoder's output - and so paces
    the pipeline: nothing is produced much faster than the client reads it, and the writer's own pacing keeps
    the stream near real time (a decode that outruns the declared rate would otherwise run minutes ahead of
    the viewer, buffered in the player). ``control`` is the MJPEG stream's ``StreamControl``, so ``/stop`` ends
    both kinds of stream the same way.
    """
    encoder = encoder or pick_encoder()
    end_s = match.start_s + match.frame_count / match.fps
    stop = control if control is not None else _NEVER_STOPPED
    reader = reader_factory(
        match.video, fps=fps, width=width, start_s=start_s, duration_s=max(0.1, end_s - start_s)
    )
    first, rest = _first_frame(reader)
    _timestamp, frame = first
    height, actual_width = frame.shape[:2]
    process = _start(
        encoder, fps=fps, width=actual_width, height=height, output="pipe:1", live=True, keyframe_s=1.0
    )
    stderr = _StderrTail(process.stderr)
    failures: list[Exception] = []

    def produce() -> None:
        period = 1.0 / max(0.01, fps)
        next_frame_at = time.monotonic()
        try:
            for timestamp, frame in _chain(first, rest):
                if stop.stopped:
                    break
                match.render_at(frame, timestamp, **overlays)
                process.stdin.write(np.ascontiguousarray(frame).tobytes())
                if pace:
                    next_frame_at += period
                    delay = next_frame_at - time.monotonic()
                    if delay > 0.0:
                        time.sleep(delay)
                    else:
                        next_frame_at = time.monotonic()  # the decode is the cap; just re-anchor
        except (BrokenPipeError, OSError, ValueError) as error:
            failures.append(error)  # reported below; the reader loop is the one that decides the response
        finally:
            with suppress(OSError, ValueError):
                process.stdin.close()

    writer = threading.Thread(target=produce, name="match-video-writer", daemon=True)
    writer.start()
    try:
        while True:
            if stop.stopped:
                break
            ready, _, _ = select.select([process.stdout], [], [], 0.5)
            if not ready:
                if process.poll() is not None:
                    break
                continue
            chunk = os.read(process.stdout.fileno(), _CHUNK_BYTES)
            if not chunk:
                break
            yield chunk
    finally:
        if process.poll() is None:
            process.kill()
        process.wait()
        writer.join(timeout=5.0)
        stderr.close()
    if failures and not stop.stopped:
        raise VideoError(f"the encoder failed: {stderr.text() or failures[0]}")


def clip_key(match_id: str, start_s: float, duration_s: float, fps: float, width: int, overlays: dict) -> str:
    """The identity of an encode: any difference in any of these is a different file."""
    layers = ",".join(sorted(name for name, on in overlays.items() if on))
    return f"{match_id}-{start_s:.2f}-{duration_s:.2f}-{fps:g}-{width}-{layers or 'none'}"


def _file_name(key: str) -> str:
    digest = hashlib.sha1(key.encode("utf-8")).hexdigest()[:8]
    safe = "".join(ch if (ch.isalnum() or ch in "-_.") else "_" for ch in key)[:80]
    return f"{safe}-{digest}.mp4"


class ClipCache:
    """Finished clips on disk, kept for a while because a player asks for the same clip more than once.

    A clip request blocks for the whole encode, so the file it produces is worth keeping: the player comes back
    for byte ranges as the viewer scrubs, and a reload asks again. Entries expire by age and by count and their
    files are deleted then; an encode that is still running holds its key (see :meth:`claim`), so two requests
    for one clip never write the same file.
    """

    def __init__(
        self,
        *,
        directory: str | Path | None = None,
        ttl_s: float = CLIP_CACHE_TTL_S,
        max_entries: int = CLIP_CACHE_MAX,
    ):
        self.directory = (
            Path(directory) if directory is not None else Path(tempfile.mkdtemp(prefix="match-video-"))
        )
        self.ttl_s = float(ttl_s)
        self.max_entries = int(max_entries)
        self._entries: dict[str, tuple[Path, float]] = {}
        self._locks: dict[str, threading.Lock] = {}
        self._lock = threading.Lock()

    def lookup(self, key: str) -> Path | None:
        """The finished clip for a key, if it is still there; expired entries are dropped as they are noticed."""
        now = time.monotonic()
        with self._lock:
            self._purge(now)
            entry = self._entries.get(key)
        if entry is None:
            return None
        path, _stored_at = entry
        if not path.exists():
            with self._lock:
                self._entries.pop(key, None)
            return None
        return path

    def target(self, key: str) -> Path:
        """Where the encode for a key should be written (not visible to lookups until :meth:`keep`)."""
        self.directory.mkdir(parents=True, exist_ok=True)
        return self.directory / _file_name(key)

    def keep(self, key: str, path: Path) -> None:
        now = time.monotonic()
        with self._lock:
            self._entries[key] = (path, now)
            self._purge(now)

    @contextmanager
    def claim(self, key: str):
        with self._lock:
            lock = self._locks.setdefault(key, threading.Lock())
        with lock:
            yield

    def _purge(self, now: float) -> None:
        expired = [key for key, (_path, stored_at) in self._entries.items() if now - stored_at > self.ttl_s]
        for key in expired:
            self._drop(key)
        if len(self._entries) > self.max_entries:
            oldest = sorted(self._entries.items(), key=lambda item: item[1][1])
            for key, _entry in oldest[: len(self._entries) - self.max_entries]:
                self._drop(key)

    def _drop(self, key: str) -> None:
        path, _stored_at = self._entries.pop(key, (None, 0.0))
        if path is not None:
            path.unlink(missing_ok=True)
        self._locks.pop(key, None)
