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


# The encoder candidates per codec, best first. ``codec=hevc`` is an explicit choice, not the default: HEVC in
# MP4 is smaller at equal quality on this hardware, but browsers only play it where the platform decodes it
# (Chromium on this box does not), so the compatible H.264 stays the default and HEVC is for downloads and
# players that know it.
ENCODERS = {
    "h264": ("h264_nvenc", "libx264"),
    "hevc": ("hevc_nvenc", "libx265"),
}
CODECS = tuple(ENCODERS)


class _NeverStops:
    stopped = False


_NEVER_STOPPED = _NeverStops()


@lru_cache(maxsize=4)
def pick_encoder(codec: str = "h264") -> str:
    """The encoder this ffmpeg can actually open for a codec: the GPU one when the driver takes it, else CPU.

    Probed with two tiny frames rather than assumed: a machine with an NVIDIA card but no encode permission
    (or a filled session table) must fall back to a working encoder, not fail every request. Cached per codec:
    the answer cannot change while the process lives.
    """
    candidates = ENCODERS.get(codec, ENCODERS["h264"])
    for encoder in candidates:
        command = [
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
            # 320x240, not a thumbnail: HEVC's NVENC refuses frames below its minimum size (64x64 fails with
            # "dimensions are less than the minimum supported value"), which would silently pick the CPU encoder.
            "-f", "lavfi", "-i", "nullsrc=size=320x240:rate=10",
            "-frames:v", "2", "-c:v", encoder, "-f", "null", "-",
        ]
        if subprocess.run(command, capture_output=True).returncode == 0:
            return encoder
    raise VideoError(f"no usable {codec} encoder available (tried {', '.join(candidates)})")


def _atempo_chain(rate: float) -> str:
    """The audio tempo filter chain for a content rate: ``atempo`` only accepts 0.5-2.0 per stage.

    Slowing down and speeding up must not turn the match into chipmunks, so the audio is retimed by stretching
    it rather than resampling it; 4x is two 2x stages, 0.25x two halves.
    """
    remaining = max(0.25, min(8.0, float(rate)))
    stages = []
    while remaining > 2.0:
        stages.append("atempo=2.0")
        remaining /= 2.0
    while remaining < 0.5:
        stages.append("atempo=0.5")
        remaining /= 0.5
    stages.append(f"atempo={remaining:.4f}")
    return ",".join(stages)


def encoder_command(
    *,
    width: int,
    height: int,
    fps: float,
    output: str,
    live: bool,
    encoder: str,
    keyframe_s: float,
    rate: float = 1.0,
    source: str | Path | None = None,
    audio: bool = False,
    audio_start_s: float = 0.0,
    audio_duration_s: float | None = None,
) -> list[str]:
    """The ffmpeg invocation both encoders share: raw BGR frames in, H.264 MP4 out.

    ``live`` picks the fragmented-MP4 flags - a stream a player can join mid-flight - and a low-latency tune;
    a clip gets the index at the front instead (``faststart``), which is what makes it seekable, and a slower,
    denser tune. The keyframe interval doubles as the fragment interval for a live stream: about a second of
    video per fragment is what keeps a viewer near the live edge without chopping the stream into confetti.

    ``rate`` is the *content* speed: the muxer's clock runs ``fps * rate`` per source second, so a 2x request
    plays twice as fast in any player - the audio, when asked for, is retimed to match (``atempo``), and the
    writer paces to the same product. ``source``/``audio`` add the match's own soundtrack from the recording,
    seek-together with the video window; ``1:a:0?`` keeps a source without an audio track working, and the
    output is bounded by ``-t`` (the window over the rate) rather than ``-shortest``: a soundtrack shorter than
    the video must not truncate the clip, and one longer must not extend a live stream past its window in a
    silent tail.
    """
    if encoder.endswith("nvenc"):
        quality = ["-preset", "p4", "-tune", "ll" if live else "hq", "-rc", "vbr", "-cq", "23"]
    elif "265" in encoder or "hevc" in encoder:
        quality = ["-preset", "veryfast" if live else "medium", "-crf", "25"]
    else:
        quality = ["-preset", "veryfast" if live else "medium", "-crf", "20"]
    has_audio = bool(audio and source is not None)
    output_rate = max(1.0, float(fps) * float(rate))
    gop = max(1, int(round(output_rate * keyframe_s)))
    finishing = (
        ["-movflags", "+frag_keyframe+empty_moov+default_base_moof", "-f", "mp4", output]
        if live
        else ["-movflags", "+faststart", "-f", "mp4", output]
    )
    command = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin", "-y",
        "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{width}x{height}", "-framerate", f"{output_rate:g}",
        "-i", "pipe:0",
    ]
    if has_audio:
        command += ["-ss", f"{float(audio_start_s):.3f}", "-i", str(source)]
    command += ["-map", "0:v:0"]
    if has_audio:
        command += ["-map", "1:a:0?"]
        if abs(float(rate) - 1.0) > 1e-6:
            command += ["-filter:a", _atempo_chain(rate)]
        command += ["-c:a", "aac", "-b:a", "128k"]
        if audio_duration_s is not None:
            command += ["-t", f"{max(0.1, float(audio_duration_s) / max(0.01, float(rate))):.3f}"]
    else:
        command += ["-an"]
    command += [
        "-c:v", encoder, *quality, "-pix_fmt", "yuv420p", "-g", str(gop), "-keyint_min", str(gop),
    ]
    if "265" in encoder or "hevc" in encoder:
        # ffmpeg tags HEVC as hev1 by default; QuickTime and Safari only accept hvc1, and it costs nothing.
        command += ["-tag:v", "hvc1"]
    command += finishing
    return command


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


def _start(
    encoder: str,
    *,
    fps: float,
    width: int,
    height: int,
    output: str,
    live: bool,
    keyframe_s: float,
    rate: float = 1.0,
    source: str | Path | None = None,
    audio: bool = False,
    audio_start_s: float = 0.0,
    audio_duration_s: float | None = None,
):
    command = encoder_command(
        width=width,
        height=height,
        fps=fps,
        output=output,
        live=live,
        encoder=encoder,
        keyframe_s=keyframe_s,
        rate=rate,
        source=source,
        audio=audio,
        audio_start_s=audio_start_s,
        audio_duration_s=audio_duration_s,
    )
    return subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)


def _audio_source(match, audio: bool) -> str | None:
    """The recording to take the soundtrack from, or None - a simulated match has no file behind it."""
    if not audio:
        return None
    video = Path(match.video)
    return str(video) if video.exists() else None


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
    rate: float = 1.0,
    audio: bool = True,
    codec: str = "h264",
    skip_frame: str | None = None,
    on_progress=None,
) -> int:
    """Encode ``[start_s, start_s + duration_s]`` of the annotated match into ``output``; returns frame count.

    The frame size is taken from the first decoded frame rather than computed: the height follows the source's
    aspect, and the reader owns that decision - a raw-video pipe that disagrees with the frames it receives
    corrupts the encode. Encoding to a file means nothing has to drain the encoder's output, so this is a
    straight loop: decode, draw, write. The match's own audio rides along when the recording is there.

    ``skip_frame`` (a decoder input option, "nokey") is for the whole-game navigator videos, where one picture
    per second is enough and decoding every frame would take the footage's own length. Only readers that accept
    it are given it.
    """
    encoder = encoder or pick_encoder(codec)
    reader_kwargs: dict = {"fps": fps, "width": width, "start_s": start_s, "duration_s": duration_s}
    if skip_frame:
        reader_kwargs["skip_frame"] = skip_frame
    reader = reader_factory(match.video, **reader_kwargs)
    first, rest = _first_frame(reader)
    _timestamp, frame = first
    height, actual_width = frame.shape[:2]
    process = _start(
        encoder,
        fps=fps,
        width=actual_width,
        height=height,
        output=str(output),
        live=False,
        keyframe_s=2.0,
        rate=rate,
        source=_audio_source(match, audio),
        audio=audio,
        audio_start_s=start_s,
        audio_duration_s=duration_s,
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
    rate: float = 1.0,
    audio: bool = True,
    codec: str = "h264",
):
    """Yield the annotated match as a fragmented MP4, chunk by chunk, until the window ends or ``control`` stops.

    A writer thread renders and feeds the encoder; this generator drains the encoder's output - and so paces
    the pipeline: nothing is produced much faster than the client reads it, and the writer's own pacing keeps
    the stream near real time (a decode that outruns the declared rate would otherwise run minutes ahead of
    the viewer, buffered in the player). ``control`` is the MJPEG stream's ``StreamControl``, so ``/stop`` ends
    both kinds of stream the same way.
    """
    encoder = encoder or pick_encoder(codec)
    end_s = match.start_s + match.frame_count / match.fps
    stop = control if control is not None else _NEVER_STOPPED
    reader = reader_factory(
        match.video, fps=fps, width=width, start_s=start_s, duration_s=max(0.1, end_s - start_s)
    )
    first, rest = _first_frame(reader)
    _timestamp, frame = first
    height, actual_width = frame.shape[:2]
    process = _start(
        encoder,
        fps=fps,
        width=actual_width,
        height=height,
        output="pipe:1",
        live=True,
        keyframe_s=1.0,
        rate=rate,
        source=_audio_source(match, audio),
        audio=audio,
        audio_start_s=start_s,
        audio_duration_s=max(0.1, end_s - start_s),
    )
    stderr = _StderrTail(process.stderr)
    failures: list[Exception] = []

    def produce() -> None:
        period = 1.0 / max(0.01, fps * max(0.01, rate))  # the muxer's clock runs fps*rate per source second
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


def clip_key(
    match_id: str,
    start_s: float,
    duration_s: float,
    fps: float,
    width: int,
    overlays: dict,
    rate: float = 1.0,
    audio: bool = True,
    codec: str = "h264",
) -> str:
    """The identity of an encode: any difference in any of these is a different file."""
    layers = ",".join(sorted(name for name, on in overlays.items() if on))
    return (
        f"{match_id}-{start_s:.2f}-{duration_s:.2f}-{fps:g}x{rate:g}-{width}-{codec}"
        f"-{layers or 'none'}-{'aud' if audio else 'mute'}"
    )


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
