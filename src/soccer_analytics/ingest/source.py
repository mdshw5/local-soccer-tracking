"""One video source, whether it is a single file or a game's set of camera clips read as one recording.

A game arrives as two or three ~30-minute camera files. The old workflow concatenated them into a merged video
(a stream copy) so that every consumer saw an ordinary single file; that copy is tens of gigabytes, a build step,
and one more thing that can fail. This module is the alternative: a :class:`ClipSource` resolves a game-clock time
to the clip that contains it and decodes clip after clip in playing order, so a window that crosses a join is one
continuous stream of frames to the caller - the camera-motion chain and the tracker live above the reader, in
Python, and never notice the file change.

Two facts, measured on the reference game (2026-10-03, three 4K HEVC clips plus a merged copy), shape the design:

* seeking the *resolved clip* (``-ss t - clip.start_s``) is pixel-identical to seeking the merged video - mean
  absolute difference 0.000 at samples on both sides of both joins - and costs the same ~1 s;
* the ffmpeg concat demuxer is not a substitute: it decodes sequentially just fine, but its input seeking is
  pathological (>90 s exactly on the clip-1 sample), so every seek must resolve to a clip first.

``game.json`` is the source spec: the manifest *is* the video. Callers pass either that path or a plain video
file; :func:`as_source` decides, and :func:`open_reader` / :func:`probe_source` / :func:`grab_source_frame` give
back what the single-file functions gave back before.
"""

from __future__ import annotations

import subprocess
import tempfile
import wave
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from soccer_analytics.ingest.ffmpeg_reader import (
    FFmpegError,
    FFmpegFrameReader,
    VideoProbe,
    extract_audio,
    grab_frame,
    probe_video,
    read_wav_mono,
)


@dataclass(frozen=True)
class ClipRef:
    """One camera file and where it begins on the game's own clock."""

    path: str
    start_s: float
    duration_s: float

    @property
    def end_s(self) -> float:
        return self.start_s + self.duration_s


@dataclass(frozen=True)
class ClipSource:
    """A game read as one continuous recording: its clips in playing order, on the game clock.

    ``spec`` is what goes into ``meta["video"]`` (the manifest path for a game); ``key_name``/``key_size`` are
    what :func:`soccer_analytics.analysis.stage_a.segment_dir_for` keys a segment directory by when there is no
    single file to stat. A record built from the manifest already carries the total size in its game id, so
    ``key_size`` is 0 and the name alone is the key.
    """

    clips: tuple[ClipRef, ...]
    duration_s: float
    spec: str
    label: str = ""
    key_name: str = ""
    key_size: int = 0

    def clip_at(self, time_s: float) -> tuple[ClipRef, float] | None:
        """The clip containing ``time_s`` and the offset within it, or ``None`` for an empty source.

        The time is clamped into the game, so a request a fraction past the last frame lands on the final clip
        rather than failing: every existing caller of a single-file reader asked with times inside the window,
        and the clamping keeps that true at the edges.
        """
        if not self.clips:
            return None
        t = max(0.0, min(float(time_s), self.duration_s))
        containing = [clip for clip in self.clips if clip.start_s <= t] or [self.clips[0]]
        clip = containing[-1]
        return clip, max(0.0, t - clip.start_s)


@dataclass(frozen=True)
class FileSource:
    """A single video file: the ordinary case, unchanged from before clip sets existed."""

    path: Path
    spec: str
    key_name: str
    key_size: int


Source = ClipSource | FileSource


def as_source(value: str | Path | Source) -> Source:
    """The source a value names: a ``game.json`` manifest is its clip set, anything else is a file.

    The game import is deliberately inside the function: ``analysis.game`` imports the reader from this package,
    and a module-level import here would be circular.
    """
    if isinstance(value, (ClipSource, FileSource)):
        return value
    path = Path(value)
    if path.name == "game.json" and path.exists():
        from soccer_analytics.analysis.game import GameRecord  # noqa: PLC0415 - breaks an import cycle

        record = GameRecord.load(path.parent)
        if record.clips and all(Path(clip.path).exists() for clip in record.clips):
            return source_from_record(record, spec=str(path))
        # A manifest whose clips are gone (or a legacy one that never recorded them) is still a game whose merged
        # video exists: fall back to the file so old archives keep working untouched.
        return FileSource(path=Path(record.output), spec=str(path), key_name=Path(record.output).stem,
                          key_size=Path(record.output).stat().st_size)
    return FileSource(path=path, spec=str(path), key_name=path.stem, key_size=path.stat().st_size)


def source_from_record(record, *, spec: str = "") -> ClipSource:
    """The clip source for a game record (duck-typed, so ``analysis.game`` stays out of this module's imports)."""
    clips = tuple(
        ClipRef(path=str(clip.path), start_s=float(clip.start_s), duration_s=float(clip.duration_s))
        for clip in record.clips
    )
    manifest = Path(spec).parent if spec else None
    key_size = sum(int(getattr(clip, "bytes", 0) or 0) for clip in record.clips)
    if not key_size:
        key_size = sum(Path(clip.path).stat().st_size for clip in record.clips)
    return ClipSource(
        clips=clips,
        duration_s=float(record.duration_s),
        spec=spec or str(record.output),
        label=str(getattr(record, "game_id", "")),
        # The game id already carries the total size (`game_<first clip>_<bytes>`), so keying by the directory
        # name keeps the same "a replaced clip is not the same game" property without repeating the number.
        key_name=manifest.name if manifest else Path(record.output).stem,
        key_size=0,
    )


class ClipFrameReader:
    """Yields ``(game_time, frame)`` by decoding the clips one after another, on one analysis grid.

    Each clip is decoded by an ordinary :class:`FFmpegFrameReader` positioned where the requested game time falls
    inside it; timestamps are shifted onto the game clock by the clip's ``start_s``. The next clip starts where
    the previous one's grid left off (clamped to the clip's own start), so a window crossing a join continues at
    exactly ``1 / fps`` per frame - the seam is one file change, not a time jump. A genuine gap between clips
    leans on the manifest's starts and shows up as a forward jump, which is what the merged timeline would show
    too because the starts are the same numbers the merge used.

    ``frames()`` also carries the single-file reader's semantics for the edges: frames are yielded for the
    requested ``[start_s, start_s + duration_s)``, and asking for one more frame than exists simply ends.
    """

    def __init__(
        self,
        source: ClipSource,
        *,
        fps: float = 10.0,
        width: int = 1920,
        start_s: float = 0.0,
        duration_s: float | None = None,
        prefer_gpu: bool = True,
        skip_frame: str | None = None,
    ):
        self.source = source
        self.fps = float(fps)
        self.width = int(width)
        self.start_s = max(0.0, float(start_s))
        self.duration_s = None if duration_s is None else float(duration_s)
        self.prefer_gpu = prefer_gpu
        self.skip_frame = skip_frame
        self.used_gpu: bool | None = None

    @property
    def probe(self) -> VideoProbe:
        return probe_source(self.source)

    def frames(self) -> Iterator[tuple[float, np.ndarray]]:
        end = self.source.duration_s if self.duration_s is None else min(
            self.source.duration_s, self.start_s + self.duration_s
        )
        step = 1.0 / self.fps
        next_t = self.start_s
        for clip in self.source.clips:
            if next_t >= end:
                break
            if clip.end_s <= next_t:
                continue
            window_start = max(next_t, clip.start_s)
            clip_stop = min(end, clip.end_s)
            local_start = window_start - clip.start_s
            # Over-ask by a step and a margin: `-t` cuts on the boundary and the frame sitting on it would be
            # lost (the same padding the per-chunk refresh uses). Frames past `clip_stop` are dropped below.
            duration = clip_stop - window_start + step + 0.5
            reader = FFmpegFrameReader(
                clip.path,
                fps=self.fps,
                width=self.width,
                start_s=local_start,
                duration_s=duration,
                prefer_gpu=self.prefer_gpu,
                skip_frame=self.skip_frame,
            )
            for local_ts, frame in reader.frames():
                game_ts = clip.start_s + local_ts
                if game_ts >= clip_stop - 1e-9:
                    break
                if game_ts < next_t - 1e-9:
                    continue
                self.used_gpu = reader.used_gpu
                yield game_ts, frame
                next_t = game_ts + step


def open_reader(
    source: str | Path | Source,
    *,
    fps: float = 10.0,
    width: int = 1920,
    start_s: float = 0.0,
    duration_s: float | None = None,
    prefer_gpu: bool = True,
    skip_frame: str | None = None,
) -> FFmpegFrameReader | ClipFrameReader:
    """A frame reader for either kind of source, with the single-file reader's own signature."""
    resolved = as_source(source)
    if isinstance(resolved, ClipSource):
        return ClipFrameReader(
            resolved,
            fps=fps,
            width=width,
            start_s=start_s,
            duration_s=duration_s,
            prefer_gpu=prefer_gpu,
            skip_frame=skip_frame,
        )
    return FFmpegFrameReader(
        resolved.path,
        fps=fps,
        width=width,
        start_s=start_s,
        duration_s=duration_s,
        prefer_gpu=prefer_gpu,
        skip_frame=skip_frame,
    )


def probe_source(source: str | Path | Source) -> VideoProbe:
    """The aggregate probe of a source: frame geometry from the first clip, whole-game duration from the manifest.

    ffprobe cannot answer for a clip set (a concat list reports ``duration=N/A``), and the manifest already
    knows the length, so the duration comes from there; everything a probe is otherwise used for (size, rate,
    codec, audio presence) is a property of the clips, which a recording split by one camera shares.
    """
    resolved = as_source(source)
    if isinstance(resolved, ClipSource):
        first = probe_video(resolved.clips[0].path)
        return VideoProbe(
            width=first.width,
            height=first.height,
            fps=first.fps,
            duration_s=float(resolved.duration_s),
            codec=first.codec,
            has_audio=first.has_audio,
        )
    return probe_video(resolved.path)


def grab_source_frame(source: str | Path | Source, time_s: float, *, width: int = 1600) -> np.ndarray | None:
    """One frame at a game-clock time: seek the containing clip, exactly as the merged file would."""
    resolved = as_source(source)
    if isinstance(resolved, ClipSource):
        hit = resolved.clip_at(time_s)
        if hit is None:
            return None
        clip, offset = hit
        return grab_frame(clip.path, offset, width=width)
    return grab_frame(resolved.path, time_s, width=width)


@dataclass(frozen=True)
class WindowPiece:
    """One clip's slice of a requested game-clock window."""

    path: str
    start_s: float  # seconds into the clip, not into the game
    duration_s: float

    @property
    def end_s(self) -> float:
        return self.start_s + self.duration_s


def window_segments(source: str | Path | Source, start_s: float, duration_s: float | None = None) -> list[WindowPiece]:
    """Split a game-clock window into per-clip pieces, in playing order.

    This is what lets the raw-ffmpeg callers (encoders, proxy builds, highlight cuts, audio extraction) stay
    fast on a never-merged game: each piece seeks inside its own clip, because seeking the ffmpeg concat demuxer
    itself decodes from the start (see the module docstring). A window inside one clip - the common case - is a
    single piece with its local offset; one crossing a join is two or three.
    """
    resolved = as_source(source)
    start = max(0.0, float(start_s))
    if isinstance(resolved, FileSource):
        if duration_s is None:
            duration = max(0.0, float(probe_video(resolved.path).duration_s) - start)
        else:
            duration = max(0.0, float(duration_s))
        return [WindowPiece(path=str(resolved.path), start_s=start, duration_s=duration)]
    end = resolved.duration_s if duration_s is None else min(resolved.duration_s, start + float(duration_s))
    pieces: list[WindowPiece] = []
    for clip in resolved.clips:
        lo, hi = max(start, clip.start_s), min(end, clip.end_s)
        if hi > lo:
            pieces.append(WindowPiece(path=clip.path, start_s=lo - clip.start_s, duration_s=hi - lo))
    return pieces


def concat_copies(parts: list[str | Path], destination: str | Path) -> Path:
    """Join normalized pieces (same codec, size, rate) by stream copy, through an ffmpeg concat listing.

    This is the cheap half of cutting a window that crosses a clip join: each piece is encoded by the ordinary
    single-file routine, and the join itself costs no second encode and no re-encode of what was just written.
    """
    destination = Path(destination)
    listing = destination.with_name(destination.name + ".concat.txt")
    try:
        lines = []
        for part in parts:
            path = str(part).replace("'", "'\\''")
            lines.append(f"file '{path}'\n")
        listing.write_text("".join(lines), encoding="utf-8")
        command = [
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin", "-y",
            "-f", "concat", "-safe", "0", "-i", str(listing), "-c", "copy", str(destination),
        ]
        result = subprocess.run(command, capture_output=True, text=True)
        if result.returncode != 0 or not destination.exists() or destination.stat().st_size == 0:
            raise FFmpegError(f"could not join the pieces into {destination}: {result.stderr.strip() or result.returncode}")
        return destination
    finally:
        listing.unlink(missing_ok=True)


def extract_source_audio(
    source: str | Path | Source,
    wav_path: str | Path,
    *,
    start_s: float = 0.0,
    duration_s: float | None = None,
    on_progress=None,
) -> Path:
    """A mono wav of a game-clock window, stitched from the clips it crosses.

    Each clip contributes its overlapping slice, extracted by the ordinary single-file routine with a seek
    inside that clip; the samples are concatenated and written once, so the whistle scan keeps reading one wav
    on one clock - exactly as it did for the merged video. The clip durations from the manifest are the same
    numbers the merged file's timeline used, so game-clock times stay correct across the stitches.
    """
    resolved = as_source(source)
    if isinstance(resolved, FileSource):
        return extract_audio(
            resolved.path, wav_path, start_s=start_s, duration_s=duration_s, on_progress=on_progress
        )
    pieces = window_segments(resolved, start_s, duration_s)
    total = sum(piece.duration_s for piece in pieces) or 1.0
    samples: list[np.ndarray] = []
    rate = 16000
    done = 0.0
    with tempfile.TemporaryDirectory() as tmp:
        for index, piece in enumerate(pieces):
            part = Path(tmp) / f"piece_{index}.wav"

            def part_progress(fraction: float, base: float = done, span: float = piece.duration_s) -> None:
                if on_progress is not None:
                    on_progress(min(1.0, (base + span * float(fraction)) / total))

            extract_audio(
                piece.path, part, start_s=piece.start_s, duration_s=piece.duration_s, on_progress=part_progress
            )
            data, rate = read_wav_mono(part)
            samples.append(data)
            done += piece.duration_s
    joined = np.concatenate(samples) if samples else np.empty(0, dtype=np.float32)
    wav_path = Path(wav_path)
    wav_path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(wav_path), "wb") as out:
        out.setnchannels(1)
        out.setsampwidth(2)
        out.setframerate(int(rate))
        out.writeframes((np.clip(joined, -1.0, 1.0) * 32767.0).astype("<i2").tobytes())
    if on_progress is not None:
        on_progress(1.0)
    return wav_path
