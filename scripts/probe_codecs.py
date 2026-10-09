"""Benchmark the annotated video codecs on this machine: H.264 vs HEVC (H.265), cost and benefit.

Two measurements, because they answer different questions:

* **end to end** - encode the same window through the real pipeline (``dashboard.video.encode_clip``) with each
  codec and time the whole request: decode, draw the overlays, encode, mux. That is what a viewer waits for.
* **rate/quality** - render the window once to a *lossless* reference, then encode that same reference with
  each codec at a few quality points and measure SSIM against it. This isolates the codec from the renderer:
  HEVC's win (or cost) is per bit, and only visible at matched quality.

The default window is the footage pane's own settings (1280 px at the source rate); the source is 3840x2160
HEVC at 60 fps on the real match. Nothing here writes to the archive: intermediates live in a temp directory.

Usage::

    python scripts/probe_codecs.py [--match <id>] [--start 1460] [--duration 6] [--fps 30] [--width 1280]
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from soccer_analytics.dashboard.stream import MATCHES_ROOT, AnnotatedMatch, parse_overlays  # noqa: E402
from soccer_analytics.dashboard.video import (  # noqa: E402
    VideoError,
    encode_clip,
    pick_encoder,
)
from soccer_analytics.ingest.ffmpeg_reader import FFmpegFrameReader  # noqa: E402

QUALITY_POINTS = (19, 23, 27)
_SSIM_RE = re.compile(r"All:([0-9.]+)")


def render_reference(match, *, start_s: float, duration_s: float, fps: float, width: int, output: Path) -> int:
    """The window drawn once and stored losslessly (FFV1): the yardstick the encodes are measured against."""
    reader = FFmpegFrameReader(match.video, fps=fps, width=width, start_s=start_s, duration_s=duration_s)
    frames = reader.frames()
    first = next(frames, None)
    if first is None:
        raise SystemExit("no frames decoded for the requested window")
    _timestamp, frame = first
    height, actual_width = frame.shape[:2]
    process = subprocess.Popen(
        [
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
            "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{actual_width}x{height}",
            "-framerate", f"{fps:g}", "-i", "pipe:0",
            # yuv420p with the standard HD matrix, tagged: the encodes are compared in *their* colour space -
            # an RGB reference would make every SSIM carry an implicit conversion nobody asked about (and the
            # conversion is lossy in itself, drowning the codec differences this script exists to show).
            "-vf", "scale=out_color_matrix=bt709:out_range=tv",
            "-pix_fmt", "yuv420p",
            "-color_range", "tv", "-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", "bt709",
            "-c:v", "ffv1", "-level", "3", str(output),
        ],
        stdin=subprocess.PIPE,
    )
    written = 0
    try:
        for timestamp, frame in _chain(first, frames):
            match.render_at(frame, timestamp, **parse_overlays(None))
            process.stdin.write(np.ascontiguousarray(frame).tobytes())
            written += 1
    finally:
        process.stdin.close()
    if process.wait() != 0:
        raise SystemExit("the lossless reference encode failed")
    return written


def _chain(first, rest):
    yield first
    yield from rest


def encode_variant(reference: Path, output: Path, encoder: str, cq: int, codec: str) -> dict:
    """Encode the reference with one encoder and quality point; returns the row's numbers.

    Quality flags follow the encoder family - NVENC's ``cq`` and x264/x265's ``crf`` are different scales, so the
    sweep is compared by measured SSIM, never by the number printed on the knob.
    """
    if encoder.endswith("nvenc"):
        quality = ["-preset", "p4", "-tune", "hq", "-rc", "vbr", "-cq", str(cq)]
    else:
        quality = ["-preset", "medium", "-crf", str(cq)]
    command = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-i", str(reference),
        "-an", "-c:v", encoder, *quality,
        "-pix_fmt", "yuv420p",
        "-color_range", "tv", "-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", "bt709",
    ]
    if codec == "hevc":
        command += ["-tag:v", "hvc1"]
    command += ["-movflags", "+faststart", "-f", "mp4", str(output)]
    started = time.monotonic()
    if subprocess.run(command, capture_output=True).returncode != 0:
        return {"encoder": encoder, "cq": cq, "seconds": float("nan"), "mb": float("nan"), "ssim": float("nan")}
    seconds = time.monotonic() - started
    ssim = subprocess.run(
        ["ffmpeg", "-hide_banner", "-i", str(output), "-i", str(reference), "-lavfi", "ssim", "-f", "null", "-"],
        capture_output=True,
        text=True,
    ).stderr
    match = _SSIM_RE.search(ssim)
    return {
        "encoder": encoder,
        "cq": cq,
        "seconds": seconds,
        "mb": output.stat().st_size / 1e6,
        "ssim": float(match.group(1)) if match else float("nan"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--match", default="2026-10-04_16-58-38-391")
    parser.add_argument("--start", type=float, default=1460.0)
    parser.add_argument("--duration", type=float, default=6.0)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--width", type=int, default=1280)
    args = parser.parse_args()

    match = AnnotatedMatch.load(args.match, root=MATCHES_ROOT)
    work = Path(tempfile.mkdtemp(prefix="codec-probe-"))
    overlays = parse_overlays(None)
    print(
        f"window: {args.match} @ {args.start:.0f}s +{args.duration:.0f}s, {args.width}px, {args.fps:g} fps "
        f"(source: {match.native_width}x{match.native_height} @ {match.native_fps:g} fps) -> {work}"
    )

    print("\n== end to end, through the real pipeline (decode + draw + encode + mux) ==")
    frames_expected = int(round(args.duration * args.fps))
    for codec in ("h264", "hevc"):
        try:
            encoder = pick_encoder(codec)
        except VideoError as error:
            print(f"{codec:>4}: unavailable ({error})")
            continue
        output = work / f"e2e_{codec}.mp4"
        started = time.monotonic()
        frames = encode_clip(
            match,
            start_s=args.start,
            duration_s=args.duration,
            fps=args.fps,
            width=args.width,
            overlays=overlays,
            output=output,
            encoder=encoder,
            codec=codec,
        )
        seconds = time.monotonic() - started
        size = output.stat().st_size / 1e6
        print(
            f"{codec:>4}: {seconds:6.1f}s for {frames} frames = {frames / seconds:5.1f} fps "
            f"({seconds / args.duration:4.2f}x realtime), {size:6.1f} MB ({size * 8 / args.duration:5.1f} Mbit/s), "
            f"encoder {encoder}"
        )
    print(f"(expected {frames_expected} frames)")

    print("\n== rate/quality against a lossless reference (encode-only; SSIM measured at full-rate) ==")
    reference = work / "reference.mkv"
    frames = render_reference(
        match, start_s=args.start, duration_s=args.duration, fps=args.fps, width=args.width, output=reference
    )
    print(f"reference: {frames} frames, {reference.stat().st_size / 1e6:.0f} MB lossless (FFV1)")
    rows = []
    for codec in ("h264", "hevc"):
        try:
            encoder = pick_encoder(codec)
        except VideoError:
            continue
        for cq in QUALITY_POINTS:
            row = encode_variant(reference, work / f"{codec}_{cq}.mp4", encoder, cq, codec)
            row["fps"] = frames / row["seconds"] if row["seconds"] > 0 else float("nan")
            rows.append(row)
            print(
                f"{codec:>4} cq{cq}: {row['seconds']:5.1f}s = {row['fps']:5.1f} fps, {row['mb']:6.1f} MB, "
                f"SSIM {row['ssim']:.6f}"
            )

    print("\n== matched-quality summary ==")
    h264_rows = [row for row in rows if "264" in row["encoder"]]
    hevc_rows = [row for row in rows if "265" in row["encoder"]]
    for baseline in h264_rows:
        candidates = [row for row in hevc_rows if row["ssim"] >= baseline["ssim"]]
        if not candidates:
            continue
        closest = min(candidates, key=lambda row: row["mb"])
        saving = 1.0 - closest["mb"] / baseline["mb"]
        print(
            f"H.264 cq{baseline['cq']} (SSIM {baseline['ssim']:.6f}, {baseline['mb']:.1f} MB) vs "
            f"HEVC cq{closest['cq']} (SSIM {closest['ssim']:.6f}, {closest['mb']:.1f} MB): "
            f"HEVC is {saving * 100:+.0f}% the size at equal-or-better SSIM"
        )


if __name__ == "__main__":
    main()
