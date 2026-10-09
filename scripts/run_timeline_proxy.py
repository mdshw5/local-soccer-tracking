"""Build the dashboard's timeline proxy for one segment, as its own process so the page stays responsive.

The proxy is a one-off pass over the video (see `dashboard/timeline.py`), so - like Stage A - it must not run inside
the render. `timeline.start_background_build` launches this and the dashboard polls ``scrubber_build.json`` for the
outcome. On success the segment directory gains a `scrubber/` folder holding the proxy and its frontend, which the
dashboard then serves as the scrubbable timeline.

Usage::

    python scripts/run_timeline_proxy.py --video /path/match.MP4 \
        --out <footage>/analysis/<id>/segments/<segment> --start 0 --duration 300
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from soccer_analytics.dashboard import timeline


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the timeline scrubber's proxy for one segment.")
    parser.add_argument("--video", required=True, help="source video file")
    parser.add_argument("--out", required=True, help="segment directory to write the proxy beside")
    parser.add_argument("--start", type=float, default=0.0, help="start offset in the source video (seconds)")
    parser.add_argument("--duration", type=float, required=True, help="length of the segment (seconds)")
    parser.add_argument("--width", type=int, default=timeline.PROXY_WIDTH, help="proxy width in pixels")
    parser.add_argument("--fps", type=float, default=timeline.PROXY_FPS, help="proxy frame rate")
    parser.add_argument("--cpu", action="store_true", help="do not use the GPU encoder")
    parser.add_argument(
        "--every-frame",
        action="store_true",
        help="decode every frame even for a long window (smoother, roughly 10x slower)",
    )
    args = parser.parse_args()

    # The dashboard polls the build state; a fraction written a few times a second is a real progress bar there.
    last_write = 0.0

    def on_progress(fraction: float) -> None:
        nonlocal last_write
        now = time.time()
        if now - last_write >= 1.0 or fraction >= 1.0:
            last_write = now
            timeline.write_build_state(args.out, progress=round(min(1.0, fraction), 4))

    try:
        proxy = timeline.build_proxy(
            args.video,
            args.out,
            start_s=args.start,
            duration_s=args.duration,
            width=args.width,
            fps=args.fps,
            prefer_gpu=not args.cpu,
            skip_frame=None if args.every_frame else timeline.proxy_skip_frame(args.duration),
            on_progress=on_progress,
        )
    except Exception as exc:  # the dashboard reads this back and falls back to a plain slider
        timeline.write_build_state(args.out, state="error", error=f"{type(exc).__name__}: {exc}")
        print(f"timeline proxy failed: {exc}", file=sys.stderr, flush=True)
        return 1

    timeline.write_build_state(args.out, state="done", error=None, progress=1.0)
    print(f"timeline proxy ready: {proxy} ({proxy.stat().st_size // 1024} KB)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
