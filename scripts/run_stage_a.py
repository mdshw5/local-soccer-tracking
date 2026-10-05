"""Run Stage A for one segment, as its own process so the dashboard stays responsive.

Usage::

    python scripts/run_stage_a.py --video /path/match.MP4 --out data/segments/<id> [--start 300] [--duration 600]

``--duration 0`` (the default) analyses from the offset to the end of the video. Progress is written to
``<out>/status.json`` by `analyse_segment`, which is what the dashboard polls. Resuming is implicit: the run starts
from the last completed chunk, so re-running this command after an interruption continues instead of starting over.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from soccer_analytics.analysis.stage_a import SegmentConfig, analyse_segment


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the heavy per-segment analysis pass (Stage A).")
    parser.add_argument("--video", required=True, help="source video file")
    parser.add_argument("--out", required=True, help="output directory for this segment")
    parser.add_argument("--start", type=float, default=0.0, help="start offset in the source video (seconds)")
    parser.add_argument("--duration", type=float, default=0.0, help="seconds to analyse; 0 = to the end of the video")
    parser.add_argument("--fps", type=float, default=5.0, help="analysis frame rate (detections per second)")
    parser.add_argument("--width", type=int, default=1920, help="analysis frame width for detection")
    parser.add_argument("--chunk-frames", type=int, default=300, help="frames per checkpoint file")
    parser.add_argument("--cpu", action="store_true", help="do not use the GPU for detection")
    args = parser.parse_args()

    config = SegmentConfig(
        fps=args.fps,
        detect_width=args.width,
        chunk_frames=args.chunk_frames,
        device="cpu" if args.cpu else 0,
    )
    status = analyse_segment(
        args.video,
        args.out,
        config=config,
        start_s=args.start,
        duration_s=args.duration,
        on_progress=lambda s: print(
            f"chunk {s['chunk']}/{s['total_chunks']} frames {s['frames_done']}/{s['total_frames']} "
            f"{s['fps']:.1f} fps eta {s['eta_s']}s",
            flush=True,
        ),
    )
    print(f"finished: {status['state']}")
    return 0 if status["state"] == "done" else 1


if __name__ == "__main__":
    raise SystemExit(main())
