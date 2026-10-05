"""Measure the grass hue window on a video, frame by frame.

The window (`tracking.team_classifier.grass_hue_window`) is what keeps the pitch out of kit colour estimates.
Its claim is measurable, so this is the tool that measures it: sample frames across a video and report, for
each, how much of the grass the band actually covers. On the real whole game (2026-10-03) the reference's
fixed +-10-around-the-mean band covered 51-94% of the grass pixels (dusk cost the most), while the widened
band holds 98%+; a run of this script is what backs the numbers in that docstring.

Usage::

    python scripts/probe_grass_window.py /path/to/game.mp4 [--frames 8]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from soccer_analytics.tracking.team_classifier import _grass_mask, grass_hue_window  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", type=Path)
    parser.add_argument("--frames", type=int, default=8, help="frames to sample, evenly spread (default 8)")
    args = parser.parse_args()

    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        raise SystemExit(f"could not open {args.video}")
    duration_s = cap.get(cv2.CAP_PROP_FRAME_COUNT) / max(cap.get(cv2.CAP_PROP_FPS), 1e-6)
    times = np.linspace(0.0, max(0.0, duration_s - 1.0), args.frames)

    print(f"{args.video.name}: {duration_s / 60:.1f} min, {args.frames} frames sampled")
    print(f"{'t':>8} {'window':>12} {'grass %':>8} {'covered':>8}")
    coverages = []
    for t in times:
        cap.set(cv2.CAP_PROP_POS_MSEC, float(t) * 1000.0)
        ok, frame = cap.read()
        if not ok:
            continue
        window = grass_hue_window(frame)
        if window is None:
            print(f"{t:>8.0f} {'-':>12} {'0.0':>8} {'-':>8}")
            continue
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = _grass_mask(hsv)
        hues = hsv[:, :, 0][mask > 0]
        coverage = np.count_nonzero((hues >= window[0]) & (hues <= window[1])) / max(1, hues.size)
        coverages.append(coverage)
        share = 100.0 * np.count_nonzero(mask) / (frame.shape[0] * frame.shape[1])
        print(f"{t:>8.0f} {str(window):>12} {share:>7.1f}% {100.0 * coverage:>7.1f}%")
    cap.release()
    if coverages:
        print(f"coverage: min {100.0 * min(coverages):.1f}%, mean {100.0 * np.mean(coverages):.1f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())