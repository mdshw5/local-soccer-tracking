"""Measure the grass hue window and the full grass mask on a video, frame by frame.

The window (`tracking.team_classifier.measure_grass`) is what keeps the pitch out of kit colour estimates. Its
claim is measurable, so this is the tool that measures it: sample frames across a video and report, for each, how
much of the grass the band actually covers and how much the full mask - which also removes bleached turf below the
saturation floor - removes. On the real whole game (2026-10-03) the reference's fixed +-10-around-the-mean band
covered 51-94% of the grass pixels (dusk cost the most), while the widened band holds 98%+; a run of this script is
what backs the numbers in that docstring.

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

from soccer_analytics.tracking.team_classifier import _grass_population, measure_grass  # noqa: E402


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
    print(f"{'t':>8} {'window':>12} {'grass %':>8} {'band':>7} {'masked':>8}")
    band_coverages = []
    mask_coverages = []
    for t in times:
        cap.set(cv2.CAP_PROP_POS_MSEC, float(t) * 1000.0)
        ok, frame = cap.read()
        if not ok:
            continue
        model = measure_grass(frame)
        if model is None:
            print(f"{t:>8.0f} {'-':>12} {'0.0':>8} {'-':>7} {'-':>8}")
            continue
        # `_grass_population` is every green pixel, bleached ones included - the population the model was measured
        # over and the honest denominator for "how much of the pitch does the mask actually remove".
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        population = _grass_population(hsv) > 0
        masked = float(np.count_nonzero(model.mask(hsv, lab) & population) / max(1, int(population.sum())))
        band_coverages.append(model.coverage)
        mask_coverages.append(masked)
        share = 100.0 * np.count_nonzero(population) / (frame.shape[0] * frame.shape[1])
        print(
            f"{t:>8.0f} {str(model.window):>12} {share:>7.1f}% "
            f"{100.0 * model.coverage:>6.1f}% {100.0 * masked:>7.1f}%"
        )
    cap.release()
    if band_coverages:
        print(
            f"band coverage:  min {100.0 * min(band_coverages):.1f}%, mean {100.0 * np.mean(band_coverages):.1f}%"
        )
        print(
            f"whole-mask coverage (incl. bleached turf): min {100.0 * min(mask_coverages):.1f}%, "
            f"mean {100.0 * np.mean(mask_coverages):.1f}%"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())