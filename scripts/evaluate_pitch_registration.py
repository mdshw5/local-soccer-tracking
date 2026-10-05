"""Evaluate a pitch-keypoint checkpoint by what registration actually needs: a correct camera pose.

Keypoint mAP is the wrong score for this job. A detector can place 30 markers plausibly and still put them on the
neighbouring pitch, which is precisely the failure mode here; what matters is whether the markers it returns register
the camera to the same place the manual calibration did. This script samples frames from a calibrated match, runs
the checkpoint over them, registers with the known tripod position, and reports the recovered position and
orientation against the manual calibration.

Run::

    python scripts/evaluate_pitch_registration.py \
        --segment data/segments/game_...__whole_game_541_4851 \
        --calibration data/matches/2026-10-04_17-28-37-430/calibration.json \
        --weights runs/pose/pitch_keypoints_finetune/weights/best.pt
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from soccer_analytics.analysis.projection import segment_poses  # noqa: E402
from soccer_analytics.analysis.stage_a import load_segment  # noqa: E402
from soccer_analytics.geometry.auto_register import register_with_position_prior  # noqa: E402
from soccer_analytics.geometry.pitch_calibration import PitchCalibration  # noqa: E402
from soccer_analytics.geometry.pitch_keypoint_yolo import (  # noqa: E402
    load_pitch_keypoint_model,
    observations_for_frames,
)
from soccer_analytics.ingest.ffmpeg_reader import grab_frame  # noqa: E402


def _angle_between(a: np.ndarray, b: np.ndarray) -> float:
    """Rotation angle in degrees between two orientation matrices."""
    cos = (np.trace(a.T @ b) - 1.0) / 2.0
    return float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--segment", required=True, type=Path)
    parser.add_argument("--calibration", required=True, type=Path, help="manual calibration used as the reference")
    parser.add_argument("--weights", type=Path, default=None, help="checkpoint to evaluate (default: installed one)")
    parser.add_argument("--length-m", type=float, default=100.0)
    parser.add_argument("--width-m", type=float, default=64.0)
    parser.add_argument("--frames", type=int, default=24)
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--end-frame", type=int, default=0)
    parser.add_argument("--imgsz", type=int, nargs="+", default=[640, 1920])
    parser.add_argument("--threshold", type=float, default=0.3)
    args = parser.parse_args()

    segment = load_segment(args.segment)
    meta = json.loads((args.segment / "meta.json").read_text())
    reference = PitchCalibration.from_json(json.loads(args.calibration.read_text()))
    q, focal = segment_poses(segment)

    stop = len(segment.time) if not args.end_frame else min(len(segment.time), args.end_frame)
    start = max(0, args.start_frame)
    indexes = sorted(set(np.linspace(start, stop - 1, args.frames).astype(int).tolist()))
    samples = []
    for index in indexes:
        frame = grab_frame(meta["video"], float(segment.time[int(index)]), width=1920)
        if frame is not None:
            samples.append((int(index), frame))

    model = load_pitch_keypoint_model(args.weights, device=0)
    observations = observations_for_frames(
        model, samples, image_sizes=tuple(args.imgsz), threshold=args.threshold
    )
    chain = {int(index): (q[int(index)], float(focal[int(index)])) for index, _ in samples}
    print(f"frames {len(samples)}, keypoints {len(observations)}")
    try:
        result = register_with_position_prior(
            observations,
            chain,
            segment.aspect,
            position_prior=tuple(float(v) for v in reference.position),
            focal_scale_prior=float(reference.focal_scale),
            length_m=args.length_m,
            width_m=args.width_m,
            correct_drift=False,
        )
    except Exception as exc:
        print(f"registration failed: {type(exc).__name__}: {exc}")
        return 1

    recovered = result.calibration
    position_error = float(np.linalg.norm(recovered.position - reference.position))
    orientation_error = _angle_between(recovered.base_rotation, reference.base_rotation)
    print(f"keypoints kept      : {result.keypoints_kept}/{result.keypoints_total}")
    print(f"position error      : {position_error:.2f} m  (reference {np.round(reference.position, 2).tolist()})")
    print(f"orientation error   : {orientation_error:.2f} deg")
    print(f"focal scale         : {recovered.focal_scale:.3f}  (reference {reference.focal_scale:.3f})")
    print(f"referenced frames   : {result.frames_used}")
    for note in result.notes:
        print(f"  - {note}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())