"""Fine-tune the pitch-keypoint pose model on a dataset built by ``build_pitch_dataset.py``.

Starts from the downloaded broadcast-trained checkpoint and continues on this camera's footage. Two augmentations
are switched off on purpose. Horizontal flip is disabled because the 32 pitch markers are not symmetric: a mirrored
image needs its keypoint indices permuted, and there is no ``flip_idx`` mapping for this template, so a flip would
train the model to put the left penalty spot on the right. Mosaic is disabled because stitching four views of a
multi-pitch scene is exactly the setting where the labels would smear across neighbouring goals - the confusion the
fine-tune exists to remove.

Run from the repository root::

    python scripts/train_pitch_keypoints.py --data data/pitch_keypoints/data.yaml --epochs 60

The fine-tuned weights land under ``runs/pose/<name>/weights/best.pt``; copy that to
``data/models/football-pitch-detection.pt`` to make the dashboard and the test server use it.
"""

from __future__ import annotations

import argparse
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_WEIGHTS = REPO_ROOT / "data" / "models" / "football-pitch-detection.pt"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, type=Path, help="data.yaml from build_pitch_dataset.py")
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS, help="checkpoint to start from")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", default="0")
    parser.add_argument("--lr0", type=float, default=0.001, help="low, because the model already detects pitches")
    parser.add_argument("--name", default="pitch_keypoints_finetune")
    parser.add_argument("--patience", type=int, default=20)
    args = parser.parse_args()

    from ultralytics import YOLO

    if not args.weights.exists():
        raise SystemExit(
            f"start weights not found: {args.weights}. Download them first with "
            "`python -c \"from soccer_analytics.geometry.pitch_keypoint_yolo import download_pitch_weights; "
            "download_pitch_weights()\"`."
        )

    model = YOLO(str(args.weights))
    model.train(
        data=str(args.data),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        # `optimizer='auto'` (the default) silently ignores lr0 and picks its own, which is wrong here: fine-tuning
        # wants a small step on top of an already-good model, not a fresh training run's learning rate.
        optimizer="AdamW",
        lr0=args.lr0,
        name=args.name,
        patience=args.patience,
        fliplr=0.0,  # keypoint indices are not flip-symmetric (see module docstring)
        mosaic=0.0,
        # Mild geometry: a pitch is a rigid plane, so heavy perspective/scale jitter teaches shapes that cannot occur.
        degrees=5.0,
        translate=0.05,
        scale=0.2,
        perspective=0.0,
        shear=0.0,
        plots=True,
        val=True,
    )
    print(f"done; weights under runs/pose/{args.name}/weights/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())