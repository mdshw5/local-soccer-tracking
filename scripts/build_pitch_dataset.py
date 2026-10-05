"""Build a YOLO-pose keypoint dataset from a manually-calibrated match.

The pitch-keypoint model from ``rustyneuron01/Real-Time-Football-Detection`` is trained on broadcast views of a
single full-size pitch, and on this footage it locks onto neighbouring goals. Fine-tuning it on this camera's own
games is the fix, and the labels do not need to be drawn by hand: a match that was calibrated by clicking a few
landmarks has a camera pose, and projecting the 32-marker template through it gives the keypoint positions on every
frame. The whole point of the fine-tune is that this dataset contains *only* the main pitch's markers - the
neighbouring goals are never labelled, so the model is never taught to fire on them.

Label quality is bounded by the calibration, and the script says so. A click-calibrated pose has a residual of a
metre or two, worth tens of pixels on far markers, so the labels are a good teaching signal for "which pitch" and
"roughly where", not a sub-pixel ground truth. Frames whose projected markers fall under a detected player are
marked not-visible rather than labelled, because the model cannot see what a person is standing on.

Output layout (Ultralytics pose)::

    <out>/images/{train,val}/*.jpg
    <out>/labels/{train,val}/*.txt
    <out>/data.yaml

Usage::

    python scripts/build_pitch_dataset.py \
        --segment data/segments/game_...__whole_game_541_4851 \
        --calibration data/matches/2026-10-04_17-28-37-430/calibration.json \
        --out data/pitch_keypoints --max-frames 1200 --width 1920
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from soccer_analytics.analysis.projection import segment_poses  # noqa: E402
from soccer_analytics.analysis.stage_a import load_segment  # noqa: E402
from soccer_analytics.geometry.pitch_calibration import PitchCalibration, pitch_to_pixels  # noqa: E402
from soccer_analytics.geometry.pitch_template import KEYPOINT_COUNT, template_for  # noqa: E402
from soccer_analytics.ingest.ffmpeg_reader import grab_frame  # noqa: E402

# A marker hidden behind a player cannot be seen, so it is not a label: the detection boxes are grown by this many
# pixels at the analysis width before testing, because a marker on a player's outline is usually hidden too.
OCCLUSION_MARGIN_PX = 6.0
# A frame needs this many visible markers to be a positive example. YOLO pose wants a box around the object, and a
# frame with one or two markers has no box worth teaching.
MIN_VISIBLE_MARKERS = 4


@dataclass
class DatasetStats:
    frames_seen: int = 0
    frames_written: int = 0
    negatives: int = 0
    skipped_few: int = 0
    skipped_grabbed: int = 0
    marker_counts: list = None  # type: ignore[assignment]

    def __post_init__(self):
        if self.marker_counts is None:
            self.marker_counts = []


def _person_boxes(segment, frame_index: int, width: int) -> list[tuple[float, float, float, float]]:
    """Stage A detection boxes for one frame, in pixels at ``width``."""
    rows = np.where(segment.det_frame == frame_index)[0]
    boxes = segment.det_box[rows] * width  # stored normalised by width on both axes
    return [tuple(float(v) for v in box) for box in boxes]


def _visible_markers(
    keypoints_px: np.ndarray,
    in_front: np.ndarray,
    frame_shape: tuple[int, int],
    boxes: list[tuple[float, float, float, float]],
    margin: float,
) -> np.ndarray:
    """Visibility per marker: in frame, in front of the camera, and not standing under a player box."""
    height, width = frame_shape
    ok = np.asarray(in_front, dtype=bool).copy()
    ok &= np.isfinite(keypoints_px).all(axis=1)
    ok &= keypoints_px[:, 0] >= 0
    ok &= keypoints_px[:, 0] < width
    ok &= keypoints_px[:, 1] >= 0
    ok &= keypoints_px[:, 1] < height
    for index in np.where(ok)[0]:
        x, y = keypoints_px[index]
        for x1, y1, x2, y2 in boxes:
            if x1 - margin <= x <= x2 + margin and y1 - margin <= y <= y2 + margin:
                ok[index] = False
                break
    return ok


def _label_line(
    keypoints_px: np.ndarray, visible: np.ndarray, frame_shape: tuple[int, int], pad: float = 8.0
) -> str | None:
    """One Ultralytics pose label line, or None when no box can be formed.

    x is normalised by width and y by height, as Ultralytics expects - which is *not* the solver's by-width
    convention, so the conversion happens here and nowhere else.
    """
    height, width = frame_shape
    shown = np.where(visible)[0]
    if len(shown) == 0:
        return None
    xs, ys = keypoints_px[shown, 0], keypoints_px[shown, 1]
    x1, x2 = max(0.0, float(xs.min()) - pad), min(float(width), float(xs.max()) + pad)
    y1, y2 = max(0.0, float(ys.min()) - pad), min(float(height), float(ys.max()) + pad)
    if x2 - x1 < 1 or y2 - y1 < 1:
        return None
    cx, cy = (x1 + x2) / 2 / width, (y1 + y2) / 2 / height
    bw, bh = (x2 - x1) / width, (y2 - y1) / height
    parts = ["0", f"{cx:.6f}", f"{cy:.6f}", f"{bw:.6f}", f"{bh:.6f}"]
    for index in range(KEYPOINT_COUNT):
        if visible[index]:
            parts += [f"{keypoints_px[index, 0] / width:.6f}", f"{keypoints_px[index, 1] / height:.6f}", "2"]
        else:
            parts += ["0", "0", "0"]
    return " ".join(parts)


def build(
    segment_dir: Path,
    calibration_path: Path,
    out_dir: Path,
    *,
    length_m: float,
    width_m: float,
    max_frames: int,
    width: int,
    val_fraction: float,
    occlusion_margin: float,
    include_negatives: bool,
    min_visible: int,
    seed: int,
    start_frame: int = 0,
    end_frame: int | None = None,
) -> DatasetStats:
    segment = load_segment(segment_dir)
    manifest = json.loads((segment_dir / "meta.json").read_text())
    video = manifest["video"]
    calibration = PitchCalibration.from_json(json.loads(calibration_path.read_text()))
    template = np.asarray(template_for(length_m, width_m))
    q, focal = segment_poses(segment)

    frame_count = len(segment.time)
    start = max(0, int(start_frame))
    stop = frame_count if end_frame is None else min(frame_count, max(start + 1, int(end_frame)))
    available = list(range(start, stop))
    if max_frames and len(available) > max_frames:
        selected = sorted(set(np.linspace(start, stop - 1, max_frames).astype(int).tolist()))
    else:
        selected = available

    # Temporal split, not random: neighbouring analysis frames are near-duplicates, and a random split would put
    # almost the same picture in both train and val, reporting a validation score that means nothing.
    split_at = int(len(selected) * (1.0 - val_fraction))
    for split in ("train", "val"):
        (out_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (out_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    stats = DatasetStats()
    for rank, frame_index in enumerate(selected):
        image = grab_frame(video, float(segment.time[frame_index]), width=width)
        stats.frames_seen += 1
        if image is None:
            stats.skipped_grabbed += 1
            continue
        # The calibration may carry a drift correction (the studio fits one); projecting through the *corrected*
        # frame is what keeps the labels on the markings late in the video. The raw chain would silently drift the
        # labels away as the match goes on.
        view_q, view_focal = calibration.corrected_frame(q[frame_index], float(focal[frame_index]), frame_index)
        keypoints_px, in_front = pitch_to_pixels(calibration, template, view_q, view_focal)
        keypoints_px = keypoints_px * width
        boxes = _person_boxes(segment, frame_index, width)
        visible = _visible_markers(keypoints_px, in_front, image.shape[:2], boxes, occlusion_margin)
        split = "train" if rank < split_at else "val"
        name = f"frame_{frame_index:06d}"
        if visible.sum() >= min_visible:
            line = _label_line(keypoints_px, visible, image.shape[:2])
            if line is None:
                stats.skipped_few += 1
                continue
            stats.frames_written += 1
            stats.marker_counts.append(int(visible.sum()))
        elif include_negatives and visible.sum() == 0:
            line = ""  # a background image: the main pitch is not in view, so nothing should be detected
            stats.negatives += 1
        else:
            stats.skipped_few += 1
            continue
        cv2.imwrite(str(out_dir / "images" / split / f"{name}.jpg"), image, [cv2.IMWRITE_JPEG_QUALITY, 90])
        (out_dir / "labels" / split / f"{name}.txt").write_text(line + "\n" if line else "")

    (out_dir / "data.yaml").write_text(
        "\n".join(
            [
                f"path: {out_dir.resolve()}",
                "train: images/train",
                "val: images/val",
                f"kpt_shape: [{KEYPOINT_COUNT}, 3]",
                "names:",
                "  0: pitch",
                "",
            ]
        )
    )
    return stats


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--segment", required=True, type=Path)
    parser.add_argument("--calibration", required=True, type=Path, help="match calibration.json")
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--length-m", type=float, default=100.0)
    parser.add_argument("--width-m", type=float, default=64.0)
    parser.add_argument("--max-frames", type=int, default=1200, help="0 for every analysis frame")
    parser.add_argument("--width", type=int, default=1920, help="stored image width")
    parser.add_argument("--val-fraction", type=float, default=0.15, help="temporal tail used for validation")
    parser.add_argument("--occlusion-margin", type=float, default=OCCLUSION_MARGIN_PX)
    parser.add_argument("--min-visible", type=int, default=MIN_VISIBLE_MARKERS)
    parser.add_argument("--include-negatives", action="store_true", help="write empty labels when no marker is visible")
    parser.add_argument("--start-frame", type=int, default=0, help="first analysis frame to label")
    parser.add_argument("--end-frame", type=int, default=0, help="last analysis frame (0 = end of segment)")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    segment_dir = args.segment if args.segment.is_absolute() else REPO_ROOT / args.segment
    stats = build(
        segment_dir,
        args.calibration if args.calibration.is_absolute() else REPO_ROOT / args.calibration,
        args.out if args.out.is_absolute() else REPO_ROOT / args.out,
        length_m=args.length_m,
        width_m=args.width_m,
        max_frames=args.max_frames,
        width=args.width,
        val_fraction=args.val_fraction,
        occlusion_margin=args.occlusion_margin,
        include_negatives=args.include_negatives,
        min_visible=args.min_visible,
        seed=args.seed,
        start_frame=args.start_frame,
        end_frame=args.end_frame or None,
    )
    counts = np.asarray(stats.marker_counts) if stats.marker_counts else np.zeros(0)
    print(
        f"frames seen {stats.frames_seen}, written {stats.frames_written} "
        f"(+{stats.negatives} negatives), skipped {stats.skipped_few} few-marker, {stats.skipped_grabbed} undecoded"
    )
    if counts.size:
        print(f"visible markers per written frame: min {counts.min()}, median {np.median(counts):.0f}, max {counts.max()}")
    print(f"dataset written to {args.out} (see data.yaml)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())