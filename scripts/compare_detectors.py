"""Benchmark detector checkpoints against each other on real footage, at the settings the pipeline uses.

The question this answers: which model should sit behind the report's analytics - the Stage A person pass (COCO
class 0) and the ball scan (COCO class 32)? Every model is run over the same sampled frames, at the width and
thresholds the pipeline uses (``stage_a.DETECT_WIDTH`` - 0 = the video's own width, the full-resolution default -
and ``PERSON_CONF``; lower confidences are accepted), and the report covers the things the analytics actually
feel:

* person counts, confidence and *size distribution* - far-side players arrive as 14-40 px boxes, and a model
  that loses them loses the shape of the game;
* pairwise detection agreement: which model finds what the other misses, overall and for small players;
* ball detections at the scan's low confidence, and how often the existing ball track has a nearby detection
  (a cheap recall proxy against the current, working pipeline);
* inference speed for both passes, since a whole game is ~21.5k frames.

The disagreement frames are saved as annotated images - the numbers suggest where to look, the images settle it.

Models are Ultralytics checkpoints (a local path or a stock name) or Roboflow-hosted checkpoints written as
``rf:<project>/<version>`` (e.g. ``rf:player-ball-detection-dq8a3/1``). The latter download and cache through
Roboflow Inference and need ``ROBOFLOW_API_KEY`` in the environment.

Usage::

    python scripts/compare_detectors.py \
        --video /srv/storage/home_video/Xbot/2026-10-03/game_16-28-37.784.mp4 \
        --start-s 540.8 --end-s 4851.2 --frames 150 \
        --models yolov8n.pt weights/yolo26n.pt yolov8s.pt \
        --segment <footage>/analysis/<id>/segments/<dir> --device 0
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from soccer_analytics.analysis.stage_a import (  # noqa: E402
    DETECT_WIDTH,
    FLOOR_WIDTH,
    MIN_PERSON_HEIGHT_PX,
    PERSON_CONF,
)
from soccer_analytics.ingest.ffmpeg_reader import grab_frame  # noqa: E402
from soccer_analytics.ingest.source import probe_source  # noqa: E402

BALL_CLASS = 32
BALL_CONF = 0.05  # what the ball scan uses
MATCH_IOU = 0.5
BALL_MATCH_DISTANCE = 0.03  # in frame-width-normalized units - a generous "same part of the pitch"

_PERSON_NAMES = {"person", "player", "players", "goalkeeper", "goalkeepers", "keeper", "referee", "referees"}
_BALL_NAMES = {"ball", "balls", "football", "soccer ball", "sports ball"}


def resolve(path: str) -> str:
    """A model name as Ultralytics takes it: a local path resolved against the repo, or a stock name."""
    candidate = REPO_ROOT / path
    return str(candidate) if candidate.exists() else path


def model_profile(model) -> dict:
    """Map a checkpoint's classes onto the two things the pipeline needs: persons and the ball.

    COCO models answer with ``classes=[0]`` / ``[32]``; football-trained models call the same things
    player/goalkeeper/referee and ball. The profile keeps the pipeline meaning intact and preserves the
    referee split, which the report uses to keep officials out of team stats.
    """
    names = {int(i): str(n).strip().lower() for i, n in model.names.items()}
    person_ids = [i for i, n in names.items() if n in _PERSON_NAMES]
    ball_ids = [i for i, n in names.items() if n in _BALL_NAMES]
    if not person_ids:  # unknown class naming - fall back to the COCO convention
        person_ids = [0]
    return {"names": names, "person_ids": person_ids, "ball_id": ball_ids[0] if ball_ids else (BALL_CLASS if len(names) >= 80 else None)}


class RoboflowModelAdapter:
    """A Roboflow-hosted checkpoint (public Universe or your workspace), run through Roboflow Inference.

    ``model_id`` is ``<project>/<version>``. Weights are downloaded once and cached under
    ``data/models/external/inference_cache``; frames are handed over as they are and the compiled
    model does its own resize to its export resolution.
    """

    def __init__(self, model_id: str) -> None:
        api_key = os.environ.get("ROBOFLOW_API_KEY")
        if not api_key:
            raise SystemExit("Set ROBOFLOW_API_KEY to benchmark Roboflow-hosted models.")
        os.environ.setdefault("MODEL_CACHE_DIR", str(REPO_ROOT / "data" / "models" / "external" / "inference_cache"))
        from inference import get_model

        self.model_id = model_id
        self._model = get_model(model_id=model_id, api_key=api_key)
        class_names = getattr(self._model, "class_names", None) or []
        self.names = {i: str(name) for i, name in enumerate(class_names)}

    def detect(self, frame: np.ndarray, conf: float, classes: list[int]):
        """One pass over a frame; x/y from the service are box centers in pixels."""
        predictions = [p for p in self._model.infer(frame, confidence=conf)[0].predictions if p.class_id in classes]
        if not predictions:
            return np.zeros((0, 4)), np.zeros(0), np.zeros(0, dtype=int)
        xyxy = np.array(
            [
                [p.x - p.width / 2, p.y - p.height / 2, p.x + p.width / 2, p.y + p.height / 2]
                for p in predictions
            ]
        )
        return xyxy, np.array([p.confidence for p in predictions]), np.array([p.class_id for p in predictions], dtype=int)


def detect(model, frame: np.ndarray, *, conf: float, classes: list[int], imgsz: int, device):
    """One detection pass as (xyxy, conf, class): Ultralytics checkpoints and RF adapters share this."""
    if isinstance(model, RoboflowModelAdapter):
        return model.detect(frame, conf, classes)
    result = model.predict(frame, imgsz=imgsz, conf=conf, classes=classes, device=device, verbose=False)[0]
    return result.boxes.xyxy.cpu().numpy(), result.boxes.conf.cpu().numpy(), result.boxes.cls.cpu().numpy().astype(int)


def run_model(model, frames: list[np.ndarray], *, device, imgsz: int, profile: dict) -> dict:
    """Both passes over every frame: persons (production conf) and balls (the scan's low conf)."""
    import torch

    people: list[np.ndarray] = []  # per frame (N, 4) xyxy in frame pixels at ``imgsz`` width
    people_conf: list[np.ndarray] = []
    people_cls: list[np.ndarray] = []
    balls: list[np.ndarray] = []
    ball_conf: list[np.ndarray] = []
    person_ms, ball_ms = [], []

    def sync() -> None:
        if device != "cpu" and torch.cuda.is_available():
            torch.cuda.synchronize()

    for frame in frames:
        sync()
        started = time.perf_counter()
        boxes, confs, cls = detect(
            model, frame, conf=PERSON_CONF, classes=profile["person_ids"], imgsz=imgsz, device=device
        )
        sync()
        person_ms.append((time.perf_counter() - started) * 1000.0)
        people.append(boxes)
        people_conf.append(confs)
        people_cls.append(cls)

        if profile["ball_id"] is None:
            balls.append(np.zeros((0, 4)))
            ball_conf.append(np.zeros(0))
            continue
        sync()
        started = time.perf_counter()
        boxes, confs, _ = detect(
            model, frame, conf=BALL_CONF, classes=[profile["ball_id"]], imgsz=imgsz, device=device
        )
        sync()
        ball_ms.append((time.perf_counter() - started) * 1000.0)
        balls.append(boxes)
        ball_conf.append(confs)

    return {
        "people": people,
        "people_conf": people_conf,
        "people_cls": people_cls,
        "balls": balls,
        "ball_conf": ball_conf,
        "person_ms": float(np.median(person_ms)),
        "ball_ms": float(np.median(ball_ms)) if ball_ms else float("nan"),
    }


def match(a: np.ndarray, b: np.ndarray) -> tuple[int, int, int]:
    """Greedy IoU matching; returns (matched, unmatched_a, unmatched_b)."""
    if len(a) == 0 or len(b) == 0:
        return 0, len(a), len(b)
    iou = np.zeros((len(a), len(b)))
    for i in range(len(a)):
        ax1, ay1, ax2, ay2 = a[i]
        for j in range(len(b)):
            bx1, by1, bx2, by2 = b[j]
            ix1, iy1 = max(ax1, bx1), max(ay1, by1)
            ix2, iy2 = min(ax2, bx2), min(ay2, by2)
            inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
            union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
            iou[i, j] = inter / union if union > 0 else 0.0
    matched = 0
    used_b = set()
    order = np.dstack(np.unravel_index(np.argsort(-iou.ravel()), iou.shape))[0]
    used_a = set()
    for i, j in order:
        if iou[i, j] < MATCH_IOU or i in used_a or j in used_b:
            continue
        used_a.add(int(i))
        used_b.add(int(j))
        matched += 1
    return matched, len(a) - matched, len(b) - matched


def person_bands(people: list[np.ndarray], width: int) -> dict[str, int]:
    """Detections by box height, expressed at 1920 width - the far side of the pitch lives under 40 px there.

    The boxes are in pixels at the sampled width, so heights are rescaled before banding: without this, a run at
    the full-resolution default would file far-side players into the buckets the 1920 run used for everybody.
    """
    scale = FLOOR_WIDTH / max(1, int(width))
    heights = np.concatenate([(b[:, 3] - b[:, 1]) * scale for b in people]) if people else np.zeros(0)
    return {
        "all": int(len(heights)),
        "h<14 (filtered)": int((heights < 14).sum()),
        "14-24": int(((heights >= 14) & (heights < 24)).sum()),
        "24-40": int(((heights >= 24) & (heights < 40)).sum()),
        "40-60": int(((heights >= 40) & (heights < 60)).sum()),
        ">=60": int((heights >= 60).sum()),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--video", required=True)
    parser.add_argument("--start-s", type=float, required=True)
    parser.add_argument("--end-s", type=float, required=True)
    parser.add_argument("--frames", type=int, default=150)
    parser.add_argument(
        "--width", type=int, default=DETECT_WIDTH,
        help="frame width to sample at; 0 (default) = the video's own width, the pipeline's full-resolution default",
    )
    parser.add_argument("--models", nargs="+", default=["yolov8n.pt", "weights/yolo26n.pt", "yolov8s.pt"])
    parser.add_argument("--segment", type=Path, default=None, help="segment dir with ball_track.json + meta.json")
    parser.add_argument("--device", default="0")
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "outputs" / "detector_compare")
    args = parser.parse_args()

    probe = probe_source(args.video)
    width = int(args.width) or int(probe.width)  # 0 = full resolution, the pipeline's default
    print(f"video {args.video}: {probe.width}x{probe.height}, {probe.duration_s:.0f}s")
    span = args.end_s - args.start_s
    times = args.start_s + (np.arange(args.frames) + 0.5) * span / args.frames
    print(f"sampling {args.frames} frames over {span:.0f}s at width {width}")

    frames: list[np.ndarray] = []
    kept_times: list[float] = []
    for t in times:
        frame = grab_frame(args.video, float(t), width=width)
        if frame is None:
            continue
        frames.append(frame)
        kept_times.append(float(t))
    print(f"read {len(frames)} frames")

    from ultralytics import YOLO

    results: dict[str, dict] = {}
    profiles: dict[str, dict] = {}
    for name in args.models:
        if name.startswith("rf:"):
            model = RoboflowModelAdapter(name[len("rf:") :])
            suffix = f"roboflow {model.model_id}"
        else:
            path = resolve(name)
            model = YOLO(path)
            suffix = Path(path).name
        profile = model_profile(model)
        profiles[name] = profile
        print(f"running {name} ({suffix}) persons={profile['person_ids']} ball={profile['ball_id']}...")
        results[name] = run_model(
            model, frames, device=args.device if args.device != "cpu" else "cpu", imgsz=width, profile=profile
        )
        del model

    # --- report -------------------------------------------------------------------------------------------
    # The floor is a physical size, so at the full-resolution default it is more pixels than the 14 measured at
    # 1920; scaling keeps "boxes below this never reach the pipeline" true at any sampled width.
    min_height = MIN_PERSON_HEIGHT_PX * width / FLOOR_WIDTH
    print("\n=== persons (model's person classes, conf {:.2f}) ===".format(PERSON_CONF))
    print(f"{'model':<30}{'dets':>7}{'>=floor':>8}{'median/frame':>13}{'conf med':>9}{'person ms':>10}{'ball ms':>9}")
    for name, r in results.items():
        above = [b[(b[:, 3] - b[:, 1]) >= min_height] for b in r["people"]]
        counts = [len(b) for b in above]
        confs = np.concatenate([c for c, b in zip(r["people_conf"], r["people"])]) if r["people"] else np.zeros(0)
        print(
            f"{name:<30}{sum(len(b) for b in r['people']):>7}{sum(counts):>8}"
            f"{np.median(counts):>13.0f}{np.median(confs) if len(confs) else 0:>9.2f}"
            f"{r['person_ms']:>10.1f}{r['ball_ms']:>9.1f}"
        )

    print("\nclass mix (person-pass detections; ball from the ball pass):")
    for name, r in results.items():
        names_map = profiles[name]["names"]
        counts: dict[int, int] = {}
        for cls in r["people_cls"]:
            for c in cls:
                counts[int(c)] = counts.get(int(c), 0) + 1
        ball_total = sum(len(b) for b in r["balls"])
        mix = ", ".join(f"{names_map.get(c, c)}={n}" for c, n in sorted(counts.items()))
        print(f"  {name:<30} {mix}; ball={ball_total}")

    print("\nheight bands (at 1920):")
    for name, r in results.items():
        print(f"  {name:<22}{person_bands(r['people'], width)}")

    print("\npairwise agreement (persons, IoU>={:.1f}):".format(MATCH_IOU))
    names = list(results)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            only_a = only_b = matched = 0
            for fa, fb in zip(results[names[i]]["people"], results[names[j]]["people"]):
                m, ua, ub = match(fa, fb)
                matched += m
                only_a += ua
                only_b += ub
            total = matched + only_a + only_b
            print(
                f"  {names[i]} vs {names[j]}: matched {matched}/{total} ({100 * matched / max(total, 1):.1f}%), "
                f"only-{Path(names[i]).stem} {only_a}, only-{Path(names[j]).stem} {only_b}"
            )

    print(f"\nball (model's ball class, conf {BALL_CONF}):")
    for name, r in results.items():
        counts = [len(b) for b in r["balls"]]
        confs = np.concatenate([c for c in r["ball_conf"]]) if r["ball_conf"] else np.zeros(0)
        print(
            f"  {name:<22} frames with a detection {sum(c > 0 for c in counts)}/{len(frames)}, "
            f"total {sum(counts)}, conf median {np.median(confs) if len(confs) else 0:.2f}"
        )

    if args.segment is not None:
        segment_dir = args.segment if args.segment.is_absolute() else REPO_ROOT / args.segment
        track = json.loads((segment_dir / "ball_track.json").read_text())
        meta = json.loads((segment_dir / "meta.json").read_text())
        tracked = {int(f["i"]): f for f in track["frames"] if f.get("status") == "tracking"}
        print(f"\nball recall proxy: detection near the tracked ball within {BALL_MATCH_DISTANCE} (width units)")
        for name, r in results.items():
            hit = total = 0
            for t, balls in zip(kept_times, r["balls"]):
                i = int(round((t - meta["start_s"]) * meta["fps"]))
                ball = tracked.get(i)
                if ball is None or not len(balls):
                    if ball is not None:
                        total += 1
                    continue
                total += 1
                center = ((balls[:, 0] + balls[:, 2]) / 2 / width, (balls[:, 1] + balls[:, 3]) / 2 / width)
                if min(np.hypot(center[0] - ball["u"], center[1] - ball["v"])) <= BALL_MATCH_DISTANCE:
                    hit += 1
            print(f"  {name:<22} {hit}/{total}")

    # --- disagreement images ------------------------------------------------------------------------------
    args.out.mkdir(parents=True, exist_ok=True)
    if len(names) >= 2:
        a, b = names[0], names[1]
        diffs = [
            abs(len(results[a]["people"][k]) - len(results[b]["people"][k])) for k in range(len(frames))
        ]
        worst = np.argsort(-np.asarray(diffs))[:3]
        for rank, k in enumerate(worst):
            image = frames[k].copy()
            for boxes, color in ((results[a]["people"][k], (0, 220, 0)), (results[b]["people"][k], (50, 50, 255))):
                for x1, y1, x2, y2 in boxes.astype(int):
                    cv2.rectangle(image, (x1, y1), (x2, y2), color, 3)
            path = args.out / f"disagree_{rank}_{Path(a).stem}_green_{Path(b).stem}_red_{kept_times[k]:.0f}s.jpg"
            cv2.imwrite(str(path), image)
            print(f"saved {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
