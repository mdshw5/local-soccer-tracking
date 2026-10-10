"""Scan a segment's video for the ball, in the background, and keep the track on disk.

The tracker (`analysis.ball`) is the logic; this is the process that feeds it a whole segment and writes what it
decided, frame by frame, where the rest of the pipeline can pick it up. It is the same shape as the whistle scan:
its own process, a status file the dashboard polls, and a result beside the segment.

The scan is expensive - two detectors at 4K, about 0.13 s per frame on this machine under load - and it is
checkpointed for exactly that reason: every ``CHECKPOINT_SECONDS`` of frames the records *and* the tracker's
state are written atomically, and a rerun resumes from the last checkpoint instead of starting the hour again.
At the 15 fps analysis rate a whole game is 2-3 hours, so a rerun that does not start from zero matters.

What a frame's record means is the tracker's state, and the states are not interchangeable:

``tracking``    a detection was accepted; ``u``/``v`` is a measurement.
``coasting``    no detection, but the forecast (camera step + learned ball velocity) is inside the frame.
``out_of_view`` the forecast left the picture - the ball is not visible; the position is a forecast.
``lost``        too long without a detection; ``u``/``v`` are null until a full-frame scan finds the ball again.

The fixed burned-in overlay regions (logo, timestamp) are blanked before every detection - the logo is otherwise a
0.9-confidence "soccer ball".

Usage::

    python scripts/run_ball_scan.py --segment <footage>/analysis/<id>/segments/<name> [--force] [--limit-frames 50]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import numpy as np  # noqa: E402

from soccer_analytics.analysis.ball import BallTrack, blank_overlays  # noqa: E402
from soccer_analytics.analysis.stage_a import load_segment  # noqa: E402
from soccer_analytics.ingest.source import open_reader  # noqa: E402

RESULT_FILE = "ball_track.json"
STATUS_FILE = "ball_scan.json"

WINDOW_PX = 1600  # the crop scanned around the prediction while tracking, at 4K
WINDOW_IMGSZ = 1280
FULL_IMGSZ = 2560
FULL_EVERY_SECONDS = 5.0  # periodic full-frame scan while tracking: a safety net against the prediction drifting wrong
CHECKPOINT_SECONDS = 120.0  # wall-clock cadence of the resume checkpoint, in frames at the scan's rate
COCO_BALL_CLASS = 32
COCO_WEIGHTS = "yolov8s.pt"
WORLD_WEIGHTS = "yolov8s-worldv2.pt"


class Status:
    """Progress file the dashboard polls; written atomically and throttled, so a reader never sees half of it."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.payload: dict = {
            "state": "running",
            "stage": "detect",
            "progress": 0.0,
            "message": "Starting...",
            "started": time.time(),
        }
        self._written = 0.0

    def update(self, *, force: bool = False, **changes) -> None:
        self.payload.update(changes)
        self.payload["updated"] = time.time()
        now = time.monotonic()
        if not force and now - self._written < 0.4:
            return
        self._written = now
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        tmp.write_text(json.dumps(self.payload))
        tmp.replace(self.path)


def make_detector(width: int):  # noqa: ANN201 - closure around the two models
    """The production detector: COCO's ball class plus an open-vocabulary model, returning frame-normalized boxes.

    Both are run because they fail differently - the COCO model is conservative and misses the smallest balls, the
    open-vocabulary one is over-eager but catches what COCO drops - and the agreement between them on the real game
    is what says the detections are real (see the module docstring of ``analysis.ball``).
    """
    from ultralytics import YOLO

    coco = YOLO(str(REPO_ROOT / COCO_WEIGHTS))
    world = YOLO(str(REPO_ROOT / WORLD_WEIGHTS))
    world.set_classes(["soccer ball", "white ball"])

    def detector(image: np.ndarray, imgsz: int, origin: tuple[int, int]) -> list[tuple]:
        out: list[tuple] = []
        for model, classes in ((coco, [COCO_BALL_CLASS]), (world, None)):
            result = model.predict(image, imgsz=imgsz, conf=0.05, classes=classes, verbose=False)[0]
            for box in result.boxes:
                x1, y1, x2, y2 = (float(v) for v in box.xyxy[0])
                out.append(
                    (
                        float(box.conf[0]),
                        (origin[0] + (x1 + x2) / 2) / width,
                        (origin[1] + (y1 + y2) / 2) / width,
                        (x2 - x1) / width,
                        (y2 - y1) / width,
                    )
                )
        return out

    return detector


def scan(
    *,
    video: str | Path,
    out_dir: str | Path,
    start_s: float,
    fps: float,
    total_frames: int,
    width: int,
    aspect: float,
    steps: np.ndarray | None = None,
    detector=None,  # noqa: ANN001 - injected for tests
    status: Status | None = None,
    limit_frames: int = 0,
) -> dict:
    """Run one scan, writing progress and - if something goes wrong - the error to the status file.

    The worker below is what actually tracks; this wrapper exists so *every* failure (a model download, a GPU
    error, a decode hiccup) reaches the dashboard instead of a process that vanished with the status file still
    saying "running" forever. A checkpoint written before the failure survives, and a retry resumes from it.
    """
    out_dir = Path(out_dir)
    status = status or Status(out_dir / STATUS_FILE)
    # Announce the start before the models load (seconds) or the video is read: the dashboard offers the button
    # again the moment a scan looks stopped, and this window is exactly when a second click would race the first.
    status.update(force=True, state="running", message="Starting...")
    try:
        return _scan_track(
            video=video,
            out_dir=out_dir,
            start_s=start_s,
            fps=fps,
            total_frames=total_frames,
            width=width,
            aspect=aspect,
            steps=steps,
            detector=detector,
            status=status,
            limit_frames=limit_frames,
        )
    except Exception as exc:  # noqa: BLE001 - whatever went wrong goes into the status for the page to show
        status.update(force=True, state="error", error=f"{type(exc).__name__}: {exc}")
        raise


def _scan_track(
    *,
    video: str | Path,
    out_dir: str | Path,
    start_s: float,
    fps: float,
    total_frames: int,
    width: int,
    aspect: float,
    steps: np.ndarray | None = None,
    detector=None,  # noqa: ANN001 - injected for tests
    status: Status | None = None,
    limit_frames: int = 0,
) -> dict:
    """Track the ball across one segment, checkpointing as it goes; returns the result payload.

    ``detector(image, imgsz, origin)`` returns ``(conf, u, v, w, h)`` detections in frame-normalized coordinates
    (injected for tests; the default builds the two-model detector). ``steps`` is the segment's per-frame camera
    step, the prediction input. A payload on disk that is not ``complete`` is resumed from, both the records and
    the tracker's state - the state matters as much as the records: resuming with a fresh tracker would re-learn
    the velocity while the records claim one continuous track.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    result_path = out_dir / RESULT_FILE
    status = status or Status(out_dir / STATUS_FILE)
    detector = detector or make_detector(width)

    records: list[dict] = []
    tracker_state: dict | None = None
    if result_path.exists():
        payload = json.loads(result_path.read_text())
        if payload.get("complete"):
            status.update(
                force=True, state="done", stage="done", progress=1.0, message="Already scanned",
                scanned=len(payload.get("frames") or []), total_frames=total_frames,
                counts=payload.get("counts") or {},
            )
            return payload
        records = payload.get("frames", [])
        tracker_state = payload.get("tracker")
        status.update(force=True, message=f"Resuming after {len(records)} frames")

    track = (
        BallTrack.from_json(tracker_state, aspect=aspect, rate=fps)
        if tracker_state
        else BallTrack(aspect=aspect, rate=fps)
    )
    resume_at = len(records)
    # Frame-count cadences in *seconds*, at this scan's own rate: more frames per second must not mean either a
    # sparser safety net or a sparser checkpoint.
    full_every = max(1, int(round(FULL_EVERY_SECONDS * fps)))
    checkpoint_every = max(1, int(round(CHECKPOINT_SECONDS * fps)))

    def save(complete: bool) -> dict:
        payload = {
            "schema": 1,
            "video": str(video),
            "fps": fps,
            "start_s": start_s,
            "frames": records,
            "tracker": track.to_json(),
            "counts": _counts(records),
            "complete": complete,
        }
        tmp = result_path.with_suffix(result_path.suffix + ".tmp")
        tmp.write_text(json.dumps(payload))
        tmp.replace(result_path)
        return payload

    if resume_at >= total_frames:
        payload = save(complete=True)
        status.update(
            force=True, state="done", stage="done", progress=1.0, message="Scan complete",
            scanned=resume_at, total_frames=total_frames, counts=payload["counts"],
        )
        return payload

    window_px = min(WINDOW_PX, width)
    # ``video`` is the segment's own source path - a game.json manifest for a never-merged game, in which case
    # the reader chains the clips and every timestamp below stays on the game clock (see ingest/source.py).
    reader = open_reader(
        video, fps=fps, width=width, start_s=start_s + resume_at / fps, duration_s=(total_frames - resume_at) / fps
    )
    started = time.monotonic()
    processed = 0
    for k, (t, frame) in enumerate(reader.frames()):
        index = resume_at + k
        if index >= total_frames:
            break
        step = None
        if steps is not None and index < len(steps):
            step = steps[index]
        blank_overlays(frame)

        predicted = track.predict(step)
        # The window scan carries the track while the ball is plausibly near its prediction: tracking, or coasting
        # for about 0.2 s (a couple of frames at 15 fps). Beyond that the ball could be anywhere, and only a
        # full-frame scan can find it and re-learn the velocity; the periodic full scan is the safety net while
        # tracking.
        coast_window = max(1, int(round(0.2 * fps)))
        window_ok = predicted is not None and (
            track.status == "tracking" or (track.status == "coasting" and track.coasted <= coast_window)
        )
        full = not window_ok or k % full_every == full_every - 1
        if full:
            detections = detector(frame, FULL_IMGSZ, (0, 0))
        else:
            win = min(window_px, frame.shape[0], frame.shape[1])
            x0 = int(np.clip(predicted[0] * width - win / 2, 0, width - win))
            y0 = int(np.clip(predicted[1] * width - win / 2, 0, frame.shape[0] - win))
            detections = detector(frame[y0 : y0 + win, x0 : x0 + win], WINDOW_IMGSZ, (x0, y0))

        state = track.update(detections, step=step, full_frame=full)
        records.append({"i": index, "t": round(float(t), 3), **state})
        processed += 1

        if processed % 25 == 0:
            done = resume_at + processed
            elapsed = time.monotonic() - started
            rate = processed / elapsed if elapsed > 0 else 0.0
            left = (total_frames - done) / rate if rate > 0 else 0.0
            status.update(
                progress=done / total_frames,
                message=f"Scanning for the ball ({done}/{total_frames}, ~{max(1, int(left / 60))} min left)",
            )
        if processed % checkpoint_every == 0:
            save(complete=False)
            status.update(force=True, progress=(resume_at + processed) / total_frames, message="Checkpoint written")

        if limit_frames and processed >= limit_frames:
            break

    complete = resume_at + processed >= total_frames
    payload = save(complete=complete)
    if complete:
        status.update(
            force=True, state="done", stage="done", progress=1.0, message="Scan complete",
            scanned=resume_at + processed, total_frames=total_frames, counts=payload["counts"],
        )
    else:
        status.update(
            force=True,
            state="partial",
            progress=(resume_at + processed) / total_frames,
            message=f"Stopped after {resume_at + processed} of {total_frames} frames - rerun to resume",
            scanned=resume_at + processed,
            total_frames=total_frames,
            counts=payload["counts"],
        )
    return payload


def _counts(records: list[dict]) -> dict:
    counts: dict[str, int] = {}
    for record in records:
        counts[record["status"]] = counts.get(record["status"], 0) + 1
    return counts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--segment", required=True, help="segment directory (with meta.json and chunk files)")
    parser.add_argument("--force", action="store_true", help="discard an existing checkpoint and start over")
    parser.add_argument("--limit-frames", type=int, default=0, help="stop after this many frames (a smoke test)")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    segment_dir = Path(args.segment)
    if not segment_dir.is_absolute():
        segment_dir = REPO_ROOT / segment_dir
    status = Status(segment_dir / STATUS_FILE)
    try:
        meta = json.loads((segment_dir / "meta.json").read_text())
        if args.force:
            for name in (RESULT_FILE, STATUS_FILE):
                (segment_dir / name).unlink(missing_ok=True)

        segment = load_segment(segment_dir)
        payload = scan(
            video=meta["video"],
            out_dir=segment_dir,
            start_s=float(meta["start_s"]),
            fps=float(meta["fps"]),
            total_frames=len(segment.time),
            width=int(meta["width"]),
            aspect=float(segment.aspect),
            steps=segment.step,
            status=status,
            limit_frames=args.limit_frames,
        )
    except Exception as exc:  # noqa: BLE001 - a background job must leave the reason in its status file
        status.update(force=True, state="error", error=f"{type(exc).__name__}: {exc}")
        print(f"error: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    counts = payload["counts"]
    total = max(1, len(payload["frames"]))
    print("frames:", total, "complete:", payload["complete"])
    print("states:", {k: f"{v} ({100 * v // total}%)" for k, v in sorted(counts.items())})
    return 0 if payload["complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
