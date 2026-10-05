"""Probe: track the ball over a stretch of the real game with tracking-by-detection, and show the trail.

This is an exploratory tool, not part of the analysis pipeline yet. It exists to answer one question on real
footage: does the two-rate scheme - a cheap detector pass on a window around the prediction each frame, and a
full-frame scan whenever the track is not confidently on the ball - hold one identity across a continuous stretch
of play, including the moments the ball leaves the picture?

It writes:

* ``track.json`` - one record per frame (source time, state, normalised position);
* ``mark_*.jpg`` - every few seconds, the frame with the recorded position drawn (green tracking, amber coasting,
  red out-of-view), so the trail can be checked against what is actually on screen;
* a ``sheet.jpg`` contact sheet of those marks, for a quick look.

Usage::

    .venv/bin/python scripts/probe_ball_track.py --start-s 700 --duration-s 180
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from soccer_analytics.analysis.ball import BallTrack, blank_overlays  # noqa: E402
from soccer_analytics.analysis.stage_a import load_segment  # noqa: E402
from soccer_analytics.ingest.ffmpeg_reader import FFmpegFrameReader  # noqa: E402

GAME = "/srv/storage/home_video/Xbot/2026-10-03/game_16-28-37.784.mp4"
SEGMENT = "data/segments/game_16-28-37.784_32823901638__whole_game_541_4851"
FPS = 5.0
COCO_BALL_CLASS = 32
WINDOW_PX = 1600  # the window scanned around the prediction while tracking, at 4K
WINDOW_IMGSZ = 1280
FULL_IMGSZ = 2560
MARK_EVERY = 15  # frames between marked stills (3 s at 5 fps)

COLOURS = {"tracking": (80, 220, 80), "coasting": (60, 190, 240), "out_of_view": (70, 70, 230), "lost": (150, 150, 150)}


def detections_in(model, frame: np.ndarray, imgsz: int, origin: tuple[int, int], width: int, world: bool) -> list:
    result = model.predict(frame, imgsz=imgsz, conf=0.05, classes=None if world else [COCO_BALL_CLASS], verbose=False)[0]
    out = []
    for box in result.boxes:
        x1, y1, x2, y2 = (float(v) for v in box.xyxy[0])
        u = (origin[0] + (x1 + x2) / 2) / width
        v = (origin[1] + (y1 + y2) / 2) / width
        w = (x2 - x1) / width
        h = (y2 - y1) / width
        out.append((float(box.conf[0]), u, v, w, h))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", default=GAME)
    parser.add_argument("--segment", default=SEGMENT)
    parser.add_argument("--start-s", type=float, default=700.0)
    parser.add_argument("--duration-s", type=float, default=180.0)
    parser.add_argument("--out", default="/tmp/xbot_frames/ball_probe/run1")
    args = parser.parse_args()

    from ultralytics import YOLO  # imported late: the probe is the only thing that needs torch

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    segment = load_segment(args.segment)
    print(f'segment: {len(segment.time)} frames, {segment.time[0]:.1f}s - {segment.time[-1]:.1f}s, aspect {segment.aspect:.4f}')

    coco = YOLO("yolov8s.pt")
    world = YOLO("yolov8s-worldv2.pt")
    world.set_classes(["soccer ball", "white ball"])

    track = BallTrack(aspect=segment.aspect)
    records: list[dict] = []
    marks: list[np.ndarray] = []
    t0 = time.perf_counter()
    scans = {"window": 0, "full": 0}

    reader = FFmpegFrameReader(Path(args.video), fps=FPS, width=3840, start_s=args.start_s, duration_s=args.duration_s)
    for count, (t, frame) in enumerate(reader.frames()):
        idx = int(round((t - float(segment.time[0])) * FPS))
        step = segment.step[idx] if 0 <= idx < len(segment.step) else None
        blank_overlays(frame)

        predicted = track.predict(step)
        # The window scan carries the track while the ball is plausibly near its prediction: tracking, or one
        # coasted frame (the prediction includes the learned velocity, so a pass stays around the window's centre;
        # measured on the real game, a kicked ball moves ~0.1 frame-widths *per frame* at onset, which a 1600 px
        # window absorbs for a frame or two). Beyond that the ball could be anywhere - a kicked ball, a ball out of
        # frame - and only a full-frame scan can find it and re-learn the velocity; a periodic full scan is the
        # safety net against the prediction quietly drifting wrong.
        window_ok = predicted is not None and (
            track.status == "tracking" or (track.status == "coasting" and track.coasted <= 1)
        )
        if window_ok and count % 25 != 24:
            scans["window"] += 1
            cx = int(np.clip(predicted[0] * frame.shape[1], WINDOW_PX / 2, frame.shape[1] - WINDOW_PX / 2))
            cy = int(np.clip(predicted[1] * frame.shape[1], WINDOW_PX / 2, frame.shape[0] - WINDOW_PX / 2))
            x0, y0 = cx - WINDOW_PX // 2, cy - WINDOW_PX // 2
            crop = frame[y0:y0 + WINDOW_PX, x0:x0 + WINDOW_PX]
            dets = detections_in(coco, crop, WINDOW_IMGSZ, (x0, y0), frame.shape[1], world=False)
            dets += detections_in(world, crop, WINDOW_IMGSZ, (x0, y0), frame.shape[1], world=True)
            state = track.update(dets, step=step, full_frame=False)
        else:
            scans["full"] += 1
            dets = detections_in(coco, frame, FULL_IMGSZ, (0, 0), frame.shape[1], world=False)
            dets += detections_in(world, frame, FULL_IMGSZ, (0, 0), frame.shape[1], world=True)
            state = track.update(dets, step=step, full_frame=True)

        records.append({"t": round(t, 3), "idx": idx, **state})
        if count % MARK_EVERY == 0:
            small = cv2.resize(frame, (1920, 1080), interpolation=cv2.INTER_AREA)
            if state["u"] is not None:
                px = int(state["u"] * 1920)
                py = int(state["v"] * 1920)
                colour = COLOURS[state["status"]]
                cv2.circle(small, (px, py), 40, colour, 3)
                cv2.drawMarker(small, (px, py), colour, cv2.MARKER_CROSS, 40, 2)
                cv2.putText(small, f"{t:.1f}s {state['status']} {state['conf']:.2f}", (px + 46, py - 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, colour, 2)
            else:
                cv2.putText(small, f"{t:.1f}s {state['status']}", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.2,
                            COLOURS["lost"], 2)
            cv2.imwrite(str(out_dir / f"mark_{count:04d}.jpg"), small)
            marks.append(small)

    wall = time.perf_counter() - t0
    n = len(records)
    counts = {}
    for r in records:
        counts[r["status"]] = counts.get(r["status"], 0) + 1
    print(f"frames {n} in {wall:.0f}s ({wall / max(n, 1) * 1000:.0f} ms/frame, both models, under load)")
    print("scans:", scans)
    print("states:", {k: f"{v} ({100 * v / max(n, 1):.0f}%)" for k, v in sorted(counts.items())})
    with open(out_dir / "track.json", "w") as fh:
        json.dump(records, fh, indent=1)

    tiles = [cv2.resize(m, (640, 360), interpolation=cv2.INTER_AREA) for m in marks]
    while len(tiles) % 4:
        tiles.append(np.zeros_like(tiles[0]))
    rows = [np.hstack(tiles[i:i + 4]) for i in range(0, len(tiles), 4)]
    sheet = np.vstack(rows)
    cv2.imwrite(str(out_dir / "sheet.jpg"), sheet)
    print("sheet:", sheet.shape, "marks:", len(marks))


if __name__ == "__main__":
    main()
