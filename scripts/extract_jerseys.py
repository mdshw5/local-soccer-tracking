"""Read shirt numbers off the footage, per tracked player. Runs in the background from the dashboard.

Strategy: pick the frames where each tracked player is largest and clearest (shirt numbers are only legible on
close-to-camera players), crop the torso band, and run EasyOCR with a digits-only alphabet. Every reading is a
candidate; the per-track majority vote in ``analysis.jerseys`` decides what - if anything - the track wears.

The heavy pipeline (projection + tracking) is rebuilt here so the script stands alone: it needs the same tracks the
report uses, or the readings would be attributed to the wrong people.

Usage::

    python scripts/extract_jerseys.py --match <match_id> --video <file> --segment <dir> [--width 1920]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import cv2
import numpy as np

from soccer_analytics.analysis import stage_b
from soccer_analytics.analysis.jerseys import JerseyCandidate, aggregate_candidates, crop_torso, sanitize_digits
from soccer_analytics.analysis.library import MatchLibrary
from soccer_analytics.analysis.projection import on_pitch_mask, project_segment
from soccer_analytics.analysis.stage_a import load_segment
from soccer_analytics.ingest.ffmpeg_reader import FFmpegFrameReader

MIN_CONFIDENCE = 0.55  # detections below this are not worth cropping
UPSAMPLE_TARGET_PX = 96.0  # torso crops smaller than this are upscaled before OCR
# The very largest boxes are near-sideline bystanders (coaches, spectators), not players - measured on the real
# game, boxes over 200 px at 1920 width are overwhelmingly off-pitch people. Numbers are read from player-sized
# boxes only, and the sharpest crops win: motion blur kills OCR on small digits.
PLAYER_MAX_HEIGHT_PX = 200.0
BLUR_REJECT_LAPLACIAN = 12.0  # variance of Laplacian below this is too blurred to read


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--match", required=True, help="match id in the archive")
    parser.add_argument("--video", required=True)
    parser.add_argument("--segment", required=True, help="segment directory from Stage A")
    parser.add_argument("--width", type=int, default=1920, help="decode width for the crops")
    parser.add_argument("--min-height-px", type=float, default=130.0, help="box height (at 1920) worth OCRing")
    parser.add_argument("--max-per-track", type=int, default=40)
    parser.add_argument("--max-crops", type=int, default=1500)
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    return parser.parse_args()


class Status:
    """Progress file the dashboard polls; written atomically so a reader never sees half a document."""

    def __init__(self, path: Path, total: int):
        self.path = path
        self.payload = {"state": "running", "crops_total": total, "crops_done": 0, "readings": 0, "started": time.time()}

    def update(self, **kwargs) -> None:
        self.payload.update(kwargs)
        self.payload["updated"] = time.time()
        tmp = self.path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(self.payload))
        tmp.replace(self.path)


def main() -> int:
    args = parse_args()
    library = MatchLibrary()
    status = Status(library.path(args.match) / "jerseys_status.json", 0)

    try:
        calibration = library.load_calibration(args.match)
        if calibration is None:
            status.update(state="error", message="no calibration saved for this match - register the pitch first")
            return 1
        record = library.load(args.match)
        segment = load_segment(args.segment)
        detections = project_segment(segment, calibration)
        report, assignment = stage_b.build_report(
            detections,
            pitch_length_m=record.pitch_length_m,
            pitch_width_m=record.pitch_width_m,
            match_frames=len(segment.time),
        )
        _ = report

        keep = on_pitch_mask(detections, record.pitch_length_m, record.pitch_width_m)
        track_of_row = np.full(len(detections.frame), -1, dtype=np.int64)
        for track_id, rows in assignment.tracks.items():
            track_of_row[rows] = track_id

        # Per track, the clearest crops across the whole segment: player-sized boxes first (the very largest are
        # near-sideline bystanders, not players), then sharpest, then spread over time so all lights appear.
        heights = detections.height_px
        selected: dict[int, list[int]] = {}
        for track_id, rows in assignment.tracks.items():
            usable = [
                int(row)
                for row in rows
                if keep[row]
                and args.min_height_px <= heights[row] <= PLAYER_MAX_HEIGHT_PX
                and detections.conf[row] >= MIN_CONFIDENCE
            ]
            usable.sort(key=lambda row: -(heights[row] * detections.conf[row]))
            if len(usable) > args.max_per_track:
                # keep the best half by quality, half spread evenly through the list so all kits/lights appear
                kept = usable[: args.max_per_track // 2]
                rest = usable[args.max_per_track // 2 :]
                stride = max(1, len(rest) // (args.max_per_track - len(kept)))
                kept += rest[::stride][: args.max_per_track - len(kept)]
                usable = kept
            if usable:
                selected[track_id] = sorted(usable)

        rows_by_frame: dict[int, list[tuple[int, int]]] = {}
        for track_id, rows in selected.items():
            for row in rows:
                rows_by_frame.setdefault(int(detections.frame[row]), []).append((track_id, row))
        total = sum(len(v) for v in rows_by_frame.values())
        if total > args.max_crops:
            # Thin evenly rather than truncating: later frames matter as much as early ones. Per-frame strides
            # barely bite when most frames hold only one or two crops (measured: 8,764 crops across 6,000 frames
            # thinned to 8,700), so the budget is enforced across the whole frame list - every stride-th frame
            # keeps its crops, the rest are dropped whole.
            stride = int(np.ceil(total / args.max_crops))
            frames_sorted = sorted(rows_by_frame)
            rows_by_frame = {frame: rows_by_frame[frame] for frame in frames_sorted[::stride]}
            total = sum(len(v) for v in rows_by_frame.values())
        status.update(crops_total=total)
        print(f"[jerseys] {len(selected)} tracks, {total} crops from {len(rows_by_frame)} frames", flush=True)
        if total == 0:
            status.update(state="done", message="no detections large or confident enough to read a number")
            library.save_jerseys(args.match, {"candidates": [], "suggestions": {}, "meta": {"crops": 0}})
            return 0

        try:
            import easyocr  # heavy import: only needed when there is actually work to do
        except ImportError:
            # The scan is the only thing that needs OCR, so the import is deferred to here - and a missing
            # install must say what to do rather than dying with a bare ModuleNotFoundError in the status file.
            status.update(
                state="error",
                message="easyocr is not installed in this environment - install it with `pip install easyocr`",
            )
            return 1

        reader = easyocr.Reader(["en"], gpu=args.device == "cuda", verbose=False)

        candidates: list[JerseyCandidate] = []
        per_track_readings: dict[int, list[JerseyCandidate]] = {}
        done = 0
        wanted_frames = sorted(rows_by_frame)
        decode_start_s = float(segment.time[wanted_frames[0]])
        decode_end_s = float(segment.time[wanted_frames[-1]])
        # det boxes in segment order, looked up through the detections' provenance index
        boxes = segment.det_box[detections.det_index]
        video_reader = FFmpegFrameReader(
            args.video, fps=float(segment.meta["fps"]), width=args.width, start_s=decode_start_s, duration_s=decode_end_s - decode_start_s + 1.0
        )
        wanted = set(wanted_frames)
        fps = float(segment.meta["fps"])
        # The reader yields SOURCE timestamps (segment.time is source seconds, offset by the segment's start in
        # the source video), so the analysis frame index is (time - start) * fps - not time * fps. Getting this
        # wrong reads every crop from the wrong frame: measured on the real game, 5455 crops yielded 27 readings
        # and 0 suggestions because the OCR was looking at grass.
        segment_start_s = float(segment.meta.get("start_s", 0.0))
        for time_s, frame in video_reader.frames():
            index = int(round((time_s - segment_start_s) * fps))
            if index not in wanted:
                continue
            for track_id, row in rows_by_frame[index]:
                crop = crop_torso(frame, tuple(boxes[row]))
                if crop is None:
                    continue
                # Motion blur makes small digits unreadable; a blurred crop wastes OCR time and produces
                # confident nonsense. The variance of the Laplacian is the standard sharpness proxy.
                grey = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
                if cv2.Laplacian(grey, cv2.CV_64F).var() < BLUR_REJECT_LAPLACIAN:
                    continue
                scale = min(4.0, max(1.0, UPSAMPLE_TARGET_PX / max(1, crop.shape[0])))
                if scale > 1.05:
                    crop = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
                best: tuple[str, float] = ("", 0.0)
                for _box, text, confidence in reader.readtext(crop, allowlist="0123456789", detail=1, paragraph=False):
                    digits = sanitize_digits(text)
                    if digits and confidence > best[1]:
                        best = (digits, float(confidence))
                if best[0]:
                    candidate = JerseyCandidate(frame=index, row=int(row), digits=best[0], confidence=best[1])
                    candidates.append(candidate)
                    per_track_readings.setdefault(track_id, []).append(candidate)
                done += 1
                if done % 25 == 0:
                    status.update(crops_done=done, readings=len(candidates))
                    print(f"[jerseys] {done}/{total} crops, {len(candidates)} readings", flush=True)
            if done >= total:
                break

        suggestions = aggregate_candidates(per_track_readings)
        payload = {
            "candidates": [vars(candidate) | {"track": track} for track, items in per_track_readings.items() for candidate in items],
            "suggestions": {str(track): entry for track, entry in sorted(suggestions.items())},
            "meta": {
                "crops": done,
                "tracks_scanned": len(selected),
                "readings": len(candidates),
                "suggested": len(suggestions),
                "min_height_px": args.min_height_px,
                "width": args.width,
            },
        }
        library.save_jerseys(args.match, payload)
        status.update(state="done", crops_done=done, readings=len(candidates), suggested=len(suggestions))
        print(f"[jerseys] done: {len(suggestions)} of {len(selected)} tracks got a number", flush=True)
        return 0
    except Exception as exc:  # a background job must record why it died
        import traceback

        status.update(state="error", message=f"{type(exc).__name__}: {exc}", traceback=traceback.format_exc()[-2000:])
        raise


if __name__ == "__main__":
    raise SystemExit(main())
