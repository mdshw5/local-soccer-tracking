"""Read shirt numbers off the footage, per tracked player. Runs in the background from the dashboard.

Strategy: give every votable track its own budget of clear crops (shirt numbers are only legible on close-to-camera
players), crop the torso band, and run EasyOCR with a digits-only alphabet. Every reading is a candidate; the
per-track majority vote in ``analysis.jerseys`` decides what - if anything - the track wears. The budget is per
track on purpose: a global crop cap starves the vote - the shipped scan kept 1368 crops across 1193 tracks (~1 per
track) while a number needs ``MIN_VOTES`` agreeing readings, so it could only ever suggest 5 tracks' numbers.

Decode dominates the runtime (~3.4x realtime over the wanted span), and OCR is ~16 ms/crop, so crops are nearly
free next to the decode that must happen anyway.

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
from soccer_analytics.analysis.jerseys import (
    MIN_VOTES,
    JerseyCandidate,
    aggregate_candidates,
    crop_torso,
    sanitize_digits,
)
from soccer_analytics.analysis.library import MatchLibrary
from soccer_analytics.analysis.projection import on_pitch_mask, project_segment
from soccer_analytics.analysis.stage_a import load_segment
from soccer_analytics.ingest.source import open_reader, probe_source  # noqa: E402

MIN_CONFIDENCE = 0.55  # detections below this are not worth cropping
UPSAMPLE_TARGET_PX = 96.0  # torso crops smaller than this are upscaled before OCR
# The very largest boxes are near-sideline bystanders (coaches, spectators), not players - measured on the real
# game, boxes over 200 px at 1920 width are overwhelmingly off-pitch people. Numbers are read from player-sized
# boxes only, and the sharpest crops win: motion blur kills OCR on small digits.
PLAYER_MAX_HEIGHT_PX = 200.0
BLUR_REJECT_LAPLACIAN = 12.0  # variance of Laplacian below this is too blurred to read
# A number stays visible for about a second of walking, so the frames around a successful reading are the most
# promising crops left to try. The main selection samples for size and spread, so a track whose number was legible
# for a moment often holds only a couple of those frames; measured on the real game, track 11270 read "6" twice and
# the frames within +-1.6 s of those readings read it a third time - exactly the vote the quorum lacked. Tracks
# with fewer than REFINE_MIN_READINGS readings get this second look; a track with no reading has no anchor at all.
REFINE_SPAN_FRAMES = 8
REFINE_MAX_FRAMES = 16
REFINE_MIN_READINGS = MIN_VOTES + 1  # below a comfortable quorum only - an extra vote is cheap insurance


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--match", required=True, help="match id in the archive")
    parser.add_argument("--video", required=True)
    parser.add_argument("--segment", required=True, help="segment directory from Stage A")
    parser.add_argument("--width", type=int, default=1920, help="decode width for the crops")
    parser.add_argument("--min-height-px", type=float, default=130.0, help="box height (at 1920) worth OCRing")
    parser.add_argument("--max-per-track", type=int, default=24, help="crops per track: half sharpest, half spread over time")
    parser.add_argument("--max-crops", type=int, default=24000, help="safety valve only - per-track budgets bound the normal case")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    return parser.parse_args()


class Status:
    """Progress file the dashboard polls; written atomically so a reader never sees half a document."""

    def __init__(self, path: Path, total: int):
        self.path = path
        self.payload = {"state": "running", "crops_total": total, "crops_done": 0, "readings": 0, "started": time.time()}

    def update(self, **kwargs) -> None:
        """Update the progress file - and never let its failure end the scan it reports on.

        The file is diagnostics: a dashboard reads it, and nothing the scan computes depends on it. On the real
        game a mid-scan migration moved the analysis directory out from under a 90-minute run and the write's
        FileNotFoundError aborted the whole scan at 12,475 of 14,602 crops - the readings were thrown away with
        it. A report that cannot be filed is worth a one-line warning, not the work.
        """
        self.payload.update(kwargs)
        self.payload["updated"] = time.time()
        tmp = self.path.with_suffix(".json.tmp")
        try:
            tmp.write_text(json.dumps(self.payload))
            tmp.replace(self.path)
        except OSError as exc:
            print(f"[jerseys] warning: could not write the status file ({type(exc).__name__}: {exc})", flush=True)


def main() -> int:
    args = parse_args()
    library = MatchLibrary()
    status = Status(library.path(args.match) / "jerseys_status.json", 0)

    try:
        calibration = library.load_calibration(args.match)
        if calibration is None:
            status.update(state="error", message="no calibration saved for this match - register the pitch first")
            return 1
        # The staleness flag compares this stamp with calibration.json's current mtime: the dashboard and the
        # stream say "the numbers were read before the pitch was last calibrated" when a re-fit happened after
        # the scan, and they cannot say it without the stamp. Captured now, because the calibration loaded here
        # is the one this scan's tracks were projected through.
        calibration_file = library.path(args.match) / "calibration.json"
        calibration_saved = calibration_file.stat().st_mtime if calibration_file.exists() else None
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
        #
        # Every track gets its OWN budget instead of a slice of one global crop count: the vote needs MIN_VOTES
        # agreeing readings, but only a fraction of torso crops show a number at all (the player may face the
        # camera, or another body occludes the shirt), so a track needs a healthy row of crops before its number
        # can win. A track with fewer usable rows than MIN_VOTES can never reach the quorum and is skipped - no
        # crop of its could ever be part of a majority.
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
            if len(usable) < MIN_VOTES:
                continue
            usable.sort(key=lambda row: -(heights[row] * detections.conf[row]))
            if len(usable) > args.max_per_track:
                # keep the best half by quality; from the rest take an even slice through TIME so the spread
                # covers the game's lights and kits (the rest is still quality-ordered, so striding it directly
                # would cluster the spread around the same few already-kept moments)
                kept = usable[: args.max_per_track // 2]
                rest = sorted(usable[args.max_per_track // 2 :], key=lambda row: detections.frame[row])
                stride = max(1, len(rest) // (args.max_per_track - len(kept)))
                kept += rest[::stride][: args.max_per_track - len(kept)]
                usable = kept
            selected[track_id] = sorted(usable)

        rows_by_frame: dict[int, list[tuple[int, int]]] = {}
        for track_id, rows in selected.items():
            for row in rows:
                rows_by_frame.setdefault(int(detections.frame[row]), []).append((track_id, row))
        total = sum(len(v) for v in rows_by_frame.values())
        if total > args.max_crops:
            # Emergency valve only: per-track budgets bound the normal case (~900 votable tracks x 24 crops), so
            # this triggers only on a pathological segment. If it does, drop whole frames evenly - never single
            # crops off the front, which would re-starve the vote for whichever tracks were listed last.
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
        # The crops are cut at the segment's clock, so the video must actually cover the segment's window: one raw
        # camera clip of a combined game keeps its own shorter clock, and decoding it at game-clock offsets reads
        # the wrong film (and silently loses every crop past the clip's end). A manifest reports the whole game's
        # span, which is the number this check wants. Cheap to check, ugly to debug.
        span = probe_source(args.video)
        if span.duration_s + 1.0 < decode_end_s:
            status.update(
                state="error",
                message=(
                    f"the video ends at {span.duration_s:.0f}s but the segment's window runs to "
                    f"{decode_end_s:.0f}s - point the scan at the game manifest or the combined game video the "
                    "segment was built from, not one raw clip"
                ),
            )
            return 1
        # det boxes in segment order, looked up through the detections' provenance index
        boxes = segment.det_box[detections.det_index]
        video_reader = open_reader(
            args.video, fps=float(segment.meta["fps"]), width=args.width, start_s=decode_start_s, duration_s=decode_end_s - decode_start_s + 1.0
        )
        wanted = set(wanted_frames)
        fps = float(segment.meta["fps"])
        # The reader yields SOURCE timestamps (segment.time is source seconds, offset by the segment's start in
        # the source video), so the analysis frame index is (time - start) * fps - not time * fps. Getting this
        # wrong reads every crop from the wrong frame: measured on the real game, 5455 crops yielded 27 readings
        # and 0 suggestions because the OCR was looking at grass.
        segment_start_s = float(segment.meta.get("start_s", 0.0))
        done_pairs: set[tuple[int, int]] = set()  # (track, analysis frame) already cropped once
        for time_s, frame in video_reader.frames():
            index = int(round((time_s - segment_start_s) * fps))
            if index not in wanted:
                continue
            for track_id, row in rows_by_frame[index]:
                done_pairs.add((track_id, index))
                crop = crop_torso(frame, tuple(boxes[row]))
                if crop is None:
                    continue
                # Motion blur makes small digits unreadable; a blurred crop wastes OCR time and produces
                # confident nonsense. The variance of the Laplacian is the standard sharpness proxy.
                gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
                if cv2.Laplacian(gray, cv2.CV_64F).var() < BLUR_REJECT_LAPLACIAN:
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

        # Second look around the frames that DID read (see REFINE_* at the top): harvest the neighboring frames
        # of each reading for tracks that do not yet sit on a comfortable quorum. Blur/quality rules apply
        # exactly as in the main pass; the extra readings vote alongside the originals, and the vote itself (not
        # this phase) decides whether the track now wears a number.
        refine_targets = {
            track_id: items for track_id, items in per_track_readings.items() if len(items) < REFINE_MIN_READINGS
        }
        refined = 0
        if refine_targets:
            wanted_refine: dict[int, list[tuple[int, int]]] = {}
            seen_pairs: set[tuple[int, int]] = set()
            for track_id, items in refine_targets.items():
                lookup: dict[int, int] = {}
                # The track's FULL observation list, not its selected crops: the frames this phase is looking for
                # are exactly the ones the selection did not pick (building this from `selected` skips every
                # neighbor, because a neighbor is by definition not selected - the first shipped version of this
                # phase found zero frames and silently did nothing).
                for row in assignment.tracks[track_id]:
                    lookup.setdefault(int(detections.frame[row]), int(row))
                extras = 0
                for item in sorted(items, key=lambda candidate: -candidate.confidence):
                    for delta in range(1, REFINE_SPAN_FRAMES + 1):
                        for frame_index in (item.frame - delta, item.frame + delta):
                            pair = (track_id, frame_index)
                            if frame_index not in lookup or pair in seen_pairs or pair in done_pairs:
                                continue
                            seen_pairs.add(pair)
                            wanted_refine.setdefault(frame_index, []).append((track_id, lookup[frame_index]))
                            extras += 1
                            if extras >= REFINE_MAX_FRAMES:
                                break
                        if extras >= REFINE_MAX_FRAMES:
                            break
                    if extras >= REFINE_MAX_FRAMES:
                        break
            # Report the phase even when it collected nothing: a silent zero is indistinguishable from the phase
            # not running at all, which is exactly how its first version hid a lookup bug.
            print(
                f"[jerseys] second look: {sum(len(v) for v in wanted_refine.values())} frames around {len(refine_targets)} tracks",
                flush=True,
            )
            if wanted_refine:
                clusters: list[list[int]] = []
                for frame_index in sorted(wanted_refine):
                    if clusters and frame_index - clusters[-1][-1] <= 2 * REFINE_SPAN_FRAMES:
                        clusters[-1].append(frame_index)
                    else:
                        clusters.append([frame_index])
                for cluster in clusters:
                    stream = open_reader(
                        args.video,
                        fps=fps,
                        width=args.width,
                        start_s=float(segment.time[cluster[0]]),
                        duration_s=(cluster[-1] - cluster[0]) / fps + 0.6,
                    )
                    remaining = {f: list(wanted_refine[f]) for f in cluster}
                    for time_s, frame in stream.frames():
                        index = int(round((time_s - segment_start_s) * fps))
                        if index not in remaining:
                            continue
                        for track_id, row in remaining.pop(index):
                            crop = crop_torso(frame, tuple(boxes[row]))
                            if crop is None:
                                continue
                            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
                            if cv2.Laplacian(gray, cv2.CV_64F).var() < BLUR_REJECT_LAPLACIAN:
                                continue
                            scale = min(4.0, max(1.0, UPSAMPLE_TARGET_PX / max(1, crop.shape[0])))
                            if scale > 1.05:
                                crop = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
                            best: tuple[str, float] = ("", 0.0)
                            for _box, text, confidence in reader.readtext(crop, allowlist="0123456789", detail=1, paragraph=False):
                                digits = sanitize_digits(text)
                                if digits and confidence > best[1]:
                                    best = (digits, float(confidence))
                            refined += 1
                            if best[0]:
                                candidate = JerseyCandidate(frame=index, row=int(row), digits=best[0], confidence=best[1])
                                candidates.append(candidate)
                                per_track_readings.setdefault(track_id, []).append(candidate)
                        if not remaining:
                            break
                status.update(crops_done=total, refined=refined, readings=len(candidates))
                print(f"[jerseys] second look done: {refined} extra crops, {len(candidates)} readings total", flush=True)

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
                "calibration_saved": calibration_saved,
                "segment": args.segment,
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
