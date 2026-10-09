"""Probe the shirt-number scan: where crops come from, and what they actually look like.

Two modes:

* Pipeline mode (``--video`` + ``--segment``): rebuilds exactly the projection+tracking
  ``scripts/extract_jerseys.py`` uses, prints the selection funnel per filter (on-pitch -> height band ->
  confidence), simulates the scan's crop selection, and - with ``--windows`` - decodes real torso crops from
  the game video via the scan's own frame mapping and writes montage PNGs so the input quality is judged by
  eye, not assumed.
* Track mode (``--tracks 928,1373``): inspects specific tracks straight from the replay boxes - no pipeline
  rebuild - cropping each track's biggest on-screen frames, OCRing them with the scan's preprocessing, and
  writing one montage per track. Answers "this player's number is clearly visible, why is it missing?"

Usage::

    python scripts/probe_jerseys.py --match <match_id> --video <game.mp4> --segment <dir> \
        --windows 9980:10160,13990:14740 --outdir outputs/jersey_probe
    python scripts/probe_jerseys.py --match <match_id> --tracks 928,1373,9238
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import cv2
import numpy as np

from soccer_analytics.analysis import stage_b
from soccer_analytics.analysis.jerseys import crop_torso
from soccer_analytics.analysis.library import MatchLibrary
from soccer_analytics.analysis.projection import on_pitch_mask, project_segment
from soccer_analytics.analysis.stage_a import load_segment
from soccer_analytics.ingest.ffmpeg_reader import FFmpegFrameReader

MIN_CONFIDENCE = 0.55
PLAYER_MAX_HEIGHT_PX = 200.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--match", required=True)
    parser.add_argument("--video", default="", help="game video; optional in track mode (segment meta supplies it)")
    parser.add_argument("--segment", default="", help="segment dir; optional in track mode (match record supplies it)")
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--min-height-px", type=float, default=130.0)
    parser.add_argument("--max-per-track", type=int, default=24)
    parser.add_argument("--max-crops", type=int, default=24000)
    parser.add_argument("--windows", default="", help="analysis-frame windows 'a:b,c:d' to montage")
    parser.add_argument("--outdir", default="outputs/jersey_probe")
    parser.add_argument("--tile-height", type=int, default=160)
    parser.add_argument("--max-tiles", type=int, default=90)
    parser.add_argument("--ocr", action="store_true", help="run easyocr on the window crops exactly like the scan")
    parser.add_argument("--tracks", default="", help="comma-separated track ids: inspect their biggest frames instead")
    parser.add_argument("--picks", type=int, default=6, help="frames to inspect per track in track mode")
    return parser.parse_args()


def track_mode(args: argparse.Namespace) -> int:
    """Inspect chosen tracks: crop their biggest on-screen moments straight from the replay boxes, OCR, montage."""
    import json

    from soccer_analytics.analysis.jerseys import sanitize_digits

    library = MatchLibrary()
    record = library.load(args.match)
    replay = library.load_replay(args.match)
    if replay is None or not replay.get("players"):
        print("no replay for this match - build the report first")
        return 1
    boxes = library.load_replay_boxes(args.match)
    segment_dir = Path(args.segment) if args.segment else Path(record.segments[0]) if record.segments else None
    if segment_dir is None:
        print("no segment known for this match; pass --segment")
        return 1
    meta = json.loads((segment_dir / "meta.json").read_text())
    fps = float(meta["fps"])
    start_s = float(meta.get("start_s", 0.0))
    video = args.video or str(meta.get("video") or "")
    if not video or not Path(video).exists():
        print(f"video not found: {video!r}; pass --video")
        return 1
    players = {int(p["track_id"]): p for p in replay["players"]}
    jerseys = library.load_jerseys(args.match)
    readings_by_track: dict[int, list] = {}
    for candidate in jerseys.get("candidates", []):
        readings_by_track.setdefault(int(candidate["track"]), []).append(candidate)

    import easyocr

    reader = easyocr.Reader(["en"], gpu=True, verbose=False)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    for tid in [int(value) for value in args.tracks.split(",") if value.strip()]:
        player = players.get(tid)
        box = boxes.get(str(tid))
        if player is None or box is None or not len(box):
            print(f"t{tid}: not in the replay payload / no boxes")
            continue
        hs = (box[:, 3] - box[:, 1]) * 1920.0
        picks = [i for i in range(len(hs)) if hs[i] >= 60.0]  # 60px is the kit-descriptor floor; colours read fine there
        picks.sort(key=lambda i: -hs[i])
        picks = picks[: max(8, args.picks * 3)]
        if len(picks) > args.picks:
            step = len(picks) / args.picks
            picks = [picks[int(i * step)] for i in range(args.picks)]
        picks = sorted(set(picks))
        # Always include the frames the scan itself read from, so a reading can be judged against its crop.
        frames_array = np.asarray(player["frames"])
        for candidate in readings_by_track.get(tid, []):
            found = np.where(frames_array == int(candidate["frame"]))[0]
            if len(found):
                picks = sorted(set(picks + [int(found[0])]))
        existing = [(c["digits"], round(c["confidence"], 2)) for c in readings_by_track.get(tid, [])]
        print(f"\nt{tid}: {len(hs)} obs, max {hs.max():.0f}px | scan readings: {existing or 'none'}")
        tiles = []
        # Decode each cluster of picks sequentially at the analysis rate - the same frame-exact path the scan
        # uses. Single-frame seeks drifted between the frame and its box (one pick cropped grass where the player
        # had moved on; the numbered frames were sliced at the edges), which made those crops silently lie about
        # what the scan actually saw.
        groups: list[list[int]] = []
        for i in picks:
            if groups and int(player["frames"][i]) - int(player["frames"][groups[-1][-1]]) <= 150:
                groups[-1].append(i)
            else:
                groups.append([i])
        for group in groups:
            first_frame = int(player["frames"][group[0]])
            group_wanted = {int(player["frames"][i]): i for i in group}
            span_frames = int(player["frames"][group[-1]]) - first_frame
            stream = FFmpegFrameReader(
                video,
                fps=fps,
                width=args.width,
                start_s=start_s + first_frame / fps,
                duration_s=span_frames / fps + 0.6,
            )
            for time_s, image in stream.frames():
                frame = int(round((time_s - start_s) * fps))
                if frame not in group_wanted:
                    continue
                i = group_wanted.pop(frame)
                crop = crop_torso(image, tuple(box[i]))
                if crop is None:
                    continue
                grey = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
                blur = float(cv2.Laplacian(grey, cv2.CV_64F).var())
                scaled = crop
                scale = min(4.0, max(1.0, 96.0 / max(1, crop.shape[0])))
                if scale > 1.05:
                    scaled = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
                best = ("", 0.0)
                outputs = []
                for _b, text, confidence in reader.readtext(scaled, allowlist="0123456789", detail=1, paragraph=False):
                    outputs.append((str(text), round(float(confidence), 2)))
                    digits = sanitize_digits(text)
                    if digits and confidence > best[1]:
                        best = (digits, float(confidence))
                print(f"   f{frame} t={time_s:.1f} h={hs[i]:.0f} blur={blur:.0f} -> {best[0] or '-':>2} {best[1]:.2f}  raw={outputs}")
                crop_dir = outdir / f"tracks_{tid}"
                crop_dir.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(crop_dir / f"f{frame:05d}_h{hs[i]:.0f}.png"), crop)
                tiles.append((tid, frame, hs[i], crop))
                if not group_wanted:
                    break
        if tiles:
            th = args.tile_height
            rendered = []
            for _t, frame, height, crop in tiles:
                scale = th / crop.shape[0]
                tile = cv2.resize(crop, (max(2, int(crop.shape[1] * scale)), th), interpolation=cv2.INTER_CUBIC)
                canvas = np.zeros((th + 22, max(tile.shape[1], 160), 3), np.uint8)
                canvas[22:, : tile.shape[1]] = tile
                cv2.putText(canvas, f"t{tid} f{frame} h{height:.0f}", (2, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 0), 1, cv2.LINE_AA)
                rendered.append(canvas)
            tile_w = max(t.shape[1] for t in rendered)
            sheet = np.zeros((th + 22, len(rendered) * tile_w, 3), np.uint8)
            for i, tile in enumerate(rendered):
                sheet[0 : th + 22, i * tile_w : i * tile_w + tile.shape[1]] = tile
            path = outdir / f"tracks_{tid}.png"
            cv2.imwrite(str(path), sheet)
            print(f"   montage: {path}")
    return 0


def main() -> int:
    args = parse_args()
    if args.tracks:
        return track_mode(args)
    if not args.video or not args.segment:
        print("--video and --segment are required unless --tracks is given")
        return 1
    library = MatchLibrary()
    record = library.load(args.match)
    calibration = library.load_calibration(args.match)
    if calibration is None:
        print("no calibration saved - register the pitch first")
        return 1
    segment = load_segment(args.segment)
    detections = project_segment(segment, calibration)
    _report, assignment = stage_b.build_report(
        detections,
        pitch_length_m=record.pitch_length_m,
        pitch_width_m=record.pitch_width_m,
        match_frames=len(segment.time),
    )
    keep = on_pitch_mask(detections, record.pitch_length_m, record.pitch_width_m)
    heights = detections.height_px

    # ---- funnel ------------------------------------------------------------------------------------------------
    conf_ok = detections.conf >= MIN_CONFIDENCE
    in_band = (heights >= args.min_height_px) & (heights <= PLAYER_MAX_HEIGHT_PX)
    print(f"detections: {len(heights)}")
    print(
        "heights px [p10 p25 p50 p75 p90 p95 p99]:",
        np.percentile(heights, [10, 25, 50, 75, 90, 95, 99]).round(0),
    )
    print(f"on-pitch: {int(keep.sum())}  +conf>= {MIN_CONFIDENCE}: {int((keep & conf_ok).sum())}  "
          f"+band {args.min_height_px:.0f}-{PLAYER_MAX_HEIGHT_PX:.0f}: {int((keep & conf_ok & in_band).sum())}")

    usable_mask = keep & conf_ok & in_band
    per_track: dict[int, list[int]] = {}
    for track_id, rows in assignment.tracks.items():
        usable = [int(row) for row in rows if usable_mask[row]]
        if usable:
            per_track[track_id] = sorted(usable)
    counts = Counter(len(v) for v in per_track.values())
    print(f"tracks total: {len(assignment.tracks)}; with >=1 usable row: {len(per_track)}")
    print("tracks by usable-row count:", dict(sorted(counts.items())[:12]), "... (10+ merged)" if max(counts) >= 10 else "")
    print(f"tracks with >=3 usable rows: {sum(1 for v in per_track.values() if len(v) >= 3)}")
    print(f"tracks with >=6 usable rows: {sum(1 for v in per_track.values() if len(v) >= 6)}")

    # The scan's own selection, verbatim: per-track budget, best half + time-spread half, votable tracks only.
    from soccer_analytics.analysis.jerseys import MIN_VOTES

    selected: dict[int, list[int]] = {}
    for track_id, usable in per_track.items():
        if len(usable) < MIN_VOTES:
            continue
        usable = sorted(usable, key=lambda row: -(heights[row] * detections.conf[row]))
        if len(usable) > args.max_per_track:
            kept = usable[: args.max_per_track // 2]
            rest = sorted(usable[args.max_per_track // 2:], key=lambda row: detections.frame[row])
            stride = max(1, len(rest) // (args.max_per_track - len(kept)))
            kept += rest[::stride][: args.max_per_track - len(kept)]
            usable = kept
        selected[track_id] = sorted(usable)
    total = sum(len(v) for v in selected.values())
    print(f"scan would select: {total} crops across {len(selected)} tracks "
          f"(cap {args.max_crops}; thinning {'ACTIVE' if total > args.max_crops else 'inactive'})")

    # The scan's frame-stride thinning, verbatim: whole frames are dropped, never single rows.
    rows_by_frame: dict[int, list[int]] = {}
    for track_id, rows in selected.items():
        for row in rows:
            rows_by_frame.setdefault(int(detections.frame[row]), []).append(row)
    if total > args.max_crops:
        stride = int(np.ceil(total / args.max_crops))
        frames_sorted = sorted(rows_by_frame)
        rows_by_frame = {frame: rows_by_frame[frame] for frame in frames_sorted[::stride]}
    after = sum(len(v) for v in rows_by_frame.values())
    per_track_after = Counter()
    for rows in rows_by_frame.values():
        for row in rows:
            per_track_after[int(assignment.track_id[row])] += 1
    after_counts = Counter(per_track_after.values())
    print(f"after thinning: {after} crops across {len(per_track_after)} tracks")
    print("crops-per-track after thinning:", dict(sorted(after_counts.items())[:14]))
    reach = sum(1 for n in per_track_after.values() if n >= 3)
    print(f"tracks that could even reach min_votes=3: {reach}")

    # ---- montage / OCR -----------------------------------------------------------------------------------------
    if not args.windows:
        return 0
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    boxes = segment.det_box[detections.det_index]
    fps = float(segment.meta["fps"])
    segment_start_s = float(segment.meta.get("start_s", 0.0))
    reader_ocr = None
    if args.ocr:
        import easyocr

        reader_ocr = easyocr.Reader(["en"], gpu=True, verbose=False)
    from soccer_analytics.analysis.jerseys import sanitize_digits

    ocr_readings: dict[int, list[tuple[str, float, int]]] = {}

    for window in args.windows.split(","):
        start_frame, _, end_frame = window.partition(":")
        first, last = int(start_frame), int(end_frame)
        wanted = sorted(f for f in rows_by_frame if first <= f <= last)
        if not wanted:
            print(f"window {first}:{last}: no selected crops")
            continue
        tiles: list[tuple[int, int, np.ndarray, float]] = []
        decode_start_s = float(segment.time[wanted[0]])
        decode_end_s = float(segment.time[wanted[-1]])
        reader = FFmpegFrameReader(
            args.video,
            fps=fps,
            width=args.width,
            start_s=decode_start_s,
            duration_s=decode_end_s - decode_start_s + 1.0,
        )
        wanted_set = set(wanted)
        for time_s, frame in reader.frames():
            index = int(round((time_s - segment_start_s) * fps))
            if index not in wanted_set:
                continue
            for row in rows_by_frame[index]:
                crop = crop_torso(frame, tuple(boxes[row]))
                if crop is None:
                    continue
                grey = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
                blur = float(cv2.Laplacian(grey, cv2.CV_64F).var())
                height = float(heights[row])
                track = int(assignment.track_id[row])
                if reader_ocr is not None:
                    # The scan's own preprocessing, verbatim, so the probe measures the shipped path.
                    if blur >= 12.0:
                        scaled = crop
                        scale = min(4.0, max(1.0, 96.0 / max(1, crop.shape[0])))
                        if scale > 1.05:
                            scaled = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
                        best = ("", 0.0)
                        for _box, text, confidence in reader_ocr.readtext(
                            scaled, allowlist="0123456789", detail=1, paragraph=False
                        ):
                            digits = sanitize_digits(text)
                            if digits and confidence > best[1]:
                                best = (digits, float(confidence))
                        if best[0]:
                            ocr_readings.setdefault(track, []).append((best[0], best[1], index))
                            print(f"  [ocr] t{track} f{index} -> '{best[0]}' {best[1]:.2f} (h={height:.0f}px blur={blur:.0f})")
                tiles.append((track, index, crop, blur))
                if len(tiles) >= args.max_tiles:
                    break
            if len(tiles) >= args.max_tiles:
                break
        if not tiles:
            print(f"window {first}:{last}: nothing decoded")
            continue
        # grid montage with labels
        th = args.tile_height
        rendered = []
        for track, index, crop, blur in tiles:
            scale = th / crop.shape[0]
            tile = cv2.resize(crop, (max(2, int(crop.shape[1] * scale)), th), interpolation=cv2.INTER_CUBIC)
            canvas = np.zeros((th + 22, max(tile.shape[1], 150), 3), np.uint8)
            canvas[22:, : tile.shape[1]] = tile
            label = f"t{track} f{index} b{blur:.0f}"
            cv2.putText(canvas, label, (2, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 0), 1, cv2.LINE_AA)
            rendered.append(canvas)
        per_row = 8
        tile_w = max(t.shape[1] for t in rendered)
        rows_n = (len(rendered) + per_row - 1) // per_row
        sheet = np.zeros((rows_n * (th + 22), per_row * tile_w, 3), np.uint8)
        for i, tile in enumerate(rendered):
            r, c = divmod(i, per_row)
            sheet[r * (th + 22):(r + 1) * (th + 22), c * tile_w:c * tile_w + tile.shape[1]] = tile
        path = outdir / f"crops_{first}_{last}.png"
        cv2.imwrite(str(path), sheet)
        blurs = np.array([t[3] for t in tiles])
        print(f"window {first}:{last}: {len(tiles)} crops -> {path}  blur p10/p50 {np.percentile(blurs, [10, 50]).round(1)}")
    if reader_ocr is not None:
        from soccer_analytics.analysis.jerseys import JerseyCandidate, aggregate_candidates

        print("per-track OCR readings in the probed windows:")
        for track, items in sorted(ocr_readings.items()):
            print(f"  t{track}: {[(d, round(c, 2), f) for d, c, f in items]}")
        votes = aggregate_candidates(
            {
                track: [JerseyCandidate(frame=f, row=-1, digits=d, confidence=c) for d, c, f in items]
                for track, items in ocr_readings.items()
            }
        )
        print(f"aggregate would assign: {votes}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
