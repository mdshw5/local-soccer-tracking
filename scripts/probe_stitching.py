"""Probe: measure how track fragments fail to stitch, with samples a human can look at.

The online pass breaks a player's track whenever the camera whips, the tracker loses them, or the ReID
association misfires; ``stage_b._stitch_tracks`` then reconnects the pieces. This probe answers the question the
stitcher's constants cannot: of the joins it *refuses*, which refuse a real continuation and why?

It runs the whole Stage B pass once over a segment (or re-reads a saved fragment bundle), then:

* saves every pre-stitch fragment's data as a bundle (``bundle.npz``) so stitcher experiments can be re-scored
  in seconds instead of re-running the 20-minute pass;
* scans all end->start fragment pairs in a loose plausibility window and reports, per gate, how many pairs are
  refused - split by whether the pair was physically in reach (a sprint plus slack), whether both sides have
  kit-color evidence, and the gap-size distribution;
* reports how many gate-passing pairs become joins (it imports the pipeline's own ``_join_fragments``, so the
  number is the stitcher's own arbitration, not a replica);
* renders contact sheets of sampled refused pairs: the frame at the end of the first fragment beside the frame
  at the start of the second, cropped around the person, so a human can say "same player" or "two people".

Usage::

    .venv/bin/python scripts/probe_stitching.py --segment <segment dir> --out /tmp/stitch_probe
    .venv/bin/python scripts/probe_stitching.py --bundle /tmp/stitch_probe/bundle.npz --samples 0
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

from soccer_analytics.analysis import stage_b  # noqa: E402
from soccer_analytics.analysis.projection import on_pitch_mask, project_segment  # noqa: E402
from soccer_analytics.analysis.stage_a import load_segment  # noqa: E402
from soccer_analytics.geometry.pitch_calibration import PitchCalibration  # noqa: E402
from soccer_analytics.ingest.source import grab_frame  # noqa: E402

DEFAULT_SEGMENT = (
    "/srv/storage/home_video/Xbot/2026-10-03/analysis/2026-10-03_game_16-28-37-784/"
    "segments/2026-10-03_game_16-28-37-784__whole_game_541_4851"
)
LOOSE_GAP_S = 30.0  # scan joins this far apart in time...
LOOSE_ENVELOPE = 2.0  # ...when the distance is within twice the stitcher's own sprint envelope
COMPARE_WIDTH = 1280  # frame width the sample crops are read at
CROP_SCALE = 2.5  # crop size around the box, in box heights

# Reason codes for a refused pair (mirrors ``_stitch_tracks``; keep in sync when that changes).
PASS, GAP, KIT, DISTANCE, DISTANCE_NO_KIT = 0, 1, 2, 3, 4
REASON_NAMES = {PASS: "pass", GAP: "gap-over-limit", KIT: "kit-mismatch", DISTANCE: "distance", DISTANCE_NO_KIT: "distance-no-kit"}


def _fragments(track_id: np.ndarray) -> dict[int, np.ndarray]:
    """The observation rows of every pre-stitch fragment, keyed by track id (ids are dense 0..T-1).

    Rows the online pass did not keep (track id -1) are not fragments: the real stitcher never sees them.
    """
    rows = np.flatnonzero(track_id >= 0)
    order = rows[np.argsort(track_id[rows], kind="stable")]
    boundaries = np.flatnonzero(np.diff(track_id[order])) + 1
    groups = np.split(order, boundaries)
    return {int(track_id[g[0]]): np.sort(g) for g in groups if len(g)}


def _bundle_ns(bundle: dict) -> object:
    """A tiny stand-in for ``PitchDetections`` carrying the arrays the stitcher's own helpers read."""
    from types import SimpleNamespace

    return SimpleNamespace(
        time=bundle["time"],
        frame=bundle["frame"],
        xy=bundle["xy"],
        sigma_m=bundle["sigma_m"],
        height_px=bundle["height_px"],
        kit=bundle["kit"],
    )


def _scan(bundle: dict) -> dict:
    """The full end->start pair scan: gate verdicts, loose-window stats, mutual-best contests."""
    ns = _bundle_ns(bundle)
    rate = stage_b.detection_rate(ns)
    max_gap = max(1, int(round(stage_b.STITCH_MAX_GAP_S * rate)))
    max_gap_loose = max(1, int(round(LOOSE_GAP_S * rate)))
    sprint = stage_b.STITCH_SPRINT_M_S / rate

    frags = _fragments(bundle["track_id"])
    frag_ids = np.array(sorted(frags), dtype=np.int64)
    first_frame = np.array([int(bundle["frame"][frags[t][0]]) for t in frag_ids])
    last_frame = np.array([int(bundle["frame"][frags[t][-1]]) for t in frag_ids])
    first_xy = np.array([bundle["xy"][frags[t][0]] for t in frag_ids])
    last_xy = np.array([bundle["xy"][frags[t][-1]] for t in frag_ids])
    kit_descriptor = {t: stage_b._track_kit_descriptor(ns, frags[t]) for t in frag_ids}
    kit_known = np.array([kit_descriptor[t] is not None for t in frag_ids])
    kit_value = np.array(
        [k if k is not None else np.full(5, np.nan) for k in (kit_descriptor[t] for t in frag_ids)], dtype=np.float64
    )

    start_order = np.argsort(first_frame, kind="stable")
    starts_sorted = first_frame[start_order]

    n = len(frag_ids)
    cost: dict[tuple[int, int], float] = {}
    records: list[tuple[int, int, int, float, int]] = []  # before_idx, after_idx, gap_frames, distance, reason
    gap_reachable = 0  # refused only by the gap limit, but inside a doubled sprint envelope
    for bi in range(n):
        end = last_frame[bi]
        lo = np.searchsorted(starts_sorted, end + 1, side="left")
        hi = np.searchsorted(starts_sorted, end + max_gap_loose, side="right")
        if lo >= hi:
            continue
        cand = start_order[lo:hi]
        gap_frames = first_frame[cand] - end
        distance = np.linalg.norm(first_xy[cand] - last_xy[bi], axis=1)
        envelope = stage_b.STITCH_BASE_M + sprint * gap_frames
        in_loose = distance <= LOOSE_ENVELOPE * envelope
        for ci in np.flatnonzero(in_loose):
            ai = int(cand[ci])
            gf, dist = int(gap_frames[ci]), float(distance[ci])
            env = float(envelope[ci])
            both_kit = bool(kit_known[bi] and kit_known[ai])
            if gf > max_gap:
                reason = GAP
                if dist <= env:
                    gap_reachable += 1
            elif both_kit:
                kd = float(np.linalg.norm(kit_value[bi] - kit_value[ai]))
                if kd > stage_b.STITCH_MAX_KIT_DISTANCE:
                    reason = KIT
                elif dist > env:
                    reason = DISTANCE
                else:
                    reason = PASS
                    cost[(int(frag_ids[bi]), int(frag_ids[ai]))] = dist / gf + 2.0 * kd
            else:
                reason = DISTANCE_NO_KIT if dist > stage_b.STITCH_BASE_M + stage_b.STITCH_NO_KIT_FACTOR * sprint * gf else PASS
                if reason == PASS:
                    cost[(int(frag_ids[bi]), int(frag_ids[ai]))] = dist / gf + 0.5
            records.append((int(frag_ids[bi]), int(frag_ids[ai]), gf, dist, reason))

    # The stitcher's own arbitration (imported, not replicated): which gate-passing pairs become joins.
    successor = stage_b._join_fragments(cost, sorted(frags))
    has_predecessor = set(successor.values())
    chains = sum(1 for head in frags if head not in has_predecessor)

    return {
        "rate": rate,
        "fragments": n,
        "passed_pairs": len(cost),
        "joins": len(successor),
        "chains_after_stitch": chains,
        "gap_reachable_refusals": gap_reachable,
        "records": records,
    }


def _summarize(stats: dict) -> str:
    rec = np.array(stats["records"], dtype=np.float64) if stats["records"] else np.zeros((0, 5))
    reasons = rec[:, 4].astype(int) if len(rec) else np.zeros(0, dtype=int)
    lines = [
        f"fragments (pre-stitch):   {stats['fragments']}",
        f"gate-passing pairs:       {stats['passed_pairs']}",
        f"joins (global matching):  {stats['joins']}",
        f"chains after stitching:   {stats['chains_after_stitch']}  "
        f"(fragments merged away: {stats['fragments'] - stats['chains_after_stitch']})",
        "",
        "refused pairs inside the loose window, by reason:",
    ]
    for code in (GAP, KIT, DISTANCE, DISTANCE_NO_KIT):
        subset = rec[reasons == code] if len(rec) else rec
        extra = ""
        if code == GAP and len(subset):
            reach = subset[subset[:, 3] <= stage_b.STITCH_BASE_M + stage_b.STITCH_SPRINT_M_S / stats["rate"] * subset[:, 2]]
            extra = f", {len(reach)} of them inside a plain sprint envelope (the gap is the only refusal)"
        lines.append(f"  {REASON_NAMES[code]:>16s}: {len(subset)}{extra}")
    if len(rec):
        gap_rec = rec[reasons == GAP]
        if len(gap_rec):
            gaps = gap_rec[:, 2] / stats["rate"]
            buckets = [(6, 8), (8, 10), (10, 15), (15, 20), (20, 31)]
            row = " ".join(f"{a}-{b}s:{int(((gaps >= a) & (gaps < b)).sum())}" for a, b in buckets)
            lines.append(f"  gap-rejection time spread (s): {row}")
    return "\n".join(lines)


def _samples_from(records: list, rate: float) -> dict[str, list[tuple]]:
    """Pick a few pairs worth looking at per refusal category: the most plausible ones first."""
    rec = np.array(records, dtype=np.float64) if records else np.zeros((0, 5))
    pick: dict[str, list[tuple]] = {"gap-reachable": [], "no-kit": [], "kit-mismatch": []}
    if not len(rec):
        return pick
    gap = rec[rec[:, 4] == GAP]
    if len(gap):
        # Only pairs a sprint could physically reach within the gap; closest first.
        envelope = stage_b.STITCH_BASE_M + stage_b.STITCH_SPRINT_M_S / rate * gap[:, 2]
        reach = gap[gap[:, 3] <= envelope]
        ratio = reach[:, 3] / envelope[gap[:, 3] <= envelope] if len(reach) else np.zeros(0)
        for row in reach[np.argsort(ratio)[:6]]:
            pick["gap-reachable"].append(tuple(int(v) for v in row[:2]) + (int(row[2]), float(row[3]), "gap"))
    nokit = rec[rec[:, 4] == DISTANCE_NO_KIT]
    if len(nokit):
        envelope = stage_b.STITCH_BASE_M + stage_b.STITCH_NO_KIT_FACTOR * stage_b.STITCH_SPRINT_M_S / rate * nokit[:, 2]
        ratio = nokit[:, 3] / envelope
        for row in nokit[np.argsort(ratio)[:6]]:
            pick["no-kit"].append(tuple(int(v) for v in row[:2]) + (int(row[2]), float(row[3]), "no-kit"))
    return pick


def _pair_sheet(bundle: dict, meta: dict, pairs: list[tuple], out_dir: Path, name: str, rate: float) -> list[Path]:
    """One row per sampled pair: end-of-first crop beside start-of-second crop, labeled."""
    source = str(meta.get("video") or "")
    if not source:
        return []
    frags = _fragments(bundle["track_id"])
    written: list[Path] = []
    rows: list[np.ndarray] = []
    for before, after, gap, dist, tag in pairs[:6]:
        row = []
        for track, row_index, when in ((before, -1, "end"), (after, 0, "start")):
            rows_of = frags.get(track)
            if rows_of is None:
                row.append(np.zeros((180, 260, 3), dtype=np.uint8))
                continue
            r = rows_of[row_index]
            frame = grab_frame(source, float(bundle["time"][r]), width=COMPARE_WIDTH)
            if frame is None:
                row.append(np.zeros((180, 260, 3), dtype=np.uint8))
                continue
            h, w = frame.shape[:2]
            x1, y1, x2, y2 = (float(v) * w for v in bundle["box"][r])
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            half = max(40.0, CROP_SCALE * (y2 - y1) / 2)
            l, rr = int(max(0, cx - half)), int(min(w, cx + half))
            t, bb = int(max(0, cy - half)), int(min(h, cy + half))
            crop = frame[t:bb, l:rr]
            crop = cv2.resize(crop, (260, 180)) if crop.size else np.zeros((180, 260, 3), dtype=np.uint8)
            cv2.putText(crop, f"{when} t{track}", (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            row.append(crop)
        strip = np.hstack(row)
        bar = np.zeros((30, strip.shape[1], 3), dtype=np.uint8)
        cv2.putText(
            bar, f"{tag}: t{before} -> t{after}  gap {gap / rate:.1f}s  dist {dist:.1f} m",
            (4, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1,
        )
        rows.append(np.vstack([strip, bar]))
    if not rows:
        return []
    width = max(r.shape[1] for r in rows)
    rows = [np.pad(r, ((0, 0), (0, width - r.shape[1]), (0, 0))) for r in rows]
    sheet = np.vstack(rows)
    path = out_dir / f"{name}.jpg"
    cv2.imwrite(str(path), sheet, [int(cv2.IMWRITE_JPEG_QUALITY), 88])
    written.append(path)
    for i, r in enumerate(rows):
        p = out_dir / f"{name}_{i}.jpg"
        cv2.imwrite(str(p), r, [int(cv2.IMWRITE_JPEG_QUALITY), 88])
        written.append(p)
    return written


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--segment", default="", help="segment directory (default: the 2026-10-03 native whole-game run)")
    parser.add_argument("--bundle", default="", help="a saved fragment bundle to re-scan instead of running Stage B")
    parser.add_argument("--out", default="/tmp/stitch_probe", help="where the bundle, report and sheets are written")
    parser.add_argument("--samples", type=int, default=6, help="pairs to render per refusal category (0 = none)")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    meta_path = out_dir / "meta.json"

    if args.bundle:
        data = np.load(args.bundle)
        bundle = {key: data[key] for key in data.files}
        meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    else:
        segment_dir = Path(args.segment or DEFAULT_SEGMENT)
        started = time.time()
        segment = load_segment(segment_dir)
        print(f"loaded segment: {len(segment.time)} frames in {time.time() - started:.0f}s", flush=True)
        cal_path = segment_dir.parent.parent / "calibration.json"
        calibration = PitchCalibration.from_json(json.loads(cal_path.read_text()))
        detections = project_segment(segment, calibration)
        keep = on_pitch_mask(detections, 100.0, 64.0)
        started = time.time()
        assignment = stage_b._track_people(detections, keep)
        print(f"online pass: {len(assignment.tracks)} fragments in {time.time() - started:.0f}s", flush=True)
        bundle = {
            "frame": detections.frame,
            "time": detections.time,
            "xy": detections.xy,
            "sigma_m": detections.sigma_m,
            "height_px": detections.height_px,
            "box": detections.box,
            "kit": detections.kit,
            "det_track": detections.det_track,
            "track_id": assignment.track_id,
        }
        meta = {
            "video": str(segment.meta.get("video") or ""),
            "fps": float(segment.meta.get("fps") or 15.0),
            "start_s": float(segment.meta.get("start_s") or 0.0),
            "segment": str(segment_dir),
            "detections": int(len(detections.frame)),
        }
        np.savez_compressed(out_dir / "bundle.npz", **bundle)
        meta_path.write_text(json.dumps(meta, indent=2))
        print(f"bundle saved to {out_dir / 'bundle.npz'}", flush=True)

    started = time.time()
    stats = _scan(bundle)
    summary = _summarize(stats)
    print(f"scan in {time.time() - started:.0f}s", flush=True)
    print(summary, flush=True)
    (out_dir / "report.txt").write_text(summary + "\n")
    np.savez_compressed(out_dir / "records.npz", records=np.array(stats["records"], dtype=np.float64))

    if args.samples:
        rate = stats["rate"]
        picks = _samples_from(stats["records"], rate)
        for name, pairs in picks.items():
            pairs = pairs[: args.samples]
            if not pairs:
                continue
            written = _pair_sheet(bundle, meta, pairs, out_dir, name, rate)
            print(f"rendered {len(written)} image(s) for {name}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
