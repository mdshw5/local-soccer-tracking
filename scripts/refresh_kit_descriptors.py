"""Recompute the stored per-detection kit descriptors of an analysed segment, without re-analysing it.

Why this exists: the kit descriptor (`analysis.kit.kit_descriptor`) is a pure function of one decoded frame and one
stored detection box - the grass mask, the torso crop and the Lab/HSV summary. Detection and camera motion are
*not* inputs to it. So when the masking changes, the fix does not require re-running the expensive Stage A pass
(a GPU detection sweep over every frame); it requires re-reading the frames the boxes were found in. That is the
difference between an hour and a few minutes, and it is why this is a separate script rather than a Stage A flag.

What it does, per chunk: decode that chunk's frames, recompute each box's descriptor with the current code, and
rewrite the chunk with only ``det_kit`` replaced. Everything else in the chunk - poses, steps, focals, boxes,
confidences - is copied through untouched, and the write is atomic (temp file + rename), so an interrupted run
leaves the segment exactly as it was and a re-run continues from the first chunk not yet refreshed.

Resuming is implicit and recorded in ``kit_refresh.json``: a chunk is only marked done after its file is replaced.
Progress only counts against the *same descriptor code* that produced it (a hash of the descriptor and the
grass-window estimator rides in the status): when the masking changes, a re-run re-reads every chunk instead of
trusting descriptors that the current code would never write.

Usage::

    python scripts/refresh_kit_descriptors.py --segment data/segments/<name> [--limit-chunks 5]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from soccer_analytics.analysis.kit import kit_descriptor  # noqa: E402
from soccer_analytics.analysis.stage_a import chunk_path, completed_chunks, load_segment  # noqa: E402
from soccer_analytics.ingest.ffmpeg_reader import FFmpegFrameReader  # noqa: E402

STATUS_FILE = "kit_refresh.json"


def _status(path: Path) -> dict:
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return {}


def _publish(path: Path, **changes) -> None:
    # `error` is cleared rather than merged: a status file that still reads "state: done, error: ..." is a lie a
    # reader has to know how to ignore, and the failure that caused it has been dealt with by the re-run.
    payload = {**_status(path), **changes, "updated": time.time()}
    payload.pop("error", None)
    if changes.get("state") == "error":
        payload["error"] = changes["error"]
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload))
    os.replace(tmp, path)


def _code_fingerprint() -> str:
    """A hash of the code that computes a descriptor: the descriptor itself and the grass-window estimator it
    masks with.

    Saved beside the progress so that chunks refreshed by an older masking are re-read instead of trusted.
    Measured on the real 2026-10-03 game: the masking changed after a refresh had completed, every stored
    descriptor on disk was stale, and the status still said "done" - so a plain re-run skipped the whole segment
    and the pipeline quietly kept reading the old descriptors. The fingerprint makes that failure impossible to
    repeat: progress only counts against the exact code that produced it.
    """
    from soccer_analytics.analysis import kit as kit_module
    from soccer_analytics.tracking import team_classifier

    digest = hashlib.sha256()
    for module in (kit_module, team_classifier):
        digest.update(Path(module.__file__).read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()[:16]


def _done_chunks(status_path: Path, total_chunks: int, fingerprint: str) -> int:
    """How many chunks an interrupted run had already rewritten, so a re-run resumes instead of repeating.

    Only trusted for a run that got far enough to be trustworthy: a status still saying "running" is assumed
    complete up to its own count (the alternative - re-reading them - is merely slower, never wrong), while a run
    that ended in "error" is trusted only up to the count it had reached when it failed. A status written by
    different descriptor code is trusted for nothing: its chunks were refreshed for a masking that no longer
    exists, and re-reading them is the only way the segment describes the pipeline that is about to read it.
    """
    status = _status(status_path)
    if not status or status.get("kit_code_hash") != fingerprint:
        return 0
    return max(0, min(int(status.get("chunks_done") or 0), total_chunks))


def refresh(segment_dir: Path, limit_chunks: int = 0) -> dict:
    """Rewrite every chunk's ``det_kit`` from the footage; returns the final status payload."""
    segment_dir = Path(segment_dir)
    status_path = segment_dir / STATUS_FILE
    segment = load_segment(segment_dir)
    video = Path(json.loads((segment_dir / "meta.json").read_text())["video"])
    meta = json.loads((segment_dir / "meta.json").read_text())
    width, fps = int(meta["width"]), float(segment.meta["fps"])
    total_chunks = completed_chunks(segment_dir)
    if total_chunks == 0:
        raise SystemExit(f"{segment_dir} has no complete chunks to refresh")
    todo = list(range(total_chunks))
    if limit_chunks:
        todo = todo[:limit_chunks]

    # `frame_grass_window` is imported here rather than at module scope so the import cost sits after the
    # argument parsing: a bad --segment should fail before OpenCV is loaded, not after.
    from soccer_analytics.analysis.kit import frame_grass_window

    done = 0
    changed = 0
    # Chunks a previous run already refreshed are counted, not re-read: re-decoding them would change nothing
    # (the pass is idempotent) and would cost the same minutes again. The count of *what changed* is reset,
    # because it describes this run's edits, not the segment's history. Progress only counts against the code
    # that produced it (see `_code_fingerprint`); a status from a different masking restarts at chunk 0.
    code_hash = _code_fingerprint()
    already = _done_chunks(status_path, total_chunks, code_hash)
    started = time.monotonic()
    # One sequential pass over the footage: the chunks are contiguous frame ranges, so the reader is positioned
    # once per chunk (at that chunk's first frame) and read forward, rather than seeking per frame.
    for index in todo:
        path = chunk_path(segment_dir, index)
        if index < already:
            # Already rewritten by a previous run: counting it keeps the progress honest without re-reading it.
            done += 1
            continue
        with np.load(path) as data:
            rows = {key: data[key] for key in data.files}
        frames = rows["det_frame"]
        times = rows["time"]
        if len(times) == 0:
            _publish(status_path, state="running", chunks_done=index + 1, total_chunks=total_chunks,
                     descriptors_changed=changed, progress=(index + 1) / total_chunks, kit_code_hash=code_hash)
            done += 1
            continue
        chunk_start = float(times[0])
        # `times[]` holds ABSOLUTE source times (the analysis window's start_s is already baked in), so this is
        # where the reader seeks - adding start_s again would ask for a time past the end of the video, which is
        # how the first run of this script "decoded 251 of 300 frames" at chunk 62.
        # The request is padded by a frame and a margin: `-t` cuts on an exact boundary and the chunk's last frame
        # sits on it, so an unpadded ask loses that frame. Frames past the chunk are ignored below, so over-asking
        # is harmless - and necessary, because the final chunk of a segment is legitimately short (the window ends
        # mid-chunk: the whole game ends at 4851.0s with 252 frames in the last chunk, not 300).
        duration = float(times[-1]) - chunk_start + 1.0 / fps + 1.0
        reader = FFmpegFrameReader(video, fps=fps, width=width, start_s=chunk_start, duration_s=duration)
        kits = np.zeros_like(rows["det_kit"])
        seen = -1
        for local, (_time, frame) in enumerate(reader.frames()):
            if local >= len(times):
                break
            seen = local
            grass = frame_grass_window(frame)
            rows_of_frame = np.where(frames == local)[0]
            for row in rows_of_frame:
                x1, y1, x2, y2 = (float(v) * width for v in rows["det_box"][row])
                kits[row] = kit_descriptor(frame, (x1, y1, x2, y2), grass)
        if seen < len(times) - 1:
            # A plain exception, not SystemExit: `main` turns this into `state="error"` in the status file, which is
            # the only thing the page reads. Raising SystemExit here would slip past that handler and leave the
            # status claiming "running" for a run that is over - the exact stale-status trap this script exists
            # alongside. Nothing is written, so the chunks already refreshed stand and a re-run continues.
            raise RuntimeError(
                f"chunk {index}: decoded {seen + 1} of {len(times)} frames from "
                f"{video.name} at t={chunk_start:.1f}s - refusing to write a partial chunk"
            )
        if len(kits):
            changed += int(np.count_nonzero(np.abs(kits - rows["det_kit"]).max(axis=1) > 1e-4))
        rows["det_kit"] = kits
        tmp = path.with_name(path.name + ".tmp.npz")
        np.savez_compressed(tmp, **rows)
        os.replace(tmp, path)

        done += 1
        elapsed = max(time.monotonic() - started, 1e-6)
        _publish(
            status_path,
            state="running",
            chunks_done=index + 1,
            total_chunks=total_chunks,
            descriptors_changed=changed,
            progress=(index + 1) / total_chunks,
            chunks_per_min=60.0 * done / elapsed,
            kit_code_hash=code_hash,
        )
        print(
            f"chunk {index + 1}/{total_chunks} refreshed ({len(rows['det_frame'])} detections, "
            f"{changed} descriptor(s) differ so far, {60.0 * done / elapsed:.1f} chunks/min)",
            flush=True,
        )
    state = "done" if done >= total_chunks else "partial"
    final = {
        "state": state,
        "chunks_done": done,
        "total_chunks": total_chunks,
        "descriptors_changed": changed,
        "kit_code_hash": code_hash,
        "updated": time.time(),
    }
    _publish(status_path, **final)
    return final


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--segment", required=True, help="segment directory (with meta.json and chunk files)")
    parser.add_argument("--limit-chunks", type=int, default=0, help="refresh only the first N chunks (a smoke test)")
    args = parser.parse_args()
    segment_dir = Path(args.segment)
    if not segment_dir.is_absolute():
        segment_dir = REPO_ROOT / segment_dir
    if not segment_dir.is_dir():
        print(f"error: {segment_dir} is not a directory", file=sys.stderr)
        return 1
    try:
        final = refresh(segment_dir, limit_chunks=args.limit_chunks)
    except Exception as exc:  # noqa: BLE001 - a background job must leave the reason where the page can show it
        _publish(segment_dir / STATUS_FILE, state="error", error=f"{type(exc).__name__}: {exc}")
        print(f"error: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print(
        f"chunks: {final['chunks_done']}/{final['total_chunks']}  "
        f"descriptors changed: {final['descriptors_changed']}  state: {final['state']}"
    )
    return 0 if final["state"] == "done" else 1


if __name__ == "__main__":
    raise SystemExit(main())