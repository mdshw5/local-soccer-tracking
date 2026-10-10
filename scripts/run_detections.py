"""Run every remaining detection pass for one match in the background, then rebuild the replay.

This is what the dashboard's "Build report + run all detections" button starts once the report itself is built:

1. the ball scan (resumes from its checkpoint; a completed one is skipped),
2. the whistle scan (skipped when an earlier scan already finished on this video),
3. the shirt-number scan (skipped when an earlier scan already finished),
4. the event detectors over the ball and player tracks (through ``analysis.detect_run``),
5. and a report + replay rebuild so the saved artifacts carry everything the scans found.

Each stage checks the previous run's own status first, so re-running after a failure never redoes finished work -
which matters because the ball scan is around an hour on a full game even though it checkpoints. Progress (and any
error) is written to ``detections.json`` beside the match; that file is what the page polls, so a stopped run is
visible instead of a process that vanished. Nothing here reads the dashboard's session: the settings the button
was pressed with travel in the command line.

Usage::

    python scripts/run_detections.py --match 2026-10-04_16-58-38-391 [--video <recording>] [--segment <dir>]
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from soccer_analytics.analysis import detect_run
from soccer_analytics.analysis import game as game_lib
from soccer_analytics.analysis.events import MIN_PROMINENCE
from soccer_analytics.analysis.library import MatchLibrary

REPO_ROOT = Path(__file__).resolve().parents[1]
STATUS_FILE = "detections.json"
BALL_STATUS_FILE = "ball_scan.json"
BALL_TRACK_FILE = "ball_track.json"
AUDIO_STATUS_FILE = "audio_scan.json"
JERSEY_STATUS_FILE = "jerseys_status.json"
POLL_S = 2.0

# Where each stage sits on the single progress bar. The ball scan is the long pole (up to an hour on a full game)
# and the rest are minutes, so the ball owns the widest slice.
BALL_SPAN = (0.0, 0.62)
AUDIO_SPAN = (0.62, 0.75)
JERSEY_SPAN = (0.75, 0.85)
EVENTS_SPAN = (0.85, 0.95)
REBUILD_SPAN = (0.95, 1.0)


def _read_json(path: Path) -> dict:
    try:
        return json.loads(Path(path).read_text())
    except (OSError, json.JSONDecodeError):
        return {}


class Status:
    """The page's progress file; written atomically and throttled, so a reader never sees half a document."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.payload: dict = {
            "state": "running",
            "stage": "ball",
            "progress": 0.0,
            "message": "Starting...",
            "stages": {},
            "started": time.time(),
            "updated": time.time(),
        }
        self._written = 0.0
        self.update(force=True)

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

    def stage(self, name: str, state: str, message: str) -> None:
        """Record one stage's outcome: ``skipped``, ``done`` or ``failed`` - and say so on the bar."""
        self.payload.setdefault("stages", {})[name] = state
        self.update(force=True, stage=name, message=message)

    def fail(self, message: str) -> None:
        self.update(force=True, state="error", error=message, message=message)

    def finish(self, message: str) -> None:
        self.update(force=True, state="done", progress=1.0, stage="done", message=message)


def _run_mirrored(
    command: list[str],
    *,
    status: Status,
    stage: str,
    base: float,
    span: float,
    message: str,
    watch: Path | None = None,
    read_progress=None,
    read_message=None,
) -> int:
    """Run a child process, mirroring its own status file into ours until it exits; returns the exit code.

    ``watch`` is the child's progress file (each scan has its own format); while the child runs, its progress and
    message are mapped onto this run's single bar, so one bar moves through the whole pipeline. A child without a
    status file (the rebuild) just holds the bar at its stage's spot.
    """
    process = subprocess.Popen(command, cwd=str(REPO_ROOT), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    while process.poll() is None:
        time.sleep(POLL_S)
        inner = _read_json(watch) if watch is not None else {}
        fraction = read_progress(inner) if (read_progress is not None and inner) else None
        detail = read_message(inner) if (read_message is not None and inner) else message
        if fraction is None:
            status.update(stage=stage, message=detail)
        else:
            status.update(stage=stage, progress=base + span * min(1.0, max(0.0, fraction)), message=detail)
    return int(process.returncode or 0)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--match", required=True, help="match id (the name of its analysis directory)")
    parser.add_argument("--video", default="", help="the recording the scans read (default: the analyzed segment's)")
    parser.add_argument("--segment", default="", help="the analyzed segment (default: the match's first one)")
    parser.add_argument("--strictness", type=float, default=MIN_PROMINENCE)
    parser.add_argument("--keep-voices", action="store_true", help="same as run_audio_scan.py's flag")
    args = parser.parse_args()

    library = MatchLibrary()
    try:
        record = library.load(args.match)
    except (OSError, KeyError, ValueError) as exc:
        print(f"cannot load match {args.match}: {exc}", file=sys.stderr)
        return 1
    segment_dir = Path(args.segment) if args.segment else (
        Path(record.segments[0]) if record.segments else None
    )
    status = Status(library.path(args.match) / STATUS_FILE)
    if segment_dir is None or not segment_dir.exists():
        status.fail("No analyzed segment for this match - run Step 1 (the heavy pass) first.")
        return 1
    segment_meta = _read_json(segment_dir / "meta.json")
    video = args.video or str(segment_meta.get("video") or "")
    if not video:
        video = str(record.sources[0]) if record.sources else ""
    if not video:
        status.fail("No video is recorded for this match, so the scans have nothing to read.")
        return 1

    # --- 1. the ball scan (the long one; resumes from its checkpoint) ---------------------------------------
    if _read_json(segment_dir / BALL_TRACK_FILE).get("complete"):
        status.stage("ball", "skipped", "The ball scan is already complete.")
    else:
        status.stage("ball", "running", "Running the ball scan (long: it checkpoints as it goes)...")
        code = _run_mirrored(
            [
                sys.executable,
                str(REPO_ROOT / "scripts" / "run_ball_scan.py"),
                "--segment",
                str(segment_dir),
                # The same recording the other stages read. The scan itself still prefers the segment's own source
                # when this machine has it (that is the clock its offsets were recorded against); this is the
                # fallback for a machine that does not, where the stored absolute path no longer resolves.
                "--video",
                video,
            ],
            status=status,
            stage="ball",
            base=BALL_SPAN[0],
            span=BALL_SPAN[1] - BALL_SPAN[0],
            message="Running the ball scan...",
            watch=segment_dir / BALL_STATUS_FILE,
            read_progress=lambda inner: inner.get("progress"),
            read_message=lambda inner: f"Ball scan: {inner.get('message') or 'running'}",
        )
        if code != 0 or _read_json(segment_dir / BALL_STATUS_FILE).get("state") != "done":
            status.fail(
                "The ball scan failed - see its status beside the segment. Press the button again to resume "
                "from its checkpoint."
            )
            return 1
        status.stage("ball", "done", "Ball scan complete.")

    # --- 2. the whistle scan ---------------------------------------------------------------------------------
    audio = library.load_audio_scan_status(args.match)
    same_video = str(audio.get("video") or video) == str(video)
    if audio.get("state") == "done" and same_video:
        status.stage("audio", "skipped", "The whistle scan already has a result for this video; skipped.")
    else:
        status.stage("audio", "running", "Running the whistle scan...")
        command = [
            sys.executable,
            str(REPO_ROOT / "scripts" / "run_audio_scan.py"),
            "--match", args.match,
            "--video", video,
            "--strictness", f"{float(args.strictness):.1f}",
        ]
        if args.keep_voices:
            command.append("--keep-voices")
        code = _run_mirrored(
            command,
            status=status,
            stage="audio",
            base=AUDIO_SPAN[0],
            span=AUDIO_SPAN[1] - AUDIO_SPAN[0],
            message="Running the whistle scan...",
            watch=library.path(args.match) / AUDIO_STATUS_FILE,
            read_progress=lambda inner: inner.get("progress"),
            read_message=lambda inner: f"Whistle scan: {inner.get('message') or 'running'}",
        )
        if code != 0 or library.load_audio_scan_status(args.match).get("state") != "done":
            status.stage("audio", "failed", "The whistle scan failed; continuing without it.")
        else:
            status.stage("audio", "done", "Whistle scan complete.")

    # --- 3. the shirt-number scan ----------------------------------------------------------------------------
    jersey = library.load_jerseys_status(args.match)
    if jersey.get("state") == "done":
        status.stage("jerseys", "skipped", "The shirt-number scan already has a result; skipped.")
    else:
        status.stage("jerseys", "running", "Reading shirt numbers from the footage...")
        code = _run_mirrored(
            [
                sys.executable,
                str(REPO_ROOT / "scripts" / "extract_jerseys.py"),
                "--match", args.match,
                "--video", video,
                "--segment", str(segment_dir),
            ],
            status=status,
            stage="jerseys",
            base=JERSEY_SPAN[0],
            span=JERSEY_SPAN[1] - JERSEY_SPAN[0],
            message="Reading shirt numbers...",
            watch=library.path(args.match) / JERSEY_STATUS_FILE,
            read_progress=lambda inner: (
                float(inner.get("crops_done", 0)) / max(1.0, float(inner.get("crops_total", 0)))
                if inner.get("crops_total")
                else None
            ),
            read_message=lambda inner: (
                f"Shirt numbers: {inner.get('crops_done', 0)}/{inner.get('crops_total', 0)} crops, "
                f"{inner.get('readings', 0)} readings"
            ),
        )
        if code != 0 or library.load_jerseys_status(args.match).get("state") != "done":
            status.stage("jerseys", "failed", "The shirt-number scan failed; continuing without it.")
        else:
            status.stage("jerseys", "done", "Shirt-number scan complete.")

    # --- 4. the event detectors ------------------------------------------------------------------------------
    status.update(force=True, stage="events", progress=EVENTS_SPAN[0], message="Detecting events...")
    game_record = game_lib.find_for_video(video)
    half_bounds = game_record.bounds() if game_record is not None else None
    try:
        result = detect_run.run_detection(library, args.match, segment_dir, video=video, half_bounds=half_bounds)
    except detect_run.MissingInput as exc:
        status.fail(str(exc))
        return 1
    except Exception as exc:  # noqa: BLE001 - whatever went wrong is what the page should show
        status.fail(f"The event detectors failed: {type(exc).__name__}: {exc}")
        return 1
    status.update(force=True, progress=EVENTS_SPAN[1])
    status.stage(
        "events",
        "done",
        f"Events: {result['detected']} candidate(s), {result['added']} new, {result['dropped']} dropped.",
    )

    # --- 5. rebuild the report + replay so the artifacts carry everything ------------------------------------
    status.update(force=True, stage="rebuild", message="Rebuilding the report and the replay...")
    code = _run_mirrored(
        [sys.executable, str(REPO_ROOT / "scripts" / "rebuild_match.py"), "--match", args.match, "--segment", str(segment_dir)],
        status=status,
        stage="rebuild",
        base=REBUILD_SPAN[0],
        span=REBUILD_SPAN[1] - REBUILD_SPAN[0],
        message="Rebuilding the report and the replay...",
    )
    if code != 0:
        status.fail("The final report + replay rebuild failed - see the terminal output of this run.")
        return 1
    status.stage("rebuild", "done", "Report and replay rebuilt with the new ball track and shirt numbers.")

    summary = ", ".join(f"{name} {state}" for name, state in status.payload.get("stages", {}).items())
    status.finish(f"All detections complete ({summary}). The report and the replay were rebuilt.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
