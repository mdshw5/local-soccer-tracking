"""Report how the whistle gates separate a match's own confirmed and rejected candidates.

The detector's thresholds are measured, not guessed, and the measurements come from the true/false verdicts a
reviewer leaves on the candidates (Step 4's buttons). As more candidates are reviewed the numbers move, and this
script re-derives them: it re-runs the detector with the gates *off*, matches every candidate to its verdict,
prints the feature table and then the gap between the two classes - which is where a threshold belongs.

Usage::

    python scripts/report_whistle_labels.py --match <match_id> [--video <file>] [--wav <file>]

``--video``/``--wav`` default to the video the candidates were scanned from (``Event.video``) and the matching
``data/cache/<stem>.wav``. A candidate with no verdict is skipped: unreviewed is not evidence.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from soccer_analytics.analysis.events import (  # noqa: E402
    LOW_GAIN_MAX,
    MIN_PROMINENCE,
    VOICE_SHARE_MAX,
    EventLog,
    detect_whistles,
)
from soccer_analytics.analysis.library import MatchLibrary  # noqa: E402
from soccer_analytics.ingest.ffmpeg_reader import read_wav_mono  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--match", required=True, help="match id in the archive")
    parser.add_argument("--video", default="", help="the recording the candidates were scanned from")
    parser.add_argument("--wav", default="", help="cached audio (default data/cache/<video stem>.wav)")
    parser.add_argument("--tolerance", type=float, default=0.3, help="seconds; how near a candidate must be")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    library = MatchLibrary()
    log = library.events(args.match)
    labeled = [e for e in log.events if e.verdict in ("true", "false")]
    if not labeled:
        print("No verdicts on this match yet - review some candidates first (Step 4, under the player).")
        return 0

    video = args.video or next((e.video for e in labeled if e.video), "")
    if not video:
        print("The labeled candidates do not record the video they came from; pass --video.")
        return 1
    wav = Path(args.wav) if args.wav else REPO_ROOT / "data" / "cache" / f"{Path(video).stem}.wav"
    if not wav.exists():
        print(f"No cached audio at {wav} - run the scan once, or pass --wav.")
        return 1

    samples, rate = read_wav_mono(wav)
    print(f"{len(labeled)} labeled candidate(s) on {Path(video).name}, audio {samples.size / rate / 60:.1f} min")
    # The gates off: every candidate the *older* stages report, each carrying its own measurements.
    candidates = detect_whistles(samples, rate, max_voice_share=None, max_low_gain=None)

    rows: list[tuple[float, str, object]] = []
    for event in labeled:
        near = [w for w in candidates if abs(w.time_s - event.time_s) < args.tolerance]
        if not near:
            print(f"  {event.time_s:8.2f} {event.verdict:>5}  no candidate within {args.tolerance}s - detector changed?")
            continue
        rows.append((event.time_s, event.verdict, min(near, key=lambda w: abs(w.time_s - event.time_s))))

    print(f"\n{'time':>8} {'label':>5} {'dur':>5} {'f0':>6} {'prom':>7} {'voice':>6} {'low_gain':>9}")
    for time_s, verdict, whistle in sorted(rows, key=lambda r: (r[1], r[2].low_gain)):
        print(
            f"{time_s:8.2f} {verdict:>5} {whistle.duration_s:5.2f} {whistle.frequency_hz:6.0f} "
            f"{whistle.prominence:7.0f} {whistle.voice_share:6.2f} {whistle.low_gain:9.2f}"
        )

    trues = [w for _t, v, w in rows if v == "true"]
    falses = [w for _t, v, w in rows if v == "false"]
    gates = (
        ("prominence", lambda w: w.prominence, MIN_PROMINENCE, False),
        ("voice_share", lambda w: w.voice_share, VOICE_SHARE_MAX, True),
        ("low_gain", lambda w: w.low_gain, LOW_GAIN_MAX, True),
    )
    print()
    for name, get, threshold, is_ceiling in gates:
        true_values = sorted(get(w) for w in trues)
        false_values = sorted(get(w) for w in falses)
        if not true_values or not false_values or threshold is None:
            continue
        rejects = (lambda v: v >= threshold) if is_ceiling else (lambda v: v < threshold)
        lost = sum(1 for value in true_values if rejects(value))
        rejected = sum(1 for value in false_values if rejects(value))
        verdict = f"keeps all {len(true_values)} confirmed" if not lost else f"WOULD DROP {lost} confirmed"
        print(
            f"{name:11} true {true_values[0]:7.2f}..{true_values[-1]:7.2f}   false {false_values[0]:7.2f}..{false_values[-1]:7.2f}"
            f"   at {threshold:6.2f} it rejects {rejected}/{len(false_values)} rejected candidates - {verdict}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
