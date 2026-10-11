"""Scan a match recording's audio for whistle candidates, in the background, reporting progress as it goes.

The dashboard used to run this inline: clicking the button froze the whole page for as long as the decode and the
transform took, which on a full game is minutes of a spinner and nothing else. It runs as its own process now and
writes ``audio_scan.json`` beside the match - the same contract as the shirt-number scan - so the page only has to
poll a file and draw a bar.

The audio is cached as a wav inside the match directory (``<match>/audio/<key>.wav``, keyed by the recording), because
the decode is the part that has to touch the whole file; a repeat scan after changing the detector's strictness only
re-runs the transform. The extraction writes through a temporary name and is renamed when it lands, so a scan never
reads half a recording.

Usage::

    python scripts/run_audio_scan.py --match <match_id> --video <file> [--strictness 50] [--keep-voices]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from soccer_analytics.analysis.events import (
    LOW_GAIN_MAX,
    MIN_PROMINENCE,
    VOICE_SHARE_MAX,
    detect_whistles,
    whistles_to_events,
)
from soccer_analytics.analysis.library import MatchLibrary, audio_cache_path
from soccer_analytics.ingest.ffmpeg_reader import extract_audio, read_wav_mono

STATUS_FILE = "audio_scan.json"
# How the bar is split between the two phases. The decode reads the whole recording, but the transform is the part
# that scales with the *content*, and on the real footage the two come out within a factor of two of each other.
EXTRACT_SHARE = 0.5


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--match", required=True, help="match id in the archive")
    parser.add_argument("--video", required=True)
    parser.add_argument("--strictness", type=float, default=MIN_PROMINENCE)
    parser.add_argument(
        "--keep-voices",
        action="store_true",
        help="do not reject blasts that bring low frequencies with them (voices and bird calls)",
    )
    parser.add_argument("--wav", default="", help="where to cache the audio (default <match>/audio/<video key>.wav)")
    return parser.parse_args()


class Status:
    """Progress file the dashboard polls; written atomically and throttled, so a reader never sees half a document."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.payload: dict = {
            "state": "running",
            "stage": "extract",
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


def scan(
    library: MatchLibrary,
    match_id: str,
    video: str,
    *,
    strictness: float = MIN_PROMINENCE,
    reject_voices: bool = True,
    wav_path: str | Path | None = None,
    status: Status | None = None,
) -> dict:
    """Run one scan, writing progress (and in the end the result) to the status file.

    Errors land in the status rather than being raised: the point of the background process is that the page can
    say what went wrong without dying with it. Returns the final payload.
    """
    status = status or Status(library.path(match_id) / STATUS_FILE)
    try:
        # The cache lives inside the match directory (the audio_cache_path rule), so clearing or moving the
        # analysis carries or resets its decoded audio with everything else; the name stays keyed by what the
        # video is (a clip set's game id / a file's stem), not by its basename alone.
        wav = Path(wav_path) if wav_path else audio_cache_path(library.path(match_id), video)
        base, span = 0.0, 1.0
        if wav.exists() and wav.stat().st_size > 1024:
            status.update(force=True, stage="detect", progress=0.0, message="Using the cached audio")
        else:
            base, span = EXTRACT_SHARE, 1.0 - EXTRACT_SHARE
            status.update(force=True, stage="extract", progress=0.0, message="Extracting the audio track")

            def extraction(fraction: float) -> None:
                status.update(
                    stage="extract",
                    progress=base * min(1.0, fraction),
                    message=f"Extracting the audio track ({fraction * 100:.0f}%)",
                )

            part = wav.with_name(wav.stem + ".part" + wav.suffix)  # .wav stays last: ffmpeg picks the muxer from it
            extract_audio(video, part, on_progress=extraction)
            part.replace(wav)  # only a complete recording takes the cache's name

        samples, rate = read_wav_mono(wav)

        def transform(fraction: float) -> None:
            status.update(
                stage="detect",
                progress=base + span * min(1.0, fraction),
                message=f"Scanning the whistle band ({fraction * 100:.0f}%)",
            )

        status.update(force=True, stage="detect", progress=base, message="Scanning the whistle band (0%)")
        whistles = detect_whistles(
            samples,
            rate,
            min_prominence=float(strictness),
            max_voice_share=VOICE_SHARE_MAX if reject_voices else None,
            max_low_gain=LOW_GAIN_MAX if reject_voices else None,
            on_progress=transform,
        )

        log = library.events(match_id)
        # Reconciled rather than merely appended: a re-scan after the gates were refined has to be able to *remove*
        # the candidates the detector no longer reports, or the queue the refinement was for never gets shorter.
        # A candidate a human confirmed is kept whatever the detector thinks now.
        added, dropped = log.reconcile_detected(whistles_to_events(whistles, video=video))
        library.save_events(match_id, log)
        status.update(
            force=True,
            state="done",
            stage="done",
            progress=1.0,
            message="Done",
            video=str(video),
            minutes=samples.size / max(1, rate) / 60.0,
            found=len(whistles),
            added=added,
            dropped=dropped,
            strongest=max((w.prominence for w in whistles), default=0.0),
            strictness=float(strictness),
            reject_voices=bool(reject_voices),
        )
    except Exception as exc:  # noqa: BLE001 - whatever went wrong goes into the status for the page to show
        status.update(force=True, state="error", error=f"{type(exc).__name__}: {exc}")
    return status.payload


def main() -> int:
    args = parse_args()
    payload = scan(
        MatchLibrary(),
        args.match,
        args.video,
        strictness=args.strictness,
        reject_voices=not args.keep_voices,
        wav_path=args.wav or None,
    )
    return 0 if payload.get("state") == "done" else 1


if __name__ == "__main__":
    raise SystemExit(main())
