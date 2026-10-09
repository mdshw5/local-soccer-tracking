"""Build one game video from the camera's clips, as its own process so the dashboard stays responsive.

Usage::

    python scripts/run_build_game.py --clip 16-28.MP4 --clip 16-58.MP4 --out <footage>/analysis/<id>

A single ``--clip`` is also accepted: a file that is already the whole game is used as it stands, with nothing
copied or re-encoded, and only the manifest is written.

Combines the clips with a stream copy (``-c copy`` - the clips come from one camera, so nothing is re-encoded)
and writes the game's manifest into the combined video's own analysis directory, beside the footage. ``--out``
must be that directory; the script derives it from the clips itself and refuses anything else, so the page and a
hand-run always write the same place. The state is written to ``build.json`` in it, which is what the page polls.
Combining three 30-minute 4K clips is minutes of work, so it belongs in the background.

The marking video is *not* built here any more: the stream server encodes it on demand from the combined file
(see the ``/game/`` route in ``dashboard/stream.py``), so there is no proxy build step and no separate build
state to go stale.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from soccer_analytics.analysis import game
from soccer_analytics.ingest.ffmpeg_reader import probe_video


def main() -> int:
    parser = argparse.ArgumentParser(description="Combine camera clips into one game video.")
    parser.add_argument("--clip", action="append", required=True, help="source clip (repeat once per clip)")
    parser.add_argument(
        "--out",
        required=True,
        help="the combined video's analysis directory (<footage>/analysis/<id>); derived from the clips and checked",
    )
    args = parser.parse_args()

    directory = Path(args.out)
    game.write_build_state(
        directory,
        state="running",
        pid=os.getpid(),
        started=time.time(),
        stage="combining" if len(args.clip) > 1 else "reading the single game video",
        error=None,
    )
    try:
        ordered, output, expected = game.locations(args.clip)
        if expected != directory:
            raise RuntimeError(
                f"{directory} does not match these clips: the game's manifest belongs in {expected} "
                "(the combined video's own analysis directory, beside the footage)"
            )

        planned = game.plan(ordered)
        if planned.problem:
            raise RuntimeError(planned.problem)
        if not output.exists():
            game.build_game(planned.clips, output)

        info = probe_video(output)
        record = game.GameRecord(
            game_id=directory.name, output=str(output.resolve()), duration_s=float(info.duration_s), clips=planned.clips
        )
        # Rebuilding the same clips must not throw the marks away: where half-time is does not change because the
        # proxy had to be rebuilt.
        manifest = directory / game.MANIFEST_FILE
        if manifest.exists():
            previous = game.GameRecord.load(directory)
            if [clip.path for clip in previous.clips] == [clip.path for clip in planned.clips]:
                record.start_s, record.half_s, record.end_s = previous.start_s, previous.half_s, previous.end_s
        record.save(directory)

        game.write_build_state(directory, stage="manifest", clips=len(planned.clips), duration_s=round(info.duration_s, 1))

        game.write_build_state(
            directory, state="done", stage="done", finished=time.time(), duration_s=round(info.duration_s, 1)
        )
        print(f"game ready: {output}", flush=True)
        return 0
    except Exception as exc:  # leave a readable state for the dashboard instead of a silently dead job
        game.write_build_state(directory, state="error", error=f"{type(exc).__name__}: {exc}")
        raise


if __name__ == "__main__":
    raise SystemExit(main())
