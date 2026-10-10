"""Prepare one game from the camera's clips, as its own process so the dashboard stays responsive.

Usage::

    python scripts/run_build_game.py --clip 16-28.MP4 --clip 16-58.MP4 --out <footage>/analysis/<id>

No video is produced: the clips are recorded in the game manifest with their playing order and their offsets on
the game clock, and every reader - Stage A, the marking stream, frame grabs - resolves a game-clock time to the
right clip itself (see ``ingest.source``). This replaced the old stream-copy combination, which wrote a duplicate
as large as the clips all over again for no reader to need. A single ``--clip`` is also accepted: the file is
already the whole game and the manifest simply says so.

``--out`` must be the game's own analysis directory (the directory the clips imply, beside the footage); the
script checks it, so the page and a hand-run always write to the same place. The state is written to ``build.json``
in it, which is what the page polls - the work is a manifest write, so it finishes immediately.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from soccer_analytics.analysis import game


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare camera clips as one game (manifest only; nothing merged).")
    parser.add_argument("--clip", action="append", required=True, help="source clip (repeat once per clip)")
    parser.add_argument(
        "--out",
        required=True,
        help="the game's analysis directory (<footage>/analysis/<id>); derived from the clips and checked",
    )
    args = parser.parse_args()

    directory = Path(args.out)
    game.write_build_state(
        directory, state="running", pid=os.getpid(), started=time.time(), stage="manifest", error=None
    )
    try:
        _ordered, _output, expected = game.locations(args.clip)
        if expected != directory:
            raise RuntimeError(
                f"{directory} does not match these clips: the game's manifest belongs in {expected} "
                "(the game's own analysis directory, beside the footage)"
            )

        record, written = game.prepare(args.clip)
        game.write_build_state(
            directory,
            state="done",
            stage="done",
            clips=len(record.clips),
            duration_s=round(record.duration_s, 1),
            finished=time.time(),
            error=None,
        )
        print(
            f"game ready: {written} ({len(record.clips)} clip(s), {record.duration_s / 60:.0f} min; "
            "the clips are read directly, no merged video is written)",
            flush=True,
        )
        return 0
    except Exception as exc:  # leave a readable state for the dashboard instead of a silently dead job
        game.write_build_state(directory, state="error", error=f"{type(exc).__name__}: {exc}")
        raise


if __name__ == "__main__":
    raise SystemExit(main())
