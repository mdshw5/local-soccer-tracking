"""Build one game video from the camera's clips, as its own process so the dashboard stays responsive.

Usage::

    python scripts/run_build_game.py --clip 16-28.MP4 --clip 16-58.MP4 --out data/games/<game_id> [--cpu]

Combines the clips with a stream copy (``-c copy`` - the clips come from one camera, so nothing is re-encoded),
writes the game's manifest, then builds the low-resolution proxy that kick-off, half-time and full-time are marked
on. The state after each step is written to ``build.json`` in the output directory, which is what the page polls.

Combining three 30-minute 4K clips is minutes of work; the proxy is a full decode and runs about as long as the
footage, exactly like a segment's scrubber, so both steps belong in the background.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from soccer_analytics.analysis import game
from soccer_analytics.dashboard import timeline
from soccer_analytics.ingest.ffmpeg_reader import probe_video


def main() -> int:
    parser = argparse.ArgumentParser(description="Combine camera clips into one game video and build its proxy.")
    parser.add_argument("--clip", action="append", required=True, help="source clip (repeat once per clip)")
    parser.add_argument("--out", required=True, help="game metadata directory (data/games/<game_id>)")
    parser.add_argument("--cpu", action="store_true", help="build the proxy without the GPU")
    args = parser.parse_args()

    directory = Path(args.out)
    game.write_build_state(directory, state="running", pid=os.getpid(), started=time.time(), stage="combining", error=None)
    try:
        ordered, output, expected = game.locations(args.clip, root=directory.parent)
        if expected != directory:
            raise RuntimeError(f"{directory} does not match these clips (expected {expected})")

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

        game.write_build_state(directory, stage="proxy", clips=len(planned.clips), duration_s=round(info.duration_s, 1))
        proxy = game.proxy_path(directory)
        if not proxy.exists():
            timeline.build_proxy(
                output,
                directory,
                start_s=0.0,
                duration_s=float(info.duration_s),
                width=game.PROXY_WIDTH,
                fps=game.PROXY_FPS,
                prefer_gpu=not args.cpu,
                # A game runs for an hour or more, so this is the keyframe-only path: one picture per second,
                # built in a couple of minutes instead of decoding 4K60 for the length of the footage.
                skip_frame=timeline.proxy_skip_frame(float(info.duration_s)),
            )

        game.write_build_state(directory, state="done", stage="done", finished=time.time())
        print(f"game ready: {output}", flush=True)
        return 0
    except Exception as exc:  # leave a readable state for the dashboard instead of a silently dead job
        game.write_build_state(directory, state="error", error=f"{type(exc).__name__}: {exc}")
        raise


if __name__ == "__main__":
    raise SystemExit(main())
