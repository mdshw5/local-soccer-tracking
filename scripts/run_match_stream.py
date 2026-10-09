"""Serve the annotated match stream: the footage with the pitch model, detections and identities drawn on it.

A small MJPEG server (no new dependencies - the stdlib's http.server, opencv and the project's own ffmpeg
reader) in front of one or more matches in the archive:

* ``/`` - an index page: every streamable match, its teams' colours, and links to play it;
* ``/stream/<match_id>.mjpg`` - the stream itself. Query parameters: ``start`` (source-clock seconds),
  ``rate`` (playback speed, 0.25-8x), ``width`` (decode width, 320-2560 px);
* ``/frame/<match_id>.jpg?t=<seconds>`` - one annotated still, for checking quickly;
* ``/matches`` - the same listing as the index, as JSON.

Each viewer gets their own decode, so two people can watch different parts of the match at different speeds.
The overlay is the honest one (see ``dashboard/stream.py``): boxes and shirt numbers come from the match's
report build, the ball from the segment's scan, and the pitch model is projected through the same corrected
camera chain the report uses. A match whose replay predates the per-player boxes is refused with the command
that rebuilds it rather than streamed without boxes.

Usage::

    python scripts/run_match_stream.py [--port 8510] [--match <match_id> ...] [--list]

The first request for a match loads its segment and camera chain (a few seconds) and then stays cached; pass
``--match`` to preload those. Stop with control-C.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from soccer_analytics.dashboard.stream import (
    DEFAULT_PORT,
    MATCHES_ROOT,
    StreamError,
    serve,
    streamable_matches,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--host", default="0.0.0.0", help="interface to bind (default: all)")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help="port to serve on")
    parser.add_argument("--root", default=str(MATCHES_ROOT), help="matches root (data/matches)")
    parser.add_argument("--width", type=int, default=1600, help="default decode width in pixels")
    parser.add_argument("--match", action="append", default=[], help="match id to preload (repeatable)")
    parser.add_argument("--list", action="store_true", help="list streamable matches and exit")
    args = parser.parse_args()

    if args.list:
        for row in streamable_matches(args.root):
            if row.get("streamable"):
                names = " vs ".join(str(name) for name in row.get("team_names") or [])
                print(f"{row['match_id']}: {names}, window {row.get('start_s', 0):.0f}s-{row.get('end_s', 0):.0f}s")
            else:
                print(f"{row['match_id']}: not streamable ({row.get('reason')})")
        return 0

    try:
        serve(args.host, args.port, root=args.root, width=args.width, preload=tuple(args.match))
    except StreamError as error:
        print(f"stream server could not start: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
