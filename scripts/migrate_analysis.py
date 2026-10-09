"""Move the repository-era analysis roots (data/matches, data/segments, data/games) beside the footage.

The dashboard now saves every analysis in an ``analysis/<id>`` directory next to the video it describes, which is
what makes a match folder portable. This script migrates what was produced before that: match records, segment
directories and game manifests move into the right match directories, and the records are rewritten with the new
(relative) paths. Run it dry first; it prints every move it would make.

Usage::

    python scripts/migrate_analysis.py            # show what would move
    python scripts/migrate_analysis.py --apply    # move it
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from soccer_analytics.analysis.library import REPO_ROOT  # noqa: E402
from soccer_analytics.analysis.migration import migrate_all  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--repo-root", default=str(REPO_ROOT), help="repository whose data/ roots are migrated")
    parser.add_argument("--apply", action="store_true", help="actually move things (default: a dry run)")
    args = parser.parse_args()

    summary = migrate_all(args.repo_root, dry_run=not args.apply)
    verb = "moved" if args.apply else "would move"
    print(
        f"\n{verb}: {summary['matches']} match(es), {summary['games']} game manifest(s), "
        f"{summary['segments']} unclaimed segment(s)"
    )
    for note in summary["skipped"]:
        print(f"skipped: {note}")
    if not args.apply:
        print("dry run - pass --apply to perform the moves")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
