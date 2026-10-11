"""Team rosters: shirt number -> player name, saved once per team and reused across matches.

The footage tells half of a player's identity and only half: the number scan (and a rebuild's stitching) reads a
*number* off a track, but a name is knowledge about the squad, not about the video - nothing in the pixels says
the #10 is called Alice. This module is where that half lives. A roster is the association ``number -> name`` for
one team, stored as a small JSON file in a shared library (``data/rosters`` by default, ``SOCCER_ROSTERS_DIR`` to
override) rather than inside one match: type the squad once, and every future match links the same roster and
gets the names for free.

A match links one roster per team in ``MatchRecord.team_rosters`` (the saved roster's name, "" for none).
:func:`team_rosters_for` turns those links into the ``{team: {number: name}}`` map :func:`jerseys.merge_numbers`
labels tracks with. The link is the only state that lives in the match; the names themselves stay in the library,
so fixing a typo in a roster propagates to every match that uses it - including ones already analyzed.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[3]
ROSTERS_DIRNAME = "rosters"
ROSTERS_ROOT_ENV = "SOCCER_ROSTERS_DIR"


def default_rosters_root() -> Path:
    """Where team rosters are kept: ``SOCCER_ROSTERS_DIR`` when set, else ``data/rosters`` in the repository."""
    override = os.environ.get(ROSTERS_ROOT_ENV, "").strip()
    return Path(override).expanduser() if override else REPO_ROOT / "data" / ROSTERS_DIRNAME


def roster_slug(name: str) -> str:
    """The file name for a roster's display name: lowercase, runs of anything else collapsed to ``-``."""
    slug = re.sub(r"[^a-z0-9]+", "-", str(name).strip().casefold()).strip("-")
    return slug[:60] or "roster"


def clean_roster(numbers: Mapping) -> dict[int, str]:
    """Normalize ``{number: name}`` into what a roster file holds: int numbers 1..99, non-empty stripped names.

    Everything that is not a usable association is dropped - an entry with no name says nothing beyond what the
    detection already said, and the library saves names, not the absence of them.
    """
    cleaned: dict[int, str] = {}
    for number, name in dict(numbers).items():
        try:
            value = int(number)
        except (TypeError, ValueError):
            continue
        text = str(name or "").strip()
        if text and 1 <= value <= 99:
            cleaned[value] = text
    return dict(sorted(cleaned.items()))


def _atomic_write(path: Path, payload) -> None:  # noqa: ANN001 - a JSON-able payload
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2))
    tmp.replace(path)


class RosterLibrary:
    """The saved team rosters, one JSON file per roster under a shared root.

    Reading tolerates a missing root, a torn file or a hand-edited one: a match linked to a roster that is not
    there is the normal state after someone deletes a file, not a crash - the match simply has no names until a
    roster is linked again.
    """

    def __init__(self, root: str | Path | None = None):
        self.root = Path(root) if root is not None else default_rosters_root()

    def path(self, name: str) -> Path:
        """The file a roster's display name maps to."""
        return self.root / f"{roster_slug(name)}.json"

    def names(self) -> list[str]:
        """Every saved roster's display name, sorted; [] when the library is empty or absent."""
        if not self.root.exists():
            return []
        found: list[str] = []
        for path in sorted(self.root.glob("*.json")):
            try:
                stored = json.loads(path.read_text()).get("name")
            except (OSError, json.JSONDecodeError):
                continue
            found.append(str(stored or path.stem))
        return sorted(found, key=str.casefold)

    def exists(self, name: str) -> bool:
        return self.path(name).exists()

    def load(self, name: str) -> dict[int, str]:
        """The roster's number -> name map; {} when the file is missing or unreadable."""
        path = self.path(name)
        if not path.exists():
            return {}
        try:
            payload = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            return {}
        return clean_roster(payload.get("numbers") or {})

    def save(self, name: str, numbers: Mapping) -> Path:
        """Write (or overwrite) a roster under ``name``, keeping only the usable associations."""
        display = str(name).strip()
        cleaned = clean_roster(numbers)
        path = self.path(display)
        _atomic_write(path, {"name": display, "numbers": {str(number): text for number, text in cleaned.items()}})
        return path

    def delete(self, name: str) -> bool:
        """Remove a roster; False when there was no file to remove."""
        path = self.path(name)
        if not path.exists():
            return False
        path.unlink()
        return True


def team_rosters_for(library: RosterLibrary, links: Sequence[str] | None) -> dict[int, dict[int, str]]:
    """The number -> name maps of a match's linked rosters, keyed by team index.

    ``links`` is a record's ``team_rosters`` list: a name per team, "" for a team with no roster. A link whose
    file has gone missing contributes an empty map - the match shows the numbers without names rather than
    refusing to load.
    """
    out: dict[int, dict[int, str]] = {}
    for team, name in enumerate(list(links or [])[:2]):
        text = str(name or "").strip()
        if text:
            out[int(team)] = library.load(text)
    return out
