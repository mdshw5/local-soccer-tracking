"""Jersey numbers: crop math, OCR sanitation and per-track voting.

The expensive part - running EasyOCR over torso crops - lives in ``scripts/extract_jerseys.py``. Everything that
decides *what number a track wears* is pure and lives here, so the voting logic is unit-tested without a model:
OCR on small, angled, motion-blurred shirt numbers is noisy, and a per-track majority vote across many frames is
what turns it into something worth showing (and worth distrusting: low agreement stays "unassigned").
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Torso band of a detection box, as a fraction of its height. Shirt numbers sit on the chest/back, so the crop
# intentionally skips the head and the legs.
TORSO_TOP = 0.14
TORSO_BOTTOM = 0.62
# A crop smaller than this at the analyzed resolution cannot contain a readable number; skip it upstream.
MIN_CROP_HEIGHT_PX = 20
# Readings that must agree before a track is reported. The scan uses this to skip tracks that cannot possibly
# reach the quorum (fewer usable crops than readings needed).
MIN_VOTES = 3


@dataclass(frozen=True)
class JerseyCandidate:
    """One OCR reading: ``digits`` seen on the torso of detection row ``row`` in analysis frame ``frame``."""

    frame: int
    row: int
    digits: str
    confidence: float


def crop_torso(frame: np.ndarray, box: tuple[float, float, float, float]) -> np.ndarray | None:
    """Crop the torso band from a frame; ``box`` is ``(x1, y1, x2, y2)`` normalized by the frame *width*.

    Detection boxes are stored as fractions of the frame width (both axes), which is also how click coordinates are
    normalized elsewhere; this is the inverse. Returns None when the crop would be empty or clipped away entirely.
    """
    height, width = frame.shape[:2]
    x1, y1, x2, y2 = (float(value) * width for value in box)
    box_height = y2 - y1
    y_top = y1 + TORSO_TOP * box_height
    y_bottom = y1 + TORSO_BOTTOM * box_height
    x0 = int(np.clip(np.floor(x1), 0, width))
    x3 = int(np.clip(np.ceil(x2), 0, width))
    y0 = int(np.clip(np.floor(y_top), 0, height))
    y3 = int(np.clip(np.ceil(y_bottom), 0, height))
    if x3 - x0 < 4 or y3 - y0 < 4:
        return None
    return frame[y0:y3, x0:x3].copy()


def sanitize_digits(text: str) -> str:
    """Keep only a plausible shirt number: 1-2 digits, 1..99, no leading zero. Anything else reads as ""."""
    digits = "".join(ch for ch in str(text) if ch.isdigit())
    digits = digits.lstrip("0") or ""
    if not 1 <= len(digits) <= 2:
        return ""
    value = int(digits)
    return "" if not 1 <= value <= 99 else digits


def aggregate_candidates(
    candidates: dict[int, list[JerseyCandidate]],
    *,
    min_votes: int = MIN_VOTES,
    min_confidence: float = 0.40,
    min_share: float = 0.5,
) -> dict[int, dict]:
    """Vote the readings of each track into one number, or leave the track unassigned.

    Three acceptance routes, each requiring the winning digits' *mean confidence* to clear its threshold - a
    cheap reading is not evidence, however many times it repeats:

    * majority (the default): at least ``min_votes`` readings and at least ``min_share`` of the valid readings.
    * strong minority: at least ``min_votes`` readings at >= 0.75 confidence and >= 40% of the valid readings.
      A clean reading moment is short, and junk crops (hands, folds, half-turned backs) can outnumber it even
      when the footage is unambiguous: track 9238 read "22" three times at ~1.0 confidence among four junk
      readings (3/7) and a plain majority rule loses it.
    * unanimous pair: exactly two readings that agree, both at >= 0.90 mean confidence. Some tracks offer a
      single clear moment of two crops (track 928: "14" twice at 0.98/1.00; track 1373: "5" twice at 1.00);
      demanding a third reading there buys no more certainty, it only loses the number.
    """
    out: dict[int, dict] = {}
    for track_id, items in candidates.items():
        readings = [item for item in items if item.digits]
        if not readings:
            continue
        by_digits: dict[str, list[JerseyCandidate]] = {}
        for item in readings:
            by_digits.setdefault(item.digits, []).append(item)
        winner, winner_items = max(by_digits.items(), key=lambda pair: (len(pair[1]), sum(i.confidence for i in pair[1])))
        mean_confidence = float(np.mean([item.confidence for item in winner_items]))
        share = len(winner_items) / len(readings)
        votes = len(winner_items)
        majority = votes >= min_votes and share >= min_share and mean_confidence >= min_confidence
        minority = votes >= min_votes and share >= 0.4 and mean_confidence >= 0.75
        pair = votes == 2 and share == 1.0 and mean_confidence >= 0.90
        if not (majority or minority or pair):
            continue
        out[int(track_id)] = {
            "number": int(winner),
            "votes": votes,
            "readings": len(readings),
            "confidence": round(mean_confidence, 2),
        }
    return out


def _roster_name(team_rosters: dict[int, dict[int, str]] | None, team: int, number) -> str:  # noqa: ANN001
    """The name a team's roster (number -> name) has for a known shirt number, "" when there is none."""
    if team_rosters is None or number is None:
        return ""
    try:
        key = int(number)
    except (TypeError, ValueError):
        return ""
    return str((team_rosters.get(int(team)) or {}).get(key) or "")


def merge_numbers(
    track_ids: list[int],
    auto: dict[int, dict] | None = None,
    manual: dict[int, dict] | None = None,
    team_of: dict[int, int] | None = None,
    team_rosters: dict[int, dict[int, str]] | None = None,
) -> dict[int, dict]:
    """Identity per track for the replay and the table. Manual entries win over OCR; team rosters supply names.

    Three sources, each contributing what it can actually know:

    * the scan (``auto``) reads a *number* off the footage and supplies it only when several crops agreed;
    * the team roster for the track's team (``team_rosters``, from ``analysis.rosters``) is the number -> name
      association typed once per squad; it names any track whose number is known, this match or a future one;
    * a manual per-track entry (``manual``) is a correction of one track and wins over both - its number beats
      the scan, its name beats the roster. A manual number with no name still gets the roster's name.

    A number never comes from the roster alone: a roster can name a known number, but it cannot claim a track
    wears one - that claim is about what the camera saw. Entries are included when they carry a number or a
    name, so a track has to be identified by *something* to appear.
    """
    auto = auto or {}
    manual = manual or {}
    team_of = team_of or {}
    team_rosters = team_rosters or {}
    out: dict[int, dict] = {}
    for track_id in track_ids:
        track = int(track_id)
        team = int(team_of.get(track, -1))
        entry = manual.get(track)
        if entry and (entry.get("number") or entry.get("name")):
            number = entry.get("number")
            name = str(entry.get("name") or "").strip()
            if not name:
                name = _roster_name(team_rosters, team, number)
            out[track] = {
                "number": number,
                "name": name,
                "source": "manual",
                "confidence": 1.0,
            }
            continue
        entry = auto.get(track)
        if entry and entry.get("number"):
            number = entry["number"]
            out[track] = {
                "number": number,
                "name": _roster_name(team_rosters, team, number),
                "source": "auto",
                "confidence": float(entry.get("confidence", 0.0)),
            }
    return out
