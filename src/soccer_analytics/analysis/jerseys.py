"""Jersey numbers: crop maths, OCR sanitation and per-track voting.

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
# A crop smaller than this at the analysed resolution cannot contain a readable number; skip it upstream.
MIN_CROP_HEIGHT_PX = 20


@dataclass(frozen=True)
class JerseyCandidate:
    """One OCR reading: ``digits`` seen on the torso of detection row ``row`` in analysis frame ``frame``."""

    frame: int
    row: int
    digits: str
    confidence: float


def crop_torso(frame: np.ndarray, box: tuple[float, float, float, float]) -> np.ndarray | None:
    """Crop the torso band from a frame; ``box`` is ``(x1, y1, x2, y2)`` normalised by the frame *width*.

    Detection boxes are stored as fractions of the frame width (both axes), which is also how click coordinates are
    normalised elsewhere; this is the inverse. Returns None when the crop would be empty or clipped away entirely.
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
    min_votes: int = 3,
    min_confidence: float = 0.40,
    min_share: float = 0.5,
) -> dict[int, dict]:
    """Vote the readings of each track into one number, or leave the track unassigned.

    A number is only reported when readings agree: at least ``min_votes`` readings back it, they are at least
    ``min_share`` of the track's valid readings, and their mean confidence clears ``min_confidence``. A track whose
    readings scatter (two-digit numbers are easily half-read) stays unassigned rather than wearing a guess.
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
        if len(winner_items) < min_votes or share < min_share or mean_confidence < min_confidence:
            continue
        out[int(track_id)] = {
            "number": int(winner),
            "votes": len(winner_items),
            "readings": len(readings),
            "confidence": round(mean_confidence, 2),
        }
    return out


def merge_numbers(
    track_ids: list[int], auto: dict[int, dict] | None = None, manual: dict[int, dict] | None = None
) -> dict[int, dict]:
    """Identity per track for the replay and the table. Manual entries win over OCR.

    A manual entry counts even with only a name, so a roster can be typed in without inventing numbers.
    """
    auto = auto or {}
    manual = manual or {}
    out: dict[int, dict] = {}
    for track_id in track_ids:
        entry = manual.get(int(track_id))
        if entry and (entry.get("number") or entry.get("name")):
            out[int(track_id)] = {
                "number": entry.get("number"),
                "name": entry.get("name") or "",
                "source": "manual",
                "confidence": 1.0,
            }
            continue
        entry = auto.get(int(track_id))
        if entry and entry.get("number"):
            out[int(track_id)] = {
                "number": entry.get("number"),
                "name": "",
                "source": "auto",
                "confidence": float(entry.get("confidence", 0.0)),
            }
    return out
