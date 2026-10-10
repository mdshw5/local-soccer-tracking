"""Jersey-number extraction: the crop math and the per-track vote, without a model.

OCR itself needs a GPU and downloaded weights, so what is tested here is everything that decides whether a reading
exists and which number a track gets - including that a track with disagreeing readings stays unassigned.
"""

from __future__ import annotations

import numpy as np

from soccer_analytics.analysis.jerseys import (
    JerseyCandidate,
    aggregate_candidates,
    crop_torso,
    merge_numbers,
    sanitize_digits,
)


def test_crop_torso_maps_box_fractions_to_pixels() -> None:
    """Det boxes are fractions of the frame *width* on both axes; the crop must use the same convention."""
    frame = np.zeros((100, 200, 3), np.uint8)
    frame[51:90, 20:80] = 255  # the expected torso band for the box below
    box = (0.10, 0.20, 0.40, 0.60)  # x1, y1, x2, y2 as fractions of width=200
    crop = crop_torso(frame, box)
    assert crop is not None
    assert crop.shape[0] > 0 and crop.shape[1] == 60
    assert (crop == 255).all(), "crop must land exactly on the band the box designates"


def test_crop_torso_rejects_boxes_that_leave_nothing() -> None:
    frame = np.zeros((100, 200, 3), np.uint8)
    assert crop_torso(frame, (1.2, 0.2, 1.4, 0.6)) is None  # fully outside the frame
    assert crop_torso(frame, (0.1, 0.2, 0.11, 0.6)) is None  # too narrow to hold a number


def test_sanitize_digits_keeps_plausible_shirt_numbers() -> None:
    assert sanitize_digits("26") == "26"
    assert sanitize_digits(" 7 ") == "7"
    assert sanitize_digits("07") == "7"  # leading zeros are OCR artifacts, not shirt numbers
    assert sanitize_digits("0") == ""
    assert sanitize_digits("123") == ""
    assert sanitize_digits("1a4") == "14"
    assert sanitize_digits("") == ""


def _candidate(number: str, confidence: float, frame: int = 0, row: int = 0) -> JerseyCandidate:
    return JerseyCandidate(frame=frame, row=row, digits=number, confidence=confidence)


def test_aggregate_needs_agreeing_readings() -> None:
    readings = [_candidate("26", 0.9)] * 4 + [_candidate("20", 0.5)]
    out = aggregate_candidates({3: readings})
    assert out[3]["number"] == 26
    assert out[3]["votes"] == 4 and out[3]["readings"] == 5


def test_aggregate_rejects_a_scattered_track() -> None:
    scattered = [_candidate("26", 0.9), _candidate("20", 0.8), _candidate("9", 0.7)]
    assert aggregate_candidates({4: scattered}) == {}, "a track with no majority must stay unassigned"


def test_aggregate_rejects_low_confidence_and_small_samples() -> None:
    assert aggregate_candidates({5: [_candidate("10", 0.2)] * 6}) == {}, "confident agreement, but not confident"
    assert aggregate_candidates({6: [_candidate("10", 0.7)] * 2}) == {}, "two middling readings are not a vote"


def test_aggregate_accepts_a_unanimous_high_confidence_pair() -> None:
    # Measured on the real game: track 928 read "14" twice at 0.98/1.00 across its only two clear crops, and a
    # three-vote quorum lost a number the footage shows plainly.
    out = aggregate_candidates({928: [_candidate("14", 0.98), _candidate("14", 1.0)]})
    assert out[928]["number"] == 14 and out[928]["votes"] == 2


def test_aggregate_accepts_a_strong_minority() -> None:
    # Measured on the real game: track 9238 read "22" three times at ~1.0 confidence among four junk readings
    # (24s and a 12), 3/7 of the crops - below the majority share, but the reading itself is unambiguous.
    readings = [_candidate("22", 1.0)] * 2 + [_candidate("24", 0.8)] * 2 + [_candidate("12", 0.3), _candidate("22", 0.95), _candidate("4", 1.0)]
    out = aggregate_candidates({9238: readings})
    assert out[9238]["number"] == 22 and out[9238]["votes"] == 3


def test_manual_entries_beat_the_scan() -> None:
    auto = {7: {"number": 26, "confidence": 0.9}, 8: {"number": 11, "confidence": 0.8}}
    manual = {7: {"number": 9, "name": "Sam"}}
    merged = merge_numbers([7, 8], auto=auto, manual=manual)
    assert merged[7] == {"number": 9, "name": "Sam", "source": "manual", "confidence": 1.0}
    assert merged[8] == {"number": 11, "name": "", "source": "auto", "confidence": 0.8}


def test_a_manual_name_without_a_number_still_counts() -> None:
    merged = merge_numbers([11], manual={11: {"number": None, "name": "Coach"}})
    assert merged[11]["name"] == "Coach" and merged[11]["number"] is None
