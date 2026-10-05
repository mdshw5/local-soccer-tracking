"""Unit tests for team classification."""

from __future__ import annotations

import numpy as np

from soccer_analytics.tracking.team_classifier import UNASSIGNED_TEAM, TeamClassifier


def _colored_frame(height: int, width: int, bgr: tuple[int, int, int]) -> np.ndarray:
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    frame[:, :] = bgr
    return frame


def test_team_classifier_separates_two_distinct_jersey_colors():
    classifier = TeamClassifier(num_teams=2, min_samples_before_fit=4)
    red_frame = _colored_frame(100, 100, (0, 0, 255))
    blue_frame = _colored_frame(100, 100, (255, 0, 0))
    bbox = (10, 10, 90, 90)

    for track_id in (1, 2):
        classifier.observe(track_id, red_frame, bbox)
    for track_id in (3, 4):
        classifier.observe(track_id, blue_frame, bbox)

    classifier.fit()

    assert classifier.is_fitted
    assert classifier.team_of(1) == classifier.team_of(2)
    assert classifier.team_of(3) == classifier.team_of(4)
    assert classifier.team_of(1) != classifier.team_of(3)


def test_team_classifier_unassigned_before_fit():
    classifier = TeamClassifier(num_teams=2, min_samples_before_fit=10)
    assert classifier.team_of(1) == UNASSIGNED_TEAM
