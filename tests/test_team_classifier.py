"""Tests for grass-aware jersey color features and robust team clustering.

The grass masking and the clustering guards are ported from
whisdev/soccer-video-detection-ai-agent.
"""

from __future__ import annotations

import cv2
import numpy as np

from soccer_analytics.tracking.team_classifier import (
    UNASSIGNED_TEAM,
    TeamClassifier,
    grass_color,
    grass_hue_window,
    kit_color_histogram,
)

GRASS_BGR = (0, 180, 0)  # OpenCV hue ~60 — pitch green
RED_BGR = (0, 0, 255)  # hue 0
BLUE_BGR = (255, 0, 0)  # hue ~120


def _solid(height: int, width: int, bgr: tuple[int, int, int]) -> np.ndarray:
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    frame[:, :] = bgr
    return frame


def _grass(hue: int, height: int, width: int) -> np.ndarray:
    """A patch of pitch grass at a given HSV hue (saturation/value high enough to read as grass)."""
    hsv = np.full((height, width, 3), (hue, 190, 150), dtype=np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def _pitch_with_player(
    kit: tuple[int, int, int],
    bbox: tuple[int, int, int, int] = (20, 20, 100, 100),
) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    """A green pitch with a kit-colored patch in the lower half of the player's torso."""
    frame = _solid(120, 120, GRASS_BGR)
    x1, y1, x2, y2 = bbox
    torso_mid = y1 + int((y2 - y1) * 0.35)
    frame[torso_mid : y2 - 10, x1 + 20 : x2 - 20] = kit
    return frame, bbox


def test_grass_hue_window_is_none_without_green():
    assert grass_hue_window(_solid(20, 20, RED_BGR)) is None


def test_grass_color_averages_only_green_pixels():
    frame = _solid(20, 20, RED_BGR)
    frame[:5, :] = GRASS_BGR
    assert tuple(round(channel) for channel in grass_color(frame)) == GRASS_BGR

    window = grass_hue_window(frame)
    assert window is not None
    assert window[0] < 60 < window[1]


def test_grass_window_covers_both_shaded_and_sunlit_grass():
    """The frame is measured, not assumed: shaded and sunlit grass sit ~25 hue apart and the mean lies between.

    On the real whole game the fixed +-10 band around the mean covered as little as 51% of the grass pixels (the
    sunlit mode also moves as the light changes late in the afternoon), so the window is widened to the measured
    population. A band that only covered the mean would mask the shaded half and let the sunlit half leak into
    every kit estimate.
    """
    frame = np.vstack([_grass(34, 30, 40), _grass(62, 30, 40)])
    window = grass_hue_window(frame)
    assert window is not None
    assert window[0] <= 35, f"shaded grass (hue 34) not covered: {window}"
    assert window[1] >= 61, f"sunlit grass (hue 62) not covered: {window}"


def test_grass_window_never_shrinks_below_the_fixed_band():
    """A frame whose grass is one tight hue mode keeps exactly the old behavior (mean hue +- 10)."""
    frame = _grass(45, 40, 40)
    window = grass_hue_window(frame)
    assert window is not None
    assert window[0] <= 35 and window[1] >= 55


def test_kit_histogram_masks_out_grass_pixels():
    crop = _solid(40, 40, GRASS_BGR)
    crop[10:30, 10:30] = RED_BGR
    grass_window = grass_hue_window(_solid(8, 8, GRASS_BGR))
    assert grass_window is not None

    masked = kit_color_histogram(crop, grass_window)
    jersey_only = kit_color_histogram(_solid(20, 20, RED_BGR))
    assert np.allclose(masked, jersey_only, atol=1e-6)


def test_kit_histogram_keeps_all_pixels_when_the_crop_is_all_grass():
    crop = _solid(20, 20, GRASS_BGR)
    grass_window = grass_hue_window(crop)
    assert grass_window is not None
    # Nothing survives the grass mask, so the unmasked crop is used instead.
    assert kit_color_histogram(crop, grass_window).sum() == kit_color_histogram(crop).sum()


def test_team_classifier_separates_kits_seen_against_grass():
    classifier = TeamClassifier(num_teams=2, min_samples_before_fit=4)
    red_frame, red_box = _pitch_with_player(RED_BGR)
    blue_frame, blue_box = _pitch_with_player(BLUE_BGR)

    for track_id in (1, 2):
        classifier.observe(track_id, red_frame, red_box)
    for track_id in (3, 4):
        classifier.observe(track_id, blue_frame, blue_box)
    classifier.fit()

    assert classifier.is_fitted
    assert classifier.team_of(1) == classifier.team_of(2)
    assert classifier.team_of(3) == classifier.team_of(4)
    assert classifier.team_of(1) != classifier.team_of(3)


def test_team_classifier_does_not_split_indistinguishable_kits():
    classifier = TeamClassifier(num_teams=2, min_samples_before_fit=4)
    frame, bbox = _pitch_with_player(RED_BGR)
    for track_id in (1, 2, 3, 4):
        classifier.observe(track_id, frame, bbox)
    classifier.fit()

    teams = {classifier.team_of(track_id) for track_id in (1, 2, 3, 4)}
    assert teams == {0}


def test_team_classifier_handles_a_single_track():
    classifier = TeamClassifier(num_teams=2, min_samples_before_fit=1)
    frame, bbox = _pitch_with_player(RED_BGR)
    classifier.observe(7, frame, bbox)
    classifier.fit()

    assert classifier.is_fitted
    assert classifier.team_of(7) == 0


def test_team_classifier_needs_enough_tracks_before_fitting():
    classifier = TeamClassifier(num_teams=2, min_samples_before_fit=3)
    frame, bbox = _pitch_with_player(RED_BGR)
    classifier.observe(1, frame, bbox)
    classifier.fit()

    assert not classifier.is_fitted
    assert classifier.team_of(1) == UNASSIGNED_TEAM


def test_team_classifier_labels_are_deterministic():
    def run() -> dict[int, int]:
        classifier = TeamClassifier(num_teams=2, min_samples_before_fit=4)
        red_frame, red_box = _pitch_with_player(RED_BGR)
        blue_frame, blue_box = _pitch_with_player(BLUE_BGR)
        for track_id in (1, 2):
            classifier.observe(track_id, red_frame, red_box)
        for track_id in (3, 4):
            classifier.observe(track_id, blue_frame, blue_box)
        classifier.fit()
        return {track_id: classifier.team_of(track_id) for track_id in (1, 2, 3, 4)}

    assert run() == run()
