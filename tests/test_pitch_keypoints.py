"""Tests for the model-agnostic pitch keypoint helpers (ported from the reference project)."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from soccer_analytics.geometry.pitch_keypoints import (
    extract_heatmap_peaks,
    homography_from_keypoints,
    refine_keypoints,
    warp_to_pitch_view,
)

# A small stand-in pitch template: 3x3 grid in "pitch units".
TEMPLATE = [(x, y) for y in (0.0, 50.0, 100.0) for x in (0.0, 100.0, 200.0)]
FRAME_SHAPE = (400, 600, 3)


def _true_homography() -> np.ndarray:
    return np.array(
        [
            [1.30, 0.10, 40.0],
            [0.05, 1.10, 25.0],
            [0.00040, 0.00015, 1.0],
        ]
    )


def _project(points: list[tuple[float, float]]) -> np.ndarray:
    array = np.array(points, dtype=np.float32).reshape(-1, 1, 2)
    return cv2.perspectiveTransform(array, _true_homography()).reshape(-1, 2)


def _detected_keypoints(indices: list[int] | None = None) -> dict[int, dict[str, float]]:
    projected = _project(TEMPLATE)
    selected = range(len(TEMPLATE)) if indices is None else indices
    return {index: {"x": float(projected[index][0]), "y": float(projected[index][1]), "p": 0.9} for index in selected}


def test_homography_from_keypoints_recovers_the_projection():
    homography = homography_from_keypoints(_detected_keypoints(), TEMPLATE)
    assert homography is not None

    template = np.array(TEMPLATE, dtype=np.float32).reshape(-1, 1, 2)
    projected = cv2.perspectiveTransform(template, homography).reshape(-1, 2)
    assert np.allclose(projected, _project(TEMPLATE), atol=1e-2)


def test_homography_from_keypoints_requires_enough_points():
    assert homography_from_keypoints(_detected_keypoints([0, 1, 2]), TEMPLATE) is None


def test_homography_from_keypoints_ignores_undetected_points():
    keypoints = _detected_keypoints()
    keypoints[4] = {"x": 12.0, "y": 34.0, "p": 0.0}
    homography = homography_from_keypoints(keypoints, TEMPLATE)
    assert homography is not None

    template = np.array(TEMPLATE, dtype=np.float32).reshape(-1, 1, 2)
    projected = cv2.perspectiveTransform(template, homography).reshape(-1, 2)
    assert np.allclose(projected, _project(TEMPLATE), atol=1e-2)


def test_refine_keypoints_fills_missing_points_from_the_template():
    keypoints = _detected_keypoints([0, 1, 2, 3, 5, 6, 7, 8])
    refined = refine_keypoints(keypoints, TEMPLATE, FRAME_SHAPE)

    assert refined[4]["source"] == "template"
    assert refined[4]["p"] == 0.0
    expected_x, expected_y = _project(TEMPLATE)[4]
    assert refined[4]["x"] == pytest.approx(float(expected_x), abs=1e-2)
    assert refined[4]["y"] == pytest.approx(float(expected_y), abs=1e-2)
    assert refined[0]["source"] == "detected"
    assert refined[0]["x"] == keypoints[0]["x"]


def test_refine_keypoints_keeps_detected_points_when_it_cannot_fit():
    keypoints = _detected_keypoints([0, 1])
    refined = refine_keypoints(keypoints, TEMPLATE, FRAME_SHAPE)
    assert set(refined) == {0, 1}


def test_extract_heatmap_peaks_finds_one_peak_per_channel():
    heatmap = np.zeros((2, 20, 30), dtype=np.float32)
    heatmap[0, 5, 6] = 0.9
    heatmap[1, 12, 18] = 0.7

    peaks = extract_heatmap_peaks(heatmap, threshold=0.2, scale=2.0)

    assert peaks[0]["x"] == 12.0
    assert peaks[0]["y"] == 10.0
    assert peaks[0]["p"] == pytest.approx(0.9, abs=1e-6)
    assert (peaks[1]["x"], peaks[1]["y"]) == (36.0, 24.0)


def test_extract_heatmap_peaks_skips_weak_channels():
    heatmap = np.full((2, 10, 10), 0.1, dtype=np.float32)
    heatmap[1, 2, 3] = 0.6

    peaks = extract_heatmap_peaks(heatmap, threshold=0.2)

    assert set(peaks) == {1}
    assert (peaks[1]["x"], peaks[1]["y"]) == (3.0, 2.0)


def test_extract_heatmap_peaks_accepts_torch_tensors():
    torch = pytest.importorskip("torch")
    heatmap = torch.zeros(1, 1, 16, 16)
    heatmap[0, 0, 4, 9] = 0.8

    peaks = extract_heatmap_peaks(heatmap, threshold=0.2)

    assert (peaks[0]["x"], peaks[0]["y"]) == (9.0, 4.0)


def test_warp_to_pitch_view_maps_template_coordinates_into_view():
    frame = np.zeros(FRAME_SHAPE, dtype=np.uint8)
    x, y = _project([(100.0, 50.0)])[0]
    frame[int(y) - 8 : int(y) + 8, int(x) - 8 : int(x) + 8] = 255

    homography = homography_from_keypoints(_detected_keypoints(), TEMPLATE)
    assert homography is not None
    warped = warp_to_pitch_view(frame, homography, (200, 100))

    assert warped.shape == (100, 200, 3)
    assert warped[50, 100].mean() > 200
