"""Automatic pitch registration: template geometry, keypoint screening, and an end-to-end recovery.

The oracle is the same synthetic gimbal camera the Stage B tests use. Its true pose, orientation and focal are
known, so the template's 32 markers can be projected into its frames exactly - which is what a perfect keypoint
detector would report. Noise and wrong-index detections are then added to it, and the test asks whether
``auto_register`` recovers the camera the simulator used.
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from synthetic_match import ASPECT, PITCH_LENGTH, PITCH_WIDTH, simulate_match

from soccer_analytics.geometry.auto_register import (
    KeypointObservation,
    auto_register,
    filter_frame,
    observations_to_landmarks,
    register_with_position_prior,
    registration_note,
)
from soccer_analytics.geometry.pitch_calibration import pitch_to_pixels
from soccer_analytics.geometry.pitch_template import (
    KEYPOINT_COUNT,
    STANDARD_LENGTH_M,
    STANDARD_WIDTH_M,
    template_for,
)


def test_template_matches_the_standard_markings() -> None:
    """The template is the pitch's own geometry, so a few markers have values worth pinning down."""
    template = template_for()
    assert len(template) == KEYPOINT_COUNT == 32
    # Corners, in the reference's top-left-to-bottom-left then mirrored order.
    assert template[0] == pytest.approx((0.0, 0.0))
    assert template[5] == pytest.approx((0.0, STANDARD_WIDTH_M))
    assert template[24] == pytest.approx((STANDARD_LENGTH_M, 0.0))
    # Left penalty spot is 11 m out, on the centre line.
    assert template[8] == pytest.approx((11.0, STANDARD_WIDTH_M / 2))
    # Right penalty spot is its mirror.
    assert template[21] == pytest.approx((STANDARD_LENGTH_M - 11.0, STANDARD_WIDTH_M / 2))
    # The centre circle's west/east points sit 9.15 m either side of the halfway line.
    assert template[30] == pytest.approx((STANDARD_LENGTH_M / 2 - 9.15, STANDARD_WIDTH_M / 2))
    assert template[31] == pytest.approx((STANDARD_LENGTH_M / 2 + 9.15, STANDARD_WIDTH_M / 2))
    # Halfway line meets both touchlines.
    assert template[13] == pytest.approx((STANDARD_LENGTH_M / 2, 0.0))
    assert template[16] == pytest.approx((STANDARD_LENGTH_M / 2, STANDARD_WIDTH_M))


def test_template_scales_independently_to_any_format() -> None:
    small = template_for(60.0, 40.0)
    assert small[8] == pytest.approx((11.0 * 60.0 / 105.0, 20.0))
    assert small[0] == pytest.approx((0.0, 0.0))
    assert small[31] == pytest.approx(((52.5 + 9.15) * 60.0 / 105.0, 20.0))
    with pytest.raises(ValueError):
        template_for(0.0, 68.0)


def test_filter_frame_rejects_a_wrong_index_but_keeps_the_frame() -> None:
    """A wrong-index keypoint is a confident, plausible-looking lie; the frame's own homography exposes it.

    Nine keypoints arranged so they lie exactly on one homography, then one of them is moved to where a *different*
    marker's pixel would be. RANSAC should keep the eight that agree and drop the odd one out.
    """
    template = [(x, y) for y in (0.0, 50.0, 100.0) for x in (0.0, 100.0, 200.0)]
    homography = np.array([[1.3, 0.1, 0.4], [0.05, 1.1, 0.25], [0.0004, 0.00015, 1.0]])
    projected = cv2.perspectiveTransform(
        np.array(template, dtype=np.float32).reshape(-1, 1, 2), homography
    ).reshape(-1, 2)
    observations = [
        KeypointObservation(0, index, float(point[0]), float(point[1]), 0.9)
        for index, point in enumerate(projected)
    ]
    observations[4] = KeypointObservation(0, 4, float(projected[7][0]), float(projected[7][1]), 0.9)

    kept = filter_frame(observations, template, ransac_threshold_px=12.0, frame_width=1920.0)
    assert 4 not in {point.index for point in kept}
    assert len(kept) == len(template) - 1


def test_filter_frame_passes_a_tiny_frame_through() -> None:
    """Fewer than four points cannot define a homography, so there is nothing to screen them against."""
    template = template_for()
    observations = [KeypointObservation(0, i, 0.5, 0.3, 0.9) for i in range(3)]
    assert filter_frame(observations, template) == observations


def test_observations_to_landmarks_uses_the_template_position() -> None:
    template = template_for(60.0, 40.0)
    landmarks = observations_to_landmarks(
        [KeypointObservation(3, 8, 0.4, 0.3, 0.9), KeypointObservation(3, 999, 0.1, 0.1, 0.9)], template
    )
    assert len(landmarks) == 1  # index 999 is off the template and ignored
    landmark = landmarks[0]
    assert (landmark.pitch_x, landmark.pitch_y) == pytest.approx(template[8])
    assert (landmark.u, landmark.v) == pytest.approx((0.4, 0.3))
    assert landmark.label == "kp8"


def _observations_from_truth(
    truth,
    frames: list[int],
    *,
    noise_px: float = 3.0,
    seed: int = 0,
    corrupt: int = 0,
) -> list[KeypointObservation]:
    """The template projected into the true camera's frames, plus noise and a few wrong-index detections."""
    rng = np.random.default_rng(seed)
    template = template_for(PITCH_LENGTH, PITCH_WIDTH)
    observations: list[KeypointObservation] = []
    for frame in frames:
        uv, front = pitch_to_pixels(truth.calibration, np.asarray(template), truth.q[frame], truth.focal[frame])
        for index, ((u, v), visible) in enumerate(zip(uv, front)):
            if not visible or not (0.0 <= u <= 1.0 and 0.0 <= v <= ASPECT):
                continue
            jitter = rng.normal(0.0, noise_px / 1920.0, 2)
            observations.append(KeypointObservation(frame, index, float(u + jitter[0]), float(v + jitter[1]), 0.9))
    if corrupt:
        # Relabel some detections with a different marker's *pixel* - a confidently wrong correspondence.
        picked = rng.choice(len(observations), size=min(corrupt, len(observations)), replace=False)
        for position in picked:
            original = observations[int(position)]
            wrong = (original.index + 7) % KEYPOINT_COUNT
            observations[int(position)] = KeypointObservation(
                original.frame, wrong, original.u, original.v, 0.9
            )
    return observations


def test_auto_register_recovers_the_simulated_camera() -> None:
    segment, truth = simulate_match(frames=600, seed=4)
    frames = [0, 100, 200, 300, 400, 500]
    observations = _observations_from_truth(truth, frames, noise_px=2.0, seed=1)
    chain = {frame: (truth.q[frame], float(truth.focal[frame])) for frame in frames}

    result = auto_register(
        observations,
        chain,
        ASPECT,
        length_m=PITCH_LENGTH,
        width_m=PITCH_WIDTH,
        correct_drift=False,
    )

    assert result.keypoints_kept >= 16
    assert result.calibration.rms_error_m < 2.0
    assert np.linalg.norm(result.calibration.position - truth.calibration.position) < 2.5
    assert result.calibration.focal_scale == pytest.approx(truth.calibration.focal_scale, rel=0.15)
    assert "rms error" in registration_note(result)


def test_auto_register_screens_out_wrong_index_keypoints() -> None:
    """The whole point of the per-frame screen: a few confident but wrongly-labelled markers must not steer the fit.

    Without screening these would be handed to the solver as ordinary landmarks; here they should be dropped before
    the fit, so the recovered camera stays close to the truth and the report says how many were discarded.
    """
    segment, truth = simulate_match(frames=600, seed=7)
    frames = [50, 150, 250, 350, 450, 550]
    clean = _observations_from_truth(truth, frames, noise_px=2.0, seed=2)
    dirty = _observations_from_truth(truth, frames, noise_px=2.0, seed=2, corrupt=12)
    chain = {frame: (truth.q[frame], float(truth.focal[frame])) for frame in frames}

    result = auto_register(
        dirty, chain, ASPECT, length_m=PITCH_LENGTH, width_m=PITCH_WIDTH, correct_drift=False
    )

    assert len(dirty) == len(clean)  # only labels changed
    assert result.keypoints_kept < len(dirty), "the corrupt keypoints should have been screened out"
    # RANSAC takes the largest agreeing subset, so a corrupted point can cost a genuine one in the same frame too;
    # what must hold is that most of the good evidence survives and the pose it yields is the simulator's camera.
    assert result.keypoints_kept >= len(clean) * 0.7
    assert np.linalg.norm(result.calibration.position - truth.calibration.position) < 0.5
    assert any("disagreed" in note for note in result.notes)


def test_prior_registration_recovers_the_camera_from_a_known_position() -> None:
    """With the tripod position known, registration only has to find the orientation - the fixed-PTZ case."""
    segment, truth = simulate_match(frames=600, seed=11)
    frames = [50, 150, 250, 350, 450, 550]
    observations = _observations_from_truth(truth, frames, noise_px=2.0, seed=4)
    chain = {frame: (truth.q[frame], float(truth.focal[frame])) for frame in frames}

    result = register_with_position_prior(
        observations,
        chain,
        ASPECT,
        position_prior=tuple(float(v) for v in truth.calibration.position),
        focal_scale_prior=float(truth.calibration.focal_scale),
        length_m=PITCH_LENGTH,
        width_m=PITCH_WIDTH,
        correct_drift=False,
    )

    assert result.keypoints_kept >= 16
    assert np.linalg.norm(result.calibration.position - truth.calibration.position) < 1.0
    assert result.calibration.focal_scale == pytest.approx(truth.calibration.focal_scale, rel=0.15)
    assert any("orientation fixed on frame" in note for note in result.notes)


def test_prior_registration_reports_when_the_pitch_cannot_be_found() -> None:
    """Detections that do not agree with the known camera must fail loudly, not return a confident pose."""
    template = template_for()
    # Four keypoints all claiming to be the same marker: no orientation can explain them on one pitch.
    observations = [KeypointObservation(0, 8, 0.2 + 0.1 * i, 0.3, 0.9) for i in range(4)]
    with pytest.raises(ValueError, match="agree on the main pitch"):
        register_with_position_prior(
            observations,
            {0: (np.eye(3), 0.82)},
            ASPECT,
            position_prior=(50.0, -2.0, 4.0),
            template=template,
            correct_drift=False,
        )


def test_auto_register_reports_too_little_evidence() -> None:
    """Three markers cannot register a camera, and the failure has to say so rather than return a bad pose."""
    template = template_for()
    observations = [KeypointObservation(0, i, 0.5, 0.3, 0.9) for i in range(3)]
    with pytest.raises(ValueError, match="survived screening"):
        auto_register(observations, {0: (np.eye(3), 0.82)}, ASPECT, template=template)


def test_auto_register_ignores_low_confidence_detections() -> None:
    segment, truth = simulate_match(frames=300, seed=9)
    frames = [10, 110, 210]
    observations = _observations_from_truth(truth, frames, noise_px=1.0, seed=3)
    observations = [
        KeypointObservation(o.frame, o.index, o.u, o.v, 0.05 if rank < 5 else o.confidence)
        for rank, o in enumerate(observations)
    ]

    result = auto_register(
        observations,
        {frame: (truth.q[frame], float(truth.focal[frame])) for frame in frames},
        ASPECT,
        length_m=PITCH_LENGTH,
        width_m=PITCH_WIDTH,
        correct_drift=False,
    )
    assert result.keypoints_total == len(observations)
    assert any("confidence floor" in note for note in result.notes)