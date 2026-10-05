"""The simulator is the oracle for Stage B tests, so it gets its own tests first."""

from __future__ import annotations

import numpy as np
import pytest

from soccer_analytics.analysis.projection import on_pitch_mask, project_segment, segment_poses
from soccer_analytics.geometry.camera_motion import apply_homography, integrate_poses  # noqa: F401

from synthetic_match import FOCAL, PITCH_LENGTH, PITCH_WIDTH, simulate_match


@pytest.fixture(scope="module")
def match():
    return simulate_match(frames=300, seed=1)


def _orientation_error(a: np.ndarray, b: np.ndarray) -> float:
    """Angle in degrees of the rotation taking a to b."""
    cos = (np.trace(a.T @ b) - 1.0) / 2.0
    return float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))))


def test_integrated_steps_reproduce_the_true_camera_orientation(match) -> None:
    segment, truth = match
    # The absolute focal length is unobservable from motion alone (a wrong constant focal biases every step by the
    # same rotation, which pitch calibration absorbs into R_base). Seed it, then check the *relative* chain.
    q, focal = segment_poses(segment, focal0=float(truth.focal[0]))
    assert np.max(np.abs(focal - truth.focal) / truth.focal) < 1e-5  # zoom is recovered to solver precision
    rel_true = np.array([truth.q[0].T @ truth.q[t] for t in range(len(q))])
    rel_est = np.array([q[0].T @ q[t] for t in range(len(q))])
    worst = max(_orientation_error(a, b) for a, b in zip(rel_true, rel_est))
    assert worst < 0.05, f"integrated orientation off by {worst:.3f} deg"
    # The remaining error is a *constant* reference-frame offset, which is exactly what R_base exists to absorb.
    offsets = [_orientation_error(q[0].T @ truth.q[0], np.eye(3)) for _ in [0]]
    constant = max(abs(_orientation_error(q[t], truth.q[t]) - offsets[0]) for t in range(len(q)))
    assert constant < 0.05, f"the reference offset is not constant (varies by {constant:.3f} deg)"


def test_integration_with_the_default_focal_keeps_relative_orientations_usable(match) -> None:
    """Without knowing the focal, the chain still returns a consistent (if offset) camera path."""
    segment, truth = match
    q, _ = segment_poses(segment)  # default focal, deliberately not the simulator's
    rel_true = np.array([truth.q[0].T @ truth.q[t] for t in range(len(q))])
    rel_est = np.array([q[0].T @ q[t] for t in range(len(q))])
    worst = max(_orientation_error(a, b) for a, b in zip(rel_true, rel_est))
    assert worst < 2.0, f"relative orientation degraded to {worst:.2f} deg with the wrong focal"


def test_the_reused_focal_chain_is_the_same_chain(match) -> None:
    """The dashboard's reruns rebuild poses with the analysis pass's own focals (no per-step search).

    That fast path must be the *same* chain, not an approximation of it: on a whole-game segment it is the
    difference between 40 s and 1 s of every rerun, so it is worth a test that pins the equivalence.

    The tolerance reflects storage, not a different algorithm: the segment's focals are float32 and the slow
    path re-*searches* each step with a bounded solver, so it lands within solver precision of the committed
    value instead of reproducing it bit-for-bit. 1e-6 is two orders below anything a focal change of that size
    can do to a rotation (~1e-6 rad); on the real whole-game segment the two chains differ by 9.5e-09 in q.
    """
    segment, _truth = match
    q_fast, focal_fast = segment_poses(segment)
    steps = [None if (not ok or i == 0) else s for i, (ok, s) in enumerate(zip(segment.ok, segment.step))]
    q_slow, focal_slow = integrate_poses(steps, float(segment.meta["default_focal"]), segment.aspect)
    assert np.allclose(q_fast, q_slow, atol=1e-6)
    assert np.allclose(focal_fast, focal_slow, rtol=1e-6, atol=1e-6)
    # and the reused focals really are the ones the chain committed, not the simulator's own zoom track
    assert np.allclose(focal_fast, segment.focal, atol=1e-6)


def _projection_for(segment, truth):
    """Project with the simulator's own reference orientation folded in, i.e. what calibration recovers."""
    from soccer_analytics.geometry.pitch_calibration import PitchCalibration

    q, focal = segment_poses(segment, focal0=float(truth.focal[0]))
    base = truth.calibration.base_rotation @ truth.q[0]
    cal = PitchCalibration(truth.calibration.position, base, truth.calibration.focal_scale, truth.calibration.aspect, 0.0, ())
    return project_segment(segment, cal, poses=(q, focal))


def test_detections_project_back_onto_true_positions(match) -> None:
    segment, truth = match
    pd = _projection_for(segment, truth)
    assert pd.valid.mean() > 0.97
    persons = truth.det_person
    true_xy = truth.positions[segment.det_frame, persons]
    err = np.linalg.norm(pd.xy[pd.valid] - true_xy[pd.valid], axis=1)
    # 3 px of foot noise: error is bounded by the same sensitivity the code reports as sigma_m.
    assert np.median(err) < 0.8
    assert np.median(err / pd.sigma_m[pd.valid]) < 1.5  # reported uncertainty is the right order of magnitude


def test_non_players_project_off_the_pitch_and_are_masked(match) -> None:
    segment, truth = match
    pd = _projection_for(segment, truth)
    keep = on_pitch_mask(pd, PITCH_LENGTH, PITCH_WIDTH)
    role = truth.role[truth.det_person]
    players_kept = keep[role < 2].mean()
    sideline_kept = keep[role == 3].mean()
    assert players_kept > 0.95, f"players wrongly dropped: {1 - players_kept:.1%}"
    assert sideline_kept < 0.10, f"sideline people wrongly kept: {sideline_kept:.1%}"


def test_simulator_is_deterministic_and_has_the_scenario_it_claims(match) -> None:
    segment, truth = match
    again, _ = simulate_match(frames=300, seed=1)
    assert np.array_equal(segment.det_box, again.det_box)
    assert set(np.unique(truth.role)) == {0, 1, 2, 3}
    assert 0.85 < truth.detected[truth.visible].mean() < 0.99  # misses exist but are not the norm
    assert truth.positions[..., 0].min() > -5 and truth.positions[..., 0].max() < PITCH_LENGTH + 8  # sideline at L + 7
    # the camera really does move: pan range of tens of degrees, as on real footage
    yaw = np.degrees(np.arctan2(truth.q[:, 0, 2], truth.q[:, 2, 2]))
    assert np.ptp(yaw) > 15
