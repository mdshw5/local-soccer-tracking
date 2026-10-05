"""The ball tracker's logic: association, camera-motion prediction, and the states a caller reads.

The distinction these tests hold is the whole reason the tracker exists: a position from a detection, a forecast
through a brief miss, and an honest "not in the picture" - and that none of those ever silently becomes another.
"""

from __future__ import annotations

import numpy as np
import pytest

from soccer_analytics.analysis.ball import MAX_COAST, BallTrack


def det(conf: float, u: float, v: float, size: float = 0.01) -> tuple:
    return (conf, u, v, size, size)


def test_predict_follows_the_step_homography() -> None:
    track = BallTrack(aspect=9 / 16, x=0.5, y=0.25)
    assert track.predict(None) == (0.5, 0.25), "a frame the chain lost predicts 'unchanged'"
    assert track.predict(np.eye(3)) == (0.5, 0.25)
    shift = np.array([[1.0, 0.0, 0.01], [0.0, 1.0, -0.02], [0.0, 0.0, 1.0]])
    u, v = track.predict(shift)
    assert u == pytest.approx(0.51)
    assert v == pytest.approx(0.23)
    # a projective step is dehomogenised, not read as if w were 1
    projective = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.2, 0.0, 1.0]])
    u, v = track.predict(projective)
    assert u == pytest.approx(0.5 / 1.1)
    assert v == pytest.approx(0.25 / 1.1)


def test_a_detection_within_the_gate_is_accepted_and_becomes_the_position() -> None:
    track = BallTrack(aspect=9 / 16, x=0.5, y=0.25)
    state = track.update([det(0.8, 0.505, 0.251)], step=np.eye(3))
    assert state["status"] == "tracking"
    assert state["u"] == pytest.approx(0.505)
    assert state["v"] == pytest.approx(0.251)
    assert state["conf"] == pytest.approx(0.8)


def test_a_confident_detection_beats_a_marginal_one_on_the_prediction() -> None:
    """Association weighs confidence against distance: the ball's detector score varies a lot frame to frame."""
    track = BallTrack(aspect=9 / 16, x=0.5, y=0.25)
    state = track.update([det(0.2, 0.5, 0.25), det(0.9, 0.51, 0.25)], step=np.eye(3))
    assert state["u"] == pytest.approx(0.51)


def test_a_detection_beyond_the_gate_coasts_instead_of_teleporting() -> None:
    """Far-away detections are *not* the tracked ball; a spectactor's white shoe 20 m across the frame is not a
    re-sighting. Inside a window-scan the track must prefer the forecast over a coincidence."""
    track = BallTrack(aspect=9 / 16, x=0.5, y=0.25)
    state = track.update([det(0.95, 0.9, 0.1)], step=np.eye(3))
    assert state["status"] == "coasting"
    assert state["u"] == pytest.approx(0.5) and state["v"] == pytest.approx(0.25), "the position is the forecast"
    assert state["conf"] == 0.0, "a coasting position carries no detection confidence"


def test_implausible_sizes_are_ignored() -> None:
    """A box a few pixels across is noise; one a tenth of the frame is a person or a bin, whatever the class says."""
    track = BallTrack(aspect=9 / 16, x=0.5, y=0.25)
    assert track.update([det(0.9, 0.5, 0.25, size=0.001)], step=np.eye(3))["status"] == "coasting"
    track2 = BallTrack(aspect=9 / 16, x=0.5, y=0.25)
    assert track2.update([det(0.9, 0.5, 0.25, size=0.2)], step=np.eye(3))["status"] == "coasting"


def test_the_track_goes_out_of_view_then_lost_then_reacquires() -> None:
    """The out-of-frame story the user described: the gimbal lags, the ball leaves the picture, and the tracker
    must say so rather than keep reporting; a full-frame scan brings it back when the ball returns."""
    step = np.array([[1.0, 0.0, 0.2], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    track = BallTrack(aspect=9 / 16, x=0.5, y=0.25)
    assert track.update([det(0.9, 0.5, 0.25)], step=np.eye(3))["status"] == "tracking"

    statuses = [track.update([], step=step)["status"] for _ in range(3)]
    # 0.7 (coasting), 0.9 (still inside), 1.1 (past the right edge and the margin)
    assert statuses == ["coasting", "coasting", "out_of_view"]
    assert track.update([], step=step)["u"] == pytest.approx(1.3), "an out-of-view forecast still tracks where"

    for _ in range(MAX_COAST):
        track.update([], step=step)
    state = track.update([], step=step)
    assert state["status"] == "lost"
    assert state["u"] is None and state["v"] is None, "a lost track reports no position at all"

    # Only a full-frame scan may restart it, and only on a confident sighting.
    assert track.update([det(0.9, 0.3, 0.2)], step=None, full_frame=False)["status"] == "lost"
    assert track.update([det(0.1, 0.3, 0.2)], step=None, full_frame=True)["status"] == "lost"
    state = track.update([det(0.6, 0.3, 0.2)], step=None, full_frame=True)
    assert state["status"] == "tracking" and state["u"] == pytest.approx(0.3)


def test_a_confident_full_frame_sighting_re_enters_while_coasting() -> None:
    """A long pass: the ball reappears on the far side of the frame while the track is still coasting. The
    window-scan gate would reject it, but the full-frame scan says it is a sighting - take it, and let the next
    frames judge whether it was really the ball."""
    track = BallTrack(aspect=9 / 16, x=0.5, y=0.25)
    track.update([det(0.9, 0.5, 0.25)], step=np.eye(3))
    state = track.update([det(0.8, 0.95, 0.4)], step=np.eye(3), full_frame=True)
    assert state["status"] == "tracking" and state["u"] == pytest.approx(0.95)


def test_a_passed_ball_is_learned_and_stays_in_the_gate() -> None:
    """The measured failure the velocity term exists for: a pass moves half a frame between analysis frames.

    A ball being kicked around the goalmouth (real footage, t=690-694) moves 0.2-0.3 of the frame width in a few
    frames. The camera step cannot explain that - it is camera motion only - so with a static prediction the
    window scan's detections sit past the gate and the track alternates tracking/coasting, re-locking by
    full-frame teleports. Once the jump is folded into a velocity, the prediction carries the ball's motion and
    every later frame is accepted through the normal gate.
    """
    track = BallTrack(aspect=9 / 16, x=0.5, y=0.25)
    track.update([det(0.9, 0.5, 0.25)], step=np.eye(3))  # locked, at rest

    # the kick: 0.08/frame (307 px/frame at 4K), straight past the gate, seen only by a full-frame scan
    state = track.update([det(0.95, 0.58, 0.25)], step=np.eye(3), full_frame=True)
    assert state["status"] == "tracking"
    assert track.vx > 0, "the jump is the ball's motion, and it is usable as a first velocity"
    kick = 0.08
    position = 0.58
    for _ in range(5):
        position += kick
        state = track.update([det(0.95, position, 0.25)], step=np.eye(3))
        assert state["status"] == "tracking", "the learned velocity must keep the pass inside the gate"
    assert track.vx == pytest.approx(kick, abs=0.01)
    # a ball that comes to rest must have its velocity fade, or the prediction runs away from it
    for _ in range(10):
        state = track.update([det(0.95, position + kick * 0.1, 0.25)], step=np.eye(3))
        assert state["status"] == "tracking"
        kick = 0.1 * kick
        position += kick
    assert abs(track.vx) < 0.02, f"velocity should decay to the new reality, got {track.vx}"


def test_velocity_is_capped_and_a_long_absence_does_not_invent_one() -> None:
    """A wild 'jump' (a mis-detection, or a wrong lock) may not turn into an absurd velocity.

    The cap is a sanity bound: 0.2/frame is 768 px at 4K, already faster than the 5 fps window can follow. And a
    re-entry after a *long* absence is a fresh sighting, not a measurement of motion - it must not fabricate one.
    """
    track = BallTrack(aspect=9 / 16, x=0.5, y=0.25)
    track.update([det(0.9, 0.5, 0.25)], step=np.eye(3))
    track.update([det(0.9, 0.5, 0.9)], step=np.eye(3), full_frame=True)  # nonsense jump in v
    assert np.hypot(track.vx, track.vy) <= BallTrack(aspect=9 / 16).max_speed + 1e-9

    other = BallTrack(aspect=9 / 16, x=0.5, y=0.25)
    other.update([det(0.9, 0.5, 0.25)], step=np.eye(3))
    for _ in range(other.max_coast + 2):
        other.update([], step=np.eye(3))  # long absence: the track goes lost
    state = other.update([det(0.9, 0.7, 0.25)], step=np.eye(3), full_frame=True)
    assert state["status"] == "tracking"
    assert other.vx == 0.0 and other.vy == 0.0, "a jump across a long gap is not a velocity"
