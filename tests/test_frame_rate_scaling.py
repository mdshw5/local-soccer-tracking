"""The analysis rate is a setting, not an assumption: every frame-count window in the pipeline is a duration.

Stage A now runs at 15 fps by default (both 60 and 30 fps sources divide by it), while older segments were built
at 5 fps - and most synthetic fixtures in this suite still are. These tests pin the two invariants that makes
safe: the same wall-clock behavior at any rate, and the tuned 5 fps behavior unchanged at the reference rate.
"""

from __future__ import annotations

import numpy as np
import pytest

from soccer_analytics.analysis.ball import MAX_COAST, BallTrack
from soccer_analytics.analysis.event_detection import BallMotion, _lag_frames
from soccer_analytics.analysis.projection import PitchDetections
from soccer_analytics.analysis.stage_a import ANALYSIS_FPS, SegmentConfig, tracker_config
from soccer_analytics.analysis.stage_b import DEFAULT_RATE, _stitch_tracks, _track_people, detection_rate

KIT_A = [0.5, 1.0, 0.0, 0.0, 0.2, 0.9]  # weight + (L, a, b, sat, val)


def _detections(rows: list[tuple[int, float, float, list[float]]], rate: float) -> PitchDetections:
    frame = np.array([r[0] for r in rows], dtype=np.int32)
    xy = np.array([[r[1], r[2]] for r in rows], dtype=np.float64)
    kit = np.array([r[3] for r in rows], dtype=np.float64)
    n = len(rows)
    return PitchDetections(
        frame=frame,
        time=frame / rate,
        xy=xy,
        sigma_m=np.full(n, 0.3),
        valid=np.ones(n, dtype=bool),
        height_px=np.full(n, 200.0),
        box=np.full((n, 4), 0.25),
        conf=np.full(n, 0.9),
        kit=kit,
        det_track=np.full(n, -1, dtype=np.int32),
        det_index=np.arange(n),
        aim_xy=np.full((int(frame.max()) + 1, 2), np.nan),
        camera_xy=np.zeros(2),
    )


def _chain(x0: float, frames: range, step: float = 0.4) -> list[tuple[int, float, float, list[float]]]:
    return [(f, x0 + step * i, 10.0, KIT_A) for i, f in enumerate(frames)]


def test_the_analysis_default_is_15_fps() -> None:
    assert ANALYSIS_FPS == 15.0
    assert SegmentConfig().fps == ANALYSIS_FPS


def test_bot_sort_buffer_is_six_seconds() -> None:
    assert tracker_config(5.0)["track_buffer"] == 30, "the tuned 5 fps value is untouched"
    assert tracker_config(15.0)["track_buffer"] == 90
    assert tracker_config(7.5)["track_buffer"] == 45


def test_ball_windows_scale_with_rate() -> None:
    slow = BallTrack(aspect=9 / 16)
    fast = BallTrack(aspect=9 / 16, rate=15.0)
    assert slow.max_coast == MAX_COAST == 12
    assert fast.max_coast == 36
    assert slow.reentry_velocity_frames == 8
    assert fast.reentry_velocity_frames == 24
    assert fast.max_speed == pytest.approx(slow.max_speed / 3.0)
    # The EMA keeps its time constant rather than adapting three times faster: (1 - g) ^ (rate ratio) is invariant.
    assert (1.0 - fast.velocity_gain) ** (fast.rate / slow.rate) == pytest.approx(1.0 - slow.velocity_gain)


def test_ball_coast_window_is_seconds_not_frames() -> None:
    for rate in (5.0, 15.0):
        track = BallTrack(aspect=9 / 16, rate=rate, x=0.5, y=0.25)
        track.update([(0.9, 0.5, 0.25, 0.01, 0.01)], step=np.eye(3))
        for _ in range(int(2.4 * rate) - 1):
            state = track.update([], step=None)
        assert state["status"] == "coasting", f"still inside the 2.4 s coast at {rate} fps"
        for _ in range(3):
            state = track.update([], step=None)
        assert state["status"] == "lost", f"past the 2.4 s coast at {rate} fps"


def test_ball_rate_survives_a_checkpoint() -> None:
    track = BallTrack(aspect=9 / 16, rate=15.0)
    again = BallTrack.from_json(track.to_json(), aspect=9 / 16)
    assert again.rate == 15.0
    assert again.max_coast == 36, "the resumed track derives its windows at the stored rate"


def test_detection_rate_reads_the_timestamps() -> None:
    at_15 = _detections([(i, 10.0, 10.0, KIT_A) for i in range(30)], rate=15.0)
    assert detection_rate(at_15) == pytest.approx(15.0)
    at_5 = _detections([(i, 10.0, 10.0, KIT_A) for i in range(30)], rate=5.0)
    assert detection_rate(at_5) == pytest.approx(5.0)
    single = _detections([(0, 10.0, 10.0, KIT_A)], rate=15.0)
    assert detection_rate(single) == DEFAULT_RATE, "no timestamps to read: the historical rate"


@pytest.mark.parametrize("rate", [5.0, 15.0])
def test_the_stitching_window_is_six_seconds(rate: float) -> None:
    """A 6 s frame gap stitches at either rate; a 6.5 s one does not - frames are not the unit, seconds are."""
    gap = int(6.0 * rate)
    rows = _chain(10.0, range(0, 8)) + _chain(13.0, range(7 + gap, 15 + gap))
    detections = _detections(rows, rate=rate)
    assignment = _track_people(detections, np.ones(len(rows), dtype=bool))
    assert len(assignment.tracks) == 2, "the online buffer (3 s) must have let the fragment die first"
    assert len(_stitch_tracks(detections, assignment).tracks) == 1

    gap = int(6.5 * rate)
    rows = _chain(10.0, range(0, 8)) + _chain(13.0, range(7 + gap, 15 + gap))
    detections = _detections(rows, rate=rate)
    assignment = _track_people(detections, np.ones(len(rows), dtype=bool))
    assert len(_stitch_tracks(detections, assignment).tracks) == 2


def test_the_online_gate_is_a_speed_not_a_distance() -> None:
    """1.2 m between consecutive samples is a 6 m/s run at 5 fps - and an 18 m/s teleport at 15 fps."""
    slow = _detections(_chain(10.0, range(0, 6), step=1.2), rate=5.0)
    assert len(_track_people(slow, np.ones(6, dtype=bool)).tracks) == 1

    fast = _detections(_chain(10.0, range(0, 6), step=1.2), rate=15.0)
    assert len(_track_people(fast, np.ones(6, dtype=bool)).tracks) > 1

    sprint = _detections(_chain(10.0, range(0, 6), step=0.45), rate=15.0)
    assert len(_track_people(sprint, np.ones(6, dtype=bool)).tracks) == 1


def test_attribution_lag_scales_with_rate() -> None:
    def motion(rate: float) -> BallMotion:
        n = int(10 * rate)
        times = np.arange(n) / rate
        blank = np.full(n, np.nan)
        return BallMotion(
            xy=np.full((n, 2), np.nan),
            measured=np.zeros(n),
            times=times,
            speed=blank,
            vx=blank,
            vy=blank,
            straight=blank,
        )

    assert _lag_frames(motion(5.0)) == 2, "0.4 s is the couple of frames the detectors were tuned with"
    assert _lag_frames(motion(15.0)) == 6
