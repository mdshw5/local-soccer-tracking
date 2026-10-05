"""Progress reporting: the fractions behind the progress bars must be monotone, bounded, and honest.

The bars are only as good as these callbacks - a pipeline that reports nothing leaves the user staring at a spinner
for minutes on a full video, and fractions that jump backwards are worse than none. The ffmpeg side is covered where
the export tests live (they exercise the real encoder).
"""

from __future__ import annotations

import numpy as np
import pytest

from soccer_analytics.analysis import stage_b
from soccer_analytics.analysis.events import detect_whistles
from soccer_analytics.analysis.projection import project_segment, segment_poses
from soccer_analytics.geometry.pitch_calibration import PitchCalibration
from synthetic_match import PITCH_LENGTH, PITCH_WIDTH, simulate_match


def _calibration(segment, truth) -> tuple[PitchCalibration, tuple]:
    q, focal = segment_poses(segment, focal0=float(truth.focal[0]))
    calibration = PitchCalibration(
        truth.calibration.position,
        truth.calibration.base_rotation @ truth.q[0],
        truth.calibration.focal_scale,
        truth.calibration.aspect,
        0.0,
        (),
    )
    return calibration, (q, focal)


def _collector() -> tuple[list[float], object]:
    calls: list[float] = []
    return calls, lambda fraction: calls.append(float(fraction))


def _assert_usable(calls: list[float]) -> None:
    assert len(calls) >= 2, "a long task that never reports is a spinner in disguise"
    assert all(0.0 <= call <= 1.0 for call in calls), "fractions must be bounded"
    assert calls == sorted(calls), "a progress bar must not jump backwards"
    assert calls[-1] == pytest.approx(1.0), "the last report must be the finished state"


def test_project_segment_reports_a_complete_monotone_ramp() -> None:
    segment, truth = simulate_match(frames=120, seed=2)
    calibration, poses = _calibration(segment, truth)
    calls, callback = _collector()
    project_segment(segment, calibration, poses=poses, on_progress=callback)
    _assert_usable(calls)


def test_detect_whistles_reports_a_complete_monotone_ramp() -> None:
    """The whistle transform is the whole cost of an audio scan, so it reports as it goes.

    The signal is long enough to need more than one transform block, which is what the block loop exists for - a
    single block would report once and the bar would be a spinner.
    """
    rng = np.random.default_rng(5)
    time = np.arange(int(70.0 * 16000)) / 16000
    audio = (0.2 * np.sin(2 * np.pi * 700.0 * time) + rng.normal(0.0, 0.05, time.size)).astype(np.float32)
    calls, callback = _collector()
    detect_whistles(audio, 16000, on_progress=callback)
    _assert_usable(calls)


def test_build_report_reports_a_complete_monotone_ramp() -> None:
    segment, truth = simulate_match(frames=120, seed=2)
    calibration, poses = _calibration(segment, truth)
    detections = project_segment(segment, calibration, poses=poses)
    calls, callback = _collector()
    stage_b.build_report(
        detections,
        pitch_length_m=PITCH_LENGTH,
        pitch_width_m=PITCH_WIDTH,
        match_frames=len(segment.time),
        on_progress=callback,
    )
    _assert_usable(calls)


def test_progress_callbacks_are_optional() -> None:
    """Every caller that does not care (tests, scripts) keeps working without a callback."""
    segment, truth = simulate_match(frames=60, seed=3)
    calibration, poses = _calibration(segment, truth)
    detections = project_segment(segment, calibration, poses=poses)
    report, _assignment = stage_b.build_report(
        detections, pitch_length_m=PITCH_LENGTH, pitch_width_m=PITCH_WIDTH, match_frames=len(segment.time)
    )
    assert report.frames_analysed == len(segment.time)
