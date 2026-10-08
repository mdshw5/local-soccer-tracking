"""Refining the camera-motion chain: the estimator choice, and that only the motion arrays change.

The refinement exists because the analysis pass prefers sparse optical flow (LK) and only falls back to descriptor
matching (SIFT) when LK *fails*. On a fast pan LK's linearisation under-estimates the rotation while still
"succeeding", so the fallback never runs and the error is integrated into the chain - the projected pitch then lags
the real markings. The refinement re-estimates each step, preferring the descriptor match whenever the motion is
large enough for LK's linearisation to matter.

Two things are worth pinning. The estimator choice is pure logic and easy to get backwards. And the write-back must
touch *only* the motion arrays - times, detections and kit descriptors are what the rest of the pipeline reads, and
a refinement that perturbed them would silently invalidate a report built from the segment.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

from soccer_analytics.analysis.stage_a import SegmentConfig, analyse_segment, load_segment
from soccer_analytics.geometry.camera_motion import MotionStep
from soccer_analytics.ingest.video_reader import VideoWriter

REPO_ROOT = Path(__file__).resolve().parents[1]
W, H = 320, 180


def _script():  # noqa: ANN202 - loaded from scripts/ without making it a package
    spec = importlib.util.spec_from_file_location(
        "refine_camera_motion", REPO_ROOT / "scripts" / "refine_camera_motion.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _step(inliers: int, ratio: float, method: str = "lk") -> MotionStep:
    return MotionStep(np.eye(3), ratio, inliers, method)


def test_choose_step_prefers_lk_for_small_motion() -> None:
    """Small motion: LK is accurate and cheap, so it is used as-is even when a descriptor match is available."""
    module = _script()
    lk, sift = _step(200, 0.9, "lk"), _step(200, 0.9, "sift")
    chosen, source = module.choose_step(lk, sift, lk_deg=0.2, large_motion_deg=1.5)
    assert chosen is lk and source == "lk"


def test_choose_step_prefers_sift_for_large_motion() -> None:
    """Large motion: LK's linearisation under-estimates the rotation, so the descriptor match wins."""
    module = _script()
    lk, sift = _step(200, 0.9, "lk"), _step(200, 0.9, "sift")
    chosen, source = module.choose_step(lk, sift, lk_deg=8.0, large_motion_deg=1.5)
    assert chosen is sift and source == "sift"


def test_choose_step_falls_back_to_lk_when_sift_is_unusable() -> None:
    """A descriptor match with too little support is not evidence; a usable LK step still is."""
    module = _script()
    lk, sift = _step(200, 0.9, "lk"), _step(5, 0.05, "sift")
    chosen, source = module.choose_step(lk, sift, lk_deg=8.0, large_motion_deg=1.5)
    assert chosen is lk and source == "lk"


def test_choose_step_reports_lost_when_neither_is_usable() -> None:
    module = _script()
    chosen, source = module.choose_step(_step(5, 0.05), _step(5, 0.05, "sift"), lk_deg=8.0, large_motion_deg=1.5)
    assert chosen is None and source == "lost"


def test_validate_step_rejects_a_step_that_is_not_a_rotation() -> None:
    """A shear is not something a pan/tilt/zoom lens can produce, and accepting it lets the chain diverge.

    This is the bug that made the first full refinement worse than the chain it replaced: the loop skipped the
    tracker's spread/focal gate, so invalid steps were accepted and the error accumulated over thousands of frames.
    """
    module = _script()
    aspect = 9 / 16
    shear = np.array([[1.0, 0.4, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    assert module.validate_step(shear, 0.82, aspect) is None


def test_validate_step_accepts_a_pure_rotation() -> None:
    module = _script()
    aspect = 9 / 16
    rotation = np.eye(3)
    result = module.validate_step(rotation, 0.82, aspect)
    assert result is not None
    assert result.focal == pytest.approx(0.82, rel=1e-6)


def _segment(tmp_path: Path):  # noqa: ANN202
    """A tiny analysed segment: a moving square so the motion estimator has something to track."""
    video = tmp_path / "clip.mp4"
    writer = VideoWriter(str(video), fps=10.0, width=W, height=H)
    for index in range(40):
        frame = np.zeros((H, W, 3), dtype=np.uint8)
        x = 20 + index * 3
        frame[60:120, x : x + 40] = 255
        writer.write(frame)
    writer.close()
    out = tmp_path / "segment"
    analyse_segment(video, out, config=SegmentConfig(fps=10.0, detect_width=W, motion_width=W, chunk_frames=20))
    return out


def test_write_back_changes_only_the_motion_arrays(tmp_path: Path) -> None:
    """The refinement must not disturb times, detections or kit descriptors - only step/focal/ok."""
    module = _script()
    out = _segment(tmp_path)
    before = load_segment(out)

    steps = [None if i == 0 else np.eye(3) for i in range(len(before.time))]
    focals = np.full(len(before.time), 0.9, dtype=np.float64)
    ok = np.ones(len(before.time), dtype=bool)
    module._write_back(out, steps, focals, ok)

    after = load_segment(out)
    assert np.allclose(after.time, before.time)
    assert np.array_equal(after.det_frame, before.det_frame)
    assert np.allclose(after.det_box, before.det_box)
    assert np.allclose(after.det_kit, before.det_kit)
    assert np.allclose(after.focal, 0.9)
    assert after.ok.all()
    # The reference frame carries no motion, so its step is the identity the writer substitutes.
    assert np.allclose(after.step[0], np.eye(3))