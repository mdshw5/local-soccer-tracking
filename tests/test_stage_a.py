"""Stage A: resume must reproduce the uninterrupted camera chain, and failures must be visible."""

from __future__ import annotations

import shutil
from pathlib import Path

import cv2
import numpy as np
import pytest

from soccer_analytics.analysis.stage_a import (
    SegmentConfig,
    analyse_segment,
    completed_chunks,
    load_segment,
    read_status,
    resolve_window,
    segment_dir_for,
)
from soccer_analytics.geometry.camera_motion import integrate_steps
from soccer_analytics.ingest.video_reader import VideoWriter

W, H = 640, 360


def test_a_window_label_gets_its_own_segment_directory(tmp_path: Path) -> None:
    """The two halves of a game are analysed separately, so they cannot share one directory."""
    video = tmp_path / "game.mp4"
    video.write_bytes(b"x" * 8)

    plain = segment_dir_for(video, tmp_path / "out")
    first = segment_dir_for(video, tmp_path / "out", window_label="first_half_120_3000")
    second = segment_dir_for(video, tmp_path / "out", window_label="second_half_3000_5900")

    assert len({plain, first, second}) == 3
    assert first.name == f"{plain.name}__first_half_120_3000"


class _FakeBoxes:
    def __init__(self, xyxy: np.ndarray, conf: np.ndarray):
        self.xyxy, self.conf = _T(xyxy), _T(conf)


class _T:
    def __init__(self, a: np.ndarray):
        self.a = a

    def cpu(self):
        return self

    def numpy(self):
        return self.a


class _FakeResult:
    def __init__(self, xyxy, conf):
        self.boxes = _FakeBoxes(np.asarray(xyxy, dtype=np.float32).reshape(-1, 4), np.asarray(conf, dtype=np.float32))


class _FakeModel:
    """Finds nothing but is called once per frame; records how many frames it saw."""

    def __init__(self, boxes_per_frame=((120.0, 80.0, 150.0, 200.0),)):
        self.calls = 0
        self.boxes = boxes_per_frame

    def predict(self, frame, **kwargs):
        self.calls += 1
        w = frame.shape[1]
        scaled = [(x1 * w / 1920, y1 * w / 1920, x2 * w / 1920, y2 * w / 1920) for x1, y1, x2, y2 in self.boxes]
        return [_FakeResult(scaled, [0.9] * len(scaled))]


def _scene() -> np.ndarray:
    rng = np.random.default_rng(11)
    layers = [cv2.GaussianBlur(rng.random((1400, 3000)).astype(np.float32), (0, 0), s) for s in (1.5, 4.0, 12.0)]
    img = sum(layer / layer.std() for layer in layers)
    return cv2.cvtColor(((img - img.min()) / (img.max() - img.min()) * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)


def test_resolve_window_zero_or_none_means_to_the_end_of_the_video() -> None:
    """The dashboard's default length is 0: pick a video, press run, analyse the whole thing."""
    assert resolve_window(0.0, 0.0, 1254.5) == (0.0, 1254.5)
    assert resolve_window(120.0, None, 1254.5) == (120.0, 1254.5)
    # a positive length is the ordinary window, clamped to the end of the video
    assert resolve_window(120.0, 300.0, 1254.5) == (120.0, 420.0)
    assert resolve_window(1200.0, 300.0, 1254.5) == (1200.0, 1254.5)
    # offsets outside the video degrade to a zero-length window rather than a negative one
    assert resolve_window(-5.0, 0.0, 100.0) == (0.0, 100.0)
    assert resolve_window(200.0, 0.0, 100.0) == (200.0, 200.0)


@pytest.fixture(scope="module")
def panning_video(tmp_path_factory) -> Path:
    """12 s, 10 fps, camera panning right by ~14 px per frame over a textured world."""
    path = tmp_path_factory.mktemp("clip") / "pan.mp4"
    scene = _scene()
    with VideoWriter(path, fps=10.0, width=W, height=H) as writer:
        for i in range(120):
            x = 200 + 14 * i
            writer.write(scene[200 : 200 + H, x : x + W].copy())
    return path


CONFIG = SegmentConfig(fps=5.0, motion_width=320, detect_width=640, chunk_frames=15, device="cpu")


def _chain_from(directory: Path):
    data = load_segment(directory)
    steps = [None if (not ok or i == 0) else s for i, (ok, s) in enumerate(zip(data.ok, data.step))]
    return data, integrate_steps(steps, data.meta["default_focal"], data.aspect)


def test_full_run_writes_chunks_status_and_detections(panning_video: Path, tmp_path: Path) -> None:
    out = tmp_path / "full"
    status = analyse_segment(panning_video, out, config=CONFIG, model=_FakeModel())
    assert status["state"] == "done"
    assert completed_chunks(out) == 4  # 60 frames / 15
    data = load_segment(out)
    assert len(data.time) == 60 and data.ok.all()
    assert len(data.det_frame) == 60  # one fake person per frame
    assert data.det_kit.shape == (60, 12)
    assert np.all(np.diff(data.time) > 0)
    assert read_status(out)["state"] == "done"


def test_zero_duration_analyses_the_whole_clip(panning_video: Path, tmp_path: Path) -> None:
    """The dashboard's default length is 0 - "to the end of the video", never "nothing"."""
    out = tmp_path / "whole"
    events: list[dict] = []
    status = analyse_segment(
        panning_video, out, config=CONFIG, model=_FakeModel(), duration_s=0.0, on_progress=events.append
    )
    assert status["state"] == "done"
    assert status["total_frames"] == 60  # the whole 12 s clip at 5 fps
    assert completed_chunks(out) == 4
    assert events and events[-1]["state"] == "done"


def test_resumed_run_reproduces_the_uninterrupted_camera_chain(panning_video: Path, tmp_path: Path) -> None:
    straight = tmp_path / "straight"
    analyse_segment(panning_video, straight, config=CONFIG, model=_FakeModel())
    _, chain_straight = _chain_from(straight)

    resumed = tmp_path / "resumed"
    seen = {"n": 0}

    def stop_after_two_chunks() -> bool:
        seen["n"] += 1
        return seen["n"] > 2 * CONFIG.chunk_frames + 3  # interrupt just into the third chunk

    status = analyse_segment(panning_video, resumed, config=CONFIG, model=_FakeModel(), should_stop=stop_after_two_chunks)
    assert status["state"] == "stopped"
    assert completed_chunks(resumed) == 2

    final = analyse_segment(panning_video, resumed, config=CONFIG, model=_FakeModel())
    assert final["state"] == "done"
    data, chain_resumed = _chain_from(resumed)
    assert len(chain_resumed) == len(chain_straight) == 60
    # Every frame, including those just after the chunk boundaries, must land where the straight run put it.
    worst = max(float(np.abs(a - b).max()) for a, b in zip(chain_resumed, chain_straight))
    assert worst < 5e-3, f"resumed chain diverged by {worst}"
    assert data.ok.all()


def test_a_stopped_run_can_be_inspected_and_finished_later(panning_video: Path, tmp_path: Path) -> None:
    out = tmp_path / "stopped"
    calls = {"n": 0}
    analyse_segment(panning_video, out, config=CONFIG, model=_FakeModel(), should_stop=lambda: (calls.__setitem__("n", calls["n"] + 1) or calls["n"] > 20))
    assert read_status(out)["state"] == "stopped"
    assert completed_chunks(out) == 1
    partial = load_segment(out)  # partial results are readable while the job is incomplete
    assert len(partial.time) == 15


def test_torn_chunk_is_not_counted_and_is_redone(panning_video: Path, tmp_path: Path) -> None:
    out = tmp_path / "torn"
    analyse_segment(panning_video, out, config=CONFIG, model=_FakeModel())
    (out / "chunk_00002.npz").write_bytes(b"not a real npz")  # simulate a crash mid-write
    assert completed_chunks(out) == 2
    analyse_segment(panning_video, out, config=CONFIG, model=_FakeModel())
    assert completed_chunks(out) == 4
    assert len(load_segment(out).time) == 60


def test_mismatched_settings_refuse_to_mix_results(panning_video: Path, tmp_path: Path) -> None:
    out = tmp_path / "mixed"
    analyse_segment(panning_video, out, config=CONFIG, model=_FakeModel())
    other = SegmentConfig(fps=4.0, motion_width=320, detect_width=640, chunk_frames=15, device="cpu")
    with pytest.raises(ValueError, match="different settings"):
        analyse_segment(panning_video, out, config=other, model=_FakeModel())


def test_detector_failure_is_recorded_not_swallowed(panning_video: Path, tmp_path: Path) -> None:
    class Boom(_FakeModel):
        def predict(self, frame, **kwargs):
            raise RuntimeError("CUDA out of memory")

    out = tmp_path / "boom"
    with pytest.raises(RuntimeError, match="CUDA"):
        analyse_segment(panning_video, out, config=CONFIG, model=Boom())
    status = read_status(out)
    assert status["state"] == "error" and "CUDA out of memory" in status["error"]


def test_overlay_detections_are_dropped(panning_video: Path, tmp_path: Path) -> None:
    # A box whose feet sit inside the bottom-right logo rectangle (fraction 0.85-0.99 x, 0.89-0.98 y of the frame).
    logo_box = ((1700.0, 900.0, 1790.0, 1030.0),)  # at 1920 wide / 1080 tall: foot at y~1029 (0.95), x~1745 (0.91)
    out = tmp_path / "logo"
    analyse_segment(panning_video, out, config=CONFIG, model=_FakeModel(logo_box))
    assert len(load_segment(out).det_frame) == 0


def test_segment_dir_changes_when_the_file_changes(tmp_path: Path) -> None:
    video = tmp_path / "a.mp4"
    video.write_bytes(b"x" * 10)
    first = segment_dir_for(video, tmp_path / "out")
    video.write_bytes(b"x" * 11)
    assert segment_dir_for(video, tmp_path / "out") != first
    shutil.rmtree(tmp_path / "out", ignore_errors=True)
