"""Stage A: resume must reproduce the uninterrupted camera chain, and failures must be visible."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import cv2
import numpy as np
import pytest

from soccer_analytics.analysis.stage_a import (
    SegmentConfig,
    analyze_segment,
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
    """The two halves of a game are analyzed separately, so they cannot share one directory."""
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

    def track(self, frame, **kwargs):
        """Same boxes as predict, with a stable id per box position (what BoT-SORT approximates)."""
        self.calls += 1
        w = frame.shape[1]
        scaled = [(x1 * w / 1920, y1 * w / 1920, x2 * w / 1920, y2 * w / 1920) for x1, y1, x2, y2 in self.boxes]
        result = _FakeResult(scaled, [0.9] * len(scaled))
        result.boxes.id = _T(np.arange(len(scaled), dtype=np.float32))
        return [result]


def _scene() -> np.ndarray:
    rng = np.random.default_rng(11)
    layers = [cv2.GaussianBlur(rng.random((1400, 3000)).astype(np.float32), (0, 0), s) for s in (1.5, 4.0, 12.0)]
    img = sum(layer / layer.std() for layer in layers)
    return cv2.cvtColor(((img - img.min()) / (img.max() - img.min()) * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)


def test_resolve_window_zero_or_none_means_to_the_end_of_the_video() -> None:
    """The dashboard's default length is 0: pick a video, press run, analyze the whole thing."""
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
    status = analyze_segment(panning_video, out, config=CONFIG, model=_FakeModel())
    assert status["state"] == "done"
    assert completed_chunks(out) == 4  # 60 frames / 15
    data = load_segment(out)
    assert len(data.time) == 60 and data.ok.all()
    assert len(data.det_frame) == 60  # one fake person per frame
    assert data.det_kit.shape == (60, 12)
    assert np.all(np.diff(data.time) > 0)
    assert read_status(out)["state"] == "done"


def test_detect_width_zero_is_full_resolution(panning_video: Path, tmp_path: Path) -> None:
    """0 asks for the source's own width: what "4K detection" means for a 4K recording.

    The resolved width has to reach both the detector's frames and ``meta`` - the dashboard's re-run conflict
    check, the kit refresh and every resume read the stored number, so a sentinel left unresolved would make them
    disagree with the frames the boxes were found in.
    """

    class _RecordingFake(_FakeModel):
        def __init__(self) -> None:
            super().__init__()
            self.widths: list[int] = []

        def track(self, frame, **kwargs):  # noqa: ANN001, ANN003
            self.widths.append(frame.shape[1])
            return super().track(frame, **kwargs)

    out = tmp_path / "native"
    model = _RecordingFake()
    config = SegmentConfig(fps=5.0, motion_width=320, detect_width=0, chunk_frames=15, device="cpu")

    status = analyze_segment(panning_video, out, config=config, model=model)

    meta = json.loads((out / "meta.json").read_text())
    assert status["state"] == "done"
    assert meta["detect_width"] == W  # the clip's own width, not the old 1920 upsample
    assert set(model.widths) == {W}, "the detector must see frames at the resolved width"


def test_zero_duration_analyzes_the_whole_clip(panning_video: Path, tmp_path: Path) -> None:
    """The dashboard's default length is 0 - "to the end of the video", never "nothing"."""
    out = tmp_path / "whole"
    events: list[dict] = []
    status = analyze_segment(
        panning_video, out, config=CONFIG, model=_FakeModel(), duration_s=0.0, on_progress=events.append
    )
    assert status["state"] == "done"
    assert status["total_frames"] == 60  # the whole 12 s clip at 5 fps
    assert completed_chunks(out) == 4
    assert events and events[-1]["state"] == "done"


def test_resumed_run_reproduces_the_uninterrupted_camera_chain(panning_video: Path, tmp_path: Path) -> None:
    straight = tmp_path / "straight"
    analyze_segment(panning_video, straight, config=CONFIG, model=_FakeModel())
    _, chain_straight = _chain_from(straight)

    resumed = tmp_path / "resumed"
    seen = {"n": 0}

    def stop_after_two_chunks() -> bool:
        seen["n"] += 1
        return seen["n"] > 2 * CONFIG.chunk_frames + 3  # interrupt just into the third chunk

    status = analyze_segment(panning_video, resumed, config=CONFIG, model=_FakeModel(), should_stop=stop_after_two_chunks)
    assert status["state"] == "stopped"
    assert completed_chunks(resumed) == 2

    final = analyze_segment(panning_video, resumed, config=CONFIG, model=_FakeModel())
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
    analyze_segment(panning_video, out, config=CONFIG, model=_FakeModel(), should_stop=lambda: (calls.__setitem__("n", calls["n"] + 1) or calls["n"] > 20))
    assert read_status(out)["state"] == "stopped"
    assert completed_chunks(out) == 1
    partial = load_segment(out)  # partial results are readable while the job is incomplete
    assert len(partial.time) == 15


def test_torn_chunk_is_not_counted_and_is_redone(panning_video: Path, tmp_path: Path) -> None:
    out = tmp_path / "torn"
    analyze_segment(panning_video, out, config=CONFIG, model=_FakeModel())
    (out / "chunk_00002.npz").write_bytes(b"not a real npz")  # simulate a crash mid-write
    assert completed_chunks(out) == 2
    analyze_segment(panning_video, out, config=CONFIG, model=_FakeModel())
    assert completed_chunks(out) == 4
    assert len(load_segment(out).time) == 60


def test_mismatched_settings_refuse_to_mix_results(panning_video: Path, tmp_path: Path) -> None:
    out = tmp_path / "mixed"
    analyze_segment(panning_video, out, config=CONFIG, model=_FakeModel())
    other = SegmentConfig(fps=4.0, motion_width=320, detect_width=640, chunk_frames=15, device="cpu")
    with pytest.raises(ValueError, match="different settings"):
        analyze_segment(panning_video, out, config=other, model=_FakeModel())


def test_detector_failure_is_recorded_not_swallowed(panning_video: Path, tmp_path: Path) -> None:
    class Boom(_FakeModel):
        def track(self, frame, **kwargs):
            raise RuntimeError("CUDA out of memory")

    out = tmp_path / "boom"
    with pytest.raises(RuntimeError, match="CUDA"):
        analyze_segment(panning_video, out, config=CONFIG, model=Boom())
    status = read_status(out)
    assert status["state"] == "error" and "CUDA out of memory" in status["error"]


def test_overlay_detections_are_dropped(panning_video: Path, tmp_path: Path) -> None:
    # A box whose feet sit inside the bottom-right logo rectangle (fraction 0.85-0.99 x, 0.89-0.98 y of the frame).
    logo_box = ((1700.0, 900.0, 1790.0, 1030.0),)  # at 1920 wide / 1080 tall: foot at y~1029 (0.95), x~1745 (0.91)
    out = tmp_path / "logo"
    analyze_segment(panning_video, out, config=CONFIG, model=_FakeModel(logo_box))
    assert len(load_segment(out).det_frame) == 0


def test_segment_dir_changes_when_the_file_changes(tmp_path: Path) -> None:
    video = tmp_path / "a.mp4"
    video.write_bytes(b"x" * 10)
    first = segment_dir_for(video, tmp_path / "out")
    video.write_bytes(b"x" * 11)
    assert segment_dir_for(video, tmp_path / "out") != first
    shutil.rmtree(tmp_path / "out", ignore_errors=True)


def _write_segment(directory: Path, video: str, frames: int = 3) -> None:
    """A minimal loadable segment: the meta plus one complete, detection-free chunk."""
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "meta.json").write_text(
        json.dumps(
            {
                "schema": 2,
                "video": video,
                "width": 640,
                "height": 360,
                "fps": 5.0,
                "motion_width": 320,
                "detect_width": 640,
                "chunk_frames": frames,
                "total_chunks": 1,
                "default_focal": 0.8,
            }
        )
    )
    np.savez(
        directory / "chunk_00000.npz",
        time=np.arange(frames) / 5.0,
        ok=np.ones(frames, dtype=bool),
        inlier=np.ones(frames, dtype=np.float32),
        step=np.tile(np.eye(3), (frames, 1, 1)),
        focal=np.full(frames, 0.8, dtype=np.float32),
        det_frame=np.zeros(0, dtype=np.int32),
        det_box=np.zeros((0, 4), dtype=np.float32),
        det_conf=np.zeros(0, dtype=np.float32),
        det_kit=np.zeros((0, 12), dtype=np.float32),
        det_track=np.zeros(0, dtype=np.int32),
    )


def test_load_segment_repoints_a_video_path_after_the_footage_tree_moved(tmp_path: Path) -> None:
    """A stored absolute path may not resolve on the machine reading the results; the sibling by name does.

    ``meta.json`` records the source as the analyzing machine saw it, so an archive copied to another root - or
    a server that mounts the same footage elsewhere - leaves every stored path carrying the old prefix. The same
    file name always sits beside the segment: a game's ``game.json`` in the analysis directory one level up, the
    combined video at the footage root three levels up. Loading re-points to those, and keeps the stored path
    when nothing matches rather than blanking it.
    """
    footage = tmp_path / "footage"
    analysis = footage / "analysis" / "2026-10-03_game_x"
    segment = analysis / "segments" / "2026-10-03_game_x__whole_game_0_10"

    # The never-merged workflow's manifest lives in the analysis directory; the stored path was written at a
    # root that does not exist here.
    _write_segment(segment, str(tmp_path / "old_root" / "analysis" / "2026-10-03_game_x" / "game.json"))
    manifest = analysis / "game.json"
    manifest.write_text("{}")
    assert Path(load_segment(segment).meta["video"]).resolve() == manifest.resolve()

    # The old workflow's combined video lives at the footage root.
    manifest.unlink()
    combined = footage / "game_16-28-37.784.mp4"
    combined.write_bytes(b"x")
    _write_segment(segment, str(tmp_path / "old_root" / "2026-10-03" / "game_16-28-37.784.mp4"))
    assert Path(load_segment(segment).meta["video"]).resolve() == combined.resolve()

    # Nothing near the segment carries the name: the stored path survives untouched.
    stored = str(tmp_path / "gone" / "raw_clip.MP4")
    _write_segment(segment, stored)
    assert load_segment(segment).meta["video"] == stored
