"""Refreshing the stored kit descriptors must not disturb anything else in the segment.

The refresh exists because the kit descriptor depends only on a frame and a stored box - so a change to the grass
mask can be applied to an analysed segment without redoing detection or camera motion. That claim is the whole
justification for the script, and it is falsifiable: if the rewrite perturbed the poses, the steps, the boxes or
the confidences, the segment would silently stop matching the report built from it. So these tests pin that the
only array that changes is ``det_kit`` - and that the refresh is idempotent, which is what makes it safe to re-run
after any further change to the masking.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from soccer_analytics.analysis.stage_a import SegmentConfig, analyse_segment, load_segment
from soccer_analytics.ingest.video_reader import VideoWriter

REPO_ROOT = Path(__file__).resolve().parents[1]
W, H = 320, 180


def _script():  # noqa: ANN202 - loaded from scripts/ without making it a package
    spec = importlib.util.spec_from_file_location(
        "refresh_kit_descriptors", REPO_ROOT / "scripts" / "refresh_kit_descriptors.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _T:
    """The `.cpu().numpy()` chain ultralytics boxes expose."""

    def __init__(self, array: np.ndarray):
        self.array = array

    def cpu(self):
        return self

    def numpy(self):
        return self.array


class _FakeBoxes:
    def __init__(self, xyxy: np.ndarray, conf: np.ndarray):
        self.xyxy, self.conf = _T(xyxy), _T(conf)


class _FakeResult:
    def __init__(self, xyxy, conf):
        self.boxes = _FakeBoxes(np.asarray(xyxy, dtype=np.float32).reshape(-1, 4), np.asarray(conf, dtype=np.float32))


class _FakeModel:
    """One person per frame, so every chunk has a detection whose descriptor can be recomputed."""

    def predict(self, frame, **kwargs):  # noqa: ANN001, ANN003
        width = frame.shape[1]
        # Boxes are given at 1920 and scaled to the frame, exactly as ultralytics reports them.
        boxes = [(0.0, 0.0, 10.0 * width / 1920, 40.0 * width / 1920)]
        return [_FakeResult(boxes, [0.9] * len(boxes))]

    def track(self, frame, **kwargs):  # noqa: ANN001, ANN003
        result = self.predict(frame)[0]
        result.boxes.id = _T(np.zeros(1, dtype=np.float32))
        return [result]


@pytest.fixture(scope="module")
def panning_video(tmp_path_factory) -> Path:
    """9 s of a camera panning right over a textured world: enough for four chunks of fifteen frames."""
    path = tmp_path_factory.mktemp("clip") / "pan.mp4"
    rng = np.random.default_rng(11)
    layers = [cv2.GaussianBlur(rng.random((900, 2000)).astype(np.float32), (0, 0), s) for s in (1.5, 4.0, 12.0)]
    scene = sum(layer / layer.std() for layer in layers)
    scene = cv2.cvtColor(((scene - scene.min()) / (scene.max() - scene.min()) * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    with VideoWriter(path, fps=10.0, width=W, height=H) as writer:
        for i in range(90):
            x = 150 + 14 * i
            writer.write(scene[150 : 150 + H, x : x + W].copy())
    return path


@pytest.fixture(scope="module")
def analysed(tmp_path_factory, panning_video: Path) -> Path:
    out = tmp_path_factory.mktemp("segment")
    analyse_segment(
        panning_video, out, config=SegmentConfig(fps=5.0, motion_width=320, detect_width=640, chunk_frames=15,
                                                  device="cpu"), model=_FakeModel(),
    )
    return out


def _snapshot(directory: Path) -> dict[str, np.ndarray]:
    with np.load(sorted(directory.glob("chunk_*.npz"))[0]) as data:
        return {key: data[key].copy() for key in data.files}


def _copy_segment(source: Path, destination: Path) -> Path:
    """A fresh copy of a segment's meta and chunks, so a test can run against its own resume history."""
    destination.mkdir(parents=True, exist_ok=True)
    for name in ("meta.json", "status.json"):
        if (source / name).exists():
            (destination / name).write_bytes((source / name).read_bytes())
    for chunk in source.glob("chunk_*.npz"):
        (destination / chunk.name).write_bytes(chunk.read_bytes())
    return destination


def test_the_refresh_changes_only_the_kit_descriptors(analysed: Path) -> None:
    """Everything the report was built from - poses, steps, focals, boxes, confidences - must come through as it was."""
    module = _script()
    before = _snapshot(analysed)

    status = module.refresh(analysed)

    after = _snapshot(analysed)
    assert set(before) == set(after)
    for key in before:
        if key == "det_kit":
            continue
        assert np.array_equal(before[key], after[key]), f"{key} was perturbed by the refresh"
    assert status["state"] == "done"
    assert status["chunks_done"] == status["total_chunks"] > 0


def test_refreshing_twice_changes_nothing(analysed: Path) -> None:
    """Idempotence is what makes the script safe to re-run: a second pass has nothing left to correct.

    Without this, every run would report a fresh pile of "changed" descriptors and the number would say nothing
    about whether the segment had actually moved. The second call also has to *skip* the work rather than redo it,
    which is what the resume count is for.
    """
    module = _script()
    module.refresh(analysed)
    first = _snapshot(analysed)["det_kit"]

    status = module.refresh(analysed)

    assert np.array_equal(first, _snapshot(analysed)["det_kit"])
    assert status["descriptors_changed"] == 0, "a second pass over already-refreshed data changed something"
    assert status["chunks_done"] == status["total_chunks"], "a resumed run must still account for every chunk"


def test_an_interrupted_run_resumes_instead_of_repeating(analysed: Path, tmp_path: Path) -> None:
    """A run that died at chunk N must not spend its minutes again on chunks 0..N-1.

    This is the reason the status file records a chunk count: the refresh takes ~25 minutes on a whole game, and a
    failure near the end (which is exactly what a decode hiccup at the last chunk looks like) must not cost the
    whole run again.
    """
    module = _script()
    fresh = _copy_segment(analysed, tmp_path / "resumed")

    module.refresh(fresh, limit_chunks=1)  # a partial run: chunk 0 only

    status = module.refresh(fresh)  # the real run: must pick up from chunk 1

    assert status["state"] == "done"
    assert status["chunks_done"] == status["total_chunks"] > 1
    # Only the chunks after the first were re-read, so the change count is this run's, not the total.
    assert 0 <= status["descriptors_changed"] <= status["chunks_done"] * len(_snapshot(fresh)["det_frame"])


def test_the_status_file_says_what_happened_and_where(analysed: Path) -> None:
    """The dashboard reads this file, so the run must leave a state, a count and the chunk total behind."""
    module = _script()
    module.refresh(analysed)

    status = json.loads((analysed / module.STATUS_FILE).read_text())

    assert status["state"] == "done"
    assert status["chunks_done"] == status["total_chunks"]
    assert status["descriptors_changed"] >= 0
    assert status["updated"] > 0
    assert "error" not in status, "a finished run must not still carry the error of an earlier failed one"


def test_a_partial_run_says_so_rather_than_claiming_the_whole_segment(analysed: Path) -> None:
    """``--limit-chunks`` exists for smoke tests, so a limited run must not report itself as finished."""
    module = _script()
    module.refresh(analysed)

    status = module.refresh(analysed, limit_chunks=1)

    assert status["state"] == "partial"
    assert status["chunks_done"] == 1 < status["total_chunks"]


def test_the_segment_still_loads_after_a_refresh(analysed: Path) -> None:
    """The guard that matters downstream: a rewritten chunk must still be a loadable segment."""
    module = _script()
    module.refresh(analysed)

    data = load_segment(analysed)

    assert len(data.det_frame) > 0
    assert data.det_kit.shape[1] == 12
    assert np.all(data.det_kit[:, 0] >= 0.0), "kit_fraction is a fraction and must stay within 0..1"


def test_the_chunk_times_are_read_as_absolute_source_times(analysed: Path, monkeypatch, tmp_path: Path) -> None:
    """The reader must be pointed at ``times[0]`` itself, not at ``start_s + times[0]``.

    ``times[]`` already carries the analysis window's offset, so adding ``start_s`` again asks for a time past the
    end of the video: the decode then returns a short run and the script refuses the chunk. That is exactly what
    happened on the whole game at chunk 62 (sought 4801.6s in a 4851.8s video, got 251 of 300 frames), and it is
    invisible in a test whose video starts at zero - hence this one, which asserts the seek argument itself.
    """
    module = _script()
    fresh = _copy_segment(analysed, tmp_path / "absolute")
    seeks: list[float] = []

    class _RecordingReader:
        def __init__(self, _path, *, fps, width, start_s, duration_s):  # noqa: ANN001, ANN003
            seeks.append(start_s)

        def frames(self):
            return iter(())

    monkeypatch.setattr(module, "FFmpegFrameReader", _RecordingReader)
    monkeypatch.setattr(module, "_done_chunks", lambda *args, **kwargs: 0)  # force every chunk to be visited

    with pytest.raises(RuntimeError, match="refusing to write a partial chunk"):
        module.refresh(fresh)  # the fake reader yields nothing, so chunk 0 fails - after recording its seek

    with np.load(sorted(fresh.glob("chunk_*.npz"))[0]) as data:
        chunk_start = float(data["time"][0])
    assert seeks, "no chunk was read"
    assert seeks[0] == pytest.approx(chunk_start, abs=0.01), (
        f"the reader was pointed at {seeks[0]:.1f}s but the chunk starts at {chunk_start:.1f}s - a doubled "
        "offset walks off the end of the video"
    )


def test_a_previous_runs_error_is_cleared_by_the_run_that_succeeds(analysed: Path, tmp_path: Path) -> None:
    """A finished status must not still carry a failure's message.

    The status file is merged key by key, so a stale ``error`` survives a successful re-run unless it is explicitly
    dropped - and "state: done, error: decoded 251 of 300 frames" is a contradiction every reader would have to know
    how to resolve. Written here by hand because provoking the real failure costs a decode.
    """
    module = _script()
    fresh = _copy_segment(analysed, tmp_path / "stale-error")
    (fresh / module.STATUS_FILE).write_text(
        json.dumps({"state": "error", "chunks_done": 1, "error": "RuntimeError: an earlier run failed"})
    )

    status = module.refresh(fresh)

    assert status["state"] == "done"
    assert "error" not in status, f"the earlier failure is still in the status: {status}"


def test_a_decode_shortfall_is_an_error_not_a_silent_zeroed_chunk(analysed: Path, tmp_path, monkeypatch) -> None:
    """A reader that returns fewer frames than the chunk holds must fail loudly, leaving the chunk untouched.

    The alternative - writing the descriptors it did compute and zeroing the rest - would quietly replace real kit
    colours with "no kit" for the missing frames, and the report would show it as a colour that was measured. It
    also has to surface as ``state="error"`` in the status file rather than escaping as a ``SystemExit``, which
    would leave the page showing a run that is still going.
    """
    module = _script()
    # Its own copy: the resume logic trusts the previous run's chunk count, and the other tests have already
    # finished the shared segment, so a chunk would be skipped and the shortfall never reached.
    fresh = _copy_segment(analysed, tmp_path / "short")
    before = _snapshot(fresh)["det_kit"]

    class _ShortReader:
        def __init__(self, *args, **kwargs):  # noqa: ANN002, ANN003
            pass

        def frames(self):
            for index in range(3):  # far fewer than the chunk holds
                yield float(index), np.zeros((8, 8, 3), dtype=np.uint8)

    monkeypatch.setattr(module, "FFmpegFrameReader", _ShortReader)

    with pytest.raises(RuntimeError, match="refusing to write a partial chunk"):
        module.refresh(fresh)

    assert np.array_equal(before, _snapshot(fresh)["det_kit"]), "the chunk was written despite the shortfall"