"""The ball scan is a background process over a whole segment: it must checkpoint, resume, and report.

Two properties make it safe to run for an hour on a whole game: the status file is what the dashboard reads, and a
stopped scan resumes from its checkpoint *with the tracker's state* - resuming with a fresh tracker would re-learn
the ball's velocity while the records claim one continuous track. Both are pinned here with an injected detector,
so the tests need no GPU, no models and no footage beyond a second of synthetic video.
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
WIDTH, HEIGHT, FPS, FRAMES = 320, 240, 5.0, 8


def _script():  # noqa: ANN202 - the scan module, loaded without making scripts/ a package
    spec = importlib.util.spec_from_file_location("run_ball_scan", REPO_ROOT / "scripts" / "run_ball_scan.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _make_video(path: Path) -> Path:
    """A second of green video: enough for the reader and the scan loop, nothing more."""
    subprocess.run(
        [
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
            "-f", "lavfi", "-i", f"color=c=green:s={WIDTH}x{HEIGHT}:r={FPS}:d={FRAMES / FPS + 0.4:.2f}",
            "-c:v", "libx264", "-pix_fmt", "yuv420p",
            str(path),
        ],
        check=True,
        capture_output=True,
    )
    return path


def _detector(state: dict):  # noqa: ANN202 - closure around the fake ball's position
    """A fake detector: a ball moving slowly right, returned regardless of the crop.

    Returns absolute frame-normalized coordinates, as the production detector does. Also records what it saw, so
    the test can check the overlay regions were blanked before it was called.
    """
    state.setdefault("seen", [])
    state.setdefault("calls", 0)

    def detect(image: np.ndarray, imgsz: int, origin: tuple[int, int]) -> list[tuple]:
        state["calls"] += 1
        if image.shape[0] == HEIGHT and image.shape[1] == WIDTH:
            # A full-frame scan: the burned-in overlay regions must already be black.
            state["seen"].append((int(image[-1, -1].sum()), int(image[-1, 0].sum())))
        u = 0.1 + 0.004 * state["calls"]
        return [(0.9, u, 0.3, 0.01, 0.01)]

    return detect


def _run(module, video: Path, out_dir: Path, detector, *, limit_frames: int = 0):  # noqa: ANN001
    return module.scan(
        video=video,
        out_dir=out_dir,
        start_s=0.0,
        fps=FPS,
        total_frames=FRAMES,
        width=WIDTH,
        aspect=HEIGHT / WIDTH,
        steps=None,
        detector=detector,
        limit_frames=limit_frames,
    )


def test_the_source_prefers_the_segments_own_video_when_this_machine_has_it(tmp_path: Path) -> None:
    """The scan decodes at the segment's clock, so its stored source wins whenever it is present.

    A machine that does not have it - the analysis was copied from another footage root, or the footage moved -
    must read the caller's recording (the page passes the same one the other scans read) instead of failing on
    the stale absolute path, which is what left the scan stuck with a bare "No such file or directory".
    """
    run = _script()
    stored = tmp_path / "stored.mp4"
    stored.write_bytes(b"x")
    other = tmp_path / "other.mp4"
    other.write_bytes(b"x")

    assert run.source_video("", {"video": str(stored)}) == str(stored)
    assert run.source_video(str(other), {"video": str(stored)}) == str(stored)  # reachable stored wins

    missing = tmp_path / "gone" / "game.json"
    assert run.source_video(str(other), {"video": str(missing)}) == str(other)  # stale stored steps aside
    assert run.source_video("", {"video": str(missing)}) == str(missing)  # ...with nothing else to read


def test_a_stale_source_fails_with_guidance_not_a_bare_enoent(tmp_path: Path, monkeypatch) -> None:
    """The stored path is the analyzing machine's own; a copy on another machine must be told what to do.

    The dashboard shows the status file's error verbatim, so naming the missing path alone - FileNotFoundError's
    own message - sends the reader hunting for a file the code cannot use anyway. The message says how to make
    the scan work instead.
    """
    run = _script()
    segment = tmp_path / "seg"
    segment.mkdir()
    (segment / "meta.json").write_text(
        json.dumps({"video": str(tmp_path / "gone" / "game.json"), "start_s": 0.0, "fps": 5.0, "width": WIDTH})
    )
    monkeypatch.setattr("sys.argv", ["run_ball_scan.py", "--segment", str(segment)])
    assert run.main() == 1
    status = json.loads((segment / "ball_scan.json").read_text())
    assert status["state"] == "error"
    assert "not on this machine" in status["error"]


def test_the_scan_writes_a_track_and_reports_to_the_status_file(tmp_path: Path) -> None:
    module = _script()
    video = _make_video(tmp_path / "tiny.mp4")
    out = tmp_path / "seg"

    payload = _run(module, video, out, _detector({}))

    assert payload["complete"] is True
    assert len(payload["frames"]) == FRAMES
    assert [r["i"] for r in payload["frames"]] == list(range(FRAMES))
    assert payload["counts"].get("tracking", 0) == FRAMES, "a visible ball must be tracked every frame"
    assert all(r["u"] is not None for r in payload["frames"])
    assert (out / module.RESULT_FILE).exists()

    status = json.loads((out / module.STATUS_FILE).read_text())
    assert status["state"] == "done" and status["progress"] == 1.0
    # The terminal status carries the coverage, so the dashboard can summarize the scan without reading the (large)
    # result file: how much of the game the ball was actually seen on is the number that says whether to trust it.
    assert status["counts"] == payload["counts"]
    assert status["scanned"] == FRAMES and status["total_frames"] == FRAMES


def test_the_overlay_regions_are_blanked_before_any_detection(tmp_path: Path) -> None:
    """The logo scores 0.9+ as 'soccer ball' - the scan must remove it before the detector can see it."""
    module = _script()
    video = _make_video(tmp_path / "tiny.mp4")
    state: dict = {}
    _run(module, video, tmp_path / "seg", _detector(state))

    assert state["seen"], "the fake detector must have been given at least one full frame"
    for logo_sum, stamp_sum in state["seen"]:
        assert logo_sum == 0, "the burned-in logo region must be black in the detector's input"
        assert stamp_sum == 0, "the timestamp region must be black in the detector's input"


def test_a_stopped_scan_resumes_without_repeating_frames(tmp_path: Path) -> None:
    """Killing an hour-long scan must cost the frames since the last checkpoint, nothing more."""
    module = _script()
    video = _make_video(tmp_path / "tiny.mp4")
    out = tmp_path / "seg"

    first = _run(module, video, out, _detector({}), limit_frames=3)
    assert first["complete"] is False
    assert [r["i"] for r in first["frames"]] == [0, 1, 2]
    status = json.loads((out / module.STATUS_FILE).read_text())
    assert status["state"] == "partial"
    assert status["scanned"] == 3 and status["total_frames"] == FRAMES

    second = _run(module, video, out, _detector({}))
    assert second["complete"] is True
    assert [r["i"] for r in second["frames"]] == list(range(FRAMES)), "no frame may be recorded twice"

    third = _run(module, video, out, _detector({}))
    assert len(third["frames"]) == FRAMES, "a completed scan is not redone"
    status = json.loads((out / module.STATUS_FILE).read_text())
    assert status["state"] == "done"


def test_the_checkpoint_carries_the_tracker_state(tmp_path: Path) -> None:
    """The resumed run must continue the *same* track: the checkpoint stores the tracker, not just records.

    A fresh tracker would still find the ball, so 'it works' is not the test - the test is that the state on disk
    is the tracker's own serialization and that resuming from it tracks without a hiccup (no coast, no lost).
    """
    module = _script()
    video = _make_video(tmp_path / "tiny.mp4")
    out = tmp_path / "seg"

    first = _run(module, video, out, _detector({}), limit_frames=4)
    tracker = first["tracker"]
    assert tracker["status"] == "tracking" and tracker["x"] is not None
    assert tracker["vx"] > 0, "the ball was moving, so a tracked scan must have learned a velocity"

    second = _run(module, video, out, _detector({}))
    resumed = second["frames"][4:]
    assert all(r["status"] == "tracking" for r in resumed), "a state-carrying resume must not drop the ball"


def test_a_crashing_detector_leaves_the_reason_in_the_status_file(tmp_path: Path) -> None:
    """The dashboard's only view of a background scan is its status file - a crash must land there.

    Without this, a model that fails to load (or a CUDA error) leaves "running" behind and the page shows a
    progress bar that will never move; the user sees a scan that is simply slow, not one that died.
    """
    module = _script()
    video = _make_video(tmp_path / "tiny.mp4")
    out = tmp_path / "seg"

    def broken(image: np.ndarray, imgsz: int, origin: tuple[int, int]) -> list[tuple]:
        raise RuntimeError("no CUDA device")

    with pytest.raises(RuntimeError, match="no CUDA device"):
        _run(module, video, out, broken)

    status = json.loads((out / module.STATUS_FILE).read_text())
    assert status["state"] == "error"
    assert "no CUDA device" in status["error"], "the status must carry the reason, not just the state"
    assert "RuntimeError" in status["error"], "the exception type is part of the reason"
