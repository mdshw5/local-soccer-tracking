"""The clip-set source: a game without a merged video must read as one continuous recording.

The reference measurements live in the docstrings; what these tests pin is the contract the rest of the pipeline
relies on - a seek lands in the right clip (and nowhere else), a sequential pass carries one analysis grid across
a join, and Stage A runs and resumes a two-clip game exactly as it runs a single file.
"""

from __future__ import annotations

import json
import math
import subprocess
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from soccer_analytics.analysis import game as game_lib
from soccer_analytics.analysis import highlights as highlights_lib
from soccer_analytics.analysis.stage_a import SegmentConfig, analyze_segment, completed_chunks, load_segment, segment_dir_for
from soccer_analytics.dashboard import timeline as timeline_lib
from soccer_analytics.dashboard.video import encode_clip
from soccer_analytics.ingest.ffmpeg_reader import FFmpegFrameReader, grab_frame, probe_video, read_wav_mono
from soccer_analytics.ingest.source import (
    ClipSource,
    FileSource,
    as_source,
    extract_source_audio,
    grab_source_frame,
    open_reader,
    probe_source,
    window_segments,
)
from soccer_analytics.ingest.video_reader import VideoWriter

W, H = 320, 180
CLIP_FPS = 10.0


def _clip(path: Path, *, seconds: float, tint: str) -> None:
    """A short panning clip with a red or blue tint: frame content names the clip it came from."""
    rng = np.random.default_rng(7)
    big = cv2.GaussianBlur(rng.random((H * 2, W * 4)).astype(np.float32), (0, 0), 2.0)
    big = ((big - big.min()) / (big.max() - big.min()) * 255).astype(np.uint8)
    with VideoWriter(path, fps=CLIP_FPS, width=W, height=H) as writer:
        for i in range(int(seconds * CLIP_FPS)):
            x = 10 + 2 * i
            frame = cv2.cvtColor(big[100 : 100 + H, x : x + W], cv2.COLOR_GRAY2BGR)
            if tint == "red":  # BGR: blue and green down, red stays - the frame averages clearly red
                frame[:, :, 0] //= 6
                frame[:, :, 1] //= 6
            else:
                frame[:, :, 2] //= 6
            writer.write(frame)


@pytest.fixture(scope="module")
def game(tmp_path_factory) -> dict:
    """Two tinted clips and the manifest that makes them one game; nothing is merged."""
    root = tmp_path_factory.mktemp("footage")
    first, second = root / "clip_a.mp4", root / "clip_b.mp4"
    _clip(first, seconds=3.0, tint="red")
    _clip(second, seconds=2.5, tint="blue")
    d1, d2 = float(probe_video(first).duration_s), float(probe_video(second).duration_s)

    directory = root / "analysis" / "game_test_1"
    directory.mkdir(parents=True)
    record = game_lib.GameRecord(
        game_id="game_test_1",
        # The merged file this game *would* have had: deliberately never created.
        output=str(root / "game_test_1_merged.mp4"),
        duration_s=d1 + d2,
        clips=[
            game_lib.Clip(path=str(first), start_s=0.0, duration_s=d1, bytes=first.stat().st_size),
            game_lib.Clip(path=str(second), start_s=d1, duration_s=d2, bytes=second.stat().st_size),
        ],
    )
    record.save(directory)
    return {"root": root, "manifest": directory / "game.json", "d1": d1, "d2": d2, "first": first, "second": second}


def test_a_manifest_resolves_to_its_clips_and_a_file_to_itself(game: dict) -> None:
    manifest = game["manifest"]
    source = as_source(manifest)
    assert isinstance(source, ClipSource)
    assert [Path(clip.path) for clip in source.clips] == [game["first"], game["second"]]
    assert source.spec == str(manifest)
    assert isinstance(as_source(game["first"]), FileSource)


def test_probe_reports_the_whole_game_from_the_manifest(game: dict) -> None:
    probe = probe_source(game["manifest"])
    assert (probe.width, probe.height) == (W, H)
    assert probe.fps == pytest.approx(CLIP_FPS, abs=0.05)
    assert probe.duration_s == pytest.approx(game["d1"] + game["d2"], abs=1e-6)


def _frames(source, **kwargs) -> list[tuple[float, np.ndarray]]:
    return list(open_reader(source, prefer_gpu=False, **kwargs).frames())


def test_the_reader_carries_one_grid_across_the_join(game: dict) -> None:
    """Times continue at 1/fps through the clip change, and the pictures change exactly there."""
    step = 1.0 / CLIP_FPS
    frames = _frames(game["manifest"], fps=CLIP_FPS, width=W)
    assert len(frames) == pytest.approx((game["d1"] + game["d2"]) * CLIP_FPS, abs=3)
    times = np.asarray([t for t, _ in frames])
    assert times[0] == pytest.approx(0.0, abs=step)
    assert np.all(np.diff(times) > 0)
    # No hole at the join: every consecutive step is one frame of the analysis grid (the last grid point before
    # a clip boundary may fall a fraction of a frame short of it, never more than a step + rounding).
    assert float(np.max(np.diff(times))) <= step + 5e-3, f"time gap at the join: {np.diff(times).max():.4f}s"

    joint = game["d1"]
    before = min(frames, key=lambda tf: abs(tf[0] - (joint - 0.3)))[1]
    after = min(frames, key=lambda tf: abs(tf[0] - (joint + 0.3)))[1]
    assert before[:, :, 2].mean() > before[:, :, 0].mean() + 40, "the frame before the join is the red clip"
    assert after[:, :, 0].mean() > after[:, :, 2].mean() + 40, "the frame after the join is the blue clip"


def test_a_window_inside_the_second_clip_never_touches_the_first(game: dict) -> None:
    joint = game["d1"]
    frames = _frames(game["manifest"], fps=CLIP_FPS, width=W, start_s=joint + 0.4, duration_s=0.6)
    assert frames
    for t, frame in frames:
        assert t >= joint + 0.4 - 1e-9
        assert frame[:, :, 0].mean() > frame[:, :, 2].mean() + 40


def test_a_single_frame_seek_resolves_to_the_containing_clip(game: dict) -> None:
    joint = game["d1"]
    before = grab_source_frame(game["manifest"], joint - 0.4, width=W)
    after = grab_source_frame(game["manifest"], joint + 0.4, width=W)
    assert before is not None and after is not None
    assert before[:, :, 2].mean() > before[:, :, 0].mean() + 40
    assert after[:, :, 0].mean() > after[:, :, 2].mean() + 40


def test_a_plain_file_reads_exactly_as_before(tmp_path: Path, game: dict) -> None:
    """The resolver must not change the ordinary single-file path: same frames, same times."""
    source = game["first"]
    direct = list(FFmpegFrameReader(source, fps=5.0, width=W, prefer_gpu=False).frames())
    resolved = _frames(source, fps=5.0, width=W)
    assert len(direct) == len(resolved) == 15
    assert all(np.array_equal(a[1], b[1]) for a, b in zip(direct, resolved))


def test_segment_dirs_key_a_manifest_by_its_game_id(game: dict) -> None:
    root = game["root"] / "segments"
    plain = segment_dir_for(game["manifest"], root)
    half = segment_dir_for(game["manifest"], root, window_label="first_half_10_200")
    assert plain.name == "game_test_1"
    assert half.name == "game_test_1__first_half_10_200"


class _T:
    def __init__(self, a: np.ndarray):
        self.a = a

    def cpu(self):
        return self

    def numpy(self):
        return self.a


class _FakeBoxes:
    def __init__(self, xyxy: np.ndarray, conf: np.ndarray):
        self.xyxy, self.conf = _T(xyxy), _T(conf)


class _FakeResult:
    def __init__(self, xyxy, conf):
        self.boxes = _FakeBoxes(np.asarray(xyxy, dtype=np.float32).reshape(-1, 4), np.asarray(conf, dtype=np.float32))


class _FakeModel:
    """One person per frame, scaled from 1920 like the detector reports."""

    def predict(self, frame, **kwargs):  # noqa: ANN001, ANN003
        width = frame.shape[1]
        boxes = [(0.0, 0.0, 10.0 * width / 1920, 40.0 * width / 1920)]
        return [_FakeResult(boxes, [0.9])]

    def track(self, frame, **kwargs):  # noqa: ANN001, ANN003
        result = self.predict(frame)[0]
        result.boxes.id = _T(np.zeros(1, dtype=np.float32))
        return [result]


CONFIG = SegmentConfig(fps=5.0, motion_width=160, detect_width=W, chunk_frames=10, device="cpu")


def test_stage_a_analyzes_a_game_without_a_merged_file(game: dict, tmp_path: Path) -> None:
    """The whole point: a manifest with no combined video runs, chunks and all."""
    out = tmp_path / "segment"
    status = analyze_segment(game["manifest"], out, config=CONFIG, model=_FakeModel())
    assert status["state"] == "done"

    meta = json.loads((out / "meta.json").read_text())
    assert meta["video"] == str(game["manifest"])
    assert not (game["root"] / "game_test_1_merged.mp4").exists(), "nothing merges the clips"

    data = load_segment(out)
    expected = (game["d1"] + game["d2"]) * CONFIG.fps
    assert len(data.time) == pytest.approx(expected, abs=2)
    assert np.all(np.diff(data.time) > 0)
    assert len(data.det_frame) == len(data.time)  # one fake person per frame
    assert completed_chunks(out) == math.ceil(len(data.time) / CONFIG.chunk_frames)


def test_stage_a_resumes_across_the_clip_join(game: dict, tmp_path: Path) -> None:
    """A resume must reproduce the straight run, even when the interruption sits inside the second clip."""
    straight = tmp_path / "straight"
    analyze_segment(game["manifest"], straight, config=CONFIG, model=_FakeModel())
    reference = load_segment(straight)

    resumed = tmp_path / "resumed"
    # Stop inside the second clip: the resumed reader starts near the join and has to chain across it.
    stop_after = int((game["d1"] + 0.6) * CONFIG.fps)
    seen = {"n": 0}

    def _stop() -> bool:
        seen["n"] += 1
        return seen["n"] > stop_after

    partial = analyze_segment(game["manifest"], resumed, config=CONFIG, model=_FakeModel(), should_stop=_stop)
    assert partial["state"] == "stopped"
    final = analyze_segment(game["manifest"], resumed, config=CONFIG, model=_FakeModel())
    assert final["state"] == "done"

    after = load_segment(resumed)
    assert len(after.time) == len(reference.time)
    assert np.allclose(after.time, reference.time, atol=1e-6)
    assert np.array_equal(after.det_frame, reference.det_frame)


# --------------------------------------------------------------------------------------------------------------
# The consumers that cut or encode a window: each has to stay fast on a never-merged game by seeking inside the
# clip that holds the second it wants, never the concat demuxer (see ingest/source.py). These tests pin the two
# shapes that matter: a window inside one clip is exactly the old single-file command, and a window crossing a
# join is the pieces encoded one by one and joined - in order, with the right sound.
# --------------------------------------------------------------------------------------------------------------


def test_window_pieces_split_at_the_clip_join(game: dict) -> None:
    d1 = game["d1"]
    pieces = window_segments(game["manifest"], d1 - 0.5, 1.0)
    assert [Path(piece.path) for piece in pieces] == [game["first"], game["second"]]
    assert pieces[0].start_s == pytest.approx(d1 - 0.5, abs=1e-6)
    assert pieces[0].duration_s == pytest.approx(0.5, abs=1e-6)
    assert pieces[1].start_s == pytest.approx(0.0, abs=1e-3)
    assert pieces[1].duration_s == pytest.approx(0.5, abs=1e-3)

    inside = window_segments(game["manifest"], d1 - 1.0, 0.5)
    assert len(inside) == 1
    assert Path(inside[0].path) == game["first"]
    assert inside[0].start_s == pytest.approx(d1 - 1.0, abs=1e-6)

    plain = window_segments(game["first"], 0.25, 0.5)
    assert len(plain) == 1
    assert plain[0].start_s == pytest.approx(0.25, abs=1e-6)
    assert plain[0].duration_s == pytest.approx(0.5, abs=1e-6)


def _clip_with_tone(path: Path, *, seconds: float, frequency: float) -> None:
    """A short video clip whose soundtrack is one pure tone: which tone is in a wav names which clip it came from."""
    subprocess.run(
        [
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
            "-f", "lavfi", "-i", f"testsrc=duration={seconds}:size={W}x{H}:rate={CLIP_FPS}",
            "-f", "lavfi", "-i", f"sine=frequency={frequency}:duration={seconds}",
            "-map", "0:v:0", "-map", "1:a:0",
            "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p", "-c:a", "aac",
            str(path),
        ],
        check=True,
    )


@pytest.fixture(scope="module")
def audio_game(tmp_path_factory) -> dict:
    """Two tone-bearing clips and their manifest: the clip-set analog of a recording with a soundtrack."""
    root = tmp_path_factory.mktemp("tone_footage")
    first, second = root / "tone_a.mp4", root / "tone_b.mp4"
    _clip_with_tone(first, seconds=2.5, frequency=440.0)
    _clip_with_tone(second, seconds=2.0, frequency=880.0)
    d1, d2 = float(probe_video(first).duration_s), float(probe_video(second).duration_s)
    directory = root / "analysis" / "game_tones_1"
    directory.mkdir(parents=True)
    record = game_lib.GameRecord(
        game_id="game_tones_1",
        output=str(root / "game_tones_1_merged.mp4"),
        duration_s=d1 + d2,
        clips=[
            game_lib.Clip(path=str(first), start_s=0.0, duration_s=d1, bytes=first.stat().st_size),
            game_lib.Clip(path=str(second), start_s=d1, duration_s=d2, bytes=second.stat().st_size),
        ],
    )
    record.save(directory)
    return {"root": root, "manifest": directory / "game.json", "d1": d1, "d2": d2}


def _dominant_hz(samples: np.ndarray, rate: int) -> float:
    window = np.hanning(len(samples))
    spectrum = np.abs(np.fft.rfft(samples * window))
    freqs = np.fft.rfftfreq(len(samples), 1.0 / rate)
    return float(freqs[int(np.argmax(spectrum))])


def test_source_audio_is_stitched_across_the_join(audio_game: dict) -> None:
    """The whistle scan's one wav on one clock: the tone before the join and the tone after it, in order."""
    manifest = str(audio_game["manifest"])
    d1 = audio_game["d1"]
    wav = extract_source_audio(manifest, audio_game["root"] / "stitched.wav", start_s=d1 - 0.8, duration_s=1.6)
    samples, rate = read_wav_mono(wav)
    assert len(samples) / rate == pytest.approx(1.6, abs=0.15)
    assert _dominant_hz(samples[: int(0.4 * rate)], rate) == pytest.approx(440.0, abs=30.0)
    assert _dominant_hz(samples[-int(0.4 * rate) :], rate) == pytest.approx(880.0, abs=30.0)


def test_encode_clip_reads_a_manifest_and_keeps_the_sound_across_the_join(audio_game: dict, tmp_path: Path) -> None:
    """The dashboard clip route on a never-merged game: frames chain across the join and the soundtrack - two
    inputs joined by ffmpeg's concat filter - comes out as one continuous track."""
    d1 = audio_game["d1"]
    match = SimpleNamespace(video=str(audio_game["manifest"]), render_at=lambda frame, timestamp, **kwargs: None)
    output = tmp_path / "clip.mp4"
    frames = encode_clip(
        match,
        start_s=d1 - 0.8,
        duration_s=1.6,
        fps=5.0,
        width=96,
        overlays={},
        output=output,
        encoder="libx264",
    )
    assert frames == 8
    probe = probe_video(output)
    assert probe.duration_s == pytest.approx(1.6, abs=0.3)
    assert probe.has_audio, "the concat filter must produce a real track, not be dropped when it cannot"


def test_the_scrubber_proxy_spans_a_manifest_join(game: dict, tmp_path: Path) -> None:
    """The timeline proxy of a window crossing the join: clipped per clip, copy-joined, and in the right order."""
    d1 = game["d1"]
    proxy = timeline_lib.build_proxy(
        game["manifest"], tmp_path / "segment", start_s=d1 - 0.6, duration_s=1.4, width=96, fps=10.0, prefer_gpu=False
    )
    assert proxy.name == "proxy.mp4" and proxy.exists()
    probe = probe_video(proxy)
    assert probe.duration_s == pytest.approx(1.4, abs=0.25)
    # Sampled well inside the file: grab_frame's seek is documented "near" a time and an end-of-file seek is a
    # known limit of its reader in general, nothing to do with the join.
    before = grab_frame(proxy, 0.2, width=96)
    after = grab_frame(proxy, 1.0, width=96)
    assert before is not None and after is not None
    assert before[:, :, 2].mean() > before[:, :, 0].mean() + 15, "first piece is the red clip"
    assert after[:, :, 0].mean() > after[:, :, 2].mean() + 15, "second piece is the blue clip"


def test_a_highlight_cut_across_the_join_holds_both_pieces(game: dict, tmp_path: Path) -> None:
    d1 = game["d1"]
    moment = highlights_lib.Moment(
        time_s=d1, start_s=d1 - 0.5, end_s=d1 + 0.9, weight=1.0, reason="crosses the join"
    )
    output = tmp_path / "moment.mp4"
    highlights_lib._encode_clip(  # noqa: SLF001 - the cut is the unit under test
        game["manifest"], moment, output, duration_s=1.4, width=96, use_gpu=False, fps=10
    )
    probe = probe_video(output)
    assert probe.duration_s == pytest.approx(1.4, abs=0.25)
    before = grab_frame(output, 0.2, width=96)
    after = grab_frame(output, 1.05, width=96)
    assert before is not None and after is not None
    assert before[:, :, 2].mean() > before[:, :, 0].mean() + 15
    assert after[:, :, 0].mean() > after[:, :, 2].mean() + 15
