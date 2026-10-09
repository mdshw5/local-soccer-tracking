"""Framing a centred clip: the crop must follow the player, stay inside the frame, and refuse what it cannot show.

The pure numbers are tested directly (sizing, smoothing, the window trimming that keeps a clip away from a gap),
and the one claim that only a real encode can settle - that the moving crop actually keeps the player in the
middle - is tested by cutting a real clip out of a synthetic video and looking for the player in the output.
"""

from __future__ import annotations

import shutil
import subprocess

import cv2
import numpy as np
import pytest

from soccer_analytics.analysis.framing import (
    DEFAULT_PLAYER_FRACTION,
    CropCommand,
    FramingPlan,
    crop_size,
    crop_times,
    plan_framing,
    sendcmd_filter,
)
from soccer_analytics.ingest.video_reader import VideoWriter


def _boxes(spans: list[tuple[float, float]]) -> np.ndarray:
    """A track of boxes: one ``(centre_x_fraction, height_fraction)`` per observation.

    Both box axes share the frame-width scale (that is how Stage A stores them), so the box is
    ``height/2.5`` wide - a person's proportions - and sits with its centre a little above mid-frame.
    """
    out = []
    for centre, height in spans:
        half_width = height / 2.0 / 2.5
        top = 0.30
        out.append([centre - half_width, top, centre + half_width, top + height])
    return np.asarray(out, dtype=np.float64)


def test_crop_size_uses_the_median_player_not_the_closest_frame() -> None:
    """One close-up frame must not zoom the whole clip in: the median height is what the crop is sized from."""
    heights = np.array([100.0, 100.0, 100.0, 400.0])  # three ordinary frames, one very close
    wide, tall = crop_size(heights, 1920, 1080, player_fraction=0.25, min_crop_height_px=100.0)
    assert tall == pytest.approx(400.0, abs=2.0), "crop height should come from 100 px / 0.25"
    assert wide == pytest.approx(400.0 * 1920 / 1080, abs=2.0)
    assert wide % 2 == 0 and tall % 2 == 0, "yuv420p needs even dimensions"


def test_crop_size_has_a_floor_and_never_exceeds_the_source() -> None:
    tiny = crop_size(np.array([10.0]), 1920, 1080, min_crop_height_px=480.0)
    assert tiny[1] == 480, "a far player must not ask for a crop that has to be blown up from nothing"
    huge = crop_size(np.array([4000.0]), 1920, 1080, min_crop_height_px=480.0)
    assert huge == (1920, 1080), "the crop cannot be larger than the frame"


def test_crop_times_are_a_clean_grid_ending_on_the_window() -> None:
    grid = crop_times(duration_s=2.0, rate_hz=5.0)
    assert grid[0] == 0.0 and grid[-1] == 2.0
    assert np.all(np.diff(grid) > 0)
    assert np.allclose(grid, np.round(grid, 3))
    # 2 s at 5 Hz is 11 points (inclusive), and the rounding must not have collapsed any of them.
    assert len(grid) == 11


def test_plan_follows_the_player_and_keeps_the_crop_inside_the_frame() -> None:
    """A player crossing the frame: the crop chases them, but never leaves the source rectangle.

    What matters is containment - the player's box has to stay inside the crop at every command - with the extra
    claim that away from the frame's edges the crop is actually *centred* on them rather than merely containing
    them. Near an edge the crop clamps (it cannot leave the source), which is why the centring check skips it.
    """
    times = np.arange(0.0, 6.0, 0.2)
    centres = np.linspace(0.1, 0.9, len(times))
    boxes = _boxes([(float(c), 0.04) for c in centres])
    plan = plan_framing(
        times,
        boxes,
        start_s=0.0,
        duration_s=6.0,
        source_width=1920,
        source_height=1080,
        player_fraction=0.25,
        min_crop_height_px=200.0,
    )
    assert plan is not None
    assert plan.truncated == ""
    assert plan.duration_s == pytest.approx(6.0, abs=1e-6)
    xs = [command.x for command in plan.commands]
    assert xs == sorted(xs), "the player only moves right, so the crop only moves right"
    assert min(xs) >= 0 and max(command.x for command in plan.commands) + plan.crop_w <= 1920
    assert all(0 <= command.y and command.y + plan.crop_h <= 1080 for command in plan.commands)

    clamped = {0, 1920 - plan.crop_w}
    for command in plan.commands:
        player = float(np.interp(command.time_s, times, centres)) * 1920
        assert command.x <= player <= command.x + plan.crop_w, "the player left the crop"
        if command.x not in clamped:  # away from the frame's edges the crop is centred on the player
            assert abs((command.x + plan.crop_w / 2) - player) < 0.20 * plan.crop_w


def test_smoothing_removes_box_jitter_from_the_crop_path() -> None:
    """Box jitter frame to frame must not become crop jitter: the smoothed path is much steadier than the raw."""
    rng = np.random.default_rng(0)
    times = np.arange(0.0, 6.0, 0.2)
    centres = 0.5 + rng.normal(0.0, 0.02, len(times)).cumsum() * 0.02  # slow drift, plus noise below
    centres = centres + rng.normal(0.0, 0.01, len(times))
    boxes = _boxes([(float(c), 0.04) for c in centres])
    plan = plan_framing(
        times, boxes, start_s=0.0, duration_s=6.0, source_width=1920, source_height=1080,
        player_fraction=0.25, min_crop_height_px=200.0,
    )
    assert plan is not None
    path = np.array([command.x for command in plan.commands], dtype=np.float64)
    raw = np.interp([command.time_s for command in plan.commands], times, centres * 1920)
    # Second difference is acceleration: following the raw centres would visibly shake.
    assert np.abs(np.diff(path, 2)).max() < np.abs(np.diff(raw, 2)).max()


def test_a_gap_in_the_track_ends_the_window_before_it() -> None:
    """Across a long gap the player's position is unknown; the clip stops rather than sliding to a guess."""
    times = np.concatenate([np.arange(0.0, 3.0, 0.2), np.arange(10.0, 13.0, 0.2)])
    boxes = _boxes([(0.3 + 0.02 * i, 0.05) for i in range(len(times))])
    plan = plan_framing(
        times, boxes, start_s=0.0, duration_s=12.0, source_width=1920, source_height=1080,
        player_fraction=0.25, min_crop_height_px=200.0,
    )
    assert plan is not None
    assert plan.truncated == "gap in the track"
    assert plan.duration_s == pytest.approx(2.8, abs=0.21), "the window ends at the last frame before the gap"
    assert plan.commands[-1].time_s == pytest.approx(plan.duration_s, abs=1e-6)


def test_a_track_that_outlasts_the_window_is_not_truncated() -> None:
    times = np.arange(0.0, 30.0, 0.2)
    boxes = _boxes([(0.5, 0.05)] * len(times))
    plan = plan_framing(
        times, boxes, start_s=10.0, duration_s=5.0, source_width=1920, source_height=1080,
        player_fraction=0.25, min_crop_height_px=200.0,
    )
    assert plan is not None
    assert plan.truncated == ""
    assert plan.duration_s == pytest.approx(5.0, abs=1e-6)


def test_plan_returns_none_for_an_empty_or_single_point_track() -> None:
    assert plan_framing(np.array([]), np.zeros((0, 4)), start_s=0.0, duration_s=5.0,
                        source_width=1920, source_height=1080) is None
    assert plan_framing(np.array([1.0]), np.zeros((1, 4)), start_s=0.0, duration_s=5.0,
                        source_width=1920, source_height=1080) is None


def test_sendcmd_filter_names_the_crop_and_puts_the_schedule_in_order() -> None:
    plan = FramingPlan(
        crop_w=640, crop_h=360,
        commands=(CropCommand(0.0, 10, 20), CropCommand(0.5, 30, 40), CropCommand(1.0, 50, 60)),
        duration_s=1.0, truncated="",
    )
    chain = sendcmd_filter(plan, scale_width=1280, fps=30)
    assert chain.startswith("sendcmd=c='0.000 crop@follow x 10, crop@follow y 20; ")
    assert "0.500 crop@follow x 30, crop@follow y 40" in chain
    assert "crop@follow=w=640:h=360:x=10:y=20" in chain
    assert chain.endswith(",scale=1280:-2:flags=lanczos,fps=30")
    # No scale/fps asked for, no scale/fps added.
    assert "scale" not in sendcmd_filter(plan)


def test_sendcmd_filter_refuses_an_empty_plan() -> None:
    with pytest.raises(ValueError):
        sendcmd_filter(FramingPlan(crop_w=100, crop_h=100, commands=(), duration_s=0.0, truncated=""))


def test_duplicate_positions_are_collapsed_but_the_ends_are_kept() -> None:
    """A standing player gets two commands, not forty: the start position and the end of the window."""
    times = np.arange(0.0, 5.0, 0.2)
    boxes = _boxes([(0.5, 0.05)] * len(times))
    plan = plan_framing(
        times, boxes, start_s=0.0, duration_s=5.0, source_width=1920, source_height=1080,
        player_fraction=0.25, min_crop_height_px=200.0,
    )
    assert plan is not None
    assert len(plan.commands) == 2, "a stationary player needs the start stated and the end held, nothing more"
    assert plan.commands[0].time_s == 0.0, "the initial position must always be stated"
    assert plan.commands[-1].time_s == pytest.approx(5.0, abs=1e-6)


@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="ffmpeg is needed to cut the clip")
def test_a_real_cut_keeps_the_moving_player_in_the_middle_of_the_frame(tmp_path) -> None:
    """The end-to-end claim: a real ffmpeg cut of a moving target stays centred on it.

    The source is a white square crossing a dark frame; the "track" is its own box. The cut is then read back and
    the square is found in every sampled frame, near the middle - which is what "cut around this player" has to
    mean for the feature to be worth anything.
    """
    source = tmp_path / "crossing.mp4"
    width, height, fps = 640, 360, 10
    duration_s = 6.0
    square = 40
    times, boxes = [], []
    with VideoWriter(source, fps=float(fps), width=width, height=height) as writer:
        for i in range(int(fps * duration_s)):
            frame = np.zeros((height, width, 3), dtype=np.uint8)
            t = i / fps
            x = int(round(60 + (width - 120 - square) * (t / duration_s)))
            y = height // 2 - square // 2
            frame[y : y + square, x : x + square] = 255
            writer.write(frame)
            if i % 2 == 0:  # the tracker sees the player every other frame, as at the analysis rate
                times.append(t)
                boxes.append([x / width, y / width, (x + square) / width, (y + square) / width])

    plan = plan_framing(
        np.asarray(times), np.asarray(boxes),
        start_s=0.0, duration_s=duration_s, source_width=width, source_height=height,
        player_fraction=0.25, min_crop_height_px=140.0, smooth_s=0.0,
    )
    assert plan is not None and plan.crop_h <= height and plan.crop_w <= width
    out = tmp_path / "centred.mp4"
    command = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin", "-y",
        "-i", str(source), "-t", f"{duration_s:.3f}",
        "-vf", sendcmd_filter(plan, scale_width=320),
        "-c:v", "libx264", "-preset", "veryfast", "-pix_fmt", "yuv420p", str(out),
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    assert result.returncode == 0, f"ffmpeg failed: {result.stderr.strip()}"
    assert out.exists() and out.stat().st_size > 1000

    capture = cv2.VideoCapture(str(out))
    found, centres = 0, []
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        mask = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) > 200
        if mask.sum() > 50:
            found += 1
            xs = np.where(mask.any(axis=0))[0]
            centres.append(float(xs.mean()) / frame.shape[1])
    capture.release()
    assert found >= int(fps * duration_s) * 0.6, f"the player is missing from {found} output frames"
    worst = max(abs(centre - 0.5) for centre in centres)
    assert worst < 0.30, f"the crop drifts away from the player (worst centre offset {worst:.2f})"


def test_the_offered_clip_length_is_always_a_float() -> None:
    """An int default against float bounds kills the page - measured live, not hypothesised.

    ``round(17.6)`` is an int in Python; a slider created with ``value=18`` while its bounds are 5.0..60.0 is
    rejected by Streamlit with ``StreamlitInvalidParameterTypeError``, so every appearance shorter than the
    default broke the dashboard until the user picked a longer one. The helper returns a float for every span, and
    clamps to the range the page offers.
    """
    from soccer_analytics.analysis.framing import DEFAULT_CLIP_SECONDS, SHORTEST_CLIP_SECONDS, default_clip_length

    for span in (0.0, 2.0, 5.0, 17.6, 18.4, 23.6, 24.0, 45.0, 600.0):
        value = default_clip_length(span)
        assert isinstance(value, float), f"span {span} produced {type(value).__name__}, which the slider rejects"
        assert SHORTEST_CLIP_SECONDS <= value <= DEFAULT_CLIP_SECONDS
    assert default_clip_length(17.6) == 18.0, "a short appearance is offered its own length, rounded"
    assert default_clip_length(2.0) == SHORTEST_CLIP_SECONDS
    assert default_clip_length(600.0) == DEFAULT_CLIP_SECONDS
    # The bounds the page passes are floats too - the whole point of the type match.
    assert isinstance(SHORTEST_CLIP_SECONDS, float) and isinstance(DEFAULT_CLIP_SECONDS, float)
