"""The game video: how the clips are combined, and what the marks mean."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from soccer_analytics.analysis.game import (
    WINDOW_FIRST,
    WINDOW_SECOND,
    WINDOW_WHOLE,
    Clip,
    GameRecord,
    build_game,
    compatibility_problem,
    concat_list_text,
    find_for_video,
    half_labels_for,
    locations,
    plan,
)
from soccer_analytics.ingest.ffmpeg_reader import FFmpegError, VideoProbe, probe_video
from soccer_analytics.ingest.video_reader import VideoWriter


def _fake_probe(duration_s: float = 1800.0, **overrides) -> VideoProbe:
    base = dict(width=3840, height=2160, fps=60.0, duration_s=duration_s, codec="hevc", has_audio=True)
    base.update(overrides)
    return VideoProbe(**base)


def _record(duration_s: float = 6000.0) -> GameRecord:
    return GameRecord(game_id="game_x", output="/videos/game_x.mp4", duration_s=duration_s)


def test_clips_are_ordered_by_name_and_offset_into_one_clock(tmp_path: Path) -> None:
    """The camera names each clip by its start time, so the names put them in playing order."""
    later = tmp_path / "17:28:37.430.MP4"
    earlier = tmp_path / "16:28:37.784.MP4"
    later.write_bytes(b"x" * 10)
    earlier.write_bytes(b"y" * 20)
    durations = {str(earlier): 1800.0, str(later): 1200.0}

    result = plan([later, earlier], probe=lambda path: _fake_probe(durations[str(path)]))

    assert result.problem is None
    assert [Path(clip.path).name for clip in result.clips] == ["16:28:37.784.MP4", "17:28:37.430.MP4"]
    assert [clip.start_s for clip in result.clips] == [0.0, 1800.0]  # the second starts where the first ends
    assert [clip.duration_s for clip in result.clips] == [1800.0, 1200.0]


def test_clips_that_would_need_a_re_encode_are_refused_with_a_reason(tmp_path: Path) -> None:
    first, second = tmp_path / "a.mp4", tmp_path / "b.mp4"
    first.write_bytes(b"a")
    second.write_bytes(b"b")

    result = plan(
        [first, second],
        probe=lambda path: _fake_probe(width=1920, height=1080) if path.name == "b.mp4" else _fake_probe(),
    )

    assert result.problem is not None
    assert "clip 2" in result.problem and "1920x1080" in result.problem


def test_concat_list_escapes_a_quote_in_a_path() -> None:
    clip = Clip(path="/videos/it's here.mp4", start_s=0.0, duration_s=1.0, bytes=1)
    assert concat_list_text([clip]) == "file '/videos/it'\\''s here.mp4'\n"


def test_concat_list_writes_absolute_paths() -> None:
    """A relative clip named after its start time reads as a protocol (``10:``), and ffmpeg then reports the
    file as missing."""
    clip = Clip(path="data/videos/10:00:00.000.MP4", start_s=0.0, duration_s=1.0, bytes=1)
    line = concat_list_text([clip])
    assert line.startswith(f"file '{Path('data/videos/10:00:00.000.MP4').resolve()}'")
    assert line != "file 'data/videos/10:00:00.000.MP4'\n"


def test_marks_must_line_up_before_they_are_used() -> None:
    game = _record()
    assert game.mark_problem() == "still to mark: kick-off, half-time, full-time"

    game.set_mark("start", 100.0)
    game.set_mark("half", 3000.0)
    game.set_mark("end", 5900.0)
    assert game.mark_problem() is None

    game.set_mark("half", 50.0)  # before kick-off
    assert "out of order" in str(game.mark_problem())


def test_halves_and_windows_come_from_the_marks() -> None:
    game = _record()
    game.set_mark("start", 120.0)
    game.set_mark("half", 3000.0)
    game.set_mark("end", 5900.0)

    assert game.half_of(120.0) == 1
    assert game.half_of(2999.9) == 1
    assert game.half_of(3000.0) == 2
    assert game.half_of(5900.0) == 2
    assert game.half_of(60.0) is None  # before kick-off
    assert game.half_of(5950.0) is None  # after the final whistle

    assert game.window(WINDOW_WHOLE) == (120.0, 5900.0)
    assert game.window(WINDOW_FIRST) == (120.0, 3000.0)
    assert game.window(WINDOW_SECOND) == (3000.0, 5900.0)
    assert game.window_label(WINDOW_FIRST) == "first_half_120_3000"
    # The labels a table of events is annotated with: outside the game is not given a half.
    assert half_labels_for(game, [60.0, 1500.0, 3000.0, 5990.0]) == ["-", "1st half", "2nd half", "-"]


def test_a_mark_is_clamped_into_the_video() -> None:
    game = _record(duration_s=100.0)
    game.set_mark("start", -5.0)
    assert game.start_s == 0.0
    game.set_mark("end", 200.0)
    assert game.end_s == 100.0
    with pytest.raises(ValueError, match="unknown mark"):
        game.set_mark("kickoff", 1.0)


def test_an_unmarked_game_has_no_window() -> None:
    with pytest.raises(ValueError, match="not been marked"):
        _record().window(WINDOW_FIRST)


def test_clearing_the_marks_leaves_the_game_unmarked() -> None:
    game = _record()
    game.set_mark("start", 10.0)
    game.set_mark("half", 20.0)
    game.set_mark("end", 30.0)

    game.clear_marks()

    assert game.bounds() is None
    assert str(game.mark_problem()).startswith("still to mark")


def test_manifest_round_trips_and_is_found_from_its_video(tmp_path: Path) -> None:
    video = tmp_path / "game_16-28-37.784.mp4"
    video.write_bytes(b"v")
    clip = Clip(path=str(tmp_path / "16:28:37.784.MP4"), start_s=0.0, duration_s=5000.0, bytes=3)
    game = GameRecord(game_id="game_16-28-37.784_30", output=str(video), duration_s=5000.0, clips=[clip])
    game.set_mark("start", 90.0)
    game.set_mark("half", 2500.0)
    game.set_mark("end", 4900.0)
    directory = tmp_path / "games" / game.game_id
    game.save(directory)

    found = find_for_video(video, tmp_path / "games")
    assert found is not None
    assert found.game_id == game.game_id
    assert found.clips == [clip]
    assert found.half_of(2600.0) == 2
    assert find_for_video(tmp_path / "someone-elses.mp4", tmp_path / "games") is None


def test_a_relative_video_path_still_finds_its_game(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A manifest may name its video relatively; matching must resolve both sides rather than compare strings."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "game_16-28-37.784.mp4").write_bytes(b"v")
    record = GameRecord(game_id="g", output="game_16-28-37.784.mp4", duration_s=10.0)
    record.save(tmp_path / "games" / "g")

    assert find_for_video("game_16-28-37.784.mp4", tmp_path / "games") is not None
    assert find_for_video(str(tmp_path / "game_16-28-37.784.mp4"), tmp_path / "games") is not None


def _write_clip(path: Path, seconds: int, colour: int, fps: int = 10) -> None:
    with VideoWriter(path, fps=float(fps), width=160, height=90) as writer:
        for index in range(seconds * fps):
            frame = np.zeros((90, 160, 3), dtype=np.uint8)
            frame[:, :] = (colour, index % 255, 40)
            writer.write(frame)


def test_build_game_copies_two_clips_into_one_stream(tmp_path: Path) -> None:
    first = tmp_path / "16:00:00.000.MP4"
    second = tmp_path / "16:30:00.000.MP4"
    _write_clip(first, 4, 10)
    _write_clip(second, 3, 60)

    planned = plan([first, second])
    assert planned.problem is None
    combined = build_game(planned.clips, tmp_path / "game.mp4")

    assert combined.exists() and combined.stat().st_size > 1000
    # The joined video is as long as the two clips together - that is the whole point of the exercise.
    assert probe_video(combined).duration_s == pytest.approx(7.0, abs=0.4)
    assert not (tmp_path / "game.mp4.tmp.mp4").exists()


def test_a_failed_build_leaves_no_half_written_game(tmp_path: Path) -> None:
    clip = Clip(path=str(tmp_path / "missing.mp4"), start_s=0.0, duration_s=1.0, bytes=1)
    output = tmp_path / "game.mp4"

    with pytest.raises(FFmpegError):
        build_game([clip], output)

    assert not output.exists()


def test_one_clip_is_the_game_video_and_is_left_exactly_as_it_is(tmp_path: Path) -> None:
    """A game already merged in an earlier pass must be usable on its own, with nothing copied or re-encoded.

    The alternative - concatenating a single input onto itself - would rewrite a match-sized file to produce a
    byte-identical one, and a run interrupted halfway through that leaves the original truncated. So the single
    clip *is* the output, and the build must leave it untouched.
    """
    only = tmp_path / "game_16-28-37.784.mp4"
    _write_clip(only, 4, 12)
    before = only.read_bytes()

    planned = plan([only])

    assert planned.problem is None, "one clip has nothing to disagree with, so it is always combinable"
    assert len(planned.clips) == 1 and planned.clips[0].start_s == 0.0
    # The combined video is the clip itself, not a new file beside it.
    ordered, output, directory = locations([only])
    assert output == only
    assert not list(tmp_path.glob("game_16-28-37.784_*.mp4"))

    assert build_game(planned.clips, output) == only
    assert only.read_bytes() == before, "the game video must not be rewritten"
    assert probe_video(only).duration_s == pytest.approx(4.0, abs=0.4)
    assert directory.name.startswith("game_"), "the manifest and proxy still get their own directory"


def test_a_single_clip_does_not_need_a_second_one_to_be_accepted() -> None:
    """The minimum that used to be two is now one: a lone file is a valid game, not a half-finished selection."""
    assert compatibility_problem([_fake_probe()]) is None
    assert compatibility_problem([]) is None
    # Two clips still have to agree with each other.
    assert compatibility_problem([_fake_probe(), _fake_probe()]) is None
    assert compatibility_problem([_fake_probe(), _fake_probe(codec="h264")]) is not None
