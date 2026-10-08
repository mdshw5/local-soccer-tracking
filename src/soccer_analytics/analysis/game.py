"""The game as one continuous video, and the match's own clock on it.

The camera writes ~30-minute clips, so a game arrives as two or three files. Analysing them as separate videos
would restart the camera-motion chain and the player identities at every join, and there would be no single clock
to hang "kick-off", "half-time" and "the final whistle" on. Combining the clips costs no re-encode - they come from
one camera, so the streams are simply copied - and afterwards everything downstream (Stage A, frame grabs, audio,
reels) sees an ordinary single video.

What the combined video does not know is when the game actually was: recording starts before kick-off and runs on
after the final whistle. Those three moments are marked by scrubbing the low-resolution proxy, and stored here, in
the *game's own clock* - seconds into the combined video. Everything else (which half an event belongs to, which
window to analyse) follows from those three numbers.

The video itself is written next to the clips it came from - it is as large as the clips are, so it belongs on the
same disk, not in this repository. The manifest and the marking proxy live under ``data/games/<game_id>/``.
"""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path

from soccer_analytics.ingest.ffmpeg_reader import FFmpegError, VideoProbe, probe_video

REPO_ROOT = Path(__file__).resolve().parents[3]
GAMES_ROOT = REPO_ROOT / "data" / "games"

MANIFEST_FILE = "game.json"
BUILD_STATE_FILE = "build.json"
PROXY_FILE = "scrubber/proxy.mp4"  # same layout as a segment's scrubber, built by the same code
PROXY_WIDTH = 640  # only has to be watchable enough to spot kick-off and the final whistle
PROXY_FPS = 4.0

MARKS = ("start", "half", "end")
HALF_LABELS = {1: "1st half", 2: "2nd half"}

WINDOW_WHOLE = "Whole game"
WINDOW_FIRST = "First half"
WINDOW_SECOND = "Second half"
WINDOW_CHOICES = (WINDOW_WHOLE, WINDOW_FIRST, WINDOW_SECOND)


@dataclass(frozen=True)
class Clip:
    """One camera file, and where it begins in the combined game video."""

    path: str
    start_s: float
    duration_s: float
    bytes: int

    def to_json(self) -> dict:
        return {"path": self.path, "start_s": round(self.start_s, 3), "duration_s": round(self.duration_s, 3), "bytes": self.bytes}

    @classmethod
    def from_json(cls, data: dict) -> "Clip":
        return cls(path=str(data["path"]), start_s=float(data["start_s"]), duration_s=float(data["duration_s"]), bytes=int(data["bytes"]))


@dataclass(frozen=True)
class ClipPlan:
    """What a set of clips would become, and why they cannot be combined as-is (``problem`` is None when they can)."""

    clips: list[Clip]
    problem: str | None = None


@dataclass
class GameRecord:
    """One game video: the clips it was made from, and the three moments that define the game on its clock."""

    game_id: str
    output: str
    duration_s: float
    clips: list[Clip] = field(default_factory=list)
    start_s: float | None = None  # kick-off
    half_s: float | None = None  # half-time
    end_s: float | None = None  # the final whistle

    # --- the clock -----------------------------------------------------------------------------------------
    def bounds(self) -> tuple[float, float, float] | None:
        """``(start, half, end)`` once all three are marked, else ``None``."""
        if self.start_s is None or self.half_s is None or self.end_s is None:
            return None
        return float(self.start_s), float(self.half_s), float(self.end_s)

    def mark_problem(self) -> str | None:
        """Why the marks cannot be used yet, or ``None`` when they line up."""
        marks = {"kick-off": self.start_s, "half-time": self.half_s, "full-time": self.end_s}
        missing = [name for name, value in marks.items() if value is None]
        if missing:
            return "still to mark: " + ", ".join(missing)
        assert self.start_s is not None and self.half_s is not None and self.end_s is not None
        if not (self.start_s < self.half_s < self.end_s):
            return "the marks are out of order - kick-off must come before half-time, which must come before full-time"
        if self.end_s > self.duration_s + 1.0:
            return "full-time is past the end of the video"
        return None

    def set_mark(self, which: str, time_s: float) -> None:
        """Record kick-off, half-time or full-time, clamped into the video."""
        if which not in MARKS:
            raise ValueError(f"unknown mark {which!r}; expected one of {MARKS}")
        setattr(self, f"{which}_s", max(0.0, min(float(time_s), float(self.duration_s))))

    def clear_marks(self) -> None:
        """Forget all three marks; the video and the clips are untouched."""
        self.start_s = self.half_s = self.end_s = None

    def half_of(self, time_s: float) -> int | None:
        """Which half a moment belongs to, or ``None`` if it is outside the marked game."""
        bounds = self.bounds()
        if bounds is None:
            return None
        start, half, end = bounds
        if time_s < start or time_s > end:
            return None
        return 1 if time_s < half else 2

    def window(self, selection: str) -> tuple[float, float]:
        """The analysed ``(start, end)`` window in game seconds for a choice from :data:`WINDOW_CHOICES`."""
        bounds = self.bounds()
        if bounds is None:
            raise ValueError("the game has not been marked yet")
        start, half, end = bounds
        if selection == WINDOW_WHOLE:
            return start, end
        if selection == WINDOW_FIRST:
            return start, half
        if selection == WINDOW_SECOND:
            return half, end
        raise ValueError(f"unknown window {selection!r}; expected one of {WINDOW_CHOICES}")

    def window_label(self, selection: str) -> str:
        """A short, filename-safe name for the window, so each half gets its own segment directory."""
        start, end = self.window(selection)
        return f"{selection.replace(' ', '_').lower()}_{start:.0f}_{end:.0f}"

    # --- persistence ---------------------------------------------------------------------------------------
    def to_json(self) -> dict:
        return {
            "game_id": self.game_id,
            "output": self.output,
            "duration_s": round(self.duration_s, 3),
            "clips": [clip.to_json() for clip in self.clips],
            "start_s": None if self.start_s is None else round(self.start_s, 3),
            "half_s": None if self.half_s is None else round(self.half_s, 3),
            "end_s": None if self.end_s is None else round(self.end_s, 3),
        }

    @classmethod
    def from_json(cls, data: dict) -> "GameRecord":
        return cls(
            game_id=str(data["game_id"]),
            output=str(data["output"]),
            duration_s=float(data["duration_s"]),
            clips=[Clip.from_json(item) for item in data.get("clips", [])],
            start_s=None if data.get("start_s") is None else float(data["start_s"]),
            half_s=None if data.get("half_s") is None else float(data["half_s"]),
            end_s=None if data.get("end_s") is None else float(data["end_s"]),
        )

    def save(self, directory: str | Path) -> Path:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / MANIFEST_FILE
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(self.to_json(), indent=2))
        os.replace(tmp, path)
        return path

    @classmethod
    def load(cls, directory: str | Path) -> "GameRecord":
        return cls.from_json(json.loads((Path(directory) / MANIFEST_FILE).read_text()))


# --------------------------------------------------------------------------------------------------------------
# Planning and building the combined video
# --------------------------------------------------------------------------------------------------------------
def sanitise(name: str) -> str:
    """A filesystem-safe version of a clip name; the camera's names carry colons."""
    return "".join(c if c.isalnum() or c in "-_." else "-" for c in name)


def order_clips(paths: Iterable[str | Path]) -> list[Path]:
    """The clips in playing order: the camera names each one by its start time, so the name sorts it."""
    return sorted({Path(p) for p in paths}, key=lambda p: p.name)


def game_id_for(paths: list[Path]) -> str:
    """Stable id for a set of clips, keyed by name *and* size so a replaced clip is not mistaken for the same game."""
    first = Path(paths[0])
    total = sum(Path(p).stat().st_size for p in paths)
    return f"game_{sanitise(first.stem)}_{total}"


def output_for(paths: list[Path]) -> Path:
    """Where the combined video goes: beside the clips, because it is as large as they are.

    A single clip is already the game video and is returned as it stands. The alternative - concatenating one
    input onto itself - would copy tens of gigabytes to produce a byte-identical file, and on a filesystem that
    is nearly full that is how you lose a match you already have.
    """
    if len(paths) == 1:
        return Path(paths[0])
    first = Path(paths[0])
    return first.parent / f"game_{sanitise(first.stem)}.mp4"


def locations(paths: Iterable[str | Path], root: str | Path = GAMES_ROOT) -> tuple[list[Path], Path, Path]:
    """The ordered clips, the combined video's path, and the directory holding this game's manifest and proxy.

    One place decides all three, so the page and the background build cannot disagree about where anything goes.
    """
    ordered = order_clips(paths)
    if not ordered:
        raise ValueError("no clips given")
    return ordered, output_for(ordered), game_dir(root, game_id_for(ordered))


def compatibility_problem(probes: list[VideoProbe]) -> str | None:
    """Why these clips cannot be concatenated with a stream copy, or ``None`` when they can.

    Combining HEVC clips from one camera is a plain stream copy; clips that differ in codec or frame size would
    need a full re-encode of tens of gigabytes, so this refuses and says why rather than doing that silently.
    One clip has nothing to disagree with: it is combined with nothing, so it is always compatible.
    """
    if len(probes) < 2:
        return None
    reference = probes[0]
    for index, probe in enumerate(probes[1:], start=2):
        reasons = []
        if probe.codec != reference.codec:
            reasons.append(f"codec {probe.codec} vs {reference.codec}")
        if (probe.width, probe.height) != (reference.width, reference.height):
            reasons.append(f"frame {probe.width}x{probe.height} vs {reference.width}x{reference.height}")
        if abs(probe.fps - reference.fps) > 0.01:
            reasons.append(f"{probe.fps:.2f} fps vs {reference.fps:.2f} fps")
        if probe.has_audio != reference.has_audio:
            reasons.append("one has audio and the other does not")
        if reasons:
            return f"clip {index} does not match clip 1 ({', '.join(reasons)})"
    return None


def plan(paths: Iterable[str | Path], *, probe: Callable[[str | Path], VideoProbe] = probe_video) -> ClipPlan:
    """Probe the clips, give each its offset in the combined video, and check they can be combined as-is."""
    ordered = order_clips(paths)
    probes = [probe(path) for path in ordered]
    clips: list[Clip] = []
    offset = 0.0
    for path, info in zip(ordered, probes):
        clips.append(Clip(path=str(path), start_s=offset, duration_s=float(info.duration_s), bytes=Path(path).stat().st_size))
        offset += float(info.duration_s)
    return ClipPlan(clips=clips, problem=compatibility_problem(probes))


def concat_list_text(clips: list[Clip]) -> str:
    """The ffmpeg concat demuxer's input file. A quote in a path is escaped the way the demuxer expects.

    Paths are written absolute on purpose: the camera names its clips after their start time, and a *relative* path
    like ``10:00:00.000.MP4`` is read as the protocol ``10:`` and fails to open (the file is reported as missing).
    """
    def quoted(path: str) -> str:
        absolute = str(Path(path).resolve())
        return "'" + absolute.replace("'", "'\\''") + "'"

    return "".join(f"file {quoted(clip.path)}\n" for clip in clips)


def build_command(list_path: str | Path, output: str | Path) -> list[str]:
    """ffmpeg command for the combination: a stream copy, so combining 75 minutes takes minutes, not hours.

    ``+faststart`` is deliberately not used: it rewrites the whole file to move the index, which for tens of
    gigabytes of 4K is a long extra pass over the network share, and nothing here streams the file over HTTP anyway.
    """
    return [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin", "-y",
        "-f", "concat", "-safe", "0", "-i", str(list_path),
        "-c", "copy", "-map", "0", str(output),
    ]


def build_game(clips: list[Clip], output: str | Path) -> Path:
    """Concatenate the clips into one video, atomically, so a torn file is never mistaken for a finished game."""
    output = Path(output)
    sources = {Path(clip.path).resolve() for clip in clips}
    if output.resolve() in sources:
        # A single already-merged game file: the "combined" video is that file. Copying it onto itself would
        # rewrite a match-sized file for nothing, and replaces the original with a truncated one if it is
        # interrupted, so there is nothing to do here and saying so is the honest outcome.
        return output
    output.parent.mkdir(parents=True, exist_ok=True)
    tmp = output.with_name(output.name + ".tmp.mp4")
    with tempfile.TemporaryDirectory() as tmpdir:
        listing = Path(tmpdir) / "clips.txt"
        listing.write_text(concat_list_text(clips))
        result = subprocess.run(build_command(listing, tmp), capture_output=True, text=True)
    if result.returncode != 0 or not tmp.exists() or tmp.stat().st_size == 0:
        tmp.unlink(missing_ok=True)
        raise FFmpegError(f"could not combine the clips: {result.stderr.strip() or f'ffmpeg exited with {result.returncode}'}")
    os.replace(tmp, output)
    return output


# --------------------------------------------------------------------------------------------------------------
# Locations and lookup
# --------------------------------------------------------------------------------------------------------------
def game_dir(root: str | Path, game_id: str) -> Path:
    return Path(root) / game_id


def proxy_path(directory: str | Path) -> Path:
    return Path(directory) / PROXY_FILE


def build_state_path(directory: str | Path) -> Path:
    return Path(directory) / BUILD_STATE_FILE


def half_labels_for(record: GameRecord, times: Iterable[float]) -> list[str]:
    """The half each moment belongs to, for annotating a table of events (``"-"`` when it is outside the game)."""
    return [HALF_LABELS.get(record.half_of(float(time_s)), "-") for time_s in times]


def clip_offset_for(record: GameRecord, video: str | Path) -> float | None:
    """Where ``video`` starts inside the game's combined recording, or ``None`` when it is not one of its clips.

    A moment's seconds are seconds of the recording it was found in, and the game's clock is the combined video's.
    A whistle scanned on ``16:58:38.391.MP4`` reports 5.3 s, which is 1805.5 s of the game - the two clocks differ
    by this offset, and anything that asks a question about the game (which half, where on the timeline) has to
    translate first. The combined video itself maps to 0.
    """
    target = Path(video).resolve()
    if Path(record.output).resolve() == target:
        return 0.0
    for clip in record.clips:
        if Path(clip.path).resolve() == target:
            return float(clip.start_s)
    return None


def game_time(record: GameRecord, time_s: float, video: str | Path) -> float | None:
    """``time_s`` of ``video`` expressed on the game's own clock, or ``None`` when the recording is not part of it.

    The one place the translation lives, so the event table, the timeline strip and the half labels cannot disagree
    about when something happened.
    """
    offset = clip_offset_for(record, video)
    return None if offset is None else float(time_s) + offset


def half_labels_for_events(record: GameRecord, events: Iterable) -> list[str]:
    """The half each event belongs to, translating each one out of its own recording's clock first.

    An event carries the recording it was found in (``Event.video``); a whistle candidate's seconds are seconds of
    a single camera file while the game's marks are on the combined video's clock. Labelling without translating
    put every audio candidate in the first half - or outside the game entirely - because 5 s of a clip is not 5 s
    of the match.
    """
    labels: list[str] = []
    for event in events:
        video = getattr(event, "video", "") or record.output
        on_game = game_time(record, float(event.time_s), video)
        labels.append("-" if on_game is None else HALF_LABELS.get(record.half_of(on_game), "-"))
    return labels


def find_for_video(video: str | Path, root: str | Path = GAMES_ROOT) -> GameRecord | None:
    """The game record whose combined video is ``video``, if it was made by this app.

    Both sides are resolved before comparing: the page passes the absolute path the video picker found, while a
    manifest may name its video relatively (older builds did), and the two must still be recognised as the same.
    """
    target = Path(video).resolve()
    root = Path(root)
    if not root.exists():
        return None
    for directory in sorted(root.iterdir()):
        manifest = directory / MANIFEST_FILE
        if not manifest.exists():
            continue
        try:
            record = GameRecord.from_json(json.loads(manifest.read_text()))
        except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError):
            continue
        if Path(record.output).resolve() == target:
            return record
    return None


# --------------------------------------------------------------------------------------------------------------
# Background build state
# --------------------------------------------------------------------------------------------------------------
def read_build_state(directory: str | Path) -> dict:
    path = build_state_path(directory)
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return {}


def write_build_state(directory: str | Path, **changes) -> dict:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    state = read_build_state(directory)
    state.update(changes)
    path = build_state_path(directory)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(state, indent=2))
    os.replace(tmp, path)
    return state


def process_alive(pid: object) -> bool:
    """Whether a pid is running. Only used to decide if a build that never wrote a finish state died."""
    if not isinstance(pid, int) or pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:  # exists but owned by someone else - still alive
        return True
    return True


def build_in_progress(directory: str | Path) -> bool:
    state = read_build_state(directory)
    return state.get("state") == "running" and process_alive(state.get("pid"))
