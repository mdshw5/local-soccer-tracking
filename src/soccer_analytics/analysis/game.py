"""The game as a set of camera clips read as one continuous recording, and the match's own clock on it.

The camera writes ~30-minute clips, so a game arrives as two or three files. Analyzing them as separate videos
would restart the camera-motion chain and the player identities at every join, and there would be no single clock
to hang "kick-off", "half-time" and "the final whistle" on. The game is therefore described by this manifest:
the clips in playing order with the game-clock second each begins at. Readers resolve a game time to the clip that
contains it and decode clip after clip (`ingest.source`), so the joins are invisible to Stage A and the scans -
and no merged video has to exist. (The old workflow stream-copied the clips into one file; `build_game` remains as
an optional utility, but nothing in the pipeline needs it and the dashboard does not offer it.)

What the clips do not know is when the game actually was: recording starts before kick-off and runs on after the
final whistle. Those three moments are marked by scrubbing the marking stream (frames straight from the clips),
and stored here, in the *game's own clock*. Everything else (which half an event belongs to, which window to
analyze) follows from those three numbers.

The manifest lives in the analysis directory of the game itself (``<footage>/analysis/<id>/game.json``), beside
the match record for the same footage, so a match directory carries its game with it. The old ``data/games`` root
stays readable for archives from before the move.
"""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field, replace
from pathlib import Path

from soccer_analytics.analysis.library import (
    ANALYSIS_DIRNAME,
    analysis_dir_for,
    resolve_path,
    store_path,
    video_roots,
)
from soccer_analytics.ingest.ffmpeg_reader import FFmpegError, VideoProbe, probe_video

REPO_ROOT = Path(__file__).resolve().parents[3]
GAMES_ROOT = REPO_ROOT / "data" / "games"

MANIFEST_FILE = "game.json"
BUILD_STATE_FILE = "build.json"
PROXY_FILE = "scrubber/proxy.mp4"  # where the marking video is cached, beside the game's manifest
PROXY_WIDTH = 640  # the marking navigator's default size: watchable enough to spot kick-off and the final whistle
PROXY_FPS = 4.0  # its default frame rate (a long game decodes keyframes only, so ~1 picture/second either way)

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
        """The analyzed ``(start, end)`` window in game seconds for a choice from :data:`WINDOW_CHOICES`."""
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
        """Write the manifest, storing its paths relative to the footage directory when they live there.

        Relative-when-beside is what lets an archive move with its footage: the manifest now sits in
        ``<footage>/analysis/<id>/`` while the videos it names sit in ``<footage>`` itself, so a copy of the
        folder resolves every name without fixing anything up.
        """
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        base = _path_base(directory)
        data = self.to_json()
        data["output"] = store_path(self.output, base)
        data["clips"] = [{**clip.to_json(), "path": store_path(clip.path, base)} for clip in self.clips]
        path = directory / MANIFEST_FILE
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(data, indent=2))
        os.replace(tmp, path)
        return path

    @classmethod
    def load(cls, directory: str | Path) -> "GameRecord":
        directory = Path(directory)
        record = cls.from_json(json.loads((directory / MANIFEST_FILE).read_text()))
        return resolve_record_paths(record, directory)


# --------------------------------------------------------------------------------------------------------------
# Planning and building the combined video
# --------------------------------------------------------------------------------------------------------------
def sanitize(name: str) -> str:
    """A filesystem-safe version of a clip name; the camera's names carry colons."""
    return "".join(c if c.isalnum() or c in "-_." else "-" for c in name)


def order_clips(paths: Iterable[str | Path]) -> list[Path]:
    """The clips in playing order: the camera names each one by its start time, so the name sorts it."""
    return sorted({Path(p) for p in paths}, key=lambda p: p.name)


def game_id_for(paths: list[Path]) -> str:
    """Stable id for a set of clips, keyed by name *and* size so a replaced clip is not mistaken for the same game."""
    first = Path(paths[0])
    total = sum(Path(p).stat().st_size for p in paths)
    return f"game_{sanitize(first.stem)}_{total}"


def output_for(paths: list[Path]) -> Path:
    """The path the combination *would* have had; nothing writes it any more.

    It survives because it is how the game's analysis directory is derived from the clips (the directory is named
    after it - see :func:`analysis_id_for`), and because a single clip is already the game video and is returned
    as it stands. A multi-clip game's manifest records the same name in ``output`` so archives that only know the
    old layout can still find their game; the file itself no longer exists, and no reader needs it to.
    """
    if len(paths) == 1:
        return Path(paths[0])
    first = Path(paths[0])
    return first.parent / f"game_{sanitize(first.stem)}.mp4"


def locations(
    paths: Iterable[str | Path], root: str | Path | None = None
) -> tuple[list[Path], Path, Path]:
    """The ordered clips, the combined video's path, and the directory holding this game's manifest and proxy.

    One place decides all three, so the page and the background build cannot disagree about where anything goes.
    Without ``root`` the manifest directory is the combined video's own analysis directory, beside the footage;
    ``root`` keeps the old "one directory per game id" layout, which the repo's first archives still use.
    """
    ordered = order_clips(paths)
    if not ordered:
        raise ValueError("no clips given")
    output = output_for(ordered)
    directory = game_dir(root, game_id_for(ordered)) if root is not None else manifest_dir_for_video(output)
    return ordered, output, directory


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
    """The legacy layout's directory for a game id (one directory per game under ``data/games``)."""
    return Path(root) / game_id


def _path_base(directory: Path) -> Path:
    """The folder a manifest's paths are stored relative to.

    The analysis layout puts the manifest at ``<footage>/analysis/<id>/`` and the videos it names in ``<footage>``
    itself, two levels up; any other directory (the legacy ``data/games/<id>``, a test's scratch folder) is its
    own base, so paths pointing outside it stay absolute.
    """
    directory = Path(directory)
    if directory.parent.name == ANALYSIS_DIRNAME:
        return directory.parent.parent
    return directory


def resolve_record_paths(record: GameRecord, directory: str | Path) -> GameRecord:
    """Re-point a record's stored paths at wherever its files are now, in place.

    Called wherever a manifest is read (``load`` and the discovery in :func:`find_for_video`), so a footage
    directory moved to another disk keeps its game clock, its clip offsets and its marking proxy: a stored path
    that no longer exists is looked up by name beside the manifest - which is where the videos sit, whether the
    manifest was written portable (relative) or by an older version (absolute at the old location).
    """
    directory = Path(directory)
    base = _path_base(directory)
    record.output = resolve_path(record.output, base)
    record.clips = [replace(clip, path=resolve_path(clip.path, base)) for clip in record.clips]
    return record


def manifest_dir_for_video(video: str | Path) -> Path:
    """Where this video's game manifest and marking proxy live: its own analysis directory, beside the footage."""
    return analysis_dir_for(video)


def discover_game_manifests(roots: list[Path] | None = None) -> list[Path]:
    """Every ``analysis/<id>/game.json`` under the footage roots (manifests written by this version)."""
    found: list[Path] = []
    for root in roots if roots is not None else video_roots():
        if root.exists():
            found += list(root.glob(f"**/{ANALYSIS_DIRNAME}/*/{MANIFEST_FILE}"))
    return found


def find_dir_by_id(game_id: str, roots: list[Path] | None = None) -> Path | None:
    """The directory holding the manifest with this game id, beside the footage first, then the legacy root.

    The id is what the stream server's URLs carry, and it is the only thing a caller with no path knows; both
    layouts are searched so a URL keeps working across the move.
    """
    for manifest in discover_game_manifests(roots):
        try:
            data = json.loads(manifest.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if str(data.get("game_id")) == game_id:
            return manifest.parent
    legacy = game_dir(GAMES_ROOT, game_id)
    return legacy if (legacy / MANIFEST_FILE).exists() else None


def manifest_dir(record: GameRecord) -> Path:
    """Where this record's manifest lives: beside its video when built by this version, else the legacy root."""
    beside = manifest_dir_for_video(record.output)
    if (beside / MANIFEST_FILE).exists():
        return beside
    legacy = game_dir(GAMES_ROOT, record.game_id)
    return legacy if (legacy / MANIFEST_FILE).exists() else beside


def proxy_path(directory: str | Path) -> Path:
    return Path(directory) / PROXY_FILE


def build_state_path(directory: str | Path) -> Path:
    return Path(directory) / BUILD_STATE_FILE


def half_labels_for(record: GameRecord, times: Iterable[float]) -> list[str]:
    """The half each moment belongs to, for annotating a table of events (``"-"`` when it is outside the game)."""
    return [HALF_LABELS.get(record.half_of(float(time_s)), "-") for time_s in times]


def prepare(paths: Iterable[str | Path]) -> tuple[GameRecord, Path]:
    """Plan the clips, keep the earlier marks, and write the manifest: no video is produced or copied.

    This is what replaced the combination. The record names every clip and where it starts on the game clock;
    Stage A, the marking stream and every frame grab resolve game time to a clip themselves, so the tens of
    gigabytes of duplicate file, the build step and its failure modes are gone. Re-running over the same clips
    preserves the marks - where half-time is does not change because the manifest was rewritten.
    """
    ordered, output, directory = locations(paths)
    planned = plan(ordered)
    if planned.problem:
        raise ValueError(f"these clips cannot form one game: {planned.problem}")
    duration = sum(float(clip.duration_s) for clip in planned.clips)
    record = GameRecord(game_id=directory.name, output=str(output), duration_s=duration, clips=planned.clips)
    manifest = directory / MANIFEST_FILE
    if manifest.exists():
        previous = GameRecord.load(directory)
        if [clip.path for clip in previous.clips] == [clip.path for clip in planned.clips]:
            record.start_s, record.half_s, record.end_s = previous.start_s, previous.half_s, previous.end_s
    record.save(directory)
    return record, directory


def manifest_for(record: GameRecord) -> Path | None:
    """The record's own manifest file, when it exists: the game's source of truth for readers."""
    path = manifest_dir(record) / MANIFEST_FILE
    return path if path.exists() else None


def clip_offset_for(record: GameRecord, video: str | Path) -> float | None:
    """Where ``video`` starts inside the game's recording, or ``None`` when it is not part of it.

    A moment's seconds are seconds of the recording it was found in, and the game's clock is the game's own. A
    whistle scanned on ``16:58:38.391.MP4`` reports 5.3 s, which is 1805.5 s of the game - the two clocks differ by
    this offset, and anything that asks a question about the game (which half, where on the timeline) has to
    translate first. The manifest itself (the game's source in the never-merged workflow) and the legacy combined
    video both map to 0.
    """
    target = Path(video).resolve()
    manifest = manifest_for(record)
    if manifest is not None and manifest.resolve() == target:
        return 0.0
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
    a single camera file while the game's marks are on the combined video's clock. Labeling without translating
    put every audio candidate in the first half - or outside the game entirely - because 5 s of a clip is not 5 s
    of the match.
    """
    labels: list[str] = []
    for event in events:
        video = getattr(event, "video", "") or record.output
        on_game = game_time(record, float(event.time_s), video)
        labels.append("-" if on_game is None else HALF_LABELS.get(record.half_of(on_game), "-"))
    return labels


def find_for_video(video: str | Path, root: str | Path | None = None) -> GameRecord | None:
    """The game record this path belongs to, if it was made by this app.

    The path may be a clip, the legacy combined video, or the game's own manifest - the never-merged workflow
    analyzes the manifest, while the picker still hands over clips. Both sides are resolved before comparing: the
    page passes the absolute path the video picker found, while a manifest may name its files relatively, and the
    two must still be recognized as the same. The video's own analysis directory is checked first; then, unless a
    specific legacy ``root`` is given, the manifests beside the rest of the footage and the old ``data/games``
    archive.
    """
    target = Path(video).resolve()
    beside = manifest_dir_for_video(video) / MANIFEST_FILE
    candidates: list[Path] = [beside] if beside.exists() else []
    if target.name == MANIFEST_FILE and target.exists():
        candidates.insert(0, target)
    if root is not None:
        base = Path(root)
        if base.exists():
            candidates += [d / MANIFEST_FILE for d in sorted(base.iterdir())]
    else:
        if GAMES_ROOT.exists():
            candidates += [d / MANIFEST_FILE for d in sorted(GAMES_ROOT.iterdir())]
        candidates += discover_game_manifests()
    seen: set[Path] = set()
    best: GameRecord | None = None
    for manifest in candidates:
        if manifest in seen or not manifest.exists():
            continue
        seen.add(manifest)
        try:
            record = GameRecord.from_json(json.loads(manifest.read_text()))
        except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError):
            continue
        resolve_record_paths(record, manifest.parent)
        if not _record_covers_path(record, manifest, target):
            continue
        # Several manifests can describe the same game (a rebuild on a new game id); a marked one carries the
        # user's kick-off/half-time/full-time work, so it wins over an unmarked duplicate.
        if best is None or (record.bounds() is not None and best.bounds() is None):
            best = record
    return best


def _record_covers_path(record: GameRecord, manifest: Path, target: Path) -> bool:
    """Whether a game's path is the target: its manifest, its legacy combined video, or one of its clips."""
    if manifest.resolve() == target:
        return True
    if Path(record.output).resolve() == target:
        return True
    return any(Path(clip.path).resolve() == target for clip in record.clips)


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
