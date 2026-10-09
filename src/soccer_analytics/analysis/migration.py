"""Move a repository-era analysis next to its footage, so every match becomes self-contained.

Before this, three flat roots in the repository held everything: ``data/matches`` the match records,
``data/segments`` the Stage A results, ``data/games`` the game manifests and marking proxies - none of them next
to the video they describe. The layout now is one ``analysis/<id>`` directory beside the footage, and this module
moves the old data into it:

* a match directory moves to the analysis directory of the video its segment was built on (or, when there is no
  readable segment, of its first recorded source); every artefact inside it moves with it;
* the segment directories the record names move into that directory's ``segments/``; the record is rewritten with
  the new relative paths and with its source pointing at the analysed video, and its id becomes the analysis
  directory's name;
* the game manifest (and its marking proxy) for that video moves in beside the record;
* segment directories and game manifests no record references are still moved to the analysis directory of their
  own video, so no computed data is left behind in the repository.

Everything is a move; nothing is deleted except the emptied legacy directories. ``dry_run`` reports what would
move without touching anything.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from soccer_analytics.analysis import game as game_lib
from soccer_analytics.analysis.library import (
    LEGACY_MATCHES_ROOT,
    REPO_ROOT,
    MatchLibrary,
    analysis_dir_for,
)


def _move_into(source: Path, target: Path, log) -> None:
    """Move a file or directory to ``target``, merging directories and replacing files; creates parents."""
    if not source.exists():
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and target.is_dir() and source.is_dir():
        for child in sorted(source.iterdir()):
            _move_into(child, target / child.name, log)
        source.rmdir()
        return
    if target.exists():
        log(f"    replacing existing {target}")
        if target.is_dir():
            shutil.rmtree(target)
        else:
            target.unlink()
    log(f"    {source} -> {target}")
    shutil.move(str(source), str(target))


def _segment_video(segment_dirs: list[str]) -> str | None:
    """The video the segments were analysed from: the first readable ``meta.json`` that names one."""
    for entry in segment_dirs:
        meta_path = Path(entry) / "meta.json"
        try:
            video = str(json.loads(meta_path.read_text()).get("video") or "")
        except (OSError, json.JSONDecodeError):
            continue
        if video:
            return video
    return None


def _game_dirs_for_video(video: str, games_root: Path) -> list[Path]:
    """Legacy game manifest directories whose combined video is ``video``, a marked one first."""
    matches: list[tuple[bool, Path]] = []
    if not games_root.exists():
        return []
    for directory in sorted(games_root.iterdir()):
        manifest = directory / game_lib.MANIFEST_FILE
        if not manifest.exists():
            continue
        try:
            record = game_lib.GameRecord.from_json(json.loads(manifest.read_text()))
        except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError):
            continue
        if Path(record.output).expanduser().resolve() == Path(video).expanduser().resolve():
            matches.append((record.bounds() is not None, directory))
    # A marked manifest is the one with the user's work in it; an unmarked duplicate is a degenerate leftover.
    matches.sort(key=lambda entry: (not entry[0], entry[1].name))
    return [directory for _marked, directory in matches]


def migrate_all(
    repo_root: str | Path = REPO_ROOT,
    *,
    dry_run: bool = False,
    log=print,
) -> dict:
    """Move every legacy match, segment and game manifest beside its footage. Returns what moved."""
    repo_root = Path(repo_root)
    matches_root = repo_root / "data" / "matches"
    segments_root = repo_root / "data" / "segments"
    games_root = repo_root / "data" / "games"

    def act(source: Path, target: Path) -> None:
        if dry_run:
            log(f"    would move {source} -> {target}")
        else:
            _move_into(source, target, log)

    def rewrite_game_manifest(target: Path) -> None:
        """Rewrite a game manifest that has just been moved, so its raw file is portable too.

        The move itself keeps the stored paths working on this machine, but they are absolute and name the old
        layout's location; re-saving stores the output and the clips relative to the footage directory, which is
        what makes a copied folder need no fixing up at all.
        """
        if dry_run:
            log("    would rewrite game.json with portable paths")
            return
        try:
            game_lib.GameRecord.load(target).save(target)
        except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
            log(f"    could not rewrite game.json: {exc}")

    summary = {"matches": 0, "segments": 0, "games": 0, "skipped": []}
    # Sources this run has already dealt with (moved, or deliberately left behind): the sweep at the end must
    # not move them again - in a dry run nothing has actually moved yet, and a left-behind duplicate would
    # otherwise overwrite the marked manifest in the target directory.
    handled: set[Path] = set()

    def _claimed(source: Path) -> bool:
        return any(source == entry or source.is_relative_to(entry) for entry in handled)

    # --- matches ---------------------------------------------------------------------------------------------
    if matches_root.exists():
        library = MatchLibrary(matches_root)
        for match_id in library.list_ids():
            record = library.load(match_id)
            legacy_dir = library.path(match_id)
            video = _segment_video(record.segments) or (record.sources[0] if record.sources else None)
            if not video:
                summary["skipped"].append(f"{match_id}: no segment or source names a video")
                continue
            target = analysis_dir_for(video)
            log(f"match {match_id}:")
            log(f"  video: {video}")
            log(f"  target: {target}")
            act(legacy_dir, target)
            handled.add(legacy_dir)

            # Segments move into the match directory's own ``segments/`` folder, and the record remembers them
            # relative to the analysis directory (save() does the relativising), so the folder is portable.
            moved_segments: list[str] = []
            for entry in record.segments:
                source = Path(entry)
                if not source.exists() and not _claimed(source):
                    summary["skipped"].append(f"{match_id}: segment not found: {source}")
                    continue
                destination = target / "segments" / source.name
                if source.parent.resolve() != destination.parent.resolve():
                    act(source, destination)
                handled.add(source)
                moved_segments.append(str(destination))
            record.segments = moved_segments

            # The game manifest for this video moves in beside the record (first choice: the marked one).
            for index, game_dir in enumerate(_game_dirs_for_video(video, games_root)):
                handled.add(game_dir)
                if index == 0:
                    act(game_dir, target)
                    rewrite_game_manifest(target)
                    summary["games"] += 1
                else:
                    log(f"  leaving duplicate game manifest at {game_dir} (same video, no marks preferred)")

            record.match_id = target.name
            record.sources = [video]
            if dry_run:
                log(f"    would rewrite match.json (id {target.name}, {len(record.segments)} segment(s))")
            else:
                library.save(record, directory=target)
            summary["matches"] += 1

    # --- anything a record did not claim ---------------------------------------------------------------------
    if segments_root.exists():
        for directory in sorted(segments_root.iterdir()):
            if not directory.is_dir() or _claimed(directory):
                continue
            video = _segment_video([str(directory)])
            if not video:
                summary["skipped"].append(f"segment {directory.name}: no meta.json naming a video")
                continue
            target = analysis_dir_for(video) / "segments" / directory.name
            log(f"segment {directory.name}:")
            act(directory, target)
            summary["segments"] += 1
        if not dry_run and segments_root.exists() and not any(segments_root.iterdir()):
            segments_root.rmdir()

    if games_root.exists():
        for directory in sorted(games_root.iterdir()):
            manifest = directory / game_lib.MANIFEST_FILE
            if not manifest.exists() or _claimed(directory):
                continue
            try:
                record = game_lib.GameRecord.from_json(json.loads(manifest.read_text()))
            except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError):
                summary["skipped"].append(f"game {directory.name}: manifest unreadable")
                continue
            target = analysis_dir_for(record.output)
            log(f"game {directory.name}:")
            act(directory, target)
            rewrite_game_manifest(target)
            summary["games"] += 1
        if not dry_run and games_root.exists() and not any(games_root.iterdir()):
            games_root.rmdir()

    if not dry_run and matches_root.exists() and not any(matches_root.iterdir()):
        matches_root.rmdir()
    return summary
