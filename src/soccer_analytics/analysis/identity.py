"""Unique players: which of the match's tracked appearances are the same person.

Stage B's tracks are honest about what a single following camera can see - and the consequence is hundreds of
appearances per match, because the camera leaves a player and meets them again. This module answers the question
those fragments raise: *which of these are the same person?*

The answer here is not a similarity score. **Appearance embeddings were measured on the real game and do not
separate footballers on this footage** (the numbers are in the README): a CLIP embedding of a player's crop puts
different players at cosine 0.93 while one player's own crops are at 0.90; a purpose-trained person-ReID model does
better but still only identifies the right track 36% of the time, because a following camera keeps the far side of
the pitch in frame and a 60-100 px motion-blurred footballer in a team kit is out of distribution for models
trained on pedestrians. Merging on that would fuse different players' clips into one "player", which is a lie the
page cannot take back.

What the footage *does* support is exact, and already built: **the shirt number**. The number scan
(``scripts/extract_jerseys.py``) reports a number for a track only when several OCR readings agree, and the user can
type one in themselves; the number plus the team identifies the player. So identities are formed from what is
known - team and number (or name) - and appearances nobody has named stay their own identity rather than being
guessed at.

The other half of the feature lives in ``analysis.framing``: a clip cut so the camera follows one of these
appearances, cropped around the player rather than around the ball.
"""

from __future__ import annotations

import numpy as np

# Boxes kept per appearance for cutting a centered clip. The crop follows the player at the video's own rate; a
# shorter trajectory would cut the corners off a run, and 240 points cover 48 s at the analysis rate, longer than
# the clip the page cuts by default.
MAX_TRAJECTORY = 240


class Appearance:
    """One tracked player, once: when they were seen and where they were in the frame.

    ``traj_t`` are source seconds and ``traj_box`` the matching ``(x1, y1, x2, y2)`` rectangles normalized by frame
    width - the player's own boxes, which is what the centered cut needs. Everything else is summary.
    """

    __slots__ = ("track_id", "team", "first_t", "last_t", "first_frame", "last_frame", "traj_t", "traj_box")

    def __init__(
        self,
        track_id: int,
        team: int,
        first_t: float = 0.0,
        last_t: float = 0.0,
        first_frame: int = 0,
        last_frame: int = 0,
        traj_t: np.ndarray | None = None,
        traj_box: np.ndarray | None = None,
    ):
        self.track_id = int(track_id)
        self.team = int(team)
        self.first_t = float(first_t)
        self.last_t = float(last_t)
        self.first_frame = int(first_frame)
        self.last_frame = int(last_frame)
        self.traj_t = traj_t
        self.traj_box = traj_box

    @property
    def span_s(self) -> float:
        """How long this appearance lasts, from its first observation to its last."""
        return max(0.0, self.last_t - self.first_t)

    def trajectory(self, *, start_s: float | None = None, max_points: int = MAX_TRAJECTORY) -> tuple[np.ndarray, np.ndarray]:
        """The (times, boxes) this appearance can be cut from, starting no earlier than ``start_s``.

        Returns empty arrays when nothing was stored, which the caller reports rather than framing a guess. The
        trajectory is thinned evenly when it is longer than ``max_points``, keeping both ends.
        """
        if self.traj_t is None or self.traj_box is None:
            return np.zeros(0), np.zeros((0, 4))
        times = np.asarray(self.traj_t, dtype=np.float64)
        boxes = np.asarray(self.traj_box, dtype=np.float64)
        if start_s is not None:
            keep = times >= float(start_s) - 1e-6
            times, boxes = times[keep], boxes[keep]
        if len(times) > max_points:
            chosen = np.unique(np.linspace(0, len(times) - 1, max_points).round().astype(int))
            times, boxes = times[chosen], boxes[chosen]
        return times, boxes


class Identity:
    """One person, as a set of appearances. ``label`` is what the roster knows them as."""

    __slots__ = ("identity_id", "team", "members", "first_t", "last_t", "grouped_by", "label")

    def __init__(
        self,
        identity_id: int,
        team: int,
        members: tuple[int, ...],
        first_t: float,
        last_t: float,
        grouped_by: str = "",
        label: str = "",
    ):
        self.identity_id = int(identity_id)
        self.team = int(team)
        self.members = tuple(int(v) for v in members)
        self.first_t = float(first_t)
        self.last_t = float(last_t)
        # How the appearances were recognized as one person: "number", "name", or "" for a single appearance.
        self.grouped_by = grouped_by
        self.label = label

    @property
    def span_s(self) -> float:
        """From the person's first appearance to their last - not the time they were on screen, which is
        the sum of the appearances' own spans (see :func:`identity_rows`)."""
        return max(0.0, self.last_t - self.first_t)


def identities_from_labels(
    appearances,
    numbers: dict[int, dict] | None = None,
) -> list[Identity]:
    """Group appearances by what is *known* about their player: the team, and the shirt number or name.

    This is exact rather than statistical. A number read off the footage - with enough agreeing readings for the
    shirt-number scan to trust it - or typed in by the user identifies the player, and two appearances wearing it
    for the same team are that person. It is the identity signal this footage supports: appearance embeddings do
    not separate players here (measured, see the module docstring), but a shirt number does.

    Appearances with no number or name stay their own identity, and two players of the same team who shared a
    number (a squad that reissued one) would be grouped together - which is why the page shows each person's
    appearance list and not only the name.
    """
    numbers = numbers or {}
    ordered = sorted(
        appearances.values() if isinstance(appearances, dict) else appearances,
        key=lambda appearance: appearance.track_id,
    )
    groups: dict[tuple, list[Appearance]] = {}
    for appearance in ordered:
        entry = numbers.get(int(appearance.track_id), {})
        number = entry.get("number")
        name = str(entry.get("name") or "").strip()
        key: tuple = (appearance.track_id,)  # an appearance nobody has named is its own identity
        if number:
            key = ("number", int(appearance.team), int(number))
        elif name:
            key = ("name", int(appearance.team), name.casefold())
        groups.setdefault(key, []).append(appearance)
    # Identities are numbered by their first appearance, so the ids do not change between page reloads.
    keys = sorted(groups, key=lambda key: groups[key][0].track_id)
    out: list[Identity] = []
    for identity_id, key in enumerate(keys):
        members = groups[key]
        track_ids = tuple(sorted(appearance.track_id for appearance in members))
        entry = numbers.get(track_ids[0], {})
        first = min(members, key=lambda appearance: (appearance.first_t, appearance.track_id))
        last = max(members, key=lambda appearance: (appearance.last_t, appearance.track_id))
        if entry.get("number"):
            label = f"#{entry['number']}" + (f" {entry['name']}" if entry.get("name") else "")
        elif entry.get("name"):
            # The user's own text, from the first appearance of the group: a name is not a field to normalize.
            label = str(entry["name"])
        else:
            label = f"track {track_ids[0]}"
        out.append(
            Identity(
                identity_id=identity_id,
                team=members[0].team,
                members=track_ids,
                first_t=first.first_t,
                last_t=last.last_t,
                grouped_by=("number" if entry.get("number") else "name") if len(track_ids) > 1 else "",
                label=label,
            )
        )
    return out


def identity_rows(
    identities: list[Identity],
    appearances,
    team_of_label,
) -> list[dict]:
    """The identity table as plain rows, longest seen first, ready for a dataframe.

    ``seen_s`` is the time the person was actually on screen - the sum of their appearances' spans, which is not
    the same as ``last_t - first_t`` when the camera was elsewhere in between. Appearances that were not measured
    cannot contribute time, so only those present in ``appearances`` count.
    """
    by_id = appearances if isinstance(appearances, dict) else {item.track_id: item for item in appearances}
    rows = []
    for identity in identities:
        seen = sum(by_id[track_id].span_s for track_id in identity.members if track_id in by_id)
        rows.append(
            {
                "player": identity.label,
                "team": team_of_label(identity.team),
                "appearances": len(identity.members),
                "first_t": identity.first_t,
                "last_t": identity.last_t,
                "seen_s": round(seen, 1),
                "grouped_by": identity.grouped_by or "-",
                "tracks": list(identity.members),
            }
        )
    rows.sort(key=lambda row: (-row["seen_s"], row["first_t"]))
    return rows


def appearances_from_players(
    players,
    *,
    frames_per_second: float,
    start_s: float = 0.0,
    fps: float | None = None,
    boxes: dict | None = None,
):
    """Build appearances from the replay payload's players, with the player's own boxes when there are any.

    The payload carries each track's frame indices and pitch positions (and the team); the *boxes* live in a
    separate small file beside it, because they are only needed here - to cut a clip that follows the player in
    the picture - and would otherwise make the payload the browser fetches tens of megabytes larger.

    The cut needs source seconds on the segment's video, which is ``start_s + frame / fps``. ``frames_per_second``
    is the rate the payload was produced at (the analysis rate); ``start_s`` is where the analyzed window begins in
    the video the clip will be cut from. Getting this wrong is the bug the shirt-number scan already paid for once:
    a frame index is not a time until it is divided by the rate and shifted by the window's start.
    """
    rate = float(fps if fps is not None else frames_per_second)
    boxes = boxes or {}
    out: dict[int, Appearance] = {}
    for player in players:
        track_id = int(player["track_id"])
        frames = np.asarray(player.get("frames") or [], dtype=np.float64)
        if len(frames) == 0:
            continue
        times = start_s + frames / max(rate, 1e-6)
        order = np.argsort(times)
        times = times[order]
        stored = boxes.get(str(track_id))
        box_array = np.asarray(stored, dtype=np.float64)[order] if stored is not None and len(stored) == len(frames) else None
        out[track_id] = Appearance(
            track_id=track_id,
            team=int(player.get("team", -1)),
            first_t=float(times[0]),
            last_t=float(times[-1]),
            first_frame=int(frames[0]),
            last_frame=int(frames[-1]),
            traj_t=times,
            traj_box=box_array,
        )
    return out


def numbers_are_stale(
    stored_tracks,
    present_tracks,
    *,
    scan_calibration_saved: float | None,
    calibration_saved: float | None,
    tolerance_s: float = 1.0,
) -> str:
    """Why the stored shirt numbers cannot be trusted for this report, or "" when they can.

    Shirt numbers are attached to *track ids*, and track ids are properties of one build: re-fitting the pitch
    moves players between tracks, so a scan read against an older fit can put a number on the wrong person. Two
    signals catch it without needing to store much: ids the scan reported that are not in this report at all
    (they cannot be anyone here), and a scan saved before the calibration that is on disk now. A scan whose
    provenance is unknown (an older file with no timestamp) is not called stale - it may well be right - but ids
    that have vanished always are.
    """
    missing = set(int(track) for track in stored_tracks) - set(int(track) for track in present_tracks)
    if missing:
        return f"{len(missing)} stored number(s) refer to track ids that are not in this report"
    if scan_calibration_saved is None or calibration_saved is None:
        return ""
    if abs(float(scan_calibration_saved) - float(calibration_saved)) > tolerance_s:
        return "the numbers were read before the pitch was last calibrated"
    return ""
