"""Small helpers that turn what is on disk into what the page shows."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from soccer_analytics.analysis.library import MatchLibrary

DEFAULT_TEAM_NAMES = ("Team 1", "Team 2")


def is_default_team_name(name: str) -> bool:
    """Whether a team is still going by a placeholder - i.e. the user has not named it yet.

    Both spellings live in the archive: the match record's own default ("Team A"/"Team B") and the one the page
    falls back to ("Team 1"/"Team 2"). A blank name counts too, so a suggestion can fill it in.
    """
    return name.strip().lower() in {"", "team a", "team b", "team 1", "team 2"}


def team_name(team: int, names: Sequence[str] | None = None) -> str:
    """What to call a team: the name the user gave it, else ``"Team 1"``/``"Team 2"``.

    Teams are numbered 0/1 by the kit clustering - the more red kit is 0 - and that numbering is what the metrics,
    the events and the replay store. The names live on the match record, so every view of a match says the same
    thing. A name is worth having because "team 0" is not something anybody can picture: the report carries each
    team's kit color, and the name is given against that color.
    """
    if team < 0:
        return "referee/other"
    if names and team < len(names) and str(names[team]).strip():
        return str(names[team]).strip()
    return DEFAULT_TEAM_NAMES[team] if team < len(DEFAULT_TEAM_NAMES) else f"Team {team + 1}"


def team_colors(team_rows: Sequence[dict]) -> list[tuple[int, int, int] | None]:
    """Each team's measured kit color, in team order (0 then 1), ``None`` where there is none to show.

    The clustering decides which kit is which color, and the report keeps it so a team can be *named* - the swatch
    and the suggested name both come from here. Reading it is defensive on purpose: an older report simply has no
    such field, and a report whose kits could not be separated has the field empty.
    """
    by_index = {int(row.get("team", -1)): row for row in team_rows}
    colors: list[tuple[int, int, int] | None] = []
    for index in (0, 1):
        rgb = by_index.get(index, {}).get("kit_rgb")
        colors.append(tuple(int(c) for c in rgb) if isinstance(rgb, (list, tuple)) and len(rgb) == 3 else None)
    return colors


def colors_were_recorded(team_rows: Sequence[dict]) -> bool:
    """Whether this report was built by a version that records kit colors at all.

    It tells the two empty cases apart: a report from before the colors existed (rebuild it and they appear) from
    one whose kits the clustering could not separate (rebuilding will not help).
    """
    return any("kit_rgb" in row for row in team_rows)


def report_from_library(library: MatchLibrary, match_id: str | None) -> dict | None:
    """Load a saved report, restoring the momentum keys to integers.

    JSON turns the minute keys into strings, and strings sort lexicographically ("10" before "2"), which would both
    reorder the momentum chart and misplace the momentum-based highlight moments.
    """
    if match_id is None:
        return None
    payload = library.load_report(match_id)
    if payload is None:
        return None
    momentum = {}
    for key, value in (payload.get("momentum") or {}).items():
        try:
            momentum[int(key)] = value
        except (TypeError, ValueError):
            continue
    payload["momentum"] = momentum
    return payload


def momentum_on_window_clock(momentum: Mapping[int, dict] | None, window_start_s: float) -> dict[float, dict]:
    """Re-key Stage B's momentum buckets from the recording's minutes onto the analyzed window's clock.

    Stage B keys its buckets by the minute of the recording - the clock its events and the highlight clips use.
    Everything on the timeline strip is drawn on the window's own clock, where the window's first frame is 0.
    On the real game the window starts at 9:00, so without this shift the curve sat nine minutes to the right of
    the events it belongs beside. Keys stay numeric rather than being rounded back to integers: the window
    boundary can sit mid-minute, and the strip positions a bucket at its minute's midpoint anyway. The buckets
    themselves are returned untouched.
    """
    return {
        round(float(minute) - float(window_start_s) / 60.0, 4): bucket
        for minute, bucket in (momentum or {}).items()
    }
