"""Unique players: grouping appearances into people, and the payload the centered cut is built from.

The identity rule itself is deliberately trivial - team plus shirt number or name - because the measured
alternative was tried and rejected (appearance embeddings do not separate players on this footage; see the
README). What these tests pin is that the grouping is *exact*, stable across reloads, and never guesses at an
appearance nobody has named; and that the geometry the centered cut needs survives the replay payload with the
right time base, which is the mistake that made every shirt-number reading come from the wrong frame once.
"""

from __future__ import annotations

import numpy as np
import pytest

from soccer_analytics.analysis.identity import (
    MAX_TRAJECTORY,
    Appearance,
    Identity,
    appearances_from_players,
    identities_from_labels,
    identity_rows,
)


def _appearance(
    track_id: int,
    *,
    team: int = 0,
    first_t: float = 0.0,
    last_t: float = 10.0,
    points: int = 4,
    with_trajectory: bool = True,
) -> Appearance:
    times = np.linspace(first_t, last_t, points)
    boxes = np.tile(np.array([0.30, 0.20, 0.34, 0.30]), (points, 1))
    return Appearance(
        track_id=track_id,
        team=team,
        first_t=first_t,
        last_t=last_t,
        first_frame=int(first_t * 5),
        last_frame=int(last_t * 5),
        traj_t=times if with_trajectory else None,
        traj_box=boxes if with_trajectory else None,
    )


def test_appearances_wearing_one_number_for_one_team_are_one_player() -> None:
    """The whole point: a player the camera met three times is one person, not three rows."""
    appearances = {
        4: _appearance(4, first_t=0.0, last_t=20.0),
        7: _appearance(7, first_t=60.0, last_t=90.0),
        11: _appearance(11, first_t=200.0, last_t=210.0),
    }
    numbers = {4: {"number": 9, "name": "Sam"}, 7: {"number": 9, "name": "Sam"}, 11: {"number": 9, "name": "Sam"}}
    identities = identities_from_labels(appearances, numbers)
    assert len(identities) == 1
    identity = identities[0]
    assert identity.members == (4, 7, 11)
    assert identity.label == "#9 Sam"
    assert identity.grouped_by == "number"
    assert identity.first_t == 0.0 and identity.last_t == 210.0


def test_the_same_number_on_two_teams_stays_two_players() -> None:
    """Both squads can field a number 9, and the kit clustering already says which shirt is whose."""
    appearances = {1: _appearance(1, team=0), 2: _appearance(2, team=1, first_t=60.0, last_t=70.0)}
    numbers = {1: {"number": 9}, 2: {"number": 9}}
    identities = identities_from_labels(appearances, numbers)
    assert len(identities) == 2
    assert {identity.team for identity in identities} == {0, 1}


def test_unnamed_appearances_are_never_merged_on_a_guess() -> None:
    appearances = {1: _appearance(1), 2: _appearance(2, first_t=60.0, last_t=80.0)}
    identities = identities_from_labels(appearances, {})
    assert [identity.members for identity in identities] == [(1,), (2,)]
    assert [identity.label for identity in identities] == ["track 1", "track 2"]
    assert all(identity.grouped_by == "" for identity in identities)


def test_a_name_alone_groups_appearances() -> None:
    """A roster can be typed without shirt numbers; the name still claims the appearances."""
    appearances = {2: _appearance(2, team=1, first_t=0.0, last_t=5.0), 5: _appearance(5, team=1, first_t=50.0, last_t=60.0)}
    identities = identities_from_labels(appearances, {2: {"name": "Sam"}, 5: {"name": "sam"}})
    assert len(identities) == 1
    assert identities[0].members == (2, 5)
    assert identities[0].grouped_by == "name"


def test_a_number_beats_a_name_when_both_are_known() -> None:
    """The number is the stronger claim: two appearances of #9 are one person even if a name was typed on one."""
    appearances = {3: _appearance(3), 8: _appearance(8, first_t=60.0, last_t=70.0)}
    numbers = {3: {"number": 9, "name": "Sam"}, 8: {"number": 9}}
    identities = identities_from_labels(appearances, numbers)
    assert len(identities) == 1
    assert identities[0].label == "#9 Sam"
    assert identities[0].grouped_by == "number"


def test_identity_ids_do_not_depend_on_dict_order() -> None:
    appearances = {8: _appearance(8, first_t=0.0, last_t=5.0), 3: _appearance(3, first_t=60.0, last_t=70.0)}
    numbers = {3: {"number": 4}, 8: {"number": 4}}
    first = identities_from_labels(appearances, numbers)
    second = identities_from_labels(dict(reversed(list(appearances.items()))), numbers)
    assert [identity.members for identity in first] == [identity.members for identity in second] == [(3, 8)]


def test_identity_rows_report_time_on_screen_not_elapsed_time() -> None:
    """A player seen for two minutes in each half was on screen for four minutes, not for the whole match."""
    appearances = {1: _appearance(1, first_t=0.0, last_t=120.0), 2: _appearance(2, first_t=1800.0, last_t=1920.0)}
    identities = identities_from_labels(appearances, {1: {"number": 7}, 2: {"number": 7}})
    rows = identity_rows(identities, appearances, lambda team: f"Team {team}")
    assert rows[0]["seen_s"] == pytest.approx(240.0)
    assert rows[0]["first_t"] == 0.0 and rows[0]["last_t"] == 1920.0
    assert rows[0]["appearances"] == 2 and rows[0]["tracks"] == [1, 2]
    assert rows[0]["team"] == "Team 0"


def test_identity_rows_skip_appearances_that_were_not_measured() -> None:
    identity = Identity(0, 0, (1, 99), 0.0, 20.0, "number", label="#9")
    rows = identity_rows([identity], {1: _appearance(1, first_t=0.0, last_t=20.0)}, str)
    assert rows[0]["seen_s"] == pytest.approx(20.0)


def test_identity_rows_are_sorted_by_time_on_screen() -> None:
    appearances = {
        1: _appearance(1, first_t=0.0, last_t=10.0),
        2: _appearance(2, first_t=0.0, last_t=100.0),
        3: _appearance(3, first_t=0.0, last_t=50.0),
    }
    identities = identities_from_labels(appearances, {})
    rows = identity_rows(identities, appearances, str)
    assert [row["player"] for row in rows] == ["track 2", "track 3", "track 1"]


def test_appearances_from_players_use_source_seconds() -> None:
    """``frames`` are analysis frames; the cut needs source seconds, so the window's start and the rate apply.

    Getting this wrong is the bug the shirt-number scan already paid for: a frame index used as a time read every
    crop from ~9 minutes later in the match.
    """
    players = [{"track_id": 5, "team": 1, "frames": [0, 5, 10]}]
    boxes = {
        "5": np.array([[0.1, 0.2, 0.2, 0.4], [0.11, 0.2, 0.21, 0.4], [0.12, 0.2, 0.22, 0.4]])
    }
    appearances = appearances_from_players(players, frames_per_second=5.0, start_s=540.0, boxes=boxes)
    five = appearances[5]
    assert five.first_t == pytest.approx(540.0)
    assert five.last_t == pytest.approx(542.0)
    assert five.traj_t is not None and five.traj_box is not None
    assert np.allclose(five.traj_box[0], [0.1, 0.2, 0.2, 0.4])
    assert five.team == 1


def test_appearances_from_players_sort_by_time_and_survive_a_missing_box() -> None:
    players = [
        {"track_id": 1, "team": 0, "frames": [10, 0]},
        {"track_id": 2, "team": 0, "frames": [0, 5]},
    ]
    boxes = {"1": np.array([[0.2, 0.2, 0.3, 0.4], [0.1, 0.2, 0.2, 0.4]])}
    appearances = appearances_from_players(players, frames_per_second=5.0, start_s=0.0, boxes=boxes)
    assert np.all(np.diff(appearances[1].traj_t) > 0), "times must be in order whichever order the frames arrived"
    assert appearances[1].traj_box is not None and appearances[1].traj_box[0][0] == pytest.approx(0.1)
    two = appearances[2]
    assert two.traj_box is None
    times, boxes = two.trajectory()
    assert len(times) == 0 and len(boxes) == 0, "an appearance with no boxes cannot be framed - say so, do not guess"


def test_a_trajectory_is_thinned_but_keeps_both_ends() -> None:
    appearance = _appearance(1, first_t=0.0, last_t=100.0, points=MAX_TRAJECTORY + 200)
    times, boxes = appearance.trajectory()
    assert len(times) == MAX_TRAJECTORY and len(boxes) == MAX_TRAJECTORY
    assert times[0] == pytest.approx(0.0) and times[-1] == pytest.approx(100.0)


def test_a_trajectory_can_start_mid_appearance_for_a_later_clip() -> None:
    appearance = _appearance(1, first_t=0.0, last_t=100.0, points=101)
    times, _boxes = appearance.trajectory(start_s=50.0)
    assert times[0] == pytest.approx(50.0, abs=0.5)
    assert times[-1] == pytest.approx(100.0)


def test_stored_numbers_are_called_stale_when_their_tracks_are_gone() -> None:
    """Numbers belong to track ids, which belong to one build: a vanished id cannot be anyone in this report."""
    from soccer_analytics.analysis.identity import numbers_are_stale

    reason = numbers_are_stale([12009, 13471], [1, 2, 3], scan_calibration_saved=None, calibration_saved=None)
    assert "not in this report" in reason
    # An unknown provenance is not itself a reason to disbelieve a scan whose ids are all present.
    assert numbers_are_stale([1], [1, 2], scan_calibration_saved=None, calibration_saved=None) == ""


def test_stored_numbers_are_stale_when_read_before_the_last_calibration() -> None:
    from soccer_analytics.analysis.identity import numbers_are_stale

    assert numbers_are_stale([1], [1], scan_calibration_saved=100.0, calibration_saved=100.0) == ""
    reason = numbers_are_stale([1], [1], scan_calibration_saved=100.0, calibration_saved=200.0)
    assert "before the pitch was last calibrated" in reason
