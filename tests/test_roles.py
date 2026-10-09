"""Role classification: the referee from range, goalkeepers from the goal pockets.""" 

from __future__ import annotations

import numpy as np

from soccer_analytics.analysis.roles import classify_roles


class _Player:
    """The fields ``classify_roles`` reads: an id, a team, and pitch positions."""

    def __init__(self, track_id: int, team: int, xs: list[float]):
        self.track_id = track_id
        self.team = team
        self.xy = np.column_stack([np.asarray(xs, dtype=np.float64), np.full(len(xs), 30.0)])


def _spread(track_id: int, team: int, lo: float, hi: float, n: int = 200) -> _Player:
    return _Player(track_id, team, list(np.linspace(lo, hi, n)))


def test_a_player_in_a_goal_pocket_is_a_goalkeeper() -> None:
    players = [_spread(10, -1, 1.0, 8.0, 120), _spread(11, 0, 20.0, 80.0, 400)]
    roles = classify_roles(players, pitch_length_m=100.0)
    assert roles[10] == {"role": "goalkeeper", "side": "left"}
    assert 11 not in roles


def test_a_goalkeeper_needs_enough_observations() -> None:
    players = [_spread(10, -1, 1.0, 8.0, 30)]
    assert classify_roles(players, pitch_length_m=100.0) == {}


def test_the_wide_ranging_unlabelled_track_is_the_referee() -> None:
    players = [_spread(20, -1, 5.0, 95.0, 300), _spread(21, -1, 40.0, 60.0, 400)]
    roles = classify_roles(players, pitch_length_m=100.0)
    assert roles[20] == {"role": "referee"} and 21 not in roles


def test_the_referee_is_the_longest_qualifying_track() -> None:
    players = [_spread(20, -1, 5.0, 95.0, 300), _spread(21, -1, 5.0, 95.0, 420)]
    roles = classify_roles(players, pitch_length_m=100.0)
    assert roles == {21: {"role": "referee"}}


def test_referee_candidates_need_kit_evidence_when_a_gate_is_given() -> None:
    players = [_spread(20, -1, 5.0, 95.0, 300), _spread(22, -1, 5.0, 95.0, 400)]
    roles = classify_roles(players, pitch_length_m=100.0, kit_evidence=lambda tid: None if tid == 22 else object())
    assert roles == {20: {"role": "referee"}}


def test_a_goalkeeper_is_never_the_referee() -> None:
    # A keeper fragment that also ranged widely (a rush out of goal) still stays a goalkeeper when the goal
    # pocket dominates its observations.
    xs = list(np.linspace(1.0, 9.0, 360)) + list(np.linspace(30.0, 95.0, 40))
    players = [_Player(30, -1, xs)]
    roles = classify_roles(players, pitch_length_m=100.0)
    assert roles[30]["role"] == "goalkeeper"


def test_a_team_labelled_track_is_never_the_referee() -> None:
    players = [_spread(20, 0, 5.0, 95.0, 400)]
    assert classify_roles(players, pitch_length_m=100.0) == {}
