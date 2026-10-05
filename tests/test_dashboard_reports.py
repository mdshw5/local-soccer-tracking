"""The report's momentum keys must survive a trip through JSON.

`save_report` writes the dict as JSON, which turns the integer minute keys into strings. Strings sort
lexicographically, so "10" would come before "2": the momentum chart would be drawn out of order and the
momentum-based highlight moments would land on the wrong minutes. `report_from_library` restores the integers.
"""

from __future__ import annotations

import json

from soccer_analytics.dashboard.reports import (
    colours_were_recorded,
    is_default_team_name,
    report_from_library,
    team_colours,
    team_name,
)


class FakeLibrary:
    """Stands in for MatchLibrary, with the same JSON round-trip the real one does on disk."""

    def __init__(self, payload: dict | None) -> None:
        self.payload = payload

    def load_report(self, match_id: str) -> dict | None:
        return None if self.payload is None else json.loads(json.dumps(self.payload))


def test_momentum_keys_are_restored_to_integers() -> None:
    payload = {
        "momentum": {
            "10": {"team_0": 0.6, "team_1": 0.4, "action_x": 30.0},
            "2": {"team_0": 0.4, "team_1": 0.6, "action_x": 40.0},
        },
        "players": [],
    }
    loaded = report_from_library(FakeLibrary(payload), "match-1")
    assert loaded is not None
    assert set(loaded["momentum"]) == {2, 10}
    assert all(isinstance(minute, int) for minute in loaded["momentum"])
    assert sorted(loaded["momentum"]) == [2, 10], "minutes must sort numerically, not as text"
    assert loaded["players"] == []


def test_missing_report_and_missing_match_are_not_errors() -> None:
    assert report_from_library(FakeLibrary(None), "match-1") is None

    class ExplodingLibrary:
        def load_report(self, match_id: str):  # pragma: no cover - must never be reached
            raise AssertionError("load_report must not be called without a match id")

    assert report_from_library(ExplodingLibrary(), None) is None


def test_report_without_momentum_still_loads() -> None:
    loaded = report_from_library(FakeLibrary({"players": [{"track_id": 1}]}), "match-1")
    assert loaded is not None and loaded["momentum"] == {} and loaded["players"] == [{"track_id": 1}]


def test_a_team_is_called_by_its_name_else_by_its_number() -> None:
    """The names the user gives live on the match record; every view asks for them through this one function.

    Teams are numbered 0/1 by the kit clustering (the more red kit is 0), which is what the metrics, the events and
    the replay store - so the fallback has to stay readable, and a blank or missing name must not print as one.
    """
    assert team_name(0, ["Reds", "Dark blues"]) == "Reds"
    assert team_name(1, ["Reds", "Dark blues"]) == "Dark blues"
    assert team_name(0, None) == "Team 1"
    assert team_name(1, []) == "Team 2"
    assert team_name(0, ["  ", "Blues"]) == "Team 1", "a blank name falls back rather than printing nothing"
    assert team_name(0, ["Reds"]) == "Reds", "one name is enough for a payload that only knows one team"
    assert team_name(-1, ["Reds", "Blues"]) == "referee/other"
    assert team_name(7, ["Reds", "Blues"]) == "Team 8", "an unexpected index still reads as a team"


def test_placeholders_are_recognised_so_a_colour_name_can_replace_them() -> None:
    """The suggestion only fills a placeholder: a team someone has already named keeps its name."""
    for placeholder in ("", "   ", "Team A", "team b", "Team 1", "team 2"):
        assert is_default_team_name(placeholder), placeholder
    for chosen in ("Reds", "Dark blues", "Thistle", "The A Team", "Team 3", "Teamsters"):
        assert not is_default_team_name(chosen), chosen


def test_team_colours_are_read_back_in_team_order() -> None:
    """The swatch and the suggested name both come from the report's own team rows."""
    rows = [
        {"team": 1, "kit_rgb": [30, 30, 200]},
        {"team": 0, "kit_rgb": [200, 30, 30]},
    ]
    assert team_colours(rows) == [(200, 30, 30), (30, 30, 200)]
    # A report whose kits could not be separated, and one that never recorded colours at all.
    assert team_colours([{"team": 0, "kit_rgb": None}, {"team": 1}]) == [None, None]
    assert team_colours([]) == [None, None]
    # A malformed value must not crash the page - it is just nothing to show.
    assert team_colours([{"team": 0, "kit_rgb": "red"}]) == [None, None]
    # Extra or missing rows: the two teams are what the page asks for, in order.
    assert team_colours([{"team": 0, "kit_rgb": [1, 2, 3]}, {"team": 2, "kit_rgb": [9, 9, 9]}]) == [(1, 2, 3), None]


def test_an_old_report_is_told_apart_from_one_that_could_not_separate_the_kits() -> None:
    """The advice differs: rebuild the report, or accept that the kits were not separable."""
    old = [{"team": 0, "name": "Team 1"}, {"team": 1, "name": "Team 2"}]
    empty = [{"team": 0, "name": "Team 1", "kit_rgb": None}, {"team": 1, "name": "Team 2", "kit_rgb": None}]
    assert not colours_were_recorded(old)
    assert colours_were_recorded(empty)
