"""The report's momentum keys must survive a trip through JSON.

`save_report` writes the dict as JSON, which turns the integer minute keys into strings. Strings sort
lexicographically, so "10" would come before "2": the momentum chart would be drawn out of order and the
momentum-based highlight moments would land on the wrong minutes. `report_from_library` restores the integers.
"""

from __future__ import annotations

import json

from soccer_analytics.dashboard.reports import report_from_library


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
