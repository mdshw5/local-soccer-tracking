"""Archives written before the project switched to American spellings still load unchanged.

The switch renamed three things that live on disk: the replay payload's ``team_colours`` key, the report
payload's ``frames_analysed`` key, and the landmark labels containing "centre". Old match directories keep
those spellings; the library translates them on load (see the ``upgrade_*`` helpers in ``analysis.library``).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from soccer_analytics.analysis.library import (
    MatchLibrary,
    analysis_dir_for,
    upgrade_landmark_label,
    upgrade_replay_payload,
)


@pytest.fixture
def match(tmp_path: Path, monkeypatch):
    root = tmp_path / "Xbot"
    root.mkdir()
    monkeypatch.setenv("SOCCER_VIDEO_ROOTS", str(root))
    video = root / "2026-10-03" / "game_16-28-37.784.mp4"
    video.parent.mkdir(parents=True)
    video.write_bytes(b"x" * 1024)
    library = MatchLibrary()
    record = library.create(video)
    return library, record.match_id, analysis_dir_for(video)


def test_a_legacy_replay_key_is_upgraded_on_load(match) -> None:
    library, match_id, directory = match
    (directory / "replay.json").write_text(json.dumps({"team_colours": [[10, 20, 30], None], "players": []}))
    payload = library.load_replay(match_id)
    assert payload is not None
    assert payload["team_colors"] == [[10, 20, 30], None]
    assert "team_colours" not in payload


def test_a_legacy_report_key_is_upgraded_on_load(match) -> None:
    library, match_id, directory = match
    (directory / "report.json").write_text(json.dumps({"frames_analysed": 3}))
    payload = library.load_report(match_id)
    assert payload is not None and payload["frames_analyzed"] == 3


def test_a_legacy_clicks_label_is_upgraded_on_load(match) -> None:
    library, match_id, directory = match
    clicks = [
        {"frame": 1, "u": 0.5, "v": 0.5, "label": "centre spot"},
        {"frame": 2, "u": 0.4, "v": 0.4, "label": "centre circle near"},
        {"frame": 3, "u": 0.3, "v": 0.3, "label": "corner near-left"},
    ]
    (directory / "clicks.json").write_text(json.dumps({"clicks": clicks, "pitch": [105.0, 68.0]}))
    labels = [click["label"] for click in library.load_clicks(match_id)]
    assert labels == ["center spot", "center circle near", "corner near-left"]


def test_a_current_key_wins_when_both_spellings_are_present() -> None:
    payload = upgrade_replay_payload({"team_colours": [[9, 9, 9]], "team_colors": [[1, 1, 1]]})
    assert payload["team_colors"] == [[1, 1, 1]]


def test_every_legacy_landmark_label_maps_onto_a_current_name() -> None:
    from soccer_analytics.dashboard.pitch_clicks import landmark_table

    names = set(landmark_table(105.0, 68.0))
    for old in (
        "centre spot",
        "centre circle near",
        "centre circle far",
        "centre circle left",
        "centre circle right",
    ):
        new = upgrade_landmark_label(old)
        assert new is not None and new != old and new in names
