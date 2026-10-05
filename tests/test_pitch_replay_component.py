"""The replay component's contract: playback stays in the browser, and what Python must supply.

Like the annotation component, this is a static frontend read as a file. The checks exist because the interesting
property is easy to lose: a playback control that round-trips to Python would re-render the whole page per frame.
"""

from __future__ import annotations

from pathlib import Path

COMPONENT = (
    Path(__file__).resolve().parents[1]
    / "src"
    / "soccer_analytics"
    / "dashboard"
    / "pitch_replay_component"
    / "index.html"
)


def test_playback_controls_exist_and_run_locally() -> None:
    html = COMPONENT.read_text()
    assert 'id="play"' in html and 'id="scrub"' in html and "requestAnimationFrame" in html
    # A play/pause or scrub must not talk to Python: no component value is ever set.
    assert "setComponentValue" not in html


def test_the_component_fetches_the_replay_by_url() -> None:
    """The payload is megabytes; it must be fetched once, not pushed through component arguments."""
    html = COMPONENT.read_text()
    assert "fetch(url)" in html or "fetch(" in html
    assert "data_url" in html


def test_shirt_numbers_arrive_as_a_separate_argument() -> None:
    """Numbers can change (roster edits) without rebuilding the replay, so they travel outside the JSON."""
    html = COMPONENT.read_text()
    assert "args.numbers" in html
    assert "entry.number" in html


def test_the_view_can_be_sized_and_shows_a_legend() -> None:
    html = COMPONENT.read_text()
    assert "setFrameHeight" in html
    assert 'id="legend"' in html


def test_the_selected_player_comes_from_python() -> None:
    html = COMPONENT.read_text()
    assert "args.selected_track" in html
    assert "selected_track" in html
