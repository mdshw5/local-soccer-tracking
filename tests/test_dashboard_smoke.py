"""Smoke test: the dashboard must render end to end without raising.

The page does real work as it renders - it probes the video, reads the segment status and loads saved artefacts - so
this catches the class of error that only appears when the script actually runs.
"""

from __future__ import annotations

from pathlib import Path

import pytest

streamlit_testing = pytest.importorskip("streamlit.testing.v1")

APP = Path(__file__).resolve().parents[1] / "src" / "soccer_analytics" / "dashboard" / "app.py"


def _no_footage(app) -> bool:
    """Whether the page gave up because this machine has no video to analyse.

    Checked by message rather than by `app.error` being non-empty: the page also raises *content* errors, such as a
    pitch calibration that came out unusable, and those are the page working, not the page failing.
    """
    return any("No video files found" in str(block.value) for block in app.error)


def test_dashboard_renders_without_error() -> None:
    app = streamlit_testing.AppTest.from_file(str(APP), default_timeout=120)
    app.run()
    assert not app.exception, [str(e.value) for e in app.exception]

    titles = [t.value for t in app.title]
    assert "Match analysis" in titles

    # Either there is no footage (reported honestly) or Step 1 offers footage and formats downstream.
    if _no_footage(app):
        pytest.skip("no footage available in this environment")
    headers = [h.value for h in app.header]
    assert any("Step 1" in h for h in headers)
    assert any("Video (newest first)" == s.label for s in app.selectbox)


def test_dashboard_offers_the_next_step_either_way() -> None:
    """The page must either offer Step 3 or say what Step 3 is waiting for - never just be silent.

    Deliberately not asserting which: the answer depends on what has already been analysed on this machine, and a
    test that only passes on a clean workspace is a test that fails the moment someone uses the tool.
    """
    app = streamlit_testing.AppTest.from_file(str(APP), default_timeout=120)
    app.run()
    assert not app.exception, [str(e.value) for e in app.exception]
    if _no_footage(app):
        pytest.skip("no footage available in this environment")

    messages = " ".join(str(block.value) for block in app.info)
    messages += " ".join(str(block.value) for block in app.warning)
    buttons = [button.label for button in app.button]
    actionable = "Build report" in buttons
    explained = any(phrase in messages for phrase in ("Run Step 1 first", "Steps 1 and 2", "Not analysed yet"))
    assert actionable or explained, f"Step 3 said neither. buttons={buttons} messages={messages!r}"
