"""The landmark component's contract with Python: what a gesture may, and may not, report.

The component is a static frontend, so these read the file rather than driving a browser. They exist because the
timeline's reactivity is easy to regress silently: scrubbing must stay in the browser, and the magnified crop must
only be re-read when the user aims at a point.
"""

from __future__ import annotations

from pathlib import Path

COMPONENT = (
    Path(__file__).resolve().parents[1]
    / "src"
    / "soccer_analytics"
    / "dashboard"
    / "field_annotation_component"
    / "index.html"
)


def test_the_frontend_is_the_scrubbable_landmark_view() -> None:
    html = COMPONENT.read_text()
    assert "<video" in html and "streamlit:setComponentValue" in html
    assert 'id="overviewVideo"' in html, "the whole-frame viewport is the timeline"


def test_scrubbing_never_reports_to_python() -> None:
    """A scrub must not ask Python for anything: a round trip per release is what re-rendered the crop."""
    html = COMPONENT.read_text()
    assert "sendValue('scrub')" not in html
    assert "commitScrub" not in html


def test_an_aiming_gesture_carries_the_frame() -> None:
    """The frame must ride with the aim, or clicking after a scrub would crop a frame that is not on screen."""
    html = COMPONENT.read_text()
    assert "frame: currentFrame()" in html


def test_the_marker_picker_tags_the_next_click() -> None:
    """A landmark can be named before it is placed, from a picker left of the zoom controls."""
    html = COMPONENT.read_text()
    assert 'id="marker-kind"' in html
    assert "label: markerKind.value" in html, "the pick is what the click is tagged with"
    assert html.index('id="marker-kind"') < html.index('id="zoom-out"'), "the picker belongs left of the zoom controls"


def test_markers_are_only_drawn_on_the_exact_crop_frame() -> None:
    """The proxy holds one picture per second, so a marker drawn on it points where the corner was ages ago.

    The still Python sends is the exact frame the crop shows, and the markers may only be drawn over it - otherwise
    a perfectly good click reads as a misplaced one, which is exactly what happened on a whip-pan frame.
    """
    html = COMPONENT.read_text()
    assert "function pictureIsExact()" in html
    marker_block = html.index("if (exact) {")
    assert marker_block < html.index("(args.marker_points || []).forEach"), "markers sit inside the exact-frame branch"
    # and the still must be loaded in video mode too, not only as the no-proxy fallback
    assert "if (args.overview_data && args.overview_data !== overviewDataSrc)" in html
    # scrubbing off the crop's frame must drop the still immediately
    assert "drawOverview();" in html.split("scrub.addEventListener('input'")[1].split("});")[0]


def test_the_viewport_follows_a_frame_moved_from_python() -> None:
    """Jumping to a landmark's frame from the list is a Python-side move; the bar and picture follow it.

    The other half of the rule matters just as much: a scrub the user made must survive the next poll, which is why
    the follow is conditional on the frame having actually changed since the last render.
    """
    html = COMPONENT.read_text()
    assert "const movedByPython = mounted && videoMode && renderedFrom !== cropFrame;" in html
    assert "} else if (movedByPython) {" in html
    follow = html.split("const movedByPython")[1].split("}\n")[0]
    assert "pendingFrame = renderedFrom;" in follow and "applyPendingFrame();" in follow


def test_a_placed_marker_can_be_dragged_onto_the_marking_it_stands_for() -> None:
    """Re-anchoring is a nudge: the placed marker carries its label, and the drag is the whole measurement."""
    html = COMPONENT.read_text()
    assert "function featureNear(x, y)" in html
    assert "canvas.addEventListener('mousedown'" in html
    assert "suppressClick = true;" in html, "a drag must not leave a fresh click behind it"
    assert "ctx.strokeText(feature.label" in html, "a marker has to say which landmark it stands for"


def test_a_dragged_marker_survives_an_unrelated_rerun() -> None:
    """Python's copy of a marker only changes on Apply, so aiming the view must not yank a drag back."""
    html = COMPONENT.read_text()
    assert "feature.dirty = true;" in html
    assert "moved.get(feature.id)" in html, "the pending drag is matched to Python's copy by id"
    apply_block = html.split("applyBtn.addEventListener('click'")[1].split("sendValue('apply')")[0]
    assert "feature.dirty = false" in apply_block, "an Apply makes Python's positions the canonical ones"
