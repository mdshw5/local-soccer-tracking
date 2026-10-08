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


def test_the_scanned_ball_layer_exists_and_distinguishes_forecasts() -> None:
    """When the payload carries a ball track, it is drawn as its own layer - and a forecast is not a sighting.

    The scan's third element per frame is the measured flag; the component is where that honesty has to land, so
    it must both read it and show it - a drawn football for a detection, a dashed ring for the tracker's forecast.
    """
    html = COMPONENT.read_text()
    assert "state.data.ball" in html
    assert "BALL_COLOUR" in html
    assert "ball[2]" in html, "the per-frame measured flag must drive the drawing"
    assert "drawSoccerBall" in html, "the ball must read as a football, not another filled circle"


def test_the_markers_wear_the_measured_kit_colour() -> None:
    """The kit colour is measured data, so the component takes it from the payload - palette only as fallback.

    The number inside a marker also has to stay readable *on* that colour: a fixed white number disappears on a
    white kit, which is why the text colour is chosen from the marker's own colour rather than hard-coded.
    """
    html = COMPONENT.read_text()
    assert "state.data.team_colours" in html, "markers must read the measured kit colours from the payload"
    assert "TEAM_COLOURS[team]" in html, "the fixed palette must remain as the fallback"
    assert "markerTextColour" in html, "the number colour must follow the kit colour, not be assumed white"


def test_the_camera_direction_line_is_drawn_from_the_camera_position() -> None:
    """The camera's direction is shown as a line from its own ground position to the aim point.

    The camera sits off the near touchline, so the view bounds must widen to include it - otherwise the line's
    origin is clipped off the edge of the canvas and the direction reads wrong.
    """
    html = COMPONENT.read_text()
    assert "state.data.camera" in html, "the line is drawn from the camera position in the payload"
    assert "CAMERA_COLOUR" in html
    assert "viewBounds" in html, "the view must expand to include the camera position"
    assert "cam[0] - MARGIN_M" in html and "cam[1] - MARGIN_M" in html


def test_the_timeline_strip_draws_momentum_and_events() -> None:
    """The strip above the pitch combines the momentum curve with the event markers, and seeks on click.

    The events and momentum arrive as component arguments (they change without the replay being rebuilt), so the
    component must read them from ``args`` and draw both - a momentum curve and one marker per event, with a manual
    tag filled and a detected candidate hollow so a machine's guess never reads as a human's certainty.
    """
    html = COMPONENT.read_text()
    assert 'id="timeline"' in html and 'id="tl"' in html
    assert "args.events" in html, "events travel as a component argument"
    assert "args.momentum" in html, "momentum travels as a component argument"
    assert "drawTimeline" in html
    assert "eventColour" in html, "each event type has its own colour"
    assert "event.source === 'manual'" in html, "a manual tag must be drawn differently from a detected one"
    assert "timeline.addEventListener('click'" in html, "clicking the strip must seek the replay"


def test_the_legend_is_redrawn_when_events_arrive_without_a_reload() -> None:
    """The legend's event chips must follow the events argument, not only the replay payload.

    Events travel as their own argument so a new tag shows up without rebuilding the replay. The legend lists a chip
    per event type, so if it were only drawn from ``loadData`` the strip would gain markers while the legend still
    listed the types it saw when the payload loaded - which is exactly what happened when the detectors first ran.
    """
    html = COMPONENT.read_text()
    render_event = html.split("Streamlit.events.addEventListener(Streamlit.RENDER_EVENT", 1)[1]
    # The else branch is the "same payload, new arguments" path - the one a new tag takes.
    else_branch = render_event.split("} else {", 1)[1].split("Streamlit.setFrameHeight", 1)[0]
    assert "renderLegend()" in else_branch, "the legend must be redrawn when the arguments change"


def test_the_pitch_and_the_clip_sit_side_by_side() -> None:
    """The clip is the footage of the moment; the pitch is the whole window. They are read together, so they share
    a row - and the pitch has to size itself from its own pane, not the page, or it would overflow the row."""
    html = COMPONENT.read_text()
    assert 'id="panes"' in html, "the two views must be laid out as panes"
    assert 'id="pitch-pane"' in html and 'id="clip-pane"' in html
    assert 'id="clip"' in html and "<video" in html, "the clip is a real video element"
    assert "pitchPane.clientWidth" in html, "the pitch must measure its own pane, not the page"


def test_the_clip_is_held_in_step_with_the_animation() -> None:
    """Sync is one-way and driven by the animation, because the animation can be scrubbed to any second of the
    match while the clip only exists for its own window.

    Two things make this work and are easy to lose: the clip's window arrives on the *strip's* clock (so the two
    can be compared at all), and the correction is a tolerance rather than a seek per frame - seeking every frame
    would stutter the video, and the two clocks run at the same rate.
    """
    html = COMPONENT.read_text()
    assert "args.clip_url" in html, "the clip arrives as a URL"
    assert "args.clip_start_s" in html and "args.clip_end_s" in html, "its window arrives on the strip's clock"
    assert "function syncClip" in html
    assert "clipTimeFor" in html, "the strip time must be translated into the clip's own time"
    assert "clip.currentTime" in html, "the clip is seeked to follow the animation"
    # A tolerance, not an equality: the two clocks drift by the gap between ticks.
    assert "0.35" in html, "the drift correction must have a tolerance"
    # Outside the window the clip must stop rather than run on past the moment it shows.
    assert "clip.pause()" in html
    # And the animation must drive it, not the other way round.
    assert "syncClip()" in html.split("function tick", 1)[1].split("requestAnimationFrame(tick)", 1)[0]


def test_the_clip_sync_can_be_turned_off() -> None:
    """A clip is worth watching on its own - the animation is a reconstruction, the clip is the footage - so the
    two can be uncoupled without losing the clip."""
    html = COMPONENT.read_text()
    assert 'id="sync"' in html, "there must be a control to uncouple the two"
    assert "state.clipSync" in html
    assert "clipSync: true" in html, "synced is the useful default"


def test_a_button_jumps_the_animation_to_the_clip_and_plays_both() -> None:
    """The clip has its own controls, so the user can drag it away from the animation; the follow button is the way
    back. It must read the *clip's* time (the animation is the thing being moved), land inside the clip's window so
    sync can hold the two, and start playback - the button's whole point is watching both together.

    It also re-couples sync when it was switched off: a button that says "match the video" while leaving the two
    free to drift again would not do what it says.
    """
    html = COMPONENT.read_text()
    assert 'id="follow-clip"' in html, "the button sits beside play"
    assert "followClipBtn.disabled = !has || state.clipStart === null" in html, (
        "without a clip (or a window to jump into) there is nothing to follow"
    )
    handler = html.split("followClipBtn.addEventListener", 1)[1]
    assert "state.clipStart + clip.currentTime" in handler, "the animation time comes from the clip's own time"
    assert "state.clipSync = true" in handler, "following re-couples sync if it was off"
    assert "setPlaying(true)" in handler, "the button starts playback"
