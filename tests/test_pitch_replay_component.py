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


def test_the_pitch_and_the_footage_stream_sit_side_by_side() -> None:
    """The footage is the annotated match stream beside the animation; the pitch is the whole window. They are
    read together, so they share a row - evenly split, since neither view is the other's annex - and the pitch
    has to size itself from its own pane, not the page, or it would overflow the row."""
    html = COMPONENT.read_text()
    assert 'id="panes"' in html, "the two views must be laid out as panes"
    assert 'id="pitch-pane"' in html and 'id="footage-pane"' in html
    assert 'id="footage"' in html and "<img" in html, "only an <img> element plays an MJPEG stream"
    pitch_css = html.split("#pitch-pane {", 1)[1].split("}", 1)[0]
    footage_css = html.split("#footage-pane {", 1)[1].split("}", 1)[0]
    assert "flex: 1 1 50%" in pitch_css and "flex: 1 1 50%" in footage_css, "equal halves of the row"
    assert "pitchPane.clientWidth" in html, "the pitch must measure its own pane, not the page"


def test_the_footage_follows_the_animation_clock() -> None:
    """An MJPEG connection cannot be seeked, so the pane follows the animation in the two ways it can: one
    annotated *still* of the exact second while paused (debounced - a scrub must not fire a decode per pixel), and
    a *stream* opened at that second, at the animation's speed, while playing.

    Two things make the times comparable and are easy to lose: the strip's clock is translated onto the
    recording's (``streamOffset``) before it is asked for, and every path that moves the animation ends in a
    draw, which is where the pane is re-synced - so the animation drives, never the other way round.
    """
    html = COMPONENT.read_text()
    assert "function footageUrl" in html
    assert "/live/" in html and "/frame/" in html, "the live encoded stream for playing, the still endpoint for paused"
    assert "state.streamOffset" in html, "the strip clock must be translated onto the recording's"
    assert "function scheduleStill" in html and "FOOTAGE_STILL_DEBOUNCE_MS" in html, "scrubbing must be debounced"
    assert "function syncFootage" in html
    draw_body = html.split("function draw()", 1)[1].split("\n      function ", 1)[0]
    assert "syncFootage();" in draw_body, "every path that moves the animation must move the footage"


def test_a_stream_that_falls_behind_is_restarted_not_left_to_drift() -> None:
    """The decode is capped by how fast the 4K footage decodes, so a playing stream can fall behind the
    animation. The pane restarts it at the animation's own second once the gap is visible - rate-limited, so a
    decode slower than real time cannot turn the pane into a loop of restarts."""
    html = COMPONENT.read_text()
    assert "FOOTAGE_DRIFT_TOLERANCE_S" in html, "a restart costs an encode: only correct a visible divergence"
    assert "FOOTAGE_MIN_RESTART_INTERVAL_MS" in html, "and never restart more often than this"
    assert "function restartFootage" in html
    assert "footageVideo.currentTime" in html, "the drift is measured off the video's own clock, not modelled"


def test_the_footage_sync_can_be_turned_off() -> None:
    """The footage is worth watching on its own - the animation is a reconstruction, the footage is the match -
    so the two can be uncoupled without losing the pane."""
    html = COMPONENT.read_text()
    assert 'id="sync"' in html, "there must be a control to uncouple the two"
    assert "state.footageSync" in html
    assert "footageSync: true" in html, "synced is the useful default"


def test_a_button_restarts_the_footage_at_the_animation_second() -> None:
    """The pane's one real control: drop the current stream and reopen it at the animation's second (the way
    back into step after a stall). It re-couples sync if it was switched off - a button that says "match the
    animation" while leaving them free to drift again would not do what it says."""
    html = COMPONENT.read_text()
    assert 'id="resync-footage"' in html, "the button sits beside play"
    handler = html.split("resyncFootageBtn.addEventListener", 1)[1]
    assert "restartFootage()" in handler, "the button restarts the stream at the animation's own second"
    assert "state.footageSync = true" in handler, "restarting re-couples sync if it was off"


def test_the_footage_overlay_toggles_choose_the_layers() -> None:
    """The footage is drawn server-side, so each toggle is a query parameter on the stream *and* the still URLs:
    switching one off reopens the pane at the animation's own second with the new layer set, rather than leaving
    the previous picture on screen until the next seek. ``none`` is the explicit spelling for a clean frame - an
    empty query value parses as \"absent\" on the server, which means everything on."""
    html = COMPONENT.read_text()
    for ident in ("ov-pitch", "ov-boxes", "ov-numbers", "ov-ball", "ov-hud", "ov-debug"):
        assert f'id="{ident}"' in html, f"a toggle must exist for {ident}"
    assert "state.footageOverlays" in html
    assert "overlays=${overlayParam()}" in html, "the layer set must travel on the URLs"
    assert html.count("overlays=${overlayParam()}") == 2, "the stream and the still both carry it"
    assert "on.length ? on.join(',') : 'none'" in html, "all layers off is spelled none, not empty"
    handler = html.split("state.footageOverlays[name] = event.target.checked;", 1)[1]
    assert "restartFootage()" in handler, "a new layer set must reopen the footage now"


def test_a_busy_stream_server_is_retried_before_giving_up() -> None:
    """Restarting the footage can lose the race for the server's decode slot (the previous connection's slot is
    only freed when the server notices nobody is reading it): the pane retries with backoff instead of declaring
    the server absent on the first refusal, and a successful frame clears the state."""
    html = COMPONENT.read_text()
    assert "FOOTAGE_RETRY_LIMIT" in html and "FOOTAGE_RETRY_BASE_MS" in html
    error_handler = html.split("function onFootageError", 1)[1].split("function onFootageLoad", 1)[0]
    assert "state.footageRetry < FOOTAGE_RETRY_LIMIT" in error_handler
    assert "openFootage();" in error_handler, "the retry reopens the footage itself, not through a counter reset"
    assert "state.footageError = true" in error_handler, "only exhausted retries call the server absent"
    load_handler = html.split("function onFootageLoad", 1)[1]
    assert "state.footageRetry = 0" in load_handler, "a frame that arrives clears the retry state"


def test_a_superseded_stream_is_aborted_not_just_ignored() -> None:
    """Chromium keeps *draining* an MJPEG fetch while its <img> exists, so merely removing the src leaves every
    superseded stream decoding - here and on the server, holding one of its slots - with nobody watching. Every
    source change therefore clears the src (which aborts the fetch) and swaps in a fresh element."""
    html = COMPONENT.read_text()
    assert "function replaceFootageImage" in html
    body = html.split("function replaceFootageImage", 1)[1].split("function setFootageSource", 1)[0]
    assert "removeAttribute('src')" in body, "clearing the src is what aborts the fetch"
    assert "replaceWith(next)" in body, "the drained element must be replaced, not reused"
    setter = html.split("function setFootageSource", 1)[1].split("\n      function ", 1)[0]
    assert "replaceFootageImage()" in setter, "every source change goes through the swap"


def test_an_abandoned_stream_is_stopped_by_the_server_not_left_draining() -> None:
    """A browser will not abort an MJPEG fetch on command - Chromium keeps draining a replaced <img> - so the
    pane tells the server to end the stream it is abandoning: before every replacement, and via a pagehide
    beacon when the page goes away. Without it every seek/toggle would leave a decode running."""
    html = COMPONENT.read_text()
    assert "function stopFootageStream" in html
    assert "/stop?token=" in html
    assert "sendBeacon" in html, "the page says goodbye on unload"
    assert "footageToken" in html, "each stream identifies itself"
    opener = html.split("function openFootage", 1)[1].split("function restartFootage", 1)[0]
    assert "stopFootageStream();" in opener, "a stream must be stopped before its replacement starts"
    assert "newFootageToken()" in opener, "the replacement gets its own token"


def test_a_repeated_render_does_not_reload_the_still() -> None:
    """Streamlit re-sends the args while a component stays unchanged, and every payload redraws the pane. While
    paused that must not swap the still <img>: the reload would abort a decode in flight for a picture that is
    already on screen - a 4K seek spent per rerun. A still that *failed* is the exception: its retry must go
    through, so a loaded-with-no-pixels image is not skipped."""
    html = COMPONENT.read_text()
    setter = html.split("function setFootageSource", 1)[1].split("\n      function ", 1)[0]
    assert "url === footage.getAttribute('src')" in setter, "the same request twice is a no-op"
    assert "!failed" in setter, "except when the picture failed to load - that retry has to proceed"


def test_the_panes_clock_comes_from_the_video_itself() -> None:
    """The pane no longer *infers* where the stream is: a <video> element reports its own timeline, so the drift
    check compares the animation's clock against a measurement - the stream's zero is the strip second it was
    asked for, and its clock starts when the first *frame* arrives, not when the request was made."""
    html = COMPONENT.read_text()
    assert "footageVideo.currentTime * state.footageRate" in html, "the display position is measured, not modelled"
    assert "state.footageOpenStrip = state.time" in html, "the stream's zero is the strip second it was asked for"
    opener = html.split("function openFootage", 1)[1].split("function restartFootage", 1)[0]
    assert "state.footageAwaitingFirstFrame = true" in opener, "the wait begins with the request"
    load_handler = html.split("footageVideo.addEventListener('loadeddata'", 1)[1]
    assert "state.footageAwaitingFirstFrame = false" in load_handler, "frame one ends the wait"
    assert "state.footageOpenStrip = state.time" in load_handler, (
        "the encoder's start-up second is re-anchored away, or a stream in step reads as permanently behind"
    )
    sync_body = html.split("function syncFootage", 1)[1].split("\n      function ", 1)[0]
    assert "state.footageAwaitingFirstFrame || footageVideo.readyState < 2" in sync_body, (
        "no drift verdict before the first frame can arrive"
    )


def test_the_playing_footage_is_full_motion_video_with_optional_sound() -> None:
    """While the animation plays, the pane plays the encoded live stream - full frame rate and the match's own
    audio - in a <video> element; the still stays up until the stream's first frame, so an encoder start-up does
    not flash black. The sound starts off (browsers block audible autoplay) and the box is the gesture that
    allows it."""
    html = COMPONENT.read_text()
    assert '<video id="footage-video"' in html, "the live stream needs a video element"
    assert "function openLiveVideo" in html and "function closeLiveVideo" in html
    assert "footageVideo.src = url" in html and "footageVideo.load()" in html, "clearing the src aborts the fetch"
    assert "const FOOTAGE_FPS = 30" in html
    assert "&fps=${FOOTAGE_FPS}&rate=${state.speed}" in html, "the live URL carries the pane's own rate"
    assert "readyState >= 2" in html, "the still gives way only once the stream has a frame"
    assert 'id="sound"' in html and "footageVideo.muted = !state.footageSound" in html


def test_the_moment_jump_follows_changing_sequences_only() -> None:
    """Python's moment picker moves the *animation* to the chosen second - the stream follows the animation, so
    one jump moves both. The jump arrives as a target plus a sequence number, and only a changed sequence acts:
    an unrelated rerun (a scan's progress, a table redraw) must not yank the playback back to the moment."""
    html = COMPONENT.read_text()
    assert "args.seek_to_s" in html and "args.seek_seq" in html, "the jump arrives as target plus sequence"
    assert "state.seekSeq" in html
    assert "pendingSeek" in html, "a jump that arrives before the payload finishes loading must still land"
    handler = html.split("seekSeq && seekSeq !== state.seekSeq", 1)
    assert len(handler) == 2, "the sequence, not the mere presence of a target, decides when to act"
    assert "setPlaying(true)" in handler[1], "jumping to a moment starts playback at it"
