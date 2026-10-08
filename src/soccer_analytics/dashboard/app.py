"""Dashboard for the gimbal-camera (single following camera) match analysis.

Workflow, in the order it appears on screen:

1. **Choose footage** - pick or browse to a video, optionally start part-way in, and run the heavy per-segment pass
   (Stage A) in the background with a progress bar and resume.
2. **Register the pitch** - click landmarks in a reference frame; the camera's motion is already known, so this
   calibrates position and orientation and reports how well it fits, per landmark.
3. **Build the report** - cheap and re-runnable: pitch-space tracking, teams, distances, momentum, and the
   separately-scanned ball track when one has run.
4. **Tag events and cut highlights** - whistle candidates from the audio, manual tags for goals/shots/saves, and the
   three highlight tiers.

Every number on the page is either measured or explicitly labelled as a guess or a manual tag. Where a metric is
beyond what the footage (and the scans built on it) can honestly support, the page says so rather than inventing one.
"""

from __future__ import annotations

import base64
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from streamlit import runtime as st_runtime

REPO_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from soccer_analytics.analysis import game as game_lib
from soccer_analytics.analysis import stage_a, stage_b
from soccer_analytics.analysis import event_detection
from soccer_analytics.analysis.event_detection import detect_events
from soccer_analytics.analysis.events import (
    EVENT_TYPES,
    MIN_PROMINENCE,
    VERDICT_FALSE,
    VERDICT_TRUE,
    Event,
)
from soccer_analytics.analysis.highlights import (
    PREVIEW_DIR,
    PREVIEW_SETTINGS,
    TIER_SECONDS,
    build_moments,
    clamp_moment,
    export_moment,
    export_reel,
    moment_for_event,
    preview_clip_name,
    select_reel,
    write_manifest,
)
from soccer_analytics.analysis.library import MatchLibrary, new_match_id
from soccer_analytics.analysis.kit import colour_hex, colour_name, suggest_team_name
from soccer_analytics.analysis.jerseys import merge_numbers
from soccer_analytics.analysis.projection import project_ball_track, project_segment, segment_poses
from soccer_analytics.analysis.stage_a import SegmentConfig, load_segment, read_status, resolve_window, segment_dir_for
from soccer_analytics.geometry.gimbal_motion import segment_has_log, segment_pose_source
from soccer_analytics.dashboard.pitch_clicks import (
    LANDMARK_HELP,
    actual_centre,
    canvas_scale,
    clamp01,
    frame_change,
    frame_points,
    inside_crop,
    landmark_table,
    landmarks_from_points,
    marker_feature,
    merge_clicked,
    order_clicks,
    overlay_homographies,
    parse_result,
    pitch_marking_polylines,
    pitch_overlay,
    point_centre,
    points_in_crop,
    projected_landmarks,
    repeated_labels_within_a_frame,
    restored_points,
    split_duplicate_clicks,
    zoom_box,
)
from soccer_analytics.dashboard import timeline
from soccer_analytics.dashboard.reports import (
    DEFAULT_TEAM_NAMES,
    colours_were_recorded,
    is_default_team_name,
    report_from_library,
    team_colours,
    team_name,
)
from soccer_analytics.dashboard.replay import build_replay, player_table_rows
from soccer_analytics.geometry.pitch_calibration import (
    MIN_CLICKS_PER_ANCHOR,
    MIN_REFERENCE_CLICKS,
    PitchCalibration,
    calibrate,
    diagnose_fit,
    format_scale_note,
)
from soccer_analytics.ingest.ffmpeg_reader import grab_frame, probe_video

SEGMENTS_ROOT = REPO_ROOT / "data" / "segments"
MATCHES_ROOT = REPO_ROOT / "data" / "matches"
ANNOTATION_COMPONENT = components.declare_component(
    "field_annotation_editor", path=str(Path(__file__).parent / "field_annotation_component")
)
REPLAY_COMPONENT = components.declare_component(
    "pitch_replay", path=str(Path(__file__).parent / "pitch_replay_component")
)
GAME_TIMELINE_COMPONENT = components.declare_component(
    "game_timeline", path=str(Path(__file__).parent / "game_timeline_component")
)

MATCH_FORMATS: dict[str, tuple[float, float]] = {  # length m, width m
    "5v5": (40.0, 25.0),
    "7v7": (50.0, 35.0),
    "9v9": (60.0, 40.0),
    "11v11": (100.0, 64.0),
}

# Zoom the landmark view starts at. The whole frame is the main viewport, so starting zoomed in is cheap: there is
# always a view that shows the whole pitch, and the magnified inset is immediately accurate enough to click.
DEFAULT_ZOOM = 8.0

# The camera rig is a pole at a fixed height, so the calibration solver pins the height instead of solving it -
# one fewer degree of freedom for the clicks to pin down, which measurably tightens a four-click fit.
CAMERA_HEIGHT_M = 4.0

# How often the fragments that watch background tasks re-read their status files. They matter only while the task
# runs: the running -> finished transition triggers one full rerun, and the fragment stops being drawn.
POLL_SECONDS = 4.0

st.set_page_config(page_title="Match analysis", layout="wide")


# --------------------------------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------------------------------
def video_roots() -> list[Path]:
    """Where match footage is looked for, newest-first within each root.

    ``data/videos`` in the repo comes first; ``SOCCER_VIDEO_ROOTS`` (colon-separated) adds machine-specific
    archives after it - the old hardcoded ``/srv/storage/...`` path only existed on the author's machine, and on
    any other box it silently halved the video picker.
    """
    extra = os.environ.get("SOCCER_VIDEO_ROOTS", "/srv/storage/home_video/Xbot")
    roots = [REPO_ROOT / "data" / "videos"]
    roots += [Path(part).expanduser() for part in extra.split(":") if part.strip()]
    return roots


def discover_videos() -> list[Path]:
    """Video files under the usual places, newest first, so the most recent match is the default."""
    found: list[Path] = []
    for root in video_roots():
        if not root.exists():
            continue
        found += list(root.rglob("*.MP4")) + list(root.rglob("*.mp4"))
    return sorted(set(found), key=lambda p: p.stat().st_mtime, reverse=True)


def _clock(seconds: float) -> str:
    """Match time as a coach reads it; hours appear only when the footage needs them."""
    total = int(max(0.0, seconds))
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    return f"{minutes}:{secs:02d}"


def _event_label(event: Event, index: int, match_clock: str | None = None) -> str:
    """One line describing an event, for the preview picker.

    The body of this string must not depend on anything that changes while the page is open - a verdict, say. A
    selectbox's labels are rendered once and handed back by the frontend as the widget's value, so a label that
    later no longer matches the option list (because a verdict changed it) leaves the page holding a string where it
    expects an index. The verdict is shown under the player and in the table instead.

    ``match_clock`` is the time on the game's own clock when the caller can work it out. A candidate's own
    ``time_s`` is a time on the recording it was found in, which is not what a coach reads out - a whistle at 5 s
    of the second camera clip is 30 minutes into the match - so the match time leads when it is known.
    """
    origin = "audio" if event.source == "audio" else (f"team {event.team}" if event.team >= 0 else "manual")
    note = f" - {event.note[:48]}" if event.note else ""
    when = match_clock if match_clock and match_clock != "-" else _clock(event.time_s)
    return f"#{index + 1}  {when}  {event.type} ({origin}){note}"


# What each preview mode costs, in the words of the person choosing. Reviewing a scan means ninety clips one after
# another, so the cheap end is the default and the labels say what is being traded away.
PREVIEW_MODE_LABELS = {
    "audio": "Sound only",
    "quick": "Light video",
    "full": "Full video",
}


def _encode(frame: np.ndarray) -> str:
    """Frame images are sent to the component as JPEG.

    Two images go over on every rerun - the crop and the whole frame - and PNG of a photographic frame is megabytes,
    which is the difference between the view following a gesture promptly and feeling stuck. The compression cannot
    move a click: the mapping is in image coordinates either way.
    """
    ok, buffer = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 88])
    if not ok:
        raise ValueError("could not encode the frame")
    return "data:image/jpeg;base64," + base64.b64encode(buffer.tobytes()).decode("ascii")


def landmark_clicker(
    crop: np.ndarray,
    overview: np.ndarray | None,
    features: list[dict],
    markers: list[tuple[float, float]],
    centre: tuple[float, float],
    zoom: float,
    box: tuple[int, int, int, int],
    frame_size: tuple[int, int],
    key: str,
    *,
    proxy_url: str = "",
    frame_count: int = 0,
    initial_frame: int = 0,
    marker_kinds: list[str] | None = None,
    mount_nonce: int = 0,
    overlay_homographies: list[dict] | None = None,
    overlay_polylines: list[list[list[float]]] | None = None,
) -> None:
    """Draw the landmark clicker over one crop, with the whole frame as its navigation viewport.

    The component owns the whole view: the whole-frame viewport where clicking aims, scrolling zooms and dragging
    pans; the magnified crop where the landmarks are clicked; and - when a proxy is available - the timeline bar
    under them both, which scrubs that same viewport.

    This only draws it. What the user did arrives as the component's widget value, which Streamlit hands to the
    *next* run before it starts - so the caller reads that (``parse_result``) and settles it into session state
    before the crop is computed, and one gesture costs one run rather than two. Nothing here can act on the value:
    by the time it is returned, the frame above has already been read and cropped.
    """
    scale = canvas_scale(crop.shape[1])
    canvas = crop if scale == 1.0 else cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    frame_width, frame_height = frame_size
    x0, y0, _w, _h = box
    placed_x, placed_y = actual_centre(box, frame_size)
    result = ANNOTATION_COMPONENT(
        frame_data=_encode(canvas),
        # Only the still-frame fallback needs the whole frame as an image; with a proxy the video is the picture.
        overview_data=_encode(overview) if overview is not None else "",
        proxy_url=proxy_url,
        frame_count=int(frame_count),
        initial_frame=int(initial_frame),
        marker_kinds=list(marker_kinds or []),
        mount_nonce=int(mount_nonce),
        # The whole-frame viewport is scrubbed and played in the browser, so Python cannot draw the pitch overlay
        # onto frames it never sees. Instead it sends sampled pitch->pixel homographies plus the markings in pitch
        # metres, and the component projects the overlay itself for whatever frame is on screen - and re-derives it
        # from the same arguments whenever the calibration changes, so a refit redraws immediately.
        overlay_homographies=list(overlay_homographies or []),
        overlay_polylines=list(overlay_polylines or []),
        initial_polygon=[],
        initial_features=features,
        marker_points=[[float(u), float(v)] for u, v in markers],
        width=int(canvas.shape[1]),
        height=int(canvas.shape[0]),
        centre_x=float(centre[0]),
        centre_y=float(centre[1]),
        zoom=float(zoom),
        actual_x=float(placed_x),
        actual_y=float(placed_y),
        box_x0=int(x0),
        box_y0=int(y0),
        scale=float(scale),
        frame_width=int(frame_width),
        frame_height=int(frame_height),
        key=key,
        default=None,
    )
    del result  # the gesture is read at the top of the next run, not from this return value


def _served_video_url(path: Path, coordinates: str) -> str | None:
    """Serve a video through Streamlit's own media endpoint so the browser can seek it.

    A component's static route ignores HTTP ``Range`` requests - it answers every request with the whole file and no
    ``Accept-Ranges`` - and a video the browser cannot seek is useless for scrubbing. The media endpoint (the one
    ``st.video`` uses) does range requests properly.

    It has to be registered on *every* run. Streamlit drops a session's media references before each run and deletes
    anything the body did not re-register, so a URL cached across runs ends up pointing at a file that has been
    swept away - the video dies with a 404. The URL is a content hash, so registering again keeps it stable and the
    iframe is not remounted. Returns ``None`` when the endpoint is not available.
    """
    if not st_runtime.exists():
        return None
    try:
        return st_runtime.get_instance().media_file_mgr.add(str(path), "video/mp4", coordinates)
    except Exception:  # a media-endpoint failure must not take the page down
        return None


def _timeline_media_url(proxy: Path, coordinates: str) -> str:
    """The segment's scrub proxy, with the component's static route as a last resort."""
    return _served_video_url(proxy, coordinates) or timeline.PROXY_FILE


def _replay_media_url(path: Path) -> str:
    """Serve the replay JSON through the media endpoint, like the timeline proxy.

    The replay is megabytes of per-frame points. Sending it through the component's *arguments* would serialise it
    onto the Streamlit websocket on every rerun, so the component is handed a URL and fetches it once. The file must
    be re-registered every run for the same reason as the proxy: Streamlit sweeps media a run did not mention.
    """
    if not st_runtime.exists():
        return ""
    try:
        return st_runtime.get_instance().media_file_mgr.add(str(path), "application/json", f"replay::{path.name}")
    except Exception:
        return ""


def replay_view(
    data_url: str,
    numbers: dict[int, dict],
    selected_track: int,
    team_names: list[str],
    key: str,
    events: list[dict] | None = None,
    momentum: dict | None = None,
    half_minute: float | None = None,
    clip_url: str = "",
    clip_start_s: float | None = None,
    clip_end_s: float | None = None,
) -> None:
    """Draw the animated pitch view; playback lives entirely in the browser.

    ``events`` and ``momentum`` are the timeline strip's data: the tagged and detected events (as JSON dicts) and
    the per-minute momentum buckets. They travel as component arguments rather than in the replay payload because
    they are small and change without the replay being rebuilt - a new tag shows up on the strip on the next rerun.

    ``clip_url`` is the generated clip shown beside the pitch, with ``clip_start_s``/``clip_end_s`` giving its
    window on the strip's own clock so the component can hold the two in step.
    """
    REPLAY_COMPONENT(
        data_url=data_url,
        numbers={str(track): entry for track, entry in numbers.items()},
        selected_track=int(selected_track),
        team_names=list(team_names),
        events=list(events or []),
        momentum=[
            {"minute": int(minute), **bucket} for minute, bucket in sorted((momentum or {}).items())
        ],
        half_minute=half_minute,
        clip_url=clip_url,
        clip_start_s=clip_start_s,
        clip_end_s=clip_end_s,
        key=key,
    )


def _team_name_editor(library: MatchLibrary, match_id: str, payload: dict) -> list[str]:
    """Show each team's kit colour and let it be named; returns the names to use from here on.

    Everything the pipeline writes calls the teams 0 and 1 (the more red kit is 0), which nobody can picture. The
    report carries the colour the clustering measured for each of them, so the swatch is the anchor: a person
    names "the red team", not "team 0", and from then on the table, the chart, the replay and the tags agree. The
    measured colour also suggests the name itself ("Reds", "Dark blues"), which turns naming into a press and an
    edit rather than a blank field - and a team that already has a name keeps it, because the suggestion only ever
    fills a placeholder.
    """
    record = library.load(match_id)
    stored = list(record.team_names)[:2] + list(DEFAULT_TEAM_NAMES[len(record.team_names) :])
    # A report written before the colours were recorded has no such field at all; one that has the field but no
    # value is a match whose kits the clustering could not separate. The two need different advice.
    team_rows = list(payload.get("teams") or [])
    colours_recorded = colours_were_recorded(team_rows)
    colours = team_colours(team_rows)
    suggestions = [suggest_team_name(rgb) if rgb is not None else "" for rgb in colours]

    st.subheader("Teams")
    st.caption(
        "The two teams are told apart by kit colour, and each keeps the same number every time this report is built. "
        "Name them here - the table below, the momentum chart, the replay and the event tags then use the names."
    )
    for index, column in enumerate(st.columns(2)):
        with column:
            rgb = colours[index]
            if rgb is not None:
                st.color_picker(
                    f"Team {index + 1} kit colour",
                    value=colour_hex(rgb),
                    disabled=True,
                    key=f"kit_swatch::{match_id}::{index}",
                    help="Measured from the match footage: the average colour of the kit pixels of the players the clustering put in this team.",
                )
                looks = f"Looks {colour_name(rgb)} in the footage."
                if suggestions[index] and is_default_team_name(stored[index]):
                    looks += f" A name for that might be **{suggestions[index]}**."
                st.caption(looks)
            elif not colours_recorded:
                st.caption(
                    f"Team {index + 1}: this report was built before kit colours were recorded, so there is no "
                    "swatch or suggested name to show yet - build the report again below and they appear."
                )
            else:
                st.caption(f"Team {index + 1}: no kit colour recorded - the kits were not separable here.")

    with st.form(f"team_names::{match_id}"):
        for index in (0, 1):
            current = stored[index]
            if is_default_team_name(current):
                current = suggestions[index] or DEFAULT_TEAM_NAMES[index]
            st.text_input(
                f"Team {index + 1} name",
                value=current,
                key=f"team_name::{match_id}::{index}",
                max_chars=40,
            )

        def _save_names() -> None:
            fresh = library.load(match_id)
            fresh.team_names = [
                str(st.session_state.get(f"team_name::{match_id}::{i}", "")).strip() or DEFAULT_TEAM_NAMES[i]
                for i in (0, 1)
            ]
            library.save(fresh)
            st.session_state["step3_flash"] = (
                "success",
                f"Teams are now {fresh.team_names[0]} (team 1) and {fresh.team_names[1]} (team 2).",
            )

        st.form_submit_button("Save team names", on_click=_save_names)
    return [
        str(st.session_state.get(f"team_name::{match_id}::{i}", stored[i])).strip() or DEFAULT_TEAM_NAMES[i]
        for i in (0, 1)
    ]


@st.cache_data(show_spinner=False)
def _load_replay_cached(path: str, mtime_ns: int) -> dict:
    """Parse the replay once per file version; it is re-read on every rerun otherwise."""
    del mtime_ns
    return json.loads(Path(path).read_text())


BALL_STATUS_STALE_S = 300.0
BALL_STATUS_FILE = "ball_scan.json"  # written by scripts/run_ball_scan.py beside the segment
BALL_TRACK_FILE = "ball_track.json"  # the scan's result, beside the status


def _ball_status(segment_dir: Path) -> dict:
    """The ball scan's status file beside the segment; {} until it has ever run."""
    try:
        return json.loads((Path(segment_dir) / BALL_STATUS_FILE).read_text())
    except (OSError, json.JSONDecodeError):
        return {}


def _ball_scan_alive(segment_dir: Path) -> bool:
    """Whether a scan is running right now: its status says so *and* it was updated recently.

    The recentness is the point. A process killed without a chance to write (a reboot, a crash) leaves "running"
    behind, and a button disabled by a dead process cannot be pressed to resume it - so anything that has not
    reported for minutes counts as stopped. The scan updates every few seconds while it runs, so the window is
    generous.
    """
    status = _ball_status(segment_dir)
    if status.get("state") != "running":
        return False
    return (time.time() - float(status.get("updated") or 0.0)) < BALL_STATUS_STALE_S


def _ball_track_for_replay(
    segment_dir: Path, calibration: PitchCalibration, q: np.ndarray, focal: np.ndarray
):  # noqa: ANN201 - (xy (F,2), measured (F,)) from project_ball_track, or None
    """The segment's ball scan projected into pitch metres, or None when it has not been scanned.

    A partial scan is used as it stands: the replay draws the frames the scan has reached, and the scan's own
    status line says whether it is finished. A scan that lands after the report was built shows up on the next
    build - which is cheap, so the page tells the user to press it again rather than rebuilding behind their back.
    """
    try:
        payload = json.loads((Path(segment_dir) / BALL_TRACK_FILE).read_text())
    except (OSError, json.JSONDecodeError):
        return None
    records = payload.get("frames") or []
    if not records:
        return None
    return project_ball_track(records, calibration, q, focal)


def _clip_offsets(video: str) -> dict[str, float]:
    """Where each of the game's clips starts inside ``video``, for mapping a moment onto the file being cut.

    A moment's seconds are seconds of its own recording: a whistle candidate was scanned on a single camera file,
    a detected event on the combined game. When the reel is cut from the selected video, the window has to be
    translated through the game manifest or the clip shows the wrong part of the match. Empty when ``video`` is not
    a combined game (a single clip has nothing to map) or the manifest cannot be read.
    """
    record = game_lib.find_for_video(video)
    if record is None or len(record.clips) < 2:
        return {}
    return {
        str(Path(clip.path).resolve()): float(clip.start_s)
        for clip in record.clips
        if Path(clip.path).exists()
    }


# The clip beside the pitch is watched, not reviewed: a 4 fps keyframe-only cut stutters badly enough to make the
# sync with the animation feel broken, so it is cut at full frame rate. It costs a decode of every frame (~11 s for
# a 10 s clip on the 4K source rather than ~3 s), which is the price of footage that actually plays.
CLIP_BESIDE_PITCH_MODE = "full"


def _clip_beside_pitch(
    library: MatchLibrary,
    match_id: str,
    replay_duration_s: float,
    chosen: str,
    event_log,
    game_record,
    window_start: float,
) -> tuple[str, float | None, float | None]:
    """The clip shown beside the pitch, and its window on the strip's clock.

    The moment is chosen from the event log and cut on request, because cutting costs a decode of the footage. The
    clip is cut from the recording the moment was found in - a whistle candidate's seconds are seconds of a single
    camera file, not of the combined game - and its window is then translated onto the strip's own clock so the
    component can hold the two in step.

    Returns ``(url, start_s, end_s)``; the URL is empty when no clip has been cut, and the window is ``None`` when
    the moment does not lie inside the analysed window (so there is nothing to sync it to).
    """
    if not event_log.events:
        return "", None, None
    preview_dir = library.highlights_dir(match_id) / PREVIEW_DIR
    # The picker is the same list the table shows, so a moment is chosen once and read everywhere. Its label leads
    # with the match clock where the game is marked, because that is the time a coach reads out.
    def _label(index: int) -> str:
        event = event_log.events[index]
        match_clock = None
        if game_record is not None and game_record.bounds() is not None:
            on_game = game_lib.game_time(game_record, event.time_s, event.video or game_record.output)
            if on_game is not None:
                half = game_record.half_of(on_game)
                marks = game_record.bounds()
                if half == 1:
                    match_clock = _clock(on_game - marks[0])
                elif half == 2:
                    match_clock = f"{_clock(on_game - marks[1])} (2H)"
        return _event_label(event, index, match_clock)

    pick_col, button_col = st.columns([3, 1])
    with pick_col:
        index = st.selectbox(
            "Clip to show beside the pitch",
            range(len(event_log.events)),
            format_func=_label,
            key=f"pitch_clip_event::{match_id}",
            help=(
                "The clip is cut from the recording the moment was found in, and plays in step with the animation "
                "while the animation is inside its window."
            ),
        )
    event = event_log.events[index]
    source = Path(event.video) if event.video else Path(chosen)
    moment = moment_for_event(event)
    # The window on the strip's clock: the moment's own recording time translated onto the game's clock and then
    # onto the analysed window's. This is what the component syncs against, so it has to be the same translation
    # the strip's markers use.
    on_game = (
        game_lib.game_time(game_record, moment.time_s, event.video)
        if game_record is not None and event.video
        else moment.time_s
    )
    strip_start = None if on_game is None else on_game - window_start
    strip_end = None if strip_start is None else strip_start + (moment.end_s - moment.start_s)
    # The strip's own clock runs 0..duration over the analysed window, so a moment outside it has nothing to be
    # synced to - the component would be asked to hold the clip against a second the animation never reaches.
    if strip_start is not None and (strip_end < 0 or strip_start > replay_duration_s):
        strip_start = strip_end = None
    with button_col:
        st.write("")  # line the button up with the selectbox
        generate = st.button(
            "Generate clip",
            key=f"pitch_clip_generate::{match_id}",
            disabled=not source.exists(),
            help="Cut this moment from the footage and play it beside the animation.",
        )
    if not source.exists():
        st.caption(f"`{source.name}` is not available, so this moment cannot be cut.")
        return "", strip_start, strip_end
    clip_file = preview_dir / preview_clip_name(moment, CLIP_BESIDE_PITCH_MODE)
    if generate and not clip_file.exists():
        bar = st.progress(0.0, text="Cutting the clip...")
        reporter = progress_reporter(bar, "Cutting the clip...")
        try:
            export_moment(
                source,
                clamp_moment(moment, probe_cached(str(source)).duration_s) or moment,
                clip_file,
                mode=CLIP_BESIDE_PITCH_MODE,
                use_gpu=True,
                progress=reporter,
                clip_offsets=_clip_offsets(str(source)),
            )
            reporter(1.0, "Clip ready")
        except Exception as exc:  # a failed cut must not take the page down
            st.error(f"Could not cut the clip: {exc}")
            return "", strip_start, strip_end
    if not clip_file.exists():
        st.caption("Press **Generate clip** to cut this moment and play it beside the animation.")
        return "", strip_start, strip_end
    url = _served_video_url(clip_file, f"pitchclip::{clip_file.name}") or ""
    return url, strip_start, strip_end


@st.cache_data(show_spinner=False)
def probe_cached(video: str):  # noqa: ANN201 - VideoProbe, kept out of the import list of callers
    """ffprobe once per video: the page re-renders on every gesture and every poll, and each probe is a process."""
    return probe_video(video)


def segment_fingerprint(segment_dir: Path, chunks: int) -> str:
    """Cache key for a segment's contents: it changes when meta, status or the set of chunks changes."""
    meta_mtime = (segment_dir / "meta.json").stat().st_mtime_ns if (segment_dir / "meta.json").exists() else 0
    status_mtime = (segment_dir / "status.json").stat().st_mtime_ns if (segment_dir / "status.json").exists() else 0
    return f"{meta_mtime}-{status_mtime}-{chunks}"


@st.cache_resource(show_spinner=False, max_entries=2)
def segment_and_poses_cached(segment_dir: str, fingerprint: str):  # noqa: ANN201
    """Load a segment and rebuild its pose chain once per process, not once per rerun.

    Step 2 is a gesture loop: every aim click re-renders the page, and loading a whole-game segment (hundreds of MB)
    and integrating 21k poses took the better part of a minute of *every* rerun. The fingerprint in the key is what
    keeps this honest: a re-analysis (new meta/status/chunks) misses the cache instead of serving stale poses.
    """
    del fingerprint
    segment = load_segment(segment_dir)
    q, focal = segment_poses(segment)
    return segment, q, focal


@st.cache_resource(show_spinner=False, max_entries=6)
def frame_cached(video: str, time_s: float, width: int):  # noqa: ANN201 - a shared, read-only frame array
    """A frame read from the source, shared across reruns.

    Aiming changes the *crop*, not the frame: without this every click paid a fresh 4K ffmpeg seek (2+ s). Callers
    treat the frame as read-only - the crop is a view, and the fit check draws on a resized copy.
    """
    return grab_frame(video, time_s, width=width)


def progress_reporter(bar, label: str):  # noqa: ANN001, ANN201 - st.progress element and a callable back
    """A throttled ``bar.progress`` updater for tasks that report often (the pipeline and ffmpeg do).

    A repaint per callback would flood the websocket; two percent or half a second apart is smooth to the eye and
    cheap. The final 100% always goes through.
    """
    state = {"last": 0.0, "fraction": -1.0}

    def report(fraction: float, text: str | None = None) -> None:
        fraction = min(1.0, max(0.0, float(fraction)))
        now = time.time()
        if fraction >= 1.0 or fraction - state["fraction"] >= 0.02 or now - state["last"] >= 0.5:
            state["last"], state["fraction"] = now, fraction
            bar.progress(fraction, text=text or label)

    return report


def timeline_ready(segment, segment_dir: Path) -> bool:
    """Whether the segment's scrub proxy is ready, starting the background build if it is not.

    The whole-frame viewport doubles as the timeline, which plays a small proxy of the segment. That proxy is a full
    pass over the video - minutes of work for 4K/60 footage - so it is built as its own process (like Stage A) and
    the viewport falls back to a still frame until it lands.
    """
    if timeline.proxy_is_ready(segment_dir):
        return True
    start_s = float(segment.meta.get("start_s", segment.time[0]))
    duration_s = float(segment.meta.get("end_s", segment.time[-1] + 1.0)) - start_s
    timeline.start_background_build(segment_dir, segment.meta["video"], start_s=start_s, duration_s=duration_s)
    return False


@st.fragment(run_every=POLL_SECONDS)
def _timeline_status(segment_dir: Path, watch_key: str) -> None:
    """Live progress of the scrub-timeline build, and the switch-over when it lands."""
    state = timeline.read_build_state(segment_dir)
    status = state.get("state")
    previous = st.session_state.get(watch_key)
    st.session_state[watch_key] = status
    if status == "error":
        st.warning(
            f"Could not build the scrub timeline ({state.get('error')}). Using a still frame and a plain frame "
            "slider - you can retry the background build below."
        )

        def _retry_build() -> None:
            timeline.write_build_state(segment_dir, state="idle", error=None)

        st.button("Retry building the scrub timeline", on_click=_retry_build, key=f"timeline_retry::{segment_dir}")
    elif status == "running":
        fraction = float(state.get("progress") or 0.0)
        st.progress(
            min(1.0, fraction),
            text="Building the scrub timeline - a one-time pass over this segment...",
        )
        st.caption(
            "The build runs in the background and this bar updates by itself. Until it lands the view below uses a "
            "still frame with a plain slider; afterwards it becomes a video you can scrub. Long segments are built "
            "from keyframes only - about one picture per second, so minutes rather than the length of the footage - "
            "and the frame you click is still read from the source at full resolution."
        )
    else:
        st.info("Preparing the scrub timeline in the background...")
    if previous == "running" and status != "running":
        st.rerun()


@st.fragment(run_every=POLL_SECONDS)
def _stage_a_status(segment_dir: Path, watch_key: str) -> None:
    """Live Stage A progress.

    Polls the status file in a fragment, so the bar moves without a manual refresh. When the run finishes, one full
    rerun replaces this fragment and lets the rest of the page - which was rendered before the change - pick the
    finished segment up.
    """
    status = read_status(segment_dir)
    state = (status or {}).get("state")
    previous = st.session_state.get(watch_key)
    st.session_state[watch_key] = state
    if status is None:
        st.info("Not analysed yet.")
    else:
        st.write(f"Stage A: **{state}**, {status.get('frames_done', 0)}/{status.get('total_frames', 0)} frames")
        if state == "running":
            fraction = status.get("frames_done", 0) / max(1, status.get("total_frames", 1))
            detail = f"{status.get('fps', 0):.1f} analysed fps"
            eta = status.get("eta_s")
            if isinstance(eta, (int, float)) and eta > 0:
                detail += f", about {_clock(float(eta))} to go"
            st.progress(min(1.0, fraction), text=detail)
        elif state == "error":
            st.error(status.get("error", "unknown error"))
        if status.get("lost"):
            st.caption(f"{status['lost']} frame(s) had no usable camera motion; they are excluded from metrics.")
    st.caption(
        f"Output directory: `{segment_dir.relative_to(REPO_ROOT)}` "
        f"({stage_a.completed_chunks(segment_dir)} chunk(s) on disk)"
    )
    if previous == "running" and state != "running":
        st.rerun()


def saved_calibration(library: MatchLibrary, match_id: str | None) -> PitchCalibration | None:
    """The calibration for this match: whatever was just clicked, else whatever is on disk.

    Returns None when there is no match selected, which is the normal state on a first visit to the page.
    """
    pending = st.session_state.get("calibration")
    if pending:
        return PitchCalibration.from_json(pending)
    if match_id is None:
        return None
    return library.load_calibration(match_id)


def fit_from_clicks(
    library: MatchLibrary,
    match_id: str,
    segment,
    q: np.ndarray,
    focal: np.ndarray,
    labelled: list[tuple[dict, str]],
    land_table: dict[str, tuple[float, float]],
    length_m: float,
    width_m: float,
) -> PitchCalibration | None:
    """Fit the calibration from the labelled clicks and save it; None when the fit failed.

    Shared by the automatic refit and the manual button, so both do exactly the same thing: solve, store in the
    session, and write the calibration and its clicks to the match record. The caller reports the outcome - the
    message differs between the two paths.

    The rig's height is known (a fixed 4 m pole), so it is pinned rather than solved - one fewer degree of freedom
    for the clicks to pin down, which measurably tightens a four-click fit.
    """
    landmarks = landmarks_from_points(labelled, land_table)
    chain = {point["frame"]: (q[point["frame"]], float(focal[point["frame"]])) for point, _ in labelled}
    try:
        calibration = calibrate(
            landmarks, chain, segment.aspect, pose_source=segment_pose_source(segment), fixed_height_m=CAMERA_HEIGHT_M
        )
    except Exception as exc:
        # Keep the cause for the page: one generic sentence hid real bugs behind "check the labels" - the solver
        # saying *why* it failed is the difference between a five-minute fix and a hunt.
        st.session_state["calib_fit_error"] = f"{type(exc).__name__}: {exc}"
        return None
    st.session_state.pop("calib_fit_error", None)
    clicks = [{"frame": lm.frame, "u": lm.u, "v": lm.v, "label": lm.label} for lm in landmarks]
    st.session_state["calibration"] = calibration.to_json()
    st.session_state["calib_landmarks"] = clicks
    library.save_calibration(match_id, calibration)
    library.save_clicks(match_id, clicks, (length_m, width_m))
    return calibration


def calibration_failure() -> str:
    """What a failed fit shows: the solver's own cause when it left one, else the generic advice."""
    cause = st.session_state.get("calib_fit_error")
    if cause:
        return f"Calibration failed: {cause}"
    return "Calibration failed - the clicks do not determine a camera. Check the labels."


def fit_message(calibration: PitchCalibration, labelled: list[tuple[dict, str]]) -> str:
    """What a successful fit says about itself: the RMS, and whether the drift got re-anchored."""
    message = f"Fit RMS error {calibration.rms_error_m:.2f} m."
    if calibration.drift is not None:
        message += (
            f" The chain is re-anchored at the {len(calibration.drift.frames)} clicked frame(s) - "
            "click landmarks later in the video to pin the drift there too."
        )
    elif len({point["frame"] for point, _label in labelled}) > 1:
        message += (
            " No drift correction was fitted: it needs at least two moments with "
            f"{MIN_CLICKS_PER_ANCHOR} or more clicks each, and the moments without enough clicks "
            "still judge the fit - check their residuals below."
        )
    return message


@st.fragment(run_every=POLL_SECONDS)
def _jersey_scan_status(library: MatchLibrary, match_id: str, watch_key: str) -> None:
    """Live progress of the background shirt-number scan; the page picks the numbers up when it lands."""
    status = library.load_jerseys_status(match_id)
    state = status.get("state")
    previous = st.session_state.get(watch_key)
    st.session_state[watch_key] = state
    if not status:
        st.caption(
            "Not scanned yet. The scan re-decodes this segment, crops the torso of every tracked player on the "
            "frames where they are largest, and reads the number with OCR. It keeps the readings per track and only "
            "reports a number when several agree. It runs in the background and takes a few minutes; the scan is a "
            "suggestion - manual entries override it."
        )
    elif state == "running":
        done, total = int(status.get("crops_done", 0)), int(status.get("crops_total", 0))
        st.progress(
            min(1.0, done / max(1, total)),
            text=f"{done}/{total} crops, {status.get('readings', 0)} readings",
        )
        st.caption("Running in the background - this bar updates by itself.")
    elif state == "error":
        st.error(f"The scan failed: {status.get('message')}")
    else:
        meta = library.load_jerseys(match_id).get("meta", {})
        st.caption(
            f"Last scan: {meta.get('crops', 0)} crops from {meta.get('tracks_scanned', 0)} tracks, "
            f"{meta.get('readings', 0)} readings, {meta.get('suggested', 0)} suggested number(s). "
            + (status.get("message") or "")
        )
    if previous == "running" and state != "running":
        st.rerun()


@st.fragment(run_every=POLL_SECONDS)
def _audio_scan_status(library: MatchLibrary, match_id: str, watch_key: str) -> None:
    """Live progress of the background whistle scan; the events table picks the candidates up when it lands."""
    status = library.load_audio_scan_status(match_id)
    state = status.get("state")
    previous = st.session_state.get(watch_key)
    st.session_state[watch_key] = state
    if not status:
        st.caption(
            "Not scanned yet. The scan decodes the recording's audio once (and caches it), then looks for narrowband "
            "blasts in the whistle band. It runs in the background and reports progress here."
        )
    elif state == "running":
        st.progress(
            min(1.0, float(status.get("progress") or 0.0)),
            text=str(status.get("message") or "Scanning..."),
        )
        st.caption("Running in the background - this bar updates by itself.")
    elif state == "error":
        st.error(f"The scan failed: {status.get('error')}")
    else:
        found = int(status.get("found", 0))
        minutes = float(status.get("minutes", 0.0))
        if found:
            already = found - int(status.get("added", found))
            dropped = int(status.get("dropped", 0))
            st.caption(
                f"Last scan: {minutes:.0f} min of audio, {found} candidate(s)"
                + (f", {already} already recorded" if already else "")
                + (f", {dropped} dropped as no longer detected" if dropped else "")
                + f", the strongest at {float(status.get('strongest', 0.0)):.0f}x the match level."
            )
        else:
            st.caption(
                f"Last scan: {minutes:.0f} min of audio, no whistle candidates. The detector wants a loud, "
                "sustained blast, so a quiet recording - or a referee a long way from the camera - can leave it "
                "with nothing to report."
            )
    if previous == "running" and state != "running":
        st.rerun()


@st.fragment(run_every=POLL_SECONDS)
def _ball_scan_status(segment_dir: Path, watch_key: str) -> None:
    """Live progress of the background ball scan; the replay draws the track once the report is rebuilt."""
    status = _ball_status(segment_dir)
    state = status.get("state")
    previous = st.session_state.get(watch_key)
    st.session_state[watch_key] = state
    if not status:
        st.caption(
            "Not scanned yet. The scan re-reads this segment's video at full resolution and follows the ball frame "
            "by frame: a window around the prediction while it holds the ball, the whole frame to re-find it after "
            "a gap, and the camera's own motion (plus the ball's velocity) as the prediction between frames. It is "
            "expensive - about an hour for a whole game - and it checkpoints on the way, so it can be stopped and "
            "resumed."
        )
    elif state == "running":
        if _ball_scan_alive(segment_dir):
            st.progress(
                min(1.0, float(status.get("progress") or 0.0)),
                text=str(status.get("message") or "Scanning..."),
            )
            st.caption("Running in the background - this bar updates by itself. It can be closed and resumed later.")
        else:
            # The process died without a chance to write (a reboot, a crash): the file still says "running" but
            # nothing has moved the bar for minutes. Saying so beats a progress bar that will never advance.
            st.warning(
                "A scan says it is running but has not reported for minutes, so it has probably stopped. The "
                "button below resumes it from its last checkpoint."
            )
    elif state == "error":
        st.error(f"The scan failed: {status.get('error')}")
    else:
        counts = status.get("counts") or {}
        scanned = int(status.get("scanned") or 0)
        if counts and scanned:
            total = int(status.get("total_frames") or 0)
            seen = 100 * counts.get("tracking", 0) / scanned
            forecast = 100 * counts.get("coasting", 0) / scanned
            off = 100 * (counts.get("out_of_view", 0) + counts.get("lost", 0)) / scanned
            summary = (
                f"Last scan: {scanned} of {total} frame(s) - the ball was seen on {seen:.0f}% of them, "
                f"forecast across a missed frame on {forecast:.0f}%, off the picture on {off:.0f}%."
            )
        else:
            # A status written by an older version of the scan has no counts in it; the replay itself does not
            # care - the track on disk is the same - so the summary degrades to the message rather than inventing
            # percentages.
            summary = f"Last scan: {status.get('message') or 'finished'}."
        if state == "partial":
            summary += " The scan was stopped early; press the button again to resume it."
        else:
            summary += " Press **Build report** to draw the track on the replay."
        st.caption(summary)
    if previous == "running" and state != "running":
        st.rerun()


def replay_section(
    library: MatchLibrary, match_id: str, segment, segment_dir: Path, video: str, calibration
) -> None:
    """The animated pitch view plus everything known about each player.

    Playback is entirely client-side (the component fetches the replay JSON once), so scrubbing and play/pause cost
    no Streamlit round trips. What *is* Python-side: the annotated player table, the roster editor and the two
    background scans (shirt numbers, ball) - the table, the editor and the number scan share the merge rule
    "manual beats scan", and only the ball scan feeds the *map* itself, through the replay payload.
    """
    replay_path = library.path(match_id) / "replay.json"
    if not replay_path.exists():
        st.info("The animated replay is built together with the report above - press **Build report**.")
        return
    replay = _load_replay_cached(str(replay_path), replay_path.stat().st_mtime_ns)
    team_names = library.load(match_id).team_names
    # The payload on disk carries whatever the names were when the report was built; the record is where renaming
    # lands, so the loaded copy is brought up to date rather than the table disagreeing with the map beside it.
    replay["team_names"] = list(team_names)
    jerseys = library.load_jerseys(match_id)
    roster = library.load_roster(match_id)
    track_ids = [int(player["track_id"]) for player in replay["players"]]
    suggestions = {int(track): entry for track, entry in (jerseys.get("suggestions") or {}).items()}
    numbers = merge_numbers(track_ids, auto=suggestions, manual=roster)
    team_of = {int(player["track_id"]): int(player["team"]) for player in replay["players"]}
    track_stats = {int(player["track_id"]): player["stats"] for player in replay["players"]}

    def player_label(track: int) -> str:
        entry = numbers.get(track, {})
        who = f"#{entry['number']}" if entry.get("number") else f"track {track}"
        if entry.get("name"):
            who += f" {entry['name']}"
        team = team_of.get(track, -1)
        return f"{who} ({team_name(team, team_names)})"

    selected = st.selectbox(
        "Highlight a player in the replay",
        [-1] + sorted(track_ids, key=lambda track: -track_stats[track]["observations"]),
        format_func=lambda track: "no highlight" if track < 0 else player_label(track),
        key=f"replay_selected::{match_id}",
    )
    # The timeline strip above the pitch draws the momentum curve and every event. The events are the same log the
    # table below shows (manual tags and detected candidates), and the momentum is the report's own per-minute
    # buckets; both are small, so they travel as component arguments and update without rebuilding the replay.
    #
    # The three things on the strip keep three different clocks, and all of them are brought onto the strip's own
    # (0-based over the analysed window) here, in one place:
    # * an event's ``time_s`` is a time on the recording it was found in - the combined game for a detected event,
    #   a single camera file for a whistle. The analysed window starts at ``segment.meta["start_s"]`` of *its own*
    #   video, so a candidate from another recording is mapped through that file's own clock, or clipped off the
    #   strip when it does not lie inside the window at all.
    # * the momentum buckets are keyed by *minutes of the analysed window* (Stage B keys on the window's own
    #   seconds), so they are already on the strip's clock.
    # * the half-time mark is a game-clock time, so it is mapped the same way as an event.
    event_log = library.events(match_id)
    report_payload = st.session_state.get("report") or report_from_library(library, match_id) or {}
    game_record_for_video = game_lib.find_for_video(video)
    half_minute = None
    if game_record_for_video is not None and game_record_for_video.bounds() is not None:
        half_minute = game_record_for_video.bounds()[1] / 60.0
    window_start = float(segment.meta.get("start_s", 0.0))
    strip_events: list[dict] = []
    for event in event_log.events:
        # The strip's clock is the analysed window's own (0 at its first frame), so an event's time is translated
        # out of its recording's clock onto the game's and then onto the window's. One helper does the first step,
        # so the strip, the table and the half labels cannot disagree.
        on_game = (
            game_lib.game_time(game_record_for_video, event.time_s, event.video)
            if game_record_for_video is not None and event.video
            else event.time_s
        )
        if on_game is None:
            continue  # found in a recording that is not part of this game
        strip_t = on_game - window_start
        if strip_t < -1.0 or strip_t > float(replay.get("duration_s", 0.0)) + 1.0:
            continue  # a moment from another recording that is not part of this analysed window
        strip_events.append({**event.to_json(), "time_s": round(max(0.0, strip_t), 3)})
    strip_half_minute = None
    if half_minute is not None:
        strip_half_minute = half_minute * 60.0 - window_start
        if strip_half_minute < 0 or strip_half_minute > float(replay.get("duration_s", 0.0)):
            strip_half_minute = None
    # The clip that plays beside the pitch. It is cut on request rather than on selection: cutting costs a decode
    # of the footage, and the animation is worth watching on its own. Once cut it is cached beside the match, so
    # re-selecting the same moment is instant and the file survives a reload.
    clip_url, clip_start_s, clip_end_s = _clip_beside_pitch(
        library,
        match_id,
        float(replay.get("duration_s", 0.0)),
        chosen,
        event_log,
        game_record_for_video,
        window_start,
    )
    replay_view(
        _replay_media_url(replay_path),
        numbers,
        selected,
        team_names,
        key=f"replay::{match_id}",
        events=strip_events,
        momentum=report_payload.get("momentum") or {},
        half_minute=None if strip_half_minute is None else strip_half_minute / 60.0,
        clip_url=clip_url,
        clip_start_s=clip_start_s,
        clip_end_s=clip_end_s,
    )
    has_ball = any(entry is not None for entry in (replay.get("ball") or []))
    excluded = int(replay.get("bystanders_excluded") or 0)
    st.caption(
        "Press play or drag the timeline. Player markers wear each team's measured kit colour; the yellow dot is "
        "the camera aim - "
        + (
            "the ball proxy used when the ball scan has not found the ball. The white football is the ball the "
            "scan tracked: drawn where a detector saw it, a dashed ring where the position is a short forecast "
            "across a missed frame."
            if has_ball
            else "the best ball proxy this footage allows (the gimbal follows the ball; the scan below can track "
            "the ball itself)."
        )
        + " Trails, shirt numbers and a pitch-usage heat map (all players, or just the selected one) are toggled "
        "above the map. Numbers come from the roster below and the automatic scan; tracks without either show their "
        "track id once they last 12 s."
        + (
            f" {excluded} bystander track(s) - touchline coaches, photographers, spectators - are excluded from "
            "the field of play: they never cover ground the way a player does."
            if excluded
            else ""
        )
    )
    only_major = st.checkbox(
        "Show only main tracks (seen for 6 s or more)",
        value=True,
        key=f"replay_major::{match_id}",
        help=(
            "The camera follows the ball, so players leave and re-enter the shot constantly and each visit becomes "
            "its own track (fragments nearby are stitched back together automatically). Short tracks are still "
            "drawn in the replay - untick to see them in the table too."
        ),
    )
    table = player_table_rows(replay, numbers)
    if only_major:
        table = table[table["seen (s)"] >= 6.0]
    st.dataframe(table, hide_index=True, use_container_width=True)

    with st.expander("Shirt numbers and names (per track)"):
        st.caption(
            "Type what you can read off the footage yourself - manual entries win over the automatic scan. "
            "Distances, speeds and the touch proxy are in the table above; a touch means the player was the nearest "
            "to the ball proxy while within 12 m of it."
        )
        editors = []
        for player in replay["players"]:
            track = int(player["track_id"])
            entry = numbers.get(track, {})
            editors.append(
                {
                    "track": track,
                    "team": team_name(player["team"], team_names),
                    "number": entry.get("number"),
                    "name": entry.get("name") or "",
                    "source": entry.get("source") or "unassigned",
                }
            )
        edited = st.data_editor(
            pd.DataFrame(editors),
            column_config={
                "track": st.column_config.NumberColumn("track", disabled=True),
                "team": st.column_config.TextColumn("team", disabled=True),
                "number": st.column_config.NumberColumn("number", min_value=1, max_value=99, step=1),
                "name": st.column_config.TextColumn("name", max_chars=30),
                "source": st.column_config.TextColumn("source", disabled=True),
            },
            hide_index=True,
            use_container_width=True,
            key=f"roster_editor::{match_id}",
        )
        if st.button("Save shirt numbers", key=f"save_roster::{match_id}"):
            new_roster: dict[int, dict] = {}
            for row in edited.to_dict("records"):
                number = row.get("number")
                number = int(number) if number is not None and not pd.isna(number) else None
                name = str(row.get("name") or "").strip()
                if number or name:
                    new_roster[int(row["track"])] = {"number": number, "name": name}
            library.save_roster(match_id, new_roster)
            st.session_state["replay_flash"] = ("success", f"Saved {len(new_roster)} player identit(ies).")
            st.rerun()

    with st.expander("Read shirt numbers from the footage (automatic scan)"):
        _jersey_scan_status(library, match_id, watch_key=f"jersey_watch::{match_id}")

        def _start_scan() -> None:
            command = [
                sys.executable,
                str(REPO_ROOT / "scripts" / "extract_jerseys.py"),
                "--match", match_id,
                "--video", video,
                "--segment", str(segment_dir),
            ]
            subprocess.Popen(command, cwd=str(REPO_ROOT), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            st.session_state["replay_flash"] = ("success", "Shirt-number scan started in the background.")

        st.button(
            "Scan for shirt numbers (background)",
            on_click=_start_scan,
            disabled=segment is None or calibration is None,
            key=f"scan_jerseys::{match_id}",
        )

    with st.expander("Track the ball in the footage (automatic scan)"):
        _ball_scan_status(segment_dir, watch_key=f"ball_watch::{match_id}")

        def _start_ball_scan() -> None:
            # The child takes seconds to boot (interpreter, torch, the segment's meta) and only then writes its
            # first status. Writing the same "Starting..." the scan itself would write closes that window: the
            # button is disabled from the moment of the click instead of briefly offering a second scan that
            # would race the first over the same checkpoint. The child's own first update overwrites this one.
            status_path = Path(segment_dir) / BALL_STATUS_FILE
            command = [
                sys.executable,
                str(REPO_ROOT / "scripts" / "run_ball_scan.py"),
                "--segment",
                str(segment_dir),
            ]
            try:
                status_path.write_text(json.dumps({"state": "running", "message": "Starting...", "updated": time.time()}))
            except OSError:
                pass  # an unwritable segment directory fails the child too, and its error lands in the terminal
            try:
                subprocess.Popen(command, cwd=str(REPO_ROOT), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            except OSError as exc:  # e.g. the interpreter vanished between the page load and the click
                status_path.write_text(json.dumps({"state": "error", "error": str(exc), "updated": time.time()}))
                st.session_state["replay_flash"] = ("warning", f"The scan could not be started: {exc}")
                return
            st.session_state["replay_flash"] = ("success", "Ball scan started in the background.")

        st.button(
            "Scan for the ball (background)",
            on_click=_start_ball_scan,
            disabled=segment is None or _ball_scan_alive(segment_dir),
            key=f"scan_ball::{match_id}",
        )
    show_flash("replay_flash")


def show_flash(key: str) -> None:
    """Show a message that a save-then-rerun action left behind.

    ``st.rerun()`` throws away the elements of the run it was called from, so a ``st.success`` printed immediately
    before it is never seen at all. Actions that save and then reload stash ``(level, text)`` under ``key`` and call
    this as the first thing on the next run.
    """
    flash = st.session_state.pop(key, None)
    if not flash:
        return
    level, message = flash
    if level == "warning":
        st.warning(message)
    else:
        st.success(message)


# --------------------------------------------------------------------------------------------------------------
# Page
# --------------------------------------------------------------------------------------------------------------
st.title("Match analysis")
st.caption(
    "Single gimbal camera: the camera's own motion is recovered first, so player positions can be placed on the pitch "
    "even though the camera pans and zooms to follow the ball."
)

library = MatchLibrary(MATCHES_ROOT)

# The preview detail is a global preference, not a per-match one: how much picture a review clip is worth is about
# the reviewer's patience and the machine's speed, which do not change from match to match. It lives in the sidebar
# so it is reachable from anywhere on the page.
with st.sidebar:
    st.subheader("Options")
    preview_mode = st.radio(
        "Preview detail",
        list(PREVIEW_SETTINGS),
        index=list(PREVIEW_SETTINGS).index("full"),
        format_func=lambda name: PREVIEW_MODE_LABELS.get(name, name),
        key="preview_mode",
        help=(
            "What each preview costs to cut, cheapest first: the sound alone is near instant (~0.2 s), the light "
            "video shows one picture a second for a few seconds' wait, and the full-size one takes about as long "
            "as the clip lasts. The verdict buttons work the same for all three."
        ),
    )

# Everything the page keeps in the session belongs to one match: the calibration just fitted, the report just
# built, the landmarks behind the residual table. Selecting another archive has to drop all of it - otherwise the
# previous match's numbers are quietly shown for the new one, which is what makes an archive look like it loaded
# when it did not.
MATCH_SCOPED_STATE = (
    "calibration",
    "report",
    "calib_points",
    "calib_landmarks",
    "calib_next_pid",
    "calib_centre",
    "calib_zoom",
    "calib_reference",
    "calib_clicks_key",
    "calib_dropped_note",
    "calib_fit_error",
    "events_flash",
    "step3_flash",
)

st.header("Match archive")
match_ids = library.list_ids()

with st.container(border=True):
    session_col, footage_col = st.columns([1, 2])
    with session_col:
        # A match saved in the last run asks to be selected. A widget's state can only be set before that widget
        # exists in the run, and the save button runs after this module has been drawn, so it leaves the id here for
        # the next run to pick up.
        pending = st.session_state.pop("archive_select_pending", None)
        if pending in match_ids:
            st.session_state["archive_selection"] = pending
        selection = st.selectbox("Archive", match_ids + ["(new match)"], index=0, key="archive_selection")
        match_id = None if selection == "(new match)" else selection
        if st.session_state.get("archive_scope") != match_id:
            st.session_state["archive_scope"] = match_id
            for scoped in MATCH_SCOPED_STATE:
                st.session_state.pop(scoped, None)
        st.caption(
            f"`{(MATCHES_ROOT / match_id).relative_to(REPO_ROOT)}`"
            if match_id
            else "No archive selected - saving the footage starts one."
        )

    with footage_col:
        video_options = discover_videos()
        if not video_options:
            searched = ", ".join(str(root) for root in video_roots())
            st.error(f"No video files found under {searched}.")
            st.stop()

        # Selecting an archive has to bring back the footage it was recorded from, or it cannot be reopened: the
        # picker would stay on whichever video happened to be newest and Step 2 would find no segment belonging to
        # the match. The widget is keyed per match rather than mutated, so each archive remembers its own choice and
        # a first visit opens on the footage the match was made from.
        default_video = 0
        if match_id is not None:
            recorded = library.load(match_id).sources
            reachable = [source for source in recorded if Path(source).exists()]
            if reachable and reachable[0] not in {str(path) for path in video_options}:
                video_options = [Path(reachable[0]), *video_options]
            if reachable:
                default_video = [str(path) for path in video_options].index(reachable[0])
            elif recorded:
                st.warning("The footage this match was recorded from is not reachable: " + ", ".join(recorded))

        video_col, probe_col = st.columns([2, 1])
        with video_col:
            chosen = st.selectbox(
                "Video (newest first)",
                [str(p) for p in video_options],
                index=default_video,
                format_func=lambda p: Path(p).name,
                key=f"source_video::{match_id}",
            )
            manual = st.text_input(
                "...or paste an absolute path", placeholder="/path/to/match.MP4", key=f"manual_video::{match_id}"
            )
            if manual.strip():
                chosen = manual.strip()
        with probe_col:
            if not Path(chosen).exists():
                st.error("File not found.")
                st.stop()
            probe = probe_cached(chosen)
            st.metric("Resolution", f"{probe.width}x{probe.height}")
            st.metric("Source FPS", f"{probe.fps:.0f}")
            st.metric("Duration", f"{probe.duration_s / 60:.1f} min")

# Saving the session: the archive entry everything below is stored against, made from the footage above.
if match_id is None:
    def _create_match() -> None:
        """Runs before the next run's body, so this module picks the new match up in that same run."""
        existed = new_match_id(chosen) in match_ids
        record = library.create(chosen)
        st.session_state["archive_select_pending"] = record.match_id
        st.session_state["session_flash"] = (
            "success",
            f"`{record.match_id}` is already archived for {Path(chosen).name} - opened it rather than saving a "
            f"second record for the same footage."
            if existed
            else f"Saved `{record.match_id}` for {Path(chosen).name}.",
        )

    st.button("Save as a new match", type="primary", on_click=_create_match)
show_flash("session_flash")

# --------------------------------------------------------------------------------------------------------------
# Step 1 - the heavy pass
# --------------------------------------------------------------------------------------------------------------
st.header("Step 1 - Run the heavy pass")
show_flash("step1_flash")

# --- one game out of several clips ----------------------------------------------------------------------------
# The camera writes ~30-minute files, so a game arrives as two or three of them. Combining them (a stream copy, no
# re-encode) gives one video with one clock, which is what everything downstream needs: one camera-motion chain, one
# set of player tracks, and one timeline to hang kick-off, half-time and full-time on.
game_record = game_lib.find_for_video(chosen)
game_window_selection: str | None = None

with st.expander("Combine clips into one game video", expanded=game_record is None):
    st.caption(
        "A game usually arrives as two or three camera files. Combining them is a plain stream copy - nothing is "
        "re-encoded - and gives one continuous video. The combined file is written next to the first clip, because "
        "it is as large as they are. If you have already merged them yourself, pick that one file on its own and "
        "it is used as it stands: nothing is copied or re-encoded."
    )
    picked_clips = st.multiselect(
        "Clips (any order - the camera's own timestamps put them in playing order)",
        [str(path) for path in video_options],
        format_func=lambda path: Path(path).name,
        key=f"game_clips::{match_id}",
    )
    if not picked_clips:
        st.info("Pick at least one clip.")
    else:
        ordered_clips, expected_output, expected_dir = game_lib.locations(picked_clips)
        planned_game = game_lib.plan(ordered_clips)
        total_minutes = sum(clip.duration_s for clip in planned_game.clips) / 60.0
        if len(planned_game.clips) == 1:
            only = Path(planned_game.clips[0].path)
            st.write(f"Already one game video: {only.name} - {total_minutes:.0f} min")
            st.caption(
                "Used as it stands. The game metadata, the marking proxy and the half-time marks all live in "
                f"`{expected_dir.name}` beside it, not in the video itself, so the file is never written to."
            )
        else:
            st.write(
                "Playing order: "
                + " -> ".join(Path(clip.path).name for clip in planned_game.clips)
                + f" - {total_minutes:.0f} min"
            )
            st.caption(f"Will write `{expected_output}`")
        if planned_game.problem:
            st.error(
                f"These clips cannot be combined: {planned_game.problem}. A combination without a re-encode needs "
                "the same codec and frame size, and re-encoding tens of gigabytes is not something to start "
                "silently."
            )
        else:

            def _build_game() -> None:
                command = [sys.executable, str(REPO_ROOT / "scripts" / "run_build_game.py")]
                for clip in ordered_clips:
                    command += ["--clip", str(clip)]
                command += ["--out", str(expected_dir)]
                subprocess.Popen(command, cwd=str(REPO_ROOT), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                st.session_state["step1_flash"] = (
                    "success",
                    "Combining the clips in the background, then building the marking proxy from keyframes (a couple "
                    "of minutes for a full game). Refresh in a moment.",
                )

            build_state = game_lib.read_build_state(expected_dir)
            running = build_state.get("state") == "running"
            label = "Use this game video" if len(planned_game.clips) == 1 else "Combine into one game video"
            st.button(
                label,
                disabled=running,
                on_click=_build_game,
                key=f"build_game::{match_id}",
            )
            if running:
                elapsed = int(max(0.0, time.time() - float(build_state.get("started", time.time()))))
                st.info(
                    f"Working in the background: {build_state.get('stage', 'combining')} ({elapsed}s so far). "
                    "Refresh in a few minutes."
                )
            elif build_state.get("state") == "error":
                st.error(f"The build failed: {build_state.get('error')}")
            elif expected_output.exists():
                if len(planned_game.clips) == 1:
                    st.success(
                        f"Ready: `{expected_output.name}`. Pick it in the video list above, then mark the game "
                        "clock below."
                    )
                else:
                    st.success(
                        f"Combined: `{expected_output.name}`. Pick it in the video list above, then mark the game "
                        "clock below."
                    )

if game_record is not None:
    game_directory = game_lib.game_dir(game_lib.GAMES_ROOT, game_record.game_id)
    st.markdown(
        f"**Game video** `{Path(game_record.output).name}` - {len(game_record.clips)} clip(s) joined, "
        f"{game_record.duration_s / 60:.0f} min"
    )
    mark_problem = game_record.mark_problem()
    if mark_problem and not mark_problem.startswith("still to mark"):
        st.warning(f"The marks cannot be used yet: {mark_problem}.")

    game_proxy = game_lib.proxy_path(game_directory)
    game_build_state = game_lib.read_build_state(game_directory)
    if not game_proxy.exists():
        if game_build_state.get("state") == "running":
            st.info(
                "Building the low-resolution marking proxy. It is built from keyframes alone - one picture per "
                "second - so a whole game takes a couple of minutes rather than the length of the footage."
            )
        elif game_build_state.get("state") == "error":
            st.error(f"The proxy build failed: {game_build_state.get('error')}")
        else:
            st.info("The marking proxy has not been built yet; start the build in the section above.")
    else:
        game_proxy_url = _served_video_url(game_proxy, coordinates=f"game::{game_record.game_id}")
        if game_proxy_url is None:
            st.warning("Could not serve the marking proxy to the browser, so the marks cannot be set here.")
        else:
            mark_result = GAME_TIMELINE_COMPONENT(
                proxy_url=game_proxy_url,
                marks={"start": game_record.start_s, "half": game_record.half_s, "end": game_record.end_s},
                key=f"game_marks::{game_record.game_id}",
                default=None,
            )
            marks_seen = st.session_state.setdefault("game_marks_seen", {})
            if isinstance(mark_result, dict) and str(mark_result.get("action")) == "mark":
                # The component's value is sticky, so it comes back on every later run: only a sequence number
                # newer than the last one handled is a new press. That is why the key itself never changes - a
                # remount would throw the playback position away in the middle of marking.
                sequence = int(mark_result.get("seq") or 0)
                if sequence > int(marks_seen.get(game_record.game_id, 0)):
                    which = str(mark_result.get("mark") or "")
                    if which in game_lib.MARKS:
                        at = float(mark_result.get("time") or 0.0)
                        game_record.set_mark(which, at)
                        game_record.save(game_directory)
                        marks_seen[game_record.game_id] = sequence
                        # The marks define the analysed window, so apply it now rather than waiting for the radio
                        # to be touched: marking kick-off and full-time is exactly how the window is chosen.
                        if game_record.bounds() is not None:
                            selection = str(
                                st.session_state.get(
                                    f"game_window::{game_record.game_id}", game_lib.WINDOW_WHOLE
                                )
                            )
                            window_start, window_end = game_record.window(selection)
                            st.session_state[f"start_s::{chosen}"] = float(window_start)
                            st.session_state[f"length_s::{chosen}"] = float(window_end - window_start)
                        st.session_state["step1_flash"] = ("success", f"Marked {which} at {_clock(at)}.")
                        st.rerun()

    st.write(
        "  ".join(
            f"**{name}** {_clock(value) if value is not None else '-'}"
            for name, value in (
                ("kick-off", game_record.start_s),
                ("half-time", game_record.half_s),
                ("full-time", game_record.end_s),
            )
        )
    )
    if game_record.bounds() is not None:
        window_key = f"game_window::{game_record.game_id}"

        def _apply_game_window() -> None:
            selection = str(st.session_state.get(window_key))
            window_start, window_end = game_record.window(selection)
            st.session_state[f"start_s::{chosen}"] = float(window_start)
            st.session_state[f"length_s::{chosen}"] = float(window_end - window_start)
            st.session_state["step1_flash"] = (
                "success",
                f"Analysing {selection.lower()}: {_clock(window_start)} to {_clock(window_end)}.",
            )

        game_window_selection = st.radio(
            "Restrict the analysis to",
            list(game_lib.WINDOW_CHOICES),
            index=0,
            horizontal=True,
            key=window_key,
            on_change=_apply_game_window,
            help=(
                "Kick-off to full-time drops the warm-up and the walk-off; one half on its own is what a per-half "
                "report needs. Each window gets its own analysis directory, so switching back and forth is cheap."
            ),
        )
        if st.button("Clear the marks", key=f"clear_marks::{game_record.game_id}"):
            game_record.clear_marks()
            game_record.save(game_directory)
            st.rerun()

span_col, length_col = st.columns(2)
# A marked game starts on kick-off rather than at the top of the recording - that is the point of marking it - and
# the window radio above overwrites these two when another window is picked.
game_marks = game_record.bounds() if game_record is not None else None
with span_col:
    # Keyed by video: the segment belongs to the footage, so one match's analysis window must not become the next
    # one's. A leftover offset would silently analyse the wrong part of a different match.
    start_s = st.number_input(
        "Start offset (s)",
        min_value=0.0,
        value=0.0 if game_marks is None else float(game_marks[0]),
        step=30.0,
        key=f"start_s::{chosen}",
    )
with length_col:
    duration_s = st.number_input(
        "Length to analyse (s)",
        min_value=0.0,
        value=0.0 if game_marks is None else float(game_marks[2] - game_marks[0]),
        step=30.0,
        key=f"length_s::{chosen}",
        help="0 analyses from the offset to the end of the video; a marked game defaults to kick-off to full-time.",
    )

window_start, window_end = resolve_window(start_s, duration_s, probe.duration_s)
st.caption(
    f"Analysing **{_clock(window_start)} to {_clock(window_end)}** "
    f"({window_end - window_start:.0f} s"
    + (", the entire video from the offset" if duration_s <= 0 else "")
    + ")."
)
if game_marks is not None:
    st.caption(
        f"Game clock: kick-off {_clock(game_marks[0])}, half-time {_clock(game_marks[1])}, full-time "
        f"{_clock(game_marks[2])}. Events and the report are labelled by half."
    )

# Each window of a marked game gets its own segment directory: the stored meta refuses a different window in the
# same directory (correctly), and a separate directory is what lets the two halves be analysed without redoing
# either - and resumed independently.
segment_dir = segment_dir_for(
    chosen,
    SEGMENTS_ROOT,
    window_label=(
        game_record.window_label(game_window_selection)
        if game_record is not None and game_window_selection
        else None
    ),
)
config = SegmentConfig()
status = read_status(segment_dir)

existing_meta: dict | None = None
meta_file = segment_dir / "meta.json"
if meta_file.exists():
    try:
        existing_meta = json.loads(meta_file.read_text())
    except (OSError, json.JSONDecodeError):
        existing_meta = None


def _same_window(meta: dict) -> bool:
    """Whether the results already in the output directory cover exactly the requested window."""
    return (
        abs(float(meta.get("start_s", -1.0)) - window_start) < 1e-6
        and abs(float(meta.get("end_s", -1.0)) - window_end) < 1e-6
    )


window_conflict = existing_meta is not None and not _same_window(existing_meta)

info_col, run_col = st.columns([2, 1])
with info_col:
    if window_conflict:
        st.warning(
            f"This video already has results for **{_clock(float(existing_meta['start_s']))} to "
            f"{_clock(float(existing_meta['end_s']))}**. There is one output directory per video, and a different "
            "window cannot be resumed into the same one - running again will move the existing results aside "
            "(renamed, not deleted)."
        )
    _stage_a_status(segment_dir, watch_key=f"stage_a_watch::{chosen}")
with run_col:
    def _run_analysis() -> None:
        """Spawn the pass from the callback, which runs before the page is drawn again - one run, not two."""
        archived_note = ""
        if existing_meta is not None and not _same_window(existing_meta):
            # `analyse_segment` refuses to mix windows in one output directory, so the old results move aside under
            # a timestamped name. Renaming (rather than deleting) means putting the earlier analysis back is a
            # rename away, which matters because re-analysis of the same footage is what created the conflict.
            archived = segment_dir.with_name(f"{segment_dir.name}_superseded_{time.strftime('%Y%m%d_%H%M%S')}")
            segment_dir.rename(archived)
            archived_note = f" The previous results are at `data/segments/{archived.name}`."
        command = [
            sys.executable,
            str(REPO_ROOT / "scripts" / "run_stage_a.py"),
            "--video", chosen,
            "--out", str(segment_dir),
            "--start", str(start_s),
            "--duration", str(duration_s),
        ]
        if match_id is not None:
            # The archive's own record of what has been produced: `summaries` reports it, and it is what tells a
            # later visit that this match already has segment results on disk.
            library.add_segment(match_id, segment_dir)
        subprocess.Popen(command, cwd=str(REPO_ROOT), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        st.session_state["step1_flash"] = (
            "success",
            "Analysis started in the background - progress updates here automatically." + archived_note,
        )

    st.button("Run analysis (background)", type="primary", on_click=_run_analysis)

# --------------------------------------------------------------------------------------------------------------
# Step 2 - pitch registration
# --------------------------------------------------------------------------------------------------------------
st.header("Step 2 - Register the pitch")
st.caption(
    "Click pitch landmarks on a frame. Four are enough; the further apart they are, the better the fit. The table "
    "below reports the error for each click, because landmarks on the far side of the pitch carry metres of "
    "uncertainty."
)

segment = None
chunks_done = stage_a.completed_chunks(segment_dir)
if chunks_done == 0:
    st.info("Run Step 1 first: the camera motion from that pass is what lets landmarks from any frame be used.")
else:
    try:
        # Cached across reruns: this is the gesture loop, and an uncached load + pose integration is the difference
        # between a click responding immediately and a click taking most of a minute on a whole-game segment.
        segment, q, focal = segment_and_poses_cached(
            str(segment_dir), segment_fingerprint(segment_dir, chunks_done)
        )
    except Exception as exc:  # a partially written segment must not break the page
        st.warning(f"Could not read the segment results: {exc}")

if segment is not None:
    # Say where the camera motion comes from: the gimbal log is a hardware measurement and does not drift, so a
    # calibration built on it holds across the whole game; the estimated chain drifts and needs the clicks to
    # re-anchor it. The user should know which one they are looking at.
    if segment_has_log(segment):
        st.caption(
            "Camera motion is taken from the gimbal's own log (yaw/pitch telemetry beside the video), so it does "
            "not drift over the game - landmarks clicked late in the match stay consistent with early ones."
        )
    else:
        st.caption(
            "Camera motion is estimated from the picture; it drifts slowly, so clicks spread across the match "
            "re-anchor it. A gimbal log beside the video would remove that drift."
        )
    # The pitch size lives on the match record, not in a widget: the calibration was clicked against those
    # dimensions, so reading them back from the record is what stops the two drifting apart across reloads.
    record = library.load(match_id)
    stored_format = record.format if record.format in MATCH_FORMATS else list(MATCH_FORMATS)[-2]
    format_col, _ = st.columns([1, 3])
    with format_col:
        format_name = st.selectbox(
            "Match format", list(MATCH_FORMATS), index=list(MATCH_FORMATS).index(stored_format)
        )
    length_m, width_m = MATCH_FORMATS[format_name]
    if (record.format, record.pitch_length_m, record.pitch_width_m) != (format_name, length_m, width_m):
        had_calibration = library.load_calibration(match_id) is not None
        record.format, record.pitch_length_m, record.pitch_width_m = format_name, length_m, width_m
        library.save(record)
        if had_calibration:
            st.warning(
                "The pitch size changed after this match was calibrated, so the saved calibration no longer matches. "
                "Click the landmarks again before relying on the report."
            )

    # q/focal come from the same cache as the segment: rebuilding the chain costs ~1 s on the whole-game segment
    # even with the fast path, and this line is on the gesture loop's hot path.
    frame_count = len(segment.time)
    land_table = landmark_table(length_m, width_m)
    names = list(land_table)

    # Clicks belong to one match, one video and one pitch size. Switching any of them must not quietly blend the
    # two, so the clicks, the view position and any in-session calibration are dropped when the combination changes.
    # The clicks saved with the last calibration are read back instead of starting empty: a page refresh starts a
    # new session, and the whole point of the saved set is to outlive the browser tab. Frames are indices into the
    # segment they were clicked on, so a set from another segment is refused (restored_points) rather than
    # renumbered into nonsense.
    clicks_key = f"{match_id}|{chosen}|{format_name}"
    if st.session_state.get("calib_clicks_key") != clicks_key:
        saved_clicks = library.load_clicks(match_id)
        restored = restored_points(saved_clicks, frame_count) if saved_clicks else None
        st.session_state["calib_clicks_key"] = clicks_key
        st.session_state["calib_points"] = restored or []
        st.session_state["calib_landmarks"] = saved_clicks if restored else []
        st.session_state["calib_next_pid"] = len(restored or [])
        st.session_state["calib_restored_count"] = len(restored or [])
        st.session_state["calib_seeded"] = []
        st.session_state["calib_centre"] = (0.5, 0.5)
        st.session_state["calib_zoom"] = DEFAULT_ZOOM
        st.session_state["calib_reference"] = int(restored[0]["frame"]) if restored else 0
        st.session_state.pop("calibration", None)

    # The clicking workspace stays on the page: it is the thing you are working in while a calibration is being
    # made, and hiding it behind a toggle would put a click between every aim and every landmark. Only the saved
    # clicks list is collapsed - it is long once a match is calibrated and is only needed to check or re-label a
    # click.
    with st.expander("What are pitch landmarks, and how do I click them?", expanded=False):
        st.markdown(
            """
The camera moves, so the app has to work out *where the camera was* before it can say where a player is. It does
that from **landmarks**: fixed markings on the pitch whose real position you already know, because the laws of the
game put them there. Halfway line, centre spot, corner flags, the goalposts, the penalty spots, the centre
circle - those are the only things in the frame with a known real-world position, so the calibration is only ever
as good as these clicks.

**What to click**

| Landmark | Click exactly on |
| --- | --- |
"""
        + "\n".join(f"| `{name}` | {LANDMARK_HELP[name]} |" for name in land_table)
        + """

**How to do it**

1. Pick a frame where the pitch markings are visible - any frame works, because the camera's motion is already
   known. Use the timeline under the whole-frame view to scrub to one, or press **Play** and pause where the
   markings are clearest.
2. **Aim the magnified view from the whole-frame view.** Click a point on the whole frame and the magnified view
   moves so that point is in the middle, reloading at whatever frame the timeline is showing; **scroll** to zoom in
   or out on it. The yellow box shows what the magnified view covers, so it doubles as a check that you are looking
   at the right part of the pitch.
3. **Click the landmark on the magnified view.** A click opens the landmark list right where you clicked - pick
   which landmark it is and the click is committed at once; there is no Apply step. Zoom in far enough that
   the corner is unmistakable: at zoom 8 a screen pixel is a pixel of the 4K frame, whereas a full-frame view is
   worth roughly a metre of ground error per pixel on the far side of the pitch.
4. **If a corner is not in shot, use the goalposts instead.** The near corner flags are often out of frame or lost
   against the grass, but the posts are easy to pick out, and the base of a post - where it meets the goal line -
   says as much about that end of the pitch as the flag does. Click **goal post left-near** / **goal post
   left-far** (or the `right` pair) in place of a corner you cannot see. The posts are 7.32 m apart, so the two of
   them together also fix the goal line's direction.
5. **The penalty spots and centre circle are usually visible too.** The penalty spots and the four points where
   the centre circle crosses the halfway and centre lines are standard markings at a range of distances from the
   camera - exactly the spread the fit is short of when the near corners are out of shot. Click the ones you can
   see clearly. The boxes are deliberately not in the list: the six-yard box is small and lost against the
   netting, and an eighteen-yard corner is a bare junction of two lines with nothing to focus on - the posts carry
   that end of the pitch instead, and the penalty spots pin the box's depth.
6. **What you cannot substitute is spread.** Four clicks that are all a long way off, or all along one line, leave
   the fit with tens of metres of doubt - measured on a simulated match, four distant landmarks were out by more
   than 50 m. Aim for landmarks at a *range* of distances from the camera: the halfway line where it meets the near
   touchline is a good near one, because a long white line is easy to pick out. Click as many as you can be sure
   of. **Six to eight spread across the frame is comfortable; four is the absolute minimum.**
7. Repeat as needed. The app discards a bad click for you, and the fit check projects the corners you never
   clicked, so you can see where it thinks they are.

Once a calibration is saved, the whole-frame view carries it too: the pitch markings and every landmark, projected
through the fit onto whichever frame is loaded. To re-anchor a later moment, press **Place the calibrated landmarks
as markers** - each landmark the calibration used lands in the magnified view where the fit expects it, and you
**drag** it onto the real marking. A drag is a measurement, so it says how far the fit has slid by that moment; it
is also far quicker than finding five markings from scratch.

**The one thing that must be consistent:** label the corners in a loop around the pitch - one touchline, then the
other (this is the order in the dropdowns: `near-left`, `near-right`, then the two `far` ones). Get a consistent
loop and the app can work out the rest.

**Then check the yellow markings** the app draws back onto the frame. They are the whole pitch - touchlines, halfway
line, both boxes, the centre circle, the penalty arcs and spots and the corner arcs - projected through your
calibration. If they land on the real markings, the registration is good. If they are mirrored, or clearly in the
wrong place, fix the labels and recalibrate - the per-landmark error table tells you which click to look at.

Be honest with yourself about the far corners: at 40-90 m, four pixels of click noise is worth several metres of
ground error in the depth direction. If the far corner is a blurred smudge, leave it out and use the landmarks you
are sure of.
"""
        )

    # The whole-frame viewport is also the timeline. It needs the proxy; while that builds (a one-time pass over
    # the segment) the viewport keeps a still frame and a plain slider picks the frame instead.
    proxy_ready = timeline_ready(segment, segment_dir)
    if not proxy_ready:
        _timeline_status(segment_dir, watch_key=f"timeline_watch::{segment_dir}")

    # A gesture arrives as the component's own widget value, and Streamlit hands widget values to the run *before* it
    # starts - so it can be read here and settled into session state before the crop is computed. That is what keeps
    # a gesture to a single run: acting on it after the component costs a second full run (another frame read from
    # the 4K source, and the whole page rendered again) because the crop is built above it.
    #
    # The value is sticky - it comes back on every later rerun - and clicks now commit the moment they happen, so
    # there is no Apply button to re-key the component after. Instead every reported gesture carries a sequence
    # number (plus the mount nonce, so a re-keyed component restarting from seq 1 is never mistaken for an old
    # gesture), and a gesture is acted on only when it has not been seen before.
    jump_nonce = int(st.session_state.get("calib_jump_nonce", 0))
    component_key = f"calib::{chosen}::{jump_nonce}"
    gesture = parse_result(st.session_state.get(component_key))
    gesture_key = f"calib_gesture::{chosen}"
    fresh = (gesture.mount, gesture.seq) != st.session_state.get(gesture_key, (-1, -1))
    if fresh:
        st.session_state[gesture_key] = (gesture.mount, gesture.seq)
    if fresh and gesture.action == "navigate":
        # Landing the view where the gesture asked is idempotent, so the value being sticky costs nothing: on a
        # later run it asks for the centre and zoom that are already stored.
        if gesture.centre is not None:
            st.session_state["calib_centre"] = gesture.centre
        if gesture.zoom is not None:
            st.session_state["calib_zoom"] = gesture.zoom
        moved_frame = frame_change(gesture.frame, st.session_state.get("calib_reference", 0), frame_count)
        if moved_frame is not None:
            st.session_state["calib_reference"] = moved_frame

    reference = int(np.clip(st.session_state.get("calib_reference", 0), 0, max(0, frame_count - 1)))
    if proxy_ready:
        st.write(
            "**Scrub the timeline to a frame where the markings are clearest, then click the frame to load it.** "
            "Scrubbing runs in the browser, so the magnified view is only re-read when you click a point - that is "
            "what keeps the drag smooth. Any frame works, because the camera's motion is already known."
        )
    else:
        reference = st.slider(
            "Frame to click on",
            0,
            max(0, frame_count - 1),
            reference,
            help="Any frame works, because the camera motion is known.",
        )

    # With a calibration saved, the whole-frame viewport can carry it: the pitch and every landmark projected
    # through the *corrected* chain onto the frame's own still. It is the same picture as the fit check further
    # down, in the place the user is actually working - which is what shows how far the fit has slid by this moment
    # without leaving the frame being clicked on. Drawn only into the still, so it appears exactly when the
    # viewport is at rest on the crop's frame, the same rule the markers follow.
    calibration_for_view = saved_calibration(library, match_id)
    overlay_on = False
    if calibration_for_view is not None:
        overlay_on = st.checkbox(
            "Show the calibration on the whole frame",
            value=True,
            key="calib_overlay_on",
            help="Projects the pitch outline and every landmark through the saved calibration onto the whole-frame "
                 "view, at whichever frame is loaded.",
        )

    full = frame_cached(chosen, float(segment.time[reference]), int(probe.width))
    if full is None:
        st.error("Could not read that frame from the video.")
    else:
        height, width = full.shape[:2]
        centre = st.session_state.get("calib_centre", (0.5, 0.5))
        zoom = float(st.session_state.get("calib_zoom", DEFAULT_ZOOM))
        box = zoom_box(width, height, zoom, centre[0], centre[1])
        x0, y0, crop_w, crop_h = box
        crop = full[y0 : y0 + crop_h, x0 : x0 + crop_w]

        # The whole frame is scrubbed as a video inside the component when the proxy exists, so scrubbing stays
        # local to the browser. The still is sent either way: the proxy is built from the camera's keyframes only
        # (one picture per second), so the component puts this *exact* frame over the coarse proxy picture whenever
        # the viewport is at rest on it - which is what keeps the markers on the corner they were clicked on.
        overview = cv2.resize(full, (1280, int(1280 * height / width)), interpolation=cv2.INTER_AREA)
        if overlay_on and calibration_for_view is not None:
            view_q, view_focal = calibration_for_view.corrected_frame(q[reference], float(focal[reference]), reference)
            overview = pitch_overlay(
                overview, calibration_for_view, view_q, view_focal, length_m, width_m, table=land_table
            )
        # The browser-side overlay: sampled homographies along the corrected chain plus the markings in pitch
        # metres, so the component can draw the pitch on whatever frame the timeline is showing - scrubbed or
        # playing - without a round trip per frame. Recomputed from the current calibration on every run, so a
        # refit redraws the overlay immediately.
        browser_overlay = (
            overlay_homographies(calibration_for_view, q, focal, length_m, width_m, frame_count)
            if overlay_on and calibration_for_view is not None
            else []
        )
        browser_polylines = (
            [line.tolist() for line in pitch_marking_polylines(length_m, width_m)] if browser_overlay else []
        )
        proxy_url = (
            _timeline_media_url(timeline.proxy_path(segment_dir), coordinates=f"timeline::{segment_dir}")
            if proxy_ready
            else ""
        )

        stored = st.session_state.setdefault("calib_points", [])
        next_pid = st.session_state.get("calib_next_pid", 0)
        scale = canvas_scale(crop_w)

        if fresh and gesture.action == "apply":
            # A click or a finished drag commits the moment it happens, so the crop just computed is the one the
            # click was made on: this is exactly the geometry that was on screen. The frame the gesture carried is
            # ignored on purpose - the points belong to the crop that is already loaded.
            if gesture.points:
                stored, next_pid = merge_clicked(stored, gesture.points, reference, box, scale, width, next_pid)
                st.session_state["calib_points"] = stored
                st.session_state["calib_next_pid"] = next_pid
            # Whatever was on screen has become real clicks; any placed markers it came from are no longer pending.
            st.session_state["calib_seeded"] = [
                seed for seed in st.session_state.get("calib_seeded", []) if seed["frame"] != reference
            ]
        elif fresh and gesture.action == "clear":
            # Clear removes the applied points in the frame of reference - the frame the crop is on - and leaves
            # the clicks of every other frame alone. The component has already emptied its own copy; this is where
            # the stored clicks follow.
            stored = [point for point in stored if point["frame"] != reference]
            st.session_state["calib_points"] = stored
            st.session_state["calib_seeded"] = [
                seed for seed in st.session_state.get("calib_seeded", []) if seed["frame"] != reference
            ]

        # A stray double-click of one landmark on one frame is resolved the way the user's last gesture meant it:
        # the newest click wins and the older one is dropped right here, before the viewport is drawn - the marker
        # that lost never even appears, and a slip never blocks the fit. The match is on the *stored* tag, which a
        # popover pick and a dropdown edit both write; an untagged click merely displays the list's fallback name,
        # and the render body cannot tell that from a deliberate choice, so untagged clicks are left alone. Other
        # frames are untouched - re-clicking a landmark later in the video is the drift-anchoring workflow.
        _kept, dropped = split_duplicate_clicks(
            (point["frame"], point["label"], point["pid"]) for point in stored if point.get("label")
        )
        if dropped:
            drop_pids = {pid for _frame, _landmark, pid in dropped}
            stored = [point for point in stored if point["pid"] not in drop_pids]
            st.session_state["calib_points"] = stored
            # One-shot: the note is popped where the calibrate button is, so it is read once and not repeated on
            # every later rerun of the same state.
            st.session_state["calib_dropped_note"] = (
                "Dropped the older duplicate click(s) automatically: "
                + ", ".join(f"`{landmark}` on frame {frame}" for frame, landmark, _pid in dropped)
                + " - a landmark counts once per frame, so clicking it again replaces the older click."
            )

        # The re-anchoring shortcut. With a calibration saved, each landmark it used can be *placed* at where the
        # fit projects it - through the corrected chain, so it already carries the drift measured so far - and
        # dragged onto the real marking. The drag is the measurement (it says how far the fit has slid by this
        # moment) and the label arrives attached, so a later moment gets anchored without hunting five markings
        # from scratch. The run happens before the features are built, so the markers appear in this same render.
        if calibration_for_view is not None:
            used_labels = {
                click.get("label")
                for click in (st.session_state.get("calib_landmarks") or library.load_clicks(match_id))
                if click.get("label")
            }
            placeable = [name for name in land_table if name in used_labels]
            if placeable and st.button(
                "Place the calibrated landmarks as markers",
                help="Puts each landmark from the saved calibration into the magnified view at the position the fit "
                     "projects it to. Drag each marker onto the real marking - the drag commits the moment you let "
                     "go, and it is what tells the fit how far it has slid by this frame.",
            ):
                seed_q, seed_focal = calibration_for_view.corrected_frame(q[reference], float(focal[reference]), reference)
                placed = [
                    spot
                    for spot in projected_landmarks(calibration_for_view, placeable, land_table, seed_q, seed_focal)
                    if inside_crop(spot["u"], spot["v"], box, scale, width)
                ]
                if placed:
                    st.session_state["calib_seeded"] = [
                        seed for seed in st.session_state.get("calib_seeded", []) if seed["frame"] != reference
                    ] + [{"frame": reference, **spot} for spot in placed]
                else:
                    st.warning(
                        "None of the landmarks from the saved calibration fall inside the magnified view - zoom "
                        "out or aim at one of them (click it on the whole frame), then press again."
                    )

        seeds_here = [seed for seed in st.session_state.get("calib_seeded", []) if seed["frame"] == reference]
        # Show this frame's existing clicks that fall inside the current crop, so they can be added to - plus any
        # placed markers, which wait here (in frame coordinates, so a zoom keeps them on their landmark) until a
        # click or a drag commits them.
        features = points_in_crop(stored, reference, box, scale, width)
        features += [marker_feature(seed["label"], seed["u"], seed["v"], box, scale, width) for seed in seeds_here]

        st.caption(
            "The magnified view is for clicking landmarks; the whole frame beside it is for aiming **and for "
            "scrubbing**. **Scrub** to a frame where the markings are clear, then **click** a point on the whole "
            "frame to load that frame and bring the point into the middle of the magnified view; **scroll** there to "
            "zoom in or out, or use the +/- buttons. **Click a landmark on the magnified view** and pick which one it "
            "is in the list that appears at the click - the click is committed the moment you pick. **Clear** removes "
            "the clicks of the frame you are looking at, and leaves the rest alone."
        )
        landmark_clicker(
            crop,
            overview,
            features,
            frame_points(stored, reference) + [(seed["u"], seed["v"]) for seed in seeds_here],
            centre,
            zoom,
            box,
            (width, height),
            key=component_key,
            proxy_url=proxy_url,
            frame_count=frame_count,
            initial_frame=reference,
            marker_kinds=names,
            mount_nonce=jump_nonce,
            overlay_homographies=browser_overlay,
            overlay_polylines=browser_polylines,
        )

    # The saved clicks and their labels: the one thing worth keeping behind a toggle, because the list is
    # long once a match is calibrated and it is only needed when a click has to be checked or re-labelled.
    # The calibrate button itself stays outside: it is the step's main action, and hiding it behind a toggle
    # meant the page read as "nothing to do here" until the expander was opened.
    stored = st.session_state.get("calib_points", [])
    ordered = order_clicks(stored)
    labelled: list[tuple[dict, str]] = []
    display_duplicates: list[str] = []

    # Callbacks run before the next run's body, so the list below is already rebuilt without the click. Doing the
    # work inline instead would need a st.rerun() to redraw it, which is a second full run for one button. They are
    # defined here, outside the expander, because the Clear button they serve sits outside it too.
    def _drop_click(pid: int) -> None:
        st.session_state["calib_points"] = [
            p for p in st.session_state.get("calib_points", []) if p["pid"] != pid
        ]

    def _goto_click(point: dict, frame_size: tuple[int, int]) -> None:
        """Show the frame a stored landmark was clicked on, centred on the click itself.

        The click list is the way back to a landmark. After a dozen clicks on scattered frames, finding one again
        by scrubbing is guesswork - and the point of going back is to check the click against the marking, which
        needs the crop on its own frame and its own spot. The frame is re-read from the source, so what comes up
        is exactly what was marked, not a nearby frame.

        The jump nonce retires the component's key: the viewport then lands on the frame the way it does on first
        mount (the same mechanism a click uses). Without it a jump to the frame already loaded would move the
        centre but leave the scrub bar where the user had left it - the whole-frame view would be somewhere else.
        """
        st.session_state["calib_reference"] = int(np.clip(point["frame"], 0, max(0, frame_count - 1)))
        st.session_state["calib_centre"] = point_centre(point["u"], point["v"], frame_size)
        st.session_state["calib_selected_pid"] = point["pid"]
        st.session_state["calib_jump_nonce"] = int(st.session_state.get("calib_jump_nonce", 0)) + 1

    def _clear_clicks() -> None:
        st.session_state["calib_points"] = []
        st.session_state["calib_landmarks"] = []
        st.session_state["calib_seeded"] = []
        st.session_state.pop("calib_fit_signature", None)
        library.clear_calibration(match_id)
        st.session_state.pop("calibration", None)

    def _retag(pid: int, widget_key: str) -> None:
        """Keep the stored tag in step with the dropdown: an edit in the list is the click's new label.

        The selectbox value is what the fit uses, but the *stored* tag is what the click carries when it is
        re-committed (a drag re-sends the marker's label) and what the duplicate drop matches on - so a correcting
        choice has to land in the stored click too, or the next drag would quietly revert it. The callback fires
        only on a real edit, which is the signal that separates a deliberate choice from the list's fallback for
        an untagged click - the render body cannot tell those apart.
        """
        choice = st.session_state.get(widget_key, "")
        if not choice:
            return
        points = st.session_state.get("calib_points", [])
        for point in points:
            if point["pid"] == pid:
                point["label"] = choice
        st.session_state["calib_points"] = points

    with st.expander("Landmark clicks", expanded=False):
        if ordered:
            restored = int(st.session_state.get("calib_restored_count", 0))
            note = " (brought back from the saved calibration)" if restored == len(ordered) else ""
            st.write(f"**{len(ordered)} landmark click(s)**{note} - tell the app which is which:")

            labelled: list[tuple[dict, str]] = []
            for index, point in enumerate(ordered):
                point_col, label_col, drop_col = st.columns([1, 3, 1])
                with point_col:
                    # Pressing the frame brings that landmark up in the crop - its own frame, centred on the click, so
                    # the marking can be checked against what was clicked without hunting for the frame again.
                    st.button(
                        f"frame {point['frame']}",
                        key=f"goto_{point['pid']}",
                        type="primary" if point["pid"] == st.session_state.get("calib_selected_pid") else "secondary",
                        on_click=_goto_click,
                        args=(point, (width, height)),
                        help="Load this landmark's own frame into the magnified view, centred on where it was clicked.",
                    )
                with label_col:
                    # The landmark is chosen at the click, in the popover that appears where it was made - so the
                    # dropdown here is only for correcting a choice afterwards. It opens on the click's own tag;
                    # an edit is persisted to the click by `_retag`, which is what makes the corrected label the
                    # click's label everywhere (including the duplicate drop below).
                    default = point.get("label") or names[0]
                    widget_key = f"landmark_label_{point['pid']}"
                    label = st.selectbox(
                        f"Landmark {index + 1}",
                        names,
                        index=names.index(default) if default in names else 0,
                        key=widget_key,
                        label_visibility="collapsed",
                        on_change=_retag,
                        args=(point["pid"], widget_key),
                    )
                    # A saved click can carry a tag the current landmark table no longer offers (the landmark set
                    # changed between versions). The dropdown silently falls back to the click-order suggestion, so
                    # without this the click would be measured against a marking it was never clicked on - say so
                    # instead of letting it through.
                    if point.get("label") and point["label"] not in names:
                        st.warning(
                            f"Tagged `{point['label']}`, which is no longer a landmark. Re-tag it (or remove the "
                            "click) before calibrating - as it stands the click would be measured against "
                            f"`{label}`."
                        )
                with drop_col:
                    st.button("Remove", key=f"drop_{point['pid']}", on_click=_drop_click, args=(point["pid"],))
                labelled.append((point, label))
                # Always describe the landmark that is actually selected: it is the moment the user picks a
                # non-default landmark that they most need to check what it means.
                st.caption(f"&nbsp;&nbsp;&nbsp;&nbsp;{LANDMARK_HELP[label]}")

            # What the *fit* uses is each click's dropdown value, and two dropdowns can still show the same
            # landmark without the stored tags agreeing: an untagged click displays the list's first entry, and a
            # click tagged with exactly that name on the same frame then matches it. Only one of the pair carries
            # a tag, so the drop above cannot see the collision - warn here instead, naming what to re-tag. The
            # fit proceeds either way: the residual table is the honest judge of what the clash costs.
            display_duplicates = repeated_labels_within_a_frame((point["frame"], label) for point, label in labelled)

            # A session with clicks on more than one moment is asking for the drift correction, and that only works if
            # each moment carries enough clicks to identify its own correction. Say which frame currently registers the
            # pose and which moments are too thin to anchor anything, rather than leaving it to the fit diagnostics.
            frames_with_clicks = sorted({point["frame"] for point, _label in labelled})
            if len(frames_with_clicks) > 1:
                counts = {frame: sum(1 for point, _l in labelled if point["frame"] == frame) for frame in frames_with_clicks}
                registered = min(frames_with_clicks, key=lambda frame: (-counts[frame], frame))
                if counts[registered] >= MIN_REFERENCE_CLICKS:
                    st.caption(
                        f"Clicks on {len(frames_with_clicks)} moments. Frame {registered} carries the most "
                        f"({counts[registered]}), so the camera pose is registered on it and the other moments pin the "
                        "drift between them."
                    )
                else:
                    st.caption(
                        f"Clicks on {len(frames_with_clicks)} moments. No single moment carries "
                        f"{MIN_REFERENCE_CLICKS} landmarks to register the pose on, so it is pooled from every click "
                        "- registering on one moment is steadier."
                    )
                sparse = [frame for frame in frames_with_clicks if counts[frame] < MIN_CLICKS_PER_ANCHOR]
                if sparse:
                    st.info(
                        "One click each at frames " + ", ".join(str(frame) for frame in sparse) + ". A single click "
                        "cannot pin a moment's correction - its ray is not enough to fix the two directions the "
                        "correction moves in - so add one more landmark at each of those moments to anchor the drift "
                        "there."
                    )

            if len(labelled) < 4:
                st.info(f"Click at least four landmarks - {4 - len(labelled)} to go.")
        else:
            st.info(
                "No landmarks clicked yet. Aim with the whole-frame view, then click a landmark on the magnified "
                "view and pick which one it is in the list that appears at the click."
            )

    dropped_note = st.session_state.pop("calib_dropped_note", "")
    if dropped_note:
        st.info(dropped_note)
    if display_duplicates:
        st.warning(
            "Two clicks are measured as the same landmark on one frame: " + ", ".join(display_duplicates)
            + ". Tag each click with its own landmark - an untagged click displays the list's first entry."
        )

    # The calibration refits itself whenever the clicks change - a click, a re-label or a removal is a new
    # measurement, and the fit is seconds, so there is nothing to wait for. The button stays for the two cases the
    # automatic pass cannot see: a refit against a changed camera-motion source, and "just fit it again".
    if len(labelled) >= 4:
        signature = [(point["frame"], point["pid"], label) for point, label in labelled]
        if st.session_state.get("calib_fit_signature") != signature:
            st.session_state["calib_fit_signature"] = signature
            calibration = fit_from_clicks(library, match_id, segment, q, focal, labelled, land_table, length_m, width_m)
            if calibration is None:
                st.session_state.pop("calib_fit_signature", None)  # let the next change retry
                st.error(calibration_failure())
            else:
                st.toast(f"Calibration updated: RMS {calibration.rms_error_m:.2f} m", icon=":material/check_circle:")

    button_col, clear_col = st.columns([2, 1])
    with button_col:
        if st.button(
            "Calibrate from these landmarks",
            type="primary",
            disabled=len(labelled) < 4,
            help="Refit the calibration from the clicks as they are labelled right now. The fit also runs "
                 "automatically whenever the clicks change; this is here for a refit against a changed camera "
                 "motion, or when you just want to run it again.",
        ):
            calibration = fit_from_clicks(library, match_id, segment, q, focal, labelled, land_table, length_m, width_m)
            if calibration is None:
                st.error(calibration_failure())
            else:
                st.success(fit_message(calibration, labelled))
    with clear_col:
        st.button("Clear all clicks", on_click=_clear_clicks)

    calibration = saved_calibration(library, match_id)
    if calibration is not None:
        used = st.session_state.get("calib_landmarks") or library.load_clicks(match_id)
        diagnosis = diagnose_fit(calibration)
        # A calibration is fitted against one camera-motion source's reference frame. If the footage now has a
        # gimbal log but the saved fit was built against the estimated chain (or vice versa), the pose it was fitted
        # to is not the pose the projection will use - measured at 24 m of error at the median - so say so and ask
        # for a refit rather than project through a stale pose.
        if segment is not None and calibration.pose_source != segment_pose_source(segment):
            st.warning(
                f"This calibration was fitted against the {calibration.pose_source} camera motion, but this "
                f"segment now uses the {segment_pose_source(segment)} motion. The two have different reference "
                "frames, so the fit is stale - press **Calibrate from these landmarks** to refit it against "
                "the current motion."
            )

        def name_of(index: int) -> str:
            """A click as the user knows it: its landmark, and the frame it came from."""
            if index < len(used):
                click = used[index]
                return f"`{click.get('label', '?')}`" + (f" (frame {click['frame']})" if "frame" in click else "")
            return f"click #{index + 1}"

        def _discard_calibration() -> None:
            library.clear_calibration(match_id)
            st.session_state.pop("calibration", None)
            st.session_state["calib_landmarks"] = []

        def _drop_outliers(indices: tuple[int, ...]) -> None:
            """Remove the clicks the fit flags, matched through the click list the fit itself used.

            The diagnosis indexes into the last fit's landmark list - the saved click list in order - so matching
            on ``(frame, label)`` rather than an index into the live session clicks keeps the removal right even
            after clicks were added on top of a stale calibration (a failed refit leaves the old residuals on
            screen). The refit then runs by itself: the click set changed, so the fit signature changed.
            """
            used_clicks = st.session_state.get("calib_landmarks") or library.load_clicks(match_id)
            wanted = {
                (int(used_clicks[i]["frame"]), str(used_clicks[i].get("label") or ""))
                for i in indices
                if i < len(used_clicks) and "frame" in used_clicks[i]
            }
            st.session_state["calib_points"] = [
                point
                for point in st.session_state.get("calib_points", [])
                if (point.get("frame"), str(point.get("label") or "")) not in wanted
            ]
            st.session_state.pop("calib_fit_signature", None)

        metric_col, discard_col = st.columns([2, 1])
        with metric_col:
            st.metric("Fit RMS error", f"{calibration.rms_error_m:.2f} m")
        with discard_col:
            st.write("")
            st.button(
                "Discard this calibration",
                help="Forget the saved fit and start the clicks again.",
                on_click=_discard_calibration,
            )

        if diagnosis.reason:
            st.error(f"**This fit is not usable.** The solver {diagnosis.reason}.")
        if diagnosis.too_few_agreeing:
            worst = ", ".join(name_of(i) for i in diagnosis.disagreeing) or "most of the clicks"
            st.error(
                f"**No pitch fits these clicks.** Only {len(diagnosis.agreeing)} of {len(calibration.residuals_m)} "
                f"agree with each other (within {diagnosis.tolerance_m:.1f} m). The ones that do not are {worst}. "
                "That is not click noise - it means at least one click is on the wrong thing, or two landmarks from "
                "different pitches have been mixed up. Check whether every click is on the *same* pitch: a venue "
                "with several goals in view makes it easy to click a corner or goal that belongs to the next pitch "
                "along, and nothing in the fit can tell you that except this."
            )
        elif diagnosis.disagreeing:
            odd = ", ".join(name_of(i) for i in diagnosis.disagreeing)
            st.warning(
                f"Fit is workable, but these clicks do not agree with the rest: {odd}. Re-click them, or drop "
                "them below and refit."
            )
        elif not diagnosis.reason:
            st.success("Good fit: every click agrees with the rest.")
        scale_note = format_scale_note(calibration)
        if scale_note:
            st.info(f"One thing to check: {scale_note}")

        x, y, z = (float(v) for v in calibration.position)
        st.caption(
            f"Camera at ({x:.0f}, {y:.0f}) m, {z:.2f} m high, lens scale {calibration.focal_scale:.2f}x."
        )
        if calibration.drift is not None and segment is not None and len(segment.time) > 1:
            anchors = calibration.drift.frames
            first, last = min(anchors), max(anchors)
            times = np.asarray(segment.time, dtype=np.float64)
            if last >= times.size:
                # The anchors are analysis-frame indices of the segment they were clicked on. A calibration saved
                # against another window (a different segment of the same video, say) cannot describe this one, and
                # indexing this segment's clock with its frames is how the page used to crash here.
                st.warning(
                    f"This match's calibration carries drift anchors up to frame {last}, but the analysed window "
                    f"holds {times.size} frames - it was solved on a different segment. Re-click the landmarks on "
                    "this one (Step 2) to fit a correction that covers it."
                )
            else:
                analysed_s = float(times[-1] - times[0])
                covered_s = float(times[last] - times[first])
                line = (
                    f"**Drift correction** is fitted: the chain is re-anchored at the {len(anchors)} clicked "
                    f"frame(s), from {_clock(times[first])} to {_clock(times[last])} of the analysed window "
                    f"({_clock(covered_s)} of {_clock(analysed_s)})."
                )
                if covered_s < 0.6 * analysed_s:
                    line += (
                        " Between anchors it is interpolated, but outside them it is held flat - so the projection "
                        "can still slide away from the markings in the parts of the video with no clicks. Clicking "
                        "the same landmarks again later in the video (any frame works) pins it there too - **Place "
                        "the calibrated landmarks as markers** above does the clicking for you, and you nudge the "
                        "markers onto the markings instead."
                    )
                st.caption(line)
        st.dataframe(
            pd.DataFrame(
                {
                    "click": [f"#{i + 1}" for i in range(len(calibration.residuals_m))],
                    "landmark": [used[i]["label"] if i < len(used) else "?" for i in range(len(calibration.residuals_m))],
                    "frame": [used[i]["frame"] if i < len(used) else "?" for i in range(len(calibration.residuals_m))],
                    "error (m)": [round(r, 2) for r in calibration.residuals_m],
                    "used in fit": [
                        "no (outlier)" if i in calibration.excluded else "yes"
                        for i in range(len(calibration.residuals_m))
                    ],
                }
            ),
            hide_index=True,
            use_container_width=True,
        )
        if calibration.excluded:
            st.warning(
                "Dropped as outliers: "
                + ", ".join(f"#{i + 1}" for i in calibration.excluded)
                + " - those clicks look wrong; check them on the frame."
            )
        outliers = sorted(set(calibration.excluded) | set(diagnosis.disagreeing))
        if outliers:
            st.button(
                f"Drop {len(outliers)} outlier click(s) and refit",
                on_click=_drop_outliers,
                args=(tuple(outliers),),
                help="Remove the clicks this fit flags - the ones the solver dropped and the ones the diagnosis "
                     "says disagree with the rest - then refit from what is left. The cleared clicks leave the "
                     "map, so re-click anything that was actually right.",
            )

# --------------------------------------------------------------------------------------------------------------
# Step 3 - report
# --------------------------------------------------------------------------------------------------------------
st.header("Step 3 - Build the tactical report")
show_flash("step3_flash")

calibration = saved_calibration(library, match_id)
if segment is None or calibration is None:
    st.info("Steps 1 and 2 are needed first: the report uses the segment results and the pitch calibration.")
else:
    # Same source of truth as Step 2: the pitch size recorded with the match, which is what was calibrated against.
    length_m, width_m = library.load(match_id).pitch_length_m, library.load(match_id).pitch_width_m
    if st.button("Build report", type="primary"):
        report_box = st.status("Building the report and the replay...", expanded=True)
        bar = st.progress(0.0, text="Projecting detections to the pitch...")
        advance = progress_reporter(bar, "Building the report...")
        with report_box:
            # The pipeline reports fractions of its own work; the bar maps them onto the whole job so it is one
            # honest line from 0 to 100 instead of three bars that each restart.
            detections = project_segment(
                segment,
                calibration,
                on_progress=lambda fraction: advance(0.40 * fraction, "Projecting detections to the pitch..."),
            )
            report, _assignment = stage_b.build_report(
                detections,
                pitch_length_m=length_m,
                pitch_width_m=width_m,
                match_frames=len(segment.time),
                on_progress=lambda fraction: advance(
                    0.40 + 0.45 * fraction, "Tracking players, teams and momentum..."
                ),
            )
            # The replay needs the same tracks, so it is built here rather than in a second pass over the segment.
            advance(0.88, "Building the replay...")
            ball = _ball_track_for_replay(segment_dir, calibration, q, focal)
            replay = build_replay(
                (length_m, width_m),
                float(segment.meta["fps"]),
                len(segment.time),
                detections.aim_xy,
                report.players,
                library.load(match_id).team_names,
                ball=ball,
                team_colours=[metrics.kit_rgb for metrics in report.teams],
                camera_xy=detections.camera_xy,
            )
            advance(0.97, "Saving the report and the replay...")
            st.session_state["report"] = {
                "teams": [vars(team) for team in report.teams],
                "players": [
                    {
                        "track_id": player.track_id,
                        "team": player.team,
                        "observations": len(player.frame),
                        "distance_m": round(player.distance_m, 1),
                        "top_speed_kmh": round(float(player.speed_kmh.max()), 1),
                        "xy": [[round(float(x), 1), round(float(y), 1)] for x, y in player.xy],
                    }
                    for player in report.players
                ],
                "momentum": report.momentum,
                "notes": report.notes,
                "pitch": [length_m, width_m],
                "detections_used": report.detections_used,
                "frames_analysed": report.frames_analysed,
            }
            library.save_report(match_id, st.session_state["report"])
            library.save_replay(match_id, replay)
        advance(1.0, "Report and replay built.")
        report_box.update(
            label=f"Report built: {len(report.players)} tracks, {len(report.teams)} teams, replay ready",
            state="complete",
            expanded=False,
        )

    payload = st.session_state.get("report") or report_from_library(library, match_id)
    if payload:
        metric_columns = st.columns(4)
        metric_columns[0].metric("Frames analysed", payload["frames_analysed"])
        metric_columns[1].metric("Player detections used", payload["detections_used"])
        metric_columns[2].metric("Players tracked", len(payload["players"]))
        metric_columns[3].metric("Tracks with a team", sum(1 for p in payload["players"] if p["team"] >= 0))
        team_names = _team_name_editor(library, match_id, payload)
        # The table says the name rather than the index, and leaves out the kit colour: it is shown as a swatch just
        # above, where it can actually be seen.
        team_rows = [
            {
                **{key: value for key, value in row.items() if key != "kit_rgb"},
                "team": team_name(int(row["team"]), team_names),
            }
            for row in payload["teams"]
        ]
        st.dataframe(pd.DataFrame(team_rows), hide_index=True, use_container_width=True)
        # The momentum curve is not drawn here any more: it is the timeline strip above the pitch, where it sits
        # on the same clock as the events and the replay. A second copy as a chart was a second thing to read.
        st.subheader("Player replay - pitch usage over time")
        replay_section(library, match_id, segment, segment_dir, chosen, calibration)
        for note in payload["notes"]:
            st.caption(f"- {note}")

# --------------------------------------------------------------------------------------------------------------
# Step 4 - events and highlights
# --------------------------------------------------------------------------------------------------------------
st.header("Step 4 - Tag events and cut highlights")

if match_id is None:
    st.info("Create a match in Step 1 to store events and highlights.")
else:
    events = library.events(match_id)
    team_names = library.load(match_id).team_names
    show_flash("events_flash")

    # These run before the next run's body, so the table below is rebuilt with the change already in it. Doing the
    # work inline instead would need a st.rerun() to redraw it, which is a second full run for one button.
    def _add_tag() -> None:
        """Store a hand tag. Its time is a time on the *selected video's* clock, and the tag says so.

        A tag is made while looking at the page's video, so that is the recording its seconds belong to - and
        stamping ``video`` is what lets the table, the timeline and the reel export translate it onto the game clock
        like any other event. Without it a tag made against a single camera clip would be read as a time on the
        combined game and land nine minutes out.
        """
        added = Event(
            time_s=float(st.session_state.get("tag_time", 0.0)),
            type=str(st.session_state.get("tag_type", EVENT_TYPES[0])),
            team=int(st.session_state.get("tag_team", -1)),
            note=str(st.session_state.get("tag_note", "")),
            video=str(Path(chosen).resolve()),
        )
        log = library.events(match_id)
        log.add(added)
        library.save_events(match_id, log)
        st.session_state["events_flash"] = ("success", f"Tagged {added.type} at {_clock(added.time_s)}.")

    def _start_audio_scan() -> None:
        """Start the whistle scan as its own process: the decode and the transform take minutes on a full game.

        The strictness and voice settings are baked into the command now, because the background process reads
        nothing from this session - the page has moved on long before the scan finishes.
        """
        command = [
            sys.executable,
            str(REPO_ROOT / "scripts" / "run_audio_scan.py"),
            "--match",
            str(match_id),
            "--video",
            str(chosen),
            "--strictness",
            f"{float(st.session_state.get('whistle_strictness', MIN_PROMINENCE)):.1f}",
        ]
        if not st.session_state.get("whistle_reject_voices", True):
            command.append("--keep-voices")
        subprocess.Popen(command, cwd=str(REPO_ROOT), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        st.session_state["events_flash"] = ("success", "Whistle scan started in the background.")

    def _discard_detected() -> None:
        log = library.events(match_id)
        removed = log.discard_detected()
        library.save_events(match_id, log)
        st.session_state["events_flash"] = (
            "success",
            f"Discarded {removed} auto-detected event(s); {len(log.events)} manual tag(s) left.",
        )

    def _detect_events() -> None:
        """Infer goals, shots, corners, penalties, clearances and tackles from the ball and player tracks.

        The ball scan and the report are the inputs, so both have to exist first; the detector is pure and quick, so
        it runs inline rather than in the background. Its output is added as review candidates (``source="ball"``),
        exactly like the whistle scan's, so the same verdict buttons and the same "discard detected" button apply.
        """
        replay = library.load_replay(match_id)
        if replay is None:
            st.session_state["events_flash"] = ("warning", "Build the report first - the detectors need the tracks.")
            return
        ball_payload = None
        try:
            ball_payload = json.loads((Path(segment_dir) / BALL_TRACK_FILE).read_text())
        except (OSError, json.JSONDecodeError):
            ball_payload = None
        if not ball_payload or not ball_payload.get("frames"):
            st.session_state["events_flash"] = (
                "warning",
                "Run the ball scan first - the detectors read the ball's track.",
            )
            return
        calibration = saved_calibration(library, match_id)
        if calibration is None:
            st.session_state["events_flash"] = ("warning", "Register the pitch first.")
            return
        segment, q, focal = segment_and_poses_cached(
            str(segment_dir), segment_fingerprint(segment_dir, stage_a.completed_chunks(segment_dir))
        )
        ball = project_ball_track(ball_payload["frames"], calibration, q, focal)
        fps = float(segment.meta["fps"])
        # Frame indices are within the analysed window, so the tracks' times need the window's own start offset -
        # the kick-off offset from Step 1. Without it every player-derived event lands `start_s` too early.
        players = event_detection.player_tracks_from_replay(replay, fps, float(segment.meta.get("start_s", 0.0)))
        times = np.asarray(segment.time, dtype=np.float64)
        # The whistle candidates are what tell a penalty from a free kick; the roster gives the shirt numbers.
        whistles = [e.time_s for e in library.events(match_id).events if e.source == "audio"]
        roster = library.load_roster(match_id)
        jerseys = library.load_jerseys(match_id)
        numbers = merge_numbers(
            [int(p["track_id"]) for p in replay.get("players", [])],
            auto={int(k): v for k, v in (jerseys.get("suggestions") or {}).items()},
            manual=roster,
        )
        detected = detect_events(
            ball[0],
            ball[1],
            times,
            players,
            (float(library.load(match_id).pitch_length_m), float(library.load(match_id).pitch_width_m)),
            whistles=whistles,
            numbers=numbers,
            half_bounds=game_marks,
            video=str(Path(chosen).resolve()),
        )
        log = library.events(match_id)
        # Reconcile rather than append: the detector's own rows are replaced, so a re-run after a fix (or after a
        # better ball scan) cannot leave the previous run's rows behind. Whistle candidates are a different source
        # and are left alone.
        added, dropped = log.reconcile_detected(detected, source="ball")
        library.save_events(match_id, log)
        stale = f", {dropped} stale removed" if dropped else ""
        st.session_state["events_flash"] = (
            "success",
            f"Detected {len(detected)} event(s); {added} new{stale}. Review them below.",
        )

    def _mark_verdict(index: int, verdict: str) -> None:
        """Record the review verdict for the moment the picker showed, then move on to the next one.

        The index arrives as a button argument rather than being read back out of session state: a selectbox's
        state is only guaranteed to be the option where it has been able to match the frontend's label against the
        options, and this has to work either way. The callback runs before the next run's body, so the log is read
        from disk here - the list below is then rebuilt with the new verdict already in it, in one run.

        Moving the picker on is the point of a review queue: ninety candidates one at a time is only bearable if
        deciding does not also mean finding the next row by hand.
        """
        log = library.events(match_id)
        try:
            number = int(index)
            log.set_verdict(number, verdict)
        except (IndexError, TypeError, ValueError):
            st.session_state["events_flash"] = ("warning", "That moment is not in the list any more - pick it again.")
            return
        library.save_events(match_id, log)
        what = {VERDICT_TRUE: "a true positive", VERDICT_FALSE: "a false positive"}.get(verdict, "unreviewed")
        st.session_state["events_flash"] = ("success", f"Marked #{number + 1} as {what}.")
        if verdict in (VERDICT_TRUE, VERDICT_FALSE):
            order = list(range(number + 1, len(log.events))) + list(range(0, number + 1))
            next_unreviewed = next((i for i in order if log.events[i].verdict == ""), None)
            if next_unreviewed is not None:
                # The next unreviewed moment, over the *current* list, so the review keeps moving without the
                # reviewer hunting for the next row. Its clip is cut on the run this triggers, as any selection is.
                st.session_state["preview_event"] = next_unreviewed
                st.session_state.pop("preview_cut_failed", None)

    def _discard_false() -> None:
        log = library.events(match_id)
        removed = log.discard_false()
        library.save_events(match_id, log)
        st.session_state["events_flash"] = (
            "success",
            f"Discarded {removed} rejected candidate(s); {len(log.events)} event(s) left.",
        )

    st.caption(
        "Whistles are found in the audio and offered as candidates. The detector is deliberately strict: it wants a "
        "loud, sustained, tonal blast, because a venue with several pitches produces a great deal of whistle-like "
        "noise at a distance - on the real sample the first version of this reported 191 candidates in five minutes. "
        "A shout or a bird of prey's call can clear that bar too, so a blast that brings its own low frequencies "
        "with it is rejected: the thresholds for that were measured against the candidates you confirmed and "
        "rejected. Each candidate's note records how far above the match's own level it sat. Goals, shots, corners, "
        "penalties, clearances and tackles can also be inferred from the ball scan and the player tracks - press "
        "**Detect events** below once the report and the ball scan exist. Saves and blocks stay yours to tag: a "
        "keeper's save and a shot wide look the same to a ball track."
    )
    tag_col, list_col = st.columns(2)
    with tag_col:
        with st.form("tag_event"):
            # The clock is named because there are two of them: the selected video's own seconds (what the player
            # shows) and the match clock (what a coach reads out). A tag is made while watching the video, so its
            # time is the video's - and the table shows the match clock beside it.
            event_time = st.number_input(
                f"Time (s) in `{Path(chosen).name}`",
                min_value=0.0,
                value=0.0,
                step=1.0,
                key="tag_time",
                help=(
                    "Seconds into the video selected in Step 1 - the time the player shows. The table adds the "
                    "match clock beside it, so a tag made on a camera clip still reads as the right minute of the "
                    "game."
                ),
            )
            event_type = st.selectbox("Type", EVENT_TYPES, index=0, key="tag_type")
            event_team = st.selectbox(
                "Team",
                [-1, 0, 1],
                format_func=lambda t: "unspecified" if t < 0 else team_name(t, team_names),
                key="tag_team",
            )
            event_note = st.text_input("Note", "", key="tag_note")
            st.form_submit_button("Add tag", on_click=_add_tag)
        st.number_input(
            "Detector strictness (x the match's own level)",
            min_value=10.0,
            max_value=2000.0,
            value=float(MIN_PROMINENCE),
            step=10.0,
            key="whistle_strictness",
            help=(
                "How far above the match's typical level in the whistle band a blast must sit to be reported. "
                "Higher means fewer, more confident candidates. Bump it up if the pitches next door dominate the "
                "list; drop it if the referee's own whistle is being missed."
            ),
        )
        st.checkbox(
            "Reject voices and calls (keep only lone tones)",
            value=True,
            key="whistle_reject_voices",
            help=(
                "A whistle puts everything into its own narrow band. A shout - at any pitch - and a bird of prey's "
                "call bring their own lower harmonics and formants with them, so a blast that lifts the region "
                "below the band is dropped as a voice or a call. The thresholds were set from this match's own "
                "confirmed and rejected candidates; turn it off if a real referee's whistle is being missed."
            ),
        )
        st.button(
            "Scan audio for whistles (background)",
            on_click=_start_audio_scan,
            disabled=library.load_audio_scan_status(match_id).get("state") == "running",
            key=f"scan_audio::{match_id}",
        )
        _audio_scan_status(library, match_id, watch_key=f"whistle_watch::{match_id}")
        st.button(
            "Detect events from the ball and player tracks",
            on_click=_detect_events,
            key=f"detect_events::{match_id}",
            help=(
                "Reads the ball scan and the player tracks to infer goals, shots, corners, penalties, clearances "
                "and tackles. Needs the report (Step 3) and the ball scan. The results are review candidates, like "
                "the whistle scan's - check them and mark the false ones."
            ),
        )
    with list_col:
        if events.events:
            counts = events.review_counts()
            hide_false = st.checkbox(
                "Hide the rejected candidates",
                value=False,
                key=f"hide_false::{match_id}",
                help=(
                    "Only changes what the table shows - the verdicts themselves are kept, and so is everything "
                    "you have not reviewed yet."
                ),
            )
            shown = [e for e in events.events if not (hide_false and e.verdict == VERDICT_FALSE)]
            # The video path is not in the table: it is long, it is the same file for every candidate, and it is
            # already named under the player when a candidate needs a different one from the selected video. The
            # team is the name the user gave it, so a row reads the way the report does.
            event_rows = [
                {
                    **{key: value for key, value in event.to_json().items() if key != "video"},
                    "team": "unspecified" if event.team < 0 else team_name(event.team, team_names),
                }
                for event in shown
            ]
            if game_marks is not None and game_record is not None:
                # The half is a question about the game clock, so it comes from the marks, not from the analysed
                # window: an event outside kick-off/full-time is labelled "-" rather than guessed at. Each event is
                # translated out of its own recording's clock first - a whistle candidate's seconds are seconds of
                # a single camera file, and 5 s of that clip is 30 minutes into the match.
                labels = game_lib.half_labels_for_events(game_record, shown)
                event_rows = [{**row, "half": label} for row, label in zip(event_rows, labels)]
                # A candidate's time is a time on *its own recording* - a camera clip's 5:00 is the game's 35:00 -
                # so the table also shows the match time a coach would read out. Events outside the marked game
                # (a whistle scanned on a clip that is not part of it) keep "-".
                match_clocks = []
                for event in shown:
                    on_game = game_lib.game_time(game_record, event.time_s, event.video or game_record.output)
                    half = None if on_game is None else game_record.half_of(on_game)
                    if half is None:
                        match_clocks.append("-")
                    elif half == 1:
                        match_clocks.append(_clock(on_game - game_marks[0]))
                    else:
                        match_clocks.append(f"{_clock(on_game - game_marks[1])} (2H)")
                event_rows = [{**row, "match clock": c} for row, c in zip(event_rows, match_clocks)]
            st.dataframe(pd.DataFrame(event_rows), hide_index=True, use_container_width=True)
            st.caption(
                f"Reviewed: {counts[VERDICT_TRUE]} true · {counts[VERDICT_FALSE]} false · "
                f"{counts['unreviewed']} still to look at."
            )
            detected = events.detected()
            if detected:
                st.button(
                    f"Discard {len(detected)} auto-detected event(s)",
                    key="discard_detected_events",
                    help=(
                        "Remove the whistle candidates the audio scan added and keep the tags you made yourself. "
                        "Re-running the scan finds them again, so nothing is lost that cannot be brought back."
                    ),
                    on_click=_discard_detected,
                )
            if counts[VERDICT_FALSE]:
                st.button(
                    f"Discard the {counts[VERDICT_FALSE]} rejected candidate(s)",
                    key="discard_false_events",
                    help=(
                        "Drop the candidates you marked as false positives and keep everything else, including "
                        "the ones you have not reviewed yet. A re-scan brings rejected candidates back."
                    ),
                    on_click=_discard_false,
                )
        else:
            st.info("No events yet.")

    # Preview a tagged moment in place: the same window a reel would cut, cut as soon as a moment is selected and
    # cached with the match so re-selecting it is instant. This is what lets a coach check a tag before committing
    # it to a reel.
    if events.events:
        st.subheader("Preview a tagged moment")
        st.caption("Pick a moment and its clip is cut and played here; clips are cached beside the match.")
        preview_dir = library.highlights_dir(match_id) / PREVIEW_DIR
        # The detail is the global sidebar option: how much picture a review clip is worth is about the reviewer's
        # patience, which does not change from match to match.
        # The picker leads with the match clock where the game is marked, because that is the time a coach reads
        # out; the candidate's own recording time is the fallback for a match with no marks.
        def _match_clock_for(event: Event) -> str | None:
            if game_record is None or game_marks is None:
                return None
            on_game = game_lib.game_time(game_record, event.time_s, event.video or game_record.output)
            if on_game is None:
                return None
            half = game_record.half_of(on_game)
            if half is None:
                return None
            return _clock(on_game - game_marks[0]) if half == 1 else f"{_clock(on_game - game_marks[1])} (2H)"

        selected = st.selectbox(
            "Moment",
            range(len(events.events)),
            format_func=lambda index: _event_label(
                events.events[index], index, _match_clock_for(events.events[index])
            ),
            key="preview_event",
        )
        preview_event = events.events[selected]
        # Cut from the recording the candidate was found in. Its timestamp is a time on *that* file's clock: a
        # single camera file's 10:00 and the combined game's 10:00 are minutes apart, so previewing one against the
        # other cuts the wrong part of the match - or, near the end of the shorter file, nothing at all.
        preview_source = Path(preview_event.video) if preview_event.video else Path(chosen)
        preview_source_missing = bool(preview_event.video) and not preview_source.exists()
        windowed = moment_for_event(preview_event)
        if preview_source_missing:
            st.error(
                f"This candidate was found in `{Path(preview_event.video).name}`, which is not available any "
                "more, so its clip cannot be cut. Re-scan the audio on the selected video for candidates on it."
            )
            preview_moment = None
        else:
            source_length = probe_cached(str(preview_source)).duration_s
            preview_moment = clamp_moment(windowed, source_length)
            if preview_moment is None:
                st.error(
                    f"This candidate sits at {_clock(preview_event.time_s)}, which is past the end of "
                    f"`{preview_source.name}` ({_clock(source_length)}) - it was found in another recording. "
                    "Select that video in Step 1, or scan this one for its own candidates."
                )
        if preview_moment is not None:
            preview_file = preview_dir / preview_clip_name(preview_moment, preview_mode)

            def _play_preview() -> None:
                """Play whatever the mode produced: an MP3 has no picture to show."""
                if preview_mode == "audio":
                    st.audio(str(preview_file))
                else:
                    st.video(str(preview_file))

            # The window is stated in the recording's own seconds (that is what the clip is cut from) with the match
            # clock beside it where the game is marked - the two differ by the clip's offset, and a coach reads the
            # match clock while the file's seconds are what the cut uses.
            match_clock = _match_clock_for(preview_event)
            when = f"{match_clock} of the match" if match_clock else f"{preview_moment.start_s:.1f}s"
            st.write(
                f"**{preview_event.type}** - {preview_moment.reason} "
                f"({when}; {preview_moment.start_s:.1f}s to {preview_moment.end_s:.1f}s of "
                f"`{preview_source.name}`, {preview_moment.end_s - preview_moment.start_s:.1f}s)"
            )
            # Where the verdict is stated for the moment on screen. The picker's own label carries it too, but
            # Streamlit only redraws a selectbox's text when its value changes, so a marker added there is a beat
            # behind the button that was just pressed.
            if preview_event.verdict:
                decided = "a true positive" if preview_event.verdict == VERDICT_TRUE else "a false positive"
                st.write(f"You marked this **{decided}** - press again to change it, or clear it below.")
            # Both of these are things the page used to leave the player to discover: that the clip came from
            # another file, and that it had to stop short of the window because the recording does.
            if Path(preview_source).resolve() != Path(chosen).resolve():
                st.caption(f"Cut from `{preview_source.name}` - the recording this candidate was found in.")
            if preview_moment is not windowed:
                st.caption(
                    "The recording covers only part of this window, so the clip ends where the footage does."
                )
            # The selection *is* the request, so there is no button to press. A failed cut is remembered, so a rerun
            # caused by some other widget does not quietly hammer ffmpeg again - that moment can be retried.
            failed = st.session_state.get("preview_cut_failed")
            if preview_file.exists():
                _play_preview()
            elif failed == preview_file.name:
                st.error("The preview for this moment could not be cut.")

                def _retry_preview() -> None:
                    st.session_state.pop("preview_cut_failed", None)

                st.button("Retry cutting the preview", on_click=_retry_preview)
            else:
                bar = st.progress(0.0, text="Cutting the preview clip...")
                reporter = progress_reporter(bar, "Cutting the preview clip...")
                try:
                    export_moment(
                        preview_source,
                        preview_moment,
                        preview_file,
                        mode=preview_mode,
                        use_gpu=True,
                        progress=reporter,
                        clip_offsets=_clip_offsets(str(preview_source)),
                    )
                except Exception as exc:  # a failed preview must not take the page down
                    st.session_state["preview_cut_failed"] = preview_file.name
                    st.error(f"Could not cut the preview: {exc}")
                if preview_file.exists():
                    reporter(1.0, "Preview ready")
                    _play_preview()
            # Review the detection this clip shows, without moving off it to hunt for a row in the table. The
            # index is passed as an argument - what the picker showed when these buttons were drawn - so the
            # verdict cannot land on a different moment if the list is rebuilt between the click and the run.
            verdict_columns = st.columns([1, 1, 1, 2])
            with verdict_columns[0]:
                st.button(
                    "True positive",
                    key=f"verdict_true::{match_id}",
                    on_click=_mark_verdict,
                    args=(selected, VERDICT_TRUE),
                    disabled=preview_event.verdict == VERDICT_TRUE,
                    help="A real stoppage: keep it, and let the reels use it.",
                )
            with verdict_columns[1]:
                st.button(
                    "False positive",
                    key=f"verdict_false::{match_id}",
                    on_click=_mark_verdict,
                    args=(selected, VERDICT_FALSE),
                    disabled=preview_event.verdict == VERDICT_FALSE,
                    help="Not a whistle's stoppage: rejected candidates are left out of the reels.",
                )
            with verdict_columns[2]:
                st.button(
                    "Clear verdict",
                    key=f"verdict_clear::{match_id}",
                    on_click=_mark_verdict,
                    args=(selected, ""),
                    disabled=not preview_event.verdict,
                    help="Back to unreviewed.",
                )

    payload = st.session_state.get("report") or report_from_library(library, match_id)
    st.subheader("Highlight reels")
    if not events.events:
        st.info("Tag an event, or scan the audio, before cutting highlights.")
    else:
        moments = build_moments(events.events, (payload or {}).get("momentum"))
        st.write(f"{len(moments)} candidate moment(s).")
        for tier, seconds in TIER_SECONDS.items():
            reel = select_reel(tier, moments, match_duration_s=probe.duration_s)
            tier_col, button_col = st.columns([3, 1])
            with tier_col:
                st.write(f"**{tier}** - {len(reel.moments)} moment(s), {reel.duration_s:.0f}s of a {seconds:.0f}s target")
            with button_col:
                export_clicked = st.button(
                    f"Export {tier}", key=f"export_{tier}", disabled=not reel.moments
                )
            if export_clicked:
                # Outside the narrow column, so the progress bar has the full width to breathe.
                out_path = library.highlights_dir(match_id) / f"{tier}.mp4"
                bar = st.progress(0.0, text=f"Cutting the {tier} reel ({len(reel.moments)} moments)...")
                reporter = progress_reporter(bar, f"Cutting the {tier} reel...")
                # Moments found in another recording are mapped onto this one through the game manifest, so a
                # whistle scanned on a camera clip lands at the right second of the combined game.
                export_reel(chosen, reel, out_path, use_gpu=True, progress=reporter, clip_offsets=_clip_offsets(chosen))
                write_manifest(reel, chosen, out_path.with_suffix(".json"))
                reporter(1.0, f"{tier} reel ready")
                st.success(f"Wrote `{out_path.relative_to(REPO_ROOT)}`")

# --------------------------------------------------------------------------------------------------------------
# Library
# --------------------------------------------------------------------------------------------------------------
st.divider()
st.header("Archive contents")
rows = library.summaries()
if rows:
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
    pick = st.selectbox("Show artefacts for", [row["match_id"] for row in rows])
    st.dataframe(pd.DataFrame(library.artifacts(pick)), hide_index=True, use_container_width=True)
else:
    st.info("No matches archived yet.")
