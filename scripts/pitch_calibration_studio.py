"""Calibration studio: produce a high-quality pitch calibration by correcting model suggestions.

The fine-tuning pipeline needs labels, the labels are only as good as the camera calibration, and a handful of
clicks on one moment is not accurate enough - measured on the reference match, its own clicks reproject 4-63 px
away. This page is the fix. It runs the pitch-keypoint model on sampled frames, drops its suggestions into the same
click editor the dashboard uses, and lets a person drag each one onto the real marking. Corrections accumulate
across many frames; the fit then solves a drift-corrected camera path from all of them, which is far more accurate
than the same number of clicks on a single frame.

Nothing here pretends the model is right. Its suggestions are a starting point to drag, not an answer: on footage
with neighbouring pitches it will sometimes place a goal's worth of markers on the wrong goal, and the person
correcting them is the point of the page.

Output: a ``calibration.json`` (and ``clicks.json``) written next to a match or a segment, ready to feed
``scripts/build_pitch_dataset.py``.

Run::

    .venv/bin/streamlit run scripts/pitch_calibration_studio.py --server.port 8507
"""

from __future__ import annotations

import base64
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
import streamlit.components.v1 as components

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from soccer_analytics.analysis.projection import segment_poses  # noqa: E402
from soccer_analytics.analysis.stage_a import load_segment  # noqa: E402
from soccer_analytics.dashboard.pitch_clicks import (  # noqa: E402
    actual_centre,
    canvas_scale,
    frame_points,
    inside_crop,
    landmark_table,
    marker_feature,
    merge_clicked,
    order_clicks,
    parse_result,
    pitch_overlay,
    points_in_crop,
    projected_landmarks,
    restored_points,
    zoom_box,
)
from soccer_analytics.geometry.pitch_calibration import Landmark, PitchCalibration, calibrate  # noqa: E402
from soccer_analytics.geometry.pitch_keypoint_yolo import (  # noqa: E402
    frame_observations,
    load_pitch_keypoint_model,
    resolve_pitch_weights,
)
from soccer_analytics.geometry.pitch_template import template_for  # noqa: E402
from soccer_analytics.ingest.ffmpeg_reader import grab_frame, probe_video  # noqa: E402

COMPONENT = components.declare_component(
    "field_annotation_editor",
    path=str(REPO_ROOT / "src" / "soccer_analytics" / "dashboard" / "field_annotation_component"),
)
SEGMENTS_ROOT = REPO_ROOT / "data" / "segments"
MATCHES_ROOT = REPO_ROOT / "data" / "matches"
MATCH_FORMATS = {"5v5": (40.0, 25.0), "7v7": (50.0, 35.0), "9v9": (60.0, 40.0), "11v11": (100.0, 64.0)}
DEFAULT_ZOOM = 6.0
SAMPLE_COUNT = 30


def _encode(frame: np.ndarray) -> str:
    ok, buffer = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 88])
    if not ok:
        raise ValueError("could not encode the frame")
    return "data:image/jpeg;base64," + base64.b64encode(buffer.tobytes()).decode("ascii")


@st.cache_resource(show_spinner="Loading the pitch-keypoint model…")
def pitch_model():
    return load_pitch_keypoint_model(device=0)


@st.cache_resource(show_spinner="Loading the segment…")
def cached_segment(segment_dir: str, fingerprint: float):
    del fingerprint
    segment = load_segment(segment_dir)
    q, focal = segment_poses(segment)
    return segment, q, focal


@st.cache_data(show_spinner=False, max_entries=32)
def cached_frame(video: str, time_s: float, width: int):
    return grab_frame(video, time_s, width=width)


@st.cache_data(show_spinner=False, max_entries=32)
def cached_width(video: str) -> int:
    return int(probe_video(video).width)


@st.cache_data(show_spinner=False, max_entries=256)
def cached_suggestions(video: str, time_s: float, frame_index: int):
    """Model keypoints for one frame, keyed by the frame (so a zoom does not re-run the model)."""
    image = cached_frame(video, time_s, 1920)
    if image is None:
        return []
    model = pitch_model()
    observations = frame_observations(model, image, frame_index)
    return [(o.index, o.u, o.v, o.confidence) for o in observations]


def template_landmarks(length_m: float, width_m: float) -> dict[str, tuple[float, float]]:
    """The 32 template markers as named landmarks, so the model's suggestions can be labelled and corrected."""
    return {f"kp{index:02d}": tuple(point) for index, point in enumerate(template_for(length_m, width_m))}


def landmark_clicker(crop, overview, features, markers, centre, zoom, box, frame_size, key, *, frame_count, initial_frame, marker_kinds):
    """Draw the click editor over one crop (same component and geometry as the dashboard's Step 2)."""
    scale = canvas_scale(crop.shape[1])
    canvas = crop if scale == 1.0 else cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    frame_width, frame_height = frame_size
    x0, y0, _w, _h = box
    placed_x, placed_y = actual_centre(box, frame_size)
    COMPONENT(
        frame_data=_encode(canvas),
        overview_data=_encode(overview) if overview is not None else "",
        proxy_url="",
        frame_count=int(frame_count),
        initial_frame=int(initial_frame),
        marker_kinds=list(marker_kinds),
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


st.set_page_config(page_title="Pitch calibration studio", layout="wide")
st.title("Pitch calibration studio")
st.caption(
    "Correct the model's suggested markers on sampled frames; the fit solves a drift-corrected camera path from "
    "all of them. The result is the calibration that `build_pitch_dataset.py` turns into training labels."
)

if resolve_pitch_weights(None) is None:
    st.error("No pitch-keypoint weights in data/models/. Run the download first.")
    st.stop()

segments = sorted(p for p in SEGMENTS_ROOT.glob("*") if (p / "meta.json").exists())
if not segments:
    st.error("No analysed segments under data/segments.")
    st.stop()

sidebar = st.sidebar
segment_dir = sidebar.selectbox("Segment", segments, format_func=lambda p: p.name)
format_name = sidebar.selectbox("Match format", list(MATCH_FORMATS), index=3)
length_m, width_m = MATCH_FORMATS[format_name]
fingerprint = max((p.stat().st_mtime for p in segment_dir.glob("chunk_*.npz")), default=0.0)
segment, q, focal = cached_segment(str(segment_dir), fingerprint)
meta = json.loads((segment_dir / "meta.json").read_text())
video = meta["video"]
frame_count = len(segment.time)

table = {**landmark_table(length_m, width_m), **template_landmarks(length_m, width_m)}
names = list(table)

sample_frames = sorted(set(np.linspace(0, frame_count - 1, SAMPLE_COUNT).astype(int).tolist()))
frame_index = sidebar.select_slider(
    "Frame to correct",
    options=sample_frames,
    value=sample_frames[len(sample_frames) // 2],
    format_func=lambda f: f"{f} (t={segment.time[f]:.0f}s)",
)

# ---- gestures (read before the crop is built, as the component hands its value to the next run) ------------------
apply_nonce = int(st.session_state.get("apply_nonce", 0))
component_key = f"studio::{segment_dir}::{apply_nonce}"
gesture = parse_result(st.session_state.get(component_key))
if gesture.action == "navigate":
    if gesture.centre is not None:
        st.session_state["centre"] = gesture.centre
    if gesture.zoom is not None:
        st.session_state["zoom"] = gesture.zoom

centre = st.session_state.get("centre", (0.5, 0.5))
zoom = float(st.session_state.get("zoom", DEFAULT_ZOOM))
stored = st.session_state.setdefault("points", [])
next_pid = st.session_state.get("next_pid", 0)

full = cached_frame(video, float(segment.time[frame_index]), cached_width(video))
show_suggestions = sidebar.checkbox("Show the model's suggested markers", value=True)

if full is None:
    st.error("Could not read that frame.")
    st.stop()

height, width = full.shape[:2]
box = zoom_box(width, height, zoom, centre[0], centre[1])
x0, y0, crop_w, crop_h = box
crop = full[y0 : y0 + crop_h, x0 : x0 + crop_w]
overview = cv2.resize(full, (1280, int(1280 * height / width)), interpolation=cv2.INTER_AREA)
scale = canvas_scale(crop_w)

seeds = [s for s in st.session_state.get("seeds", []) if s["frame"] == frame_index]
if gesture.action == "apply":
    if gesture.points:
        stored, next_pid = merge_clicked(stored, gesture.points, frame_index, box, scale, width, next_pid)
        st.session_state["points"] = stored
        st.session_state["next_pid"] = next_pid
    st.session_state["seeds"] = [s for s in st.session_state.get("seeds", []) if s["frame"] != frame_index]
    apply_nonce += 1
    st.session_state["apply_nonce"] = apply_nonce
    component_key = f"studio::{segment_dir}::{apply_nonce}"
    seeds = []

if show_suggestions and not seeds and not any(p["frame"] == frame_index for p in stored):
    suggestions = cached_suggestions(video, float(segment.time[frame_index]), frame_index)
    if suggestions:
        st.session_state["seeds"] = st.session_state.get("seeds", []) + [
            {"frame": frame_index, "label": f"kp{index:02d}", "u": u, "v": v, "confidence": confidence}
            for index, u, v, confidence in suggestions
        ]
        seeds = [s for s in st.session_state["seeds"] if s["frame"] == frame_index]

features = points_in_crop(stored, frame_index, box, scale, width)
features += [marker_feature(seed["label"], seed["u"], seed["v"], box, scale, width) for seed in seeds]
markers = frame_points(stored, frame_index) + [(seed["u"], seed["v"]) for seed in seeds]

landmark_clicker(
    crop, overview, features, markers, centre, zoom, box, (width, height), component_key,
    frame_count=frame_count, initial_frame=frame_index, marker_kinds=names,
)

st.caption(
    f"Frame {frame_index} of {frame_count}. Magenta markers are the model's suggestions (labels are its keypoint "
    "indices); drag each onto the real marking and press **Apply**. Add more with the **Marker** picker."
)

# ---- fit and save ------------------------------------------------------------------------------------------------
ordered = order_clicks(stored)
st.write(f"**{len(ordered)} landmark click(s)** across {len({p['frame'] for p in ordered})} frame(s).")

if len(ordered) >= 4 and st.button("Fit calibration from these landmarks", type="primary"):
    landmarks = [
        Landmark(p["frame"], p["u"], p["v"], table[p["label"]][0], table[p["label"]][1], p["label"])
        for p in ordered
        if p.get("label") in table
    ]
    chain = {p["frame"]: (q[p["frame"]], float(focal[p["frame"]])) for p in ordered if p["frame"] < frame_count}
    try:
        calibration = calibrate(landmarks, chain, segment.aspect, correct_drift=True)
    except Exception as exc:
        st.error(f"Calibration failed: {exc}")
    else:
        st.session_state["calibration"] = calibration.to_json()
        st.success(f"Fit RMS error {calibration.rms_error_m:.2f} m.")
        st.write("Camera position", np.round(calibration.position, 2).tolist(), "focal scale", round(calibration.focal_scale, 3))
        if calibration.drift is not None:
            st.write(f"Drift correction anchored at {len(calibration.drift.frames)} moment(s).")

calibration = PitchCalibration.from_json(st.session_state["calibration"]) if "calibration" in st.session_state else None
if calibration is not None:
    view_q, view_focal = calibration.corrected_frame(q[frame_index], float(focal[frame_index]), frame_index)
    overlay = pitch_overlay(overview, calibration, view_q, view_focal, length_m, width_m)
    st.image(overlay[:, :, ::-1], caption="Current calibration projected onto this frame", use_container_width=True)

    save_col, _ = st.columns([1, 3])
    with save_col:
        matches = sorted(p.name for p in MATCHES_ROOT.glob("*") if p.is_dir())
        target = st.selectbox("Save into", ["<segment dir>", *matches])
        if st.button("Save calibration"):
            out_dir = segment_dir if target == "<segment dir>" else MATCHES_ROOT / target
            out_dir.mkdir(parents=True, exist_ok=True)
            (out_dir / "calibration.json").write_text(json.dumps(calibration.to_json(), indent=2))
            clicks = [{"frame": p["frame"], "u": p["u"], "v": p["v"], "label": p["label"]} for p in ordered]
            (out_dir / "clicks.json").write_text(json.dumps({"clicks": clicks, "pitch": [length_m, width_m]}, indent=2))
            st.success(f"Saved calibration.json and clicks.json to {out_dir}")
else:
    st.info("Correct the markers on a few moments, then press **Fit calibration**.")