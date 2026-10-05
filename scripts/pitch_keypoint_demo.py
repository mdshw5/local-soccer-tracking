"""Test server: pitch-keypoint detection and automatic registration on a real segment.

This is a diagnostic page, not the main dashboard. Its job is to make the automatic registration's evidence
visible: which markers the YOLO pitch model finds, where the saved calibration thinks the pitch is, and what a
prior-constrained registration makes of it. On footage where other goals and kickwalls share the frame, the model
can lock onto a neighbouring structure, and that is exactly the failure this page exists to show.

Run::

    .venv/bin/streamlit run scripts/pitch_keypoint_demo.py --server.port 8506
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import streamlit as st

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from soccer_analytics.analysis.projection import segment_poses  # noqa: E402
from soccer_analytics.analysis.stage_a import load_segment  # noqa: E402
from soccer_analytics.geometry.auto_register import (  # noqa: E402
    auto_register,
    register_with_position_prior,
    registration_note,
)
from soccer_analytics.geometry.pitch_calibration import PitchCalibration, pitch_to_pixels  # noqa: E402
from soccer_analytics.geometry.pitch_keypoint_yolo import (  # noqa: E402
    frame_observations,
    load_pitch_keypoint_model,
    observations_for_frames,
    resolve_pitch_weights,
)
from soccer_analytics.geometry.pitch_template import template_for  # noqa: E402
from soccer_analytics.ingest.ffmpeg_reader import grab_frame  # noqa: E402

SEGMENTS_ROOT = REPO_ROOT / "data" / "segments"
MATCHES_ROOT = REPO_ROOT / "data" / "matches"


@st.cache_resource(show_spinner="Loading the pitch-keypoint model…")
def pitch_model():
    return load_pitch_keypoint_model(device=0)


@st.cache_resource(show_spinner="Loading the segment…")
def cached_segment(segment_dir: str, fingerprint: float):
    del fingerprint
    segment = load_segment(segment_dir)
    q, focal = segment_poses(segment)
    return segment, q, focal


@st.cache_data(show_spinner=False, max_entries=24)
def cached_frame(video: str, time_s: float, width: int):
    return grab_frame(video, time_s, width=width)


def draw_keypoints(image: np.ndarray, observations, *, width: int) -> np.ndarray:
    """Model detections as magenta crosses with their template index and confidence."""
    out = image.copy()
    for observation in observations:
        x, y = int(observation.u * width), int(observation.v * width)
        cv2.drawMarker(out, (x, y), (255, 0, 255), cv2.MARKER_CROSS, 20, 3)
        text = f"{observation.index} ({observation.confidence:.2f})"
        for colour, thickness in (((0, 0, 0), 4), ((255, 255, 255), 1)):
            cv2.putText(out, text, (x + 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.55, colour, thickness)
    return out


def draw_template(image: np.ndarray, calibration: PitchCalibration, q, focal, template, *, width: int) -> np.ndarray:
    """The template projected through a calibration, in yellow, with its indices."""
    out = image.copy()
    uv, in_front = pitch_to_pixels(calibration, np.asarray(template), q, focal)
    for index, ((u, v), visible) in enumerate(zip(uv, in_front)):
        if not visible or not (0.0 <= u <= 1.0 and 0.0 <= v <= image.shape[0] / width):
            continue
        x, y = int(u * width), int(v * width)
        cv2.circle(out, (x, y), 9, (0, 255, 255), 2)
        for colour, thickness in (((0, 0, 0), 3), ((0, 255, 255), 1)):
            cv2.putText(out, str(index), (x + 10, y + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, thickness)
    return out


def to_display(image: np.ndarray, width: int = 1280) -> np.ndarray:
    scale = width / image.shape[1]
    return cv2.resize(image, (width, int(round(image.shape[0] * scale))))


st.set_page_config(page_title="Pitch keypoints & auto-registration", layout="wide")
st.title("Pitch keypoints & automatic registration")
st.caption(
    "A diagnostic view of the YOLO pitch-keypoint model on this footage, and of registration built on it. "
    "The model's weights come from `tmoklc/scorevisionv1` (the model used by "
    "`rustyneuron01/Real-Time-Football-Detection`)."
)

weights = resolve_pitch_weights(None)
if weights is None:
    st.error(
        "No pitch-keypoint weights found. Put `football-pitch-detection.pt` in `data/models/` "
        "(see `soccer_analytics.geometry.pitch_keypoint_yolo.download_pitch_weights`)."
    )
    st.stop()

segments = sorted(p for p in SEGMENTS_ROOT.glob("*") if (p / "meta.json").exists())
if not segments:
    st.warning("No analysed segments under data/segments. Run Stage A first (dashboard Step 1).")
    st.stop()

segment_dir = st.sidebar.selectbox("Segment", segments, format_func=lambda p: p.name)
meta = json.loads((segment_dir / "meta.json").read_text())
length_m = st.sidebar.number_input("Pitch length (m)", value=100.0, step=1.0)
width_m = st.sidebar.number_input("Pitch width (m)", value=64.0, step=1.0)

fingerprint = max((p.stat().st_mtime for p in segment_dir.glob("chunk_*.npz")), default=0.0)
segment, q, focal = cached_segment(str(segment_dir), fingerprint)
frame_count = len(segment.time)
frame_index = st.sidebar.slider("Frame", 0, frame_count - 1, min(frame_count // 2, frame_count - 1))
image_sizes = st.sidebar.multiselect("Model input sizes", [640, 960, 1280, 1920], default=[640, 1920])
threshold = st.sidebar.slider("Keypoint confidence floor", 0.0, 1.0, 0.30, 0.05)

frame = cached_frame(meta["video"], float(segment.time[frame_index]), 1920)
if frame is None:
    st.error("Could not decode that frame.")
    st.stop()

model = pitch_model()
observations = frame_observations(model, frame, frame_index, image_sizes=tuple(image_sizes), threshold=threshold)

left, right = st.columns(2)
with left:
    st.subheader("Model keypoints (magenta)")
    st.image(draw_keypoints(frame, observations, width=1920)[:, :, ::-1], use_container_width=True)
    st.caption(f"{len(observations)} keypoint(s) over the confidence floor on frame {frame_index}.")

with right:
    st.subheader("Saved calibration (yellow)")
    matches = sorted(p for p in MATCHES_ROOT.glob("*") if (p / "calibration.json").exists())
    options = ["none"] + [p.name for p in matches]
    choice = st.selectbox("Match calibration", options)
    if choice != "none":
        calibration = PitchCalibration.from_json(
            json.loads((MATCHES_ROOT / choice / "calibration.json").read_text())
        )
        template = template_for(length_m, width_m)
        st.image(
            draw_template(frame, calibration, q[frame_index], float(focal[frame_index]), template, width=1920)[:, :, ::-1],
            use_container_width=True,
        )
        st.caption(
            f"Camera position {np.round(calibration.position, 2).tolist()}, focal scale "
            f"{calibration.focal_scale:.3f}, fit rms {calibration.rms_error_m:.2f} m."
        )
    else:
        st.info("Pick a saved calibration to project its pitch template over the frame.")

st.divider()
st.header("Automatic registration")
st.caption(
    "Samples frames across the segment, detects keypoints, screens each frame against its own homography, then "
    "registers. With a known fixed camera position the search is constrained to an orientation, which is what stops "
    "it locking onto a neighbouring pitch."
)

col_a, col_b, col_c, col_d = st.columns(4)
position_x = col_a.number_input("Camera x (m)", value=length_m / 2, step=0.5)
position_y = col_b.number_input("Camera y (m)", value=-2.0, step=0.5)
position_z = col_c.number_input("Camera z (m)", value=4.0, step=0.1)
focal_prior = col_d.number_input("Focal scale prior", value=1.0, step=0.01)
sample_count = st.slider("Frames to sample", 4, 32, 16)
use_prior = st.checkbox("Constrain to the known camera position", value=True)

if st.button("Run automatic registration", type="primary"):
    sample_frames = sorted(set(np.linspace(0, frame_count - 1, sample_count).astype(int).tolist()))
    samples = []
    progress = st.progress(0.0, text="Decoding frames…")
    for rank, index in enumerate(sample_frames):
        decoded = cached_frame(meta["video"], float(segment.time[int(index)]), 1920)
        if decoded is not None:
            samples.append((int(index), decoded))
        progress.progress((rank + 1) / len(sample_frames), text=f"Decoded {rank + 1}/{len(sample_frames)}")
    progress.progress(1.0, text="Detecting keypoints…")
    observations = observations_for_frames(
        model, samples, image_sizes=tuple(image_sizes), threshold=threshold,
        on_progress=lambda f: progress.progress(f, text=f"Detecting keypoints… {f:.0%}"),
    )
    chain = {int(index): (q[int(index)], float(focal[int(index)])) for index, _ in samples}
    st.write(f"{len(observations)} keypoint(s) detected across {len(samples)} frame(s).")
    try:
        if use_prior:
            result = register_with_position_prior(
                observations, chain, segment.aspect,
                position_prior=(position_x, position_y, position_z),
                focal_scale_prior=float(focal_prior),
                length_m=length_m, width_m=width_m, correct_drift=True,
            )
        else:
            result = auto_register(
                observations, chain, segment.aspect,
                length_m=length_m, width_m=width_m, correct_drift=True,
            )
    except Exception as exc:
        st.error(f"Registration failed: {exc}")
        st.stop()
    st.success(registration_note(result))
    st.write("Camera position", np.round(result.calibration.position, 2).tolist())
    st.write("Focal scale", round(result.calibration.focal_scale, 3), " fit rms (m)", round(result.calibration.rms_error_m, 2))
    for note in result.notes:
        st.write("-", note)
    st.image(
        draw_template(
            frame, result.calibration, q[frame_index], float(focal[frame_index]), template_for(length_m, width_m),
            width=1920,
        )[:, :, ::-1],
        use_container_width=True,
    )
    with st.expander("Per-frame screening"):
        st.dataframe(
            [{"frame": frame, **counts} for frame, counts in result.frames.items()],
            use_container_width=True,
        )