"""Pitch-keypoint inference with the YOLO-pose model from rustyneuron01/Real-Time-Football-Detection.

That project (and the Roboflow `sports` library it vendors) ships a YOLO-pose model, `football-pitch-detection.pt`,
hosted on Hugging Face at ``tmoklc/scorevisionv1``. Unlike the HRNet checkpoint in
:mod:`soccer_analytics.geometry.pitch_keypoint_model`, this one is actually downloadable, and it is the keypoint
source automatic registration uses by default.

The model has one class (``pitch``) and 32 keypoints, in the order of Roboflow's
``SoccerPitchConfiguration.vertices`` - the same order as :mod:`soccer_analytics.geometry.pitch_template`, so output
keypoint ``i`` is template marker ``i``. The model was trained against a 120x70 diagram whose penalty-box
proportions are internally inconsistent, but a detector predicts the *visual* marking, not the diagram: the
template supplies the world coordinate, and it is the match's own pitch format that matters there.

The weights are fetched on demand rather than shipped, the same rule the rest of the project follows.
"""

from __future__ import annotations

import shutil
import urllib.request
from pathlib import Path

import numpy as np

from soccer_analytics.geometry.auto_register import KeypointObservation
from soccer_analytics.geometry.pitch_keypoint_model import MODELS_DIR

DEFAULT_WEIGHTS_NAME = "football-pitch-detection.pt"
HF_REPO_ID = "tmoklc/scorevisionv1"
HF_FILENAME = "football-pitch-detection.pt"
HF_URL = f"https://huggingface.co/{HF_REPO_ID}/resolve/main/{HF_FILENAME}"

# The model is a single-class pose net over the ground plane. Its own detector threshold applies; below this
# keypoint confidence a marker is treated as a guess and dropped. At the widest input the model missed whole frames
# that it saw at 640 and vice versa, so both sizes are run and merged - cheap on a GPU, and it recovers far more of
# a moving camera's moments than either size alone.
DEFAULT_IMAGE_SIZES = (640, 1920)
DEFAULT_KEYPOINT_THRESHOLD = 0.30


def resolve_pitch_weights(weights: str | Path | None = None) -> str | None:
    """The pitch-keypoint checkpoint to use: an explicit path, or ``football-pitch-detection.pt`` in models.

    Deliberately does *not* fall back to whatever ``*.pt`` happens to sit in ``data/models/``. That directory is
    also where a person-detection checkpoint lives, and treating it as a pitch-keypoint model produces a model that
    silently detects nothing - the failure this resolver used to cause. Absent the named file, the caller downloads
    it instead.
    """
    if weights is not None:
        return str(weights) if Path(weights).exists() else None
    named = MODELS_DIR / DEFAULT_WEIGHTS_NAME
    return str(named) if named.exists() else None


def download_pitch_weights(destination: str | Path | None = None, *, force: bool = False) -> Path:
    """Fetch the YOLO pitch model from Hugging Face into ``data/models/`` (or ``destination``).

    Uses ``urllib`` rather than ``huggingface_hub`` so it works without an extra dependency. The download goes to a
    temporary file and is moved into place, so an interrupted fetch never leaves a half model that looks loadable.
    """
    target = Path(destination) if destination is not None else MODELS_DIR / DEFAULT_WEIGHTS_NAME
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and not force:
        return target
    tmp = target.with_suffix(target.suffix + ".part")
    with urllib.request.urlopen(HF_URL, timeout=600) as response, tmp.open("wb") as handle:
        shutil.copyfileobj(response, handle)
    tmp.replace(target)
    return target


def load_pitch_keypoint_model(weights: str | Path | None = None, device: str | int | None = None):
    """Load the YOLO pitch model, downloading it if it is absent. Returns an Ultralytics ``YOLO``."""
    from ultralytics import YOLO

    path = resolve_pitch_weights(weights)
    if path is None:
        path = str(download_pitch_weights())
    model = YOLO(path)
    if device is not None:
        model.to(device=device)
    return model


def frame_observations(
    model,
    frame: np.ndarray,
    frame_index: int,
    *,
    image_sizes: tuple[int, ...] = DEFAULT_IMAGE_SIZES,
    threshold: float = DEFAULT_KEYPOINT_THRESHOLD,
) -> list[KeypointObservation]:
    """Template keypoints detected in one frame, as solver observations.

    The frame is passed to the model at each of ``image_sizes`` and the results are merged, keeping the
    highest-confidence reading of each template index. Coordinates are returned in the solver's convention (both
    axes divided by the frame *width*), independent of the analysis resolution.
    """
    width = float(frame.shape[1])
    best: dict[int, KeypointObservation] = {}
    for image_size in image_sizes:
        result = model(frame, imgsz=int(image_size), verbose=False)[0]
        keypoints = getattr(result, "keypoints", None)
        if keypoints is None or keypoints.xy is None or keypoints.xy.numel() == 0:
            continue
        xy = keypoints.xy.cpu().numpy().reshape(-1, 2)
        conf = (
            keypoints.conf.cpu().numpy().reshape(-1)
            if keypoints.conf is not None
            else np.ones(len(xy), dtype=np.float32)
        )
        for index, ((x, y), confidence) in enumerate(zip(xy, conf)):
            if index >= 32 or float(confidence) < threshold:
                continue
            previous = best.get(index)
            if previous is not None and previous.confidence >= float(confidence):
                continue
            best[index] = KeypointObservation(
                frame=frame_index,
                index=index,
                u=float(x) / width,
                v=float(y) / width,
                confidence=float(confidence),
            )
    return [best[index] for index in sorted(best)]


def observations_for_frames(
    model,
    frames: list[tuple[int, np.ndarray]],
    *,
    image_sizes: tuple[int, ...] = DEFAULT_IMAGE_SIZES,
    threshold: float = DEFAULT_KEYPOINT_THRESHOLD,
    on_progress=None,
) -> list[KeypointObservation]:
    """Detections for several ``(frame_index, frame_bgr)`` pairs, in one flat list."""
    out: list[KeypointObservation] = []
    for rank, (index, frame) in enumerate(frames):
        out.extend(frame_observations(model, frame, index, image_sizes=image_sizes, threshold=threshold))
        if on_progress is not None:
            on_progress((rank + 1) / max(1, len(frames)))
    return out