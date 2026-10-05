"""Pitch-keypoint helpers: heatmap peaks, template refinement, bird's-eye warp.

Adapted from whisdev/soccer-video-detection-ai-agent's keypoint pipeline. Their
version is hard-wired to an HRNet heatmap model whose weights are not
distributed, so only the model-agnostic parts are ported here: they work with
*any* keypoint source (an HRNet heatmap, a YOLO-pose model, or hand-placed
points) paired with a pitch template.

Conventions
-----------
* A keypoint is ``{"x": float, "y": float, "p": float}`` in image pixels, where
  ``p`` is the detection confidence (0.0 means "believed but not seen").
* A template is a sequence of ``(x, y)`` pitch coordinates; a keypoint with
  index ``i`` corresponds to ``template[i]``.
* The homography returned by :func:`homography_from_keypoints` maps *template
  (pitch) coordinates -> image pixels*.
"""

from __future__ import annotations

from collections.abc import Sequence

import cv2
import numpy as np

Keypoint = dict[str, float]
Template = Sequence[tuple[float, float]]

DEFAULT_PEAK_THRESHOLD = 0.2
DEFAULT_RANSAC_THRESHOLD = 3.0
MIN_HOMOGRAPHY_POINTS = 4


def _to_numpy(heatmap) -> np.ndarray:
    """Accepts a torch tensor (C, H, W) / (1, C, H, W) or a numpy array."""
    if hasattr(heatmap, "detach"):
        heatmap = heatmap.detach().cpu().numpy()
    arr = np.asarray(heatmap, dtype=np.float32)
    if arr.ndim == 4:
        arr = arr[0]
    elif arr.ndim == 2:
        arr = arr[None]
    return arr


def extract_heatmap_peaks(
    heatmap,
    threshold: float = DEFAULT_PEAK_THRESHOLD,
    scale: float = 1.0,
) -> dict[int, Keypoint]:
    """Strongest local maximum per heatmap channel, like the reference project's
    `_extract_keypoints` (max-pool 3x3 local-maxima test, then top-1).

    `scale` multiplies heatmap coordinates to reach image pixels (the reference
    model emits 540x960 heatmaps for 1080x1920 frames, i.e. `scale=2`).
    """
    planes = _to_numpy(heatmap)
    peaks: dict[int, Keypoint] = {}
    kernel = np.ones((3, 3), dtype=np.uint8)
    for channel, plane in enumerate(planes):
        local_max = cv2.dilate(plane, kernel)
        mask = (plane >= local_max) & (plane >= threshold)
        if not mask.any():
            continue
        ys, xs = np.nonzero(mask)
        best = int(plane[ys, xs].argmax())
        peaks[channel] = {
            "x": float(xs[best]) * scale,
            "y": float(ys[best]) * scale,
            "p": float(plane[ys[best], xs[best]]),
        }
    return peaks


def _detected_pairs(
    keypoints: dict[int, Keypoint],
    template: Template,
) -> list[tuple[int, tuple[float, float], tuple[float, float]]]:
    """(index, template point, image point) for every confidently detected keypoint."""
    pairs = []
    for index, point in keypoints.items():
        if not 0 <= index < len(template) or point.get("p", 0.0) <= 0.0:
            continue
        pairs.append((index, (float(template[index][0]), float(template[index][1])), (point["x"], point["y"])))
    return pairs


def homography_from_keypoints(
    keypoints: dict[int, Keypoint],
    template: Template,
    min_points: int = MIN_HOMOGRAPHY_POINTS,
    ransac_threshold: float = DEFAULT_RANSAC_THRESHOLD,
) -> np.ndarray | None:
    """Template(pitch) -> image homography from detected keypoints, or None.

    Returns None when fewer than `min_points` keypoints were detected or the fit
    fails, so callers can fall back to the calibrated/manual homography.
    """
    pairs = _detected_pairs(keypoints, template)
    if len(pairs) < min_points:
        return None
    source = np.array([pair[1] for pair in pairs], dtype=np.float32)
    target = np.array([pair[2] for pair in pairs], dtype=np.float32)
    homography, _ = cv2.findHomography(source, target, cv2.RANSAC, ransac_threshold)
    return homography


def refine_keypoints(
    keypoints: dict[int, Keypoint],
    template: Template,
    frame_shape: tuple[int, ...],
    min_points: int = MIN_HOMOGRAPHY_POINTS,
    ransac_threshold: float = DEFAULT_RANSAC_THRESHOLD,
) -> dict[int, Keypoint]:
    """Fill undetected template points by projecting the template through the fit.

    Detected keypoints are kept as-is; missing ones are added when their
    template-projected position lands inside the frame, tagged with
    ``"source": "template"`` (detected ones get ``"source": "detected"``).
    Mirrors the reference project's `_apply_homography_refinement`.
    """
    refined = {
        index: {**point, "source": "detected"}
        for index, point in keypoints.items()
        if point.get("p", 0.0) > 0.0
    }
    homography = homography_from_keypoints(keypoints, template, min_points, ransac_threshold)
    if homography is None:
        return refined

    height, width = frame_shape[0], frame_shape[1]
    template_array = np.array(template, dtype=np.float32).reshape(-1, 1, 2)
    projected = cv2.perspectiveTransform(template_array, homography).reshape(-1, 2)
    for index, (x, y) in enumerate(projected):
        if index in refined:
            continue
        if 0.0 <= x < width and 0.0 <= y < height:
            refined[index] = {"x": float(x), "y": float(y), "p": 0.0, "source": "template"}
    return refined


def warp_to_pitch_view(
    frame: np.ndarray,
    pitch_to_image: np.ndarray,
    output_size: tuple[int, int],
):
    """Rectifies a frame into pitch (bird's-eye) coordinates.

    `pitch_to_image` is the homography from :func:`homography_from_keypoints`;
    `output_size` is (width, height) in template units.
    """
    width, height = int(output_size[0]), int(output_size[1])
    image_to_pitch = np.linalg.inv(np.asarray(pitch_to_image, dtype=np.float64))
    return cv2.warpPerspective(frame, image_to_pitch, (width, height))
