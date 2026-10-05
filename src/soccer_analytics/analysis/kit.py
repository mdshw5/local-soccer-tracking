"""Compact per-detection kit descriptor, small enough to store for every detection of a match.

`tracking.team_classifier.kit_colour_histogram` is a 256-bin HSV histogram; storing it for ~300k detections would be
hundreds of MB. This keeps what team clustering actually needs in 12 floats.
"""

from __future__ import annotations

import cv2
import numpy as np

from soccer_analytics.tracking.team_classifier import MIN_SATURATION, MIN_VALUE, grass_hue_window, torso_crop

DESCRIPTOR_SIZE = 12
HUE_BINS = 6
MIN_CROP_PIXELS = 24
# Descriptor layout, for readers of the stored arrays.
DESCRIPTOR_FIELDS = (
    "kit_fraction", "L", "a", "b", "saturation", "value",
    *(f"hue_{i}" for i in range(HUE_BINS)),
)


def frame_grass_window(frame: np.ndarray) -> tuple[int, int] | None:
    """Grass hue window for a frame, estimated on a thumbnail (it is the same answer, ~20x cheaper)."""
    scale = 320.0 / max(frame.shape[1], 1)
    thumb = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) if scale < 1 else frame
    return grass_hue_window(thumb)


def kit_descriptor(
    frame: np.ndarray, bbox: tuple[float, float, float, float], grass_hues: tuple[int, int] | None
) -> np.ndarray:
    """12-float descriptor of the torso colour with grass masked out; zeros if the crop is unusable."""
    out = np.zeros(DESCRIPTOR_SIZE, dtype=np.float32)
    crop = torso_crop(frame, bbox)
    if crop is None or crop.shape[0] * crop.shape[1] < MIN_CROP_PIXELS:
        return out
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    kit = np.ones(crop.shape[:2], dtype=bool)
    if grass_hues is not None:
        grass = cv2.inRange(
            hsv, np.array([grass_hues[0], MIN_SATURATION, MIN_VALUE]), np.array([grass_hues[1], 255, 255])
        ).astype(bool)
        kit &= ~grass
    fraction = float(kit.mean())
    out[0] = fraction
    if kit.sum() < MIN_CROP_PIXELS // 2:
        return out  # all grass-coloured: leave the colour fields at zero so it never looks like a kit
    lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB).astype(np.float32)
    out[1:4] = lab[kit].mean(axis=0) / 255.0
    out[4] = hsv[..., 1][kit].mean() / 255.0
    out[5] = hsv[..., 2][kit].mean() / 255.0
    hue = hsv[..., 0][kit].astype(np.float32)
    weight = hsv[..., 1][kit].astype(np.float32) / 255.0  # grey kit pixels carry no hue information
    hist, _ = np.histogram(hue, bins=HUE_BINS, range=(0, 180), weights=weight)
    total = hist.sum()
    out[6:] = hist / total if total > 1e-6 else 0.0
    return out
