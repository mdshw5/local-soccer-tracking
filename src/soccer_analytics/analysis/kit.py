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

# --------------------------------------------------------------------------------------------------------------
# Naming a team by its kit
# --------------------------------------------------------------------------------------------------------------
# The clustering order is already deterministic - the more red kit is team 0 - but nobody says "team 0". What a
# person needs is which *colour* each team wears, so the stored descriptor is turned back into a colour here, and
# into the word a coach would use for it. The names are the anchor for naming teams: "Reds" means something
# against the colour, an index does not.
# Hue bands in OpenCV's 0-179 scale; each band's upper edge is where the name after it starts.
_HUE_BANDS = (
    (10, "red"), (22, "orange"), (33, "yellow"), (85, "green"),
    (100, "teal"), (126, "blue"), (151, "purple"), (170, "pink"), (180, "red"),
)
GREY_SATURATION = 40  # below this a kit has no colour to name, only a lightness


def kit_rgb(descriptor: np.ndarray | None) -> tuple[int, int, int] | None:
    """The mean kit colour a descriptor describes, as RGB, or None when it holds no kit.

    The descriptor keeps lightness and two colour axes (Lab) instead of raw pixels so that a team's colour can be
    summarised in a handful of floats; this is that summary read back out, for showing a swatch and for naming.
    """
    if descriptor is None or len(descriptor) < 4 or float(descriptor[0]) <= 0.0:
        return None
    lab = np.clip(np.asarray(descriptor[1:4], dtype=np.float32) * 255.0, 0.0, 255.0).astype(np.uint8)
    bgr = cv2.cvtColor(lab.reshape(1, 1, 3), cv2.COLOR_LAB2BGR)[0, 0]
    return int(bgr[2]), int(bgr[1]), int(bgr[0])


def colour_hex(rgb: tuple[int, int, int] | None) -> str:
    """``#rrggbb`` for a colour, or the empty string when there is none."""
    if rgb is None:
        return ""
    return "#{:02x}{:02x}{:02x}".format(*(int(np.clip(channel, 0, 255)) for channel in rgb))


def colour_name(rgb: tuple[int, int, int]) -> str:
    """What a person would call this colour: ``"red"``, ``"dark blue"``, ``"white"``...

    Kits are mostly the flat colours this gets right - red, white, black, yellow, blue, green - with lightness
    carried in the word ("dark", "light") rather than left out, because a dark blue and a light blue kit on the
    same pitch are two different teams.
    """
    pixel = np.uint8([[[int(np.clip(c, 0, 255)) for c in rgb]]])
    hue, saturation, value = (int(v) for v in cv2.cvtColor(pixel, cv2.COLOR_RGB2HSV)[0, 0])
    if saturation < GREY_SATURATION:
        if value >= 215:
            return "white"
        if value >= 160:
            return "light grey"
        return "grey" if value >= 90 else "black"
    base = next(name for edge, name in _HUE_BANDS if hue < edge)
    if value < 110:
        return f"dark {base}"
    if value > 215 and saturation < 120:
        return f"light {base}"
    return base


def suggest_team_name(rgb: tuple[int, int, int]) -> str:
    """A name a team could go by, from the colour of its kit: ``"Reds"``, ``"Dark blues"``, ``"Whites"``.

    This is a starting point for a field the user can edit, so it aims at what someone on the touchline would say
    rather than at being clever. Plural because that is how teams are named in the game ("the Blues"), and the
    colour word is kept whole - "light greys" - because a dark blue and a light blue kit on one pitch are two
    different teams, and a name that drops the lightness would describe both of them.
    """
    word = colour_name(rgb)
    return f"{word[0].upper()}{word[1:]}s"


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
