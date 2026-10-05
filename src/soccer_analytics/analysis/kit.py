"""Compact per-detection kit descriptor, small enough to store for every detection of a match.

`tracking.team_classifier.kit_colour_histogram` is a 256-bin HSV histogram; storing it for ~300k detections would be
hundreds of MB. This keeps what team clustering actually needs in 12 floats.
"""

from __future__ import annotations

import cv2
import numpy as np

from soccer_analytics.tracking.team_classifier import (
    MIN_SATURATION,
    MIN_VALUE,
    GrassModel,
    measure_grass,
    torso_crop,
)

DESCRIPTOR_SIZE = 12
HUE_BINS = 6
MIN_CROP_PIXELS = 24
# Confidence stored for a crop that was *entirely* grass. See `kit_descriptor`: the two causes (no kit in view, and
# a genuinely green kit) are indistinguishable, so this is deliberately above the trust floor the tracker uses and
# deliberately low - a green kit should cluster with the other green kits rather than vanish, while never outvoting
# a clean reading.
ALL_GRASS_KIT_FRACTION = 0.25
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


def frame_grass_model(frame: np.ndarray) -> GrassModel | None:
    """The frame's grass, measured on a thumbnail (the per-tile bands need area, but not full resolution).

    Measured at 320 px wide, so a 4x3 tile is ~80 px across - ample for a few thousand grass pixels per tile, and
    twenty times cheaper. A tile that comes back unmeasured at that size borrows the frame-wide band, which is the
    correct answer for it anyway.
    """
    scale = 320.0 / max(frame.shape[1], 1)
    thumb = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) if scale < 1 else frame
    return measure_grass(thumb)


def _grass_mask_for(
    crop: np.ndarray,
    grass: GrassModel | tuple[int, int] | None,
    band: tuple[int, int] | None,
    hsv: np.ndarray,
    lab: np.ndarray,
) -> np.ndarray:
    """Boolean "this is grass" mask for a crop, from a model (spatial) or a plain hue pair (flat).

    ``hsv`` and ``lab`` are the crop's own conversions, passed in so the caller converts once and reuses both for the
    descriptor's colour statistics.
    """
    if isinstance(grass, GrassModel):
        return grass.mask(hsv, lab, band)
    if grass is None:
        return np.zeros(crop.shape[:2], dtype=bool)
    return cv2.inRange(
        hsv, np.array([grass[0], MIN_SATURATION, MIN_VALUE]), np.array([grass[1], 255, 255])
    ).astype(bool)


def kit_descriptor(
    frame: np.ndarray,
    bbox: tuple[float, float, float, float],
    grass: GrassModel | tuple[int, int] | None,
) -> np.ndarray:
    """12-float descriptor of the torso colour with grass masked out; zeros if the crop is unusable.

    The mask is taken with the grass *behind this player*: when ``grass`` is a :class:`GrassModel`, the band of the
    tile the box sits in is used rather than the frame-wide one, so a player on shaded turf is masked against
    shaded turf instead of against a band stretched to also cover the sunlit half of the pitch.

    A crop that comes back entirely grass gets its colour measured from the raw crop with a low confidence, rather
    than being left at zero. The two reasons for an all-grass crop cannot be told apart - the box may hold no kit at
    all (an occlusion, a badly drawn box) or the player may simply be wearing green - and leaving it at zero is
    not neutral between them: it removes the green team from the clustering entirely, where those players then land
    in whichever of the two real teams is nearer in grey space. Reporting them at low confidence keeps them with
    their own kind and lets `TrackAssignment.kit_quality` mark the split as weak.
    """
    out = np.zeros(DESCRIPTOR_SIZE, dtype=np.float32)
    crop = torso_crop(frame, bbox)
    if crop is None or crop.shape[0] * crop.shape[1] < MIN_CROP_PIXELS:
        return out
    band = grass.band_for(bbox, frame.shape) if isinstance(grass, GrassModel) else None
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB)
    kit = ~_grass_mask_for(crop, grass, band, hsv, lab)
    if int(kit.sum()) < MIN_CROP_PIXELS // 2:
        kit = np.ones(crop.shape[:2], dtype=bool)
        out[0] = ALL_GRASS_KIT_FRACTION
    else:
        out[0] = float(kit.mean())
    lab = lab.astype(np.float32)
    out[1:4] = lab[kit].mean(axis=0) / 255.0
    out[4] = hsv[..., 1][kit].mean() / 255.0
    out[5] = hsv[..., 2][kit].mean() / 255.0
    hue = hsv[..., 0][kit].astype(np.float32)
    weight = hsv[..., 1][kit].astype(np.float32) / 255.0  # grey kit pixels carry no hue information
    hist, _ = np.histogram(hue, bins=HUE_BINS, range=(0, 180), weights=weight)
    total = hist.sum()
    out[6:] = hist / total if total > 1e-6 else 0.0
    return out
