"""The spatial grass model: per-tile hue bands, bleached turf, and what happens to a green kit.

`grass_hue_window` answers "what hue is this frame's grass" with one band, and it has to stretch that band to cover
every lighting mode in the picture. `measure_grass` answers the same question once per tile, so the band used to mask
a player is the band of the grass actually behind them. These tests pin the parts that differ from the single-band
behaviour, because those are the parts that change a stored kit descriptor.
"""

from __future__ import annotations

import cv2
import numpy as np

from soccer_analytics.analysis.kit import ALL_GRASS_KIT_FRACTION, kit_descriptor, kit_rgb
from soccer_analytics.analysis.stage_b import MIN_KIT_FOR_COST
from soccer_analytics.tracking.team_classifier import (
    GRASS_TILES,
    WEAK_GRASS_SATURATION,
    _grass_mask,
    grass_hue_window,
    measure_grass,
)

RED_BGR = (0, 0, 255)


def _grass(hue: int, height: int, width: int, saturation: int = 190, value: int = 150) -> np.ndarray:
    hsv = np.full((height, width, 3), (hue, saturation, value), dtype=np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def _two_mode_pitch() -> np.ndarray:
    """240x240: the top two tile-rows are shaded turf (hue 34), the bottom row is sunlit (hue 62)."""
    frame = np.zeros((240, 240, 3), dtype=np.uint8)
    frame[:160] = _grass(34, 160, 240)
    frame[160:] = _grass(62, 80, 240)
    return frame


def test_frame_wide_band_stays_the_single_band_answer():
    """Backwards compatibility that matters: the stored-descriptor comparison is against this number."""
    frame = _two_mode_pitch()
    assert grass_hue_window(frame) == measure_grass(frame).window


def test_each_tile_measures_its_own_lighting_mode():
    """The point of the model: a shaded tile gets a shaded band, not one stretched to also cover the sunlit half.

    On the single-band answer a shaded player is masked with a band ~50 hue wide, which is wide enough to eat a
    greenish away kit standing on that shaded turf. Locally it is ~20 wide.
    """
    model = measure_grass(_two_mode_pitch())
    assert model is not None
    rows, cols = GRASS_TILES[1], GRASS_TILES[0]
    assert model.tiles.shape == (rows, cols, 2)
    shaded = model.tiles[0, 0]
    sunlit = model.tiles[rows - 1, 0]
    assert shaded[1] < 50, f"shaded tile band reached into the sunlit mode: {shaded}"
    assert sunlit[0] > 40, f"sunlit tile band reached into the shaded mode: {sunlit}"
    assert (shaded[1] - shaded[0]) < (model.hue_hi - model.hue_lo), "no tightening over the frame-wide band"
    assert model.tile_measured.all(), "every tile is pure grass here"


def test_the_band_used_is_the_band_under_the_player():
    frame = _two_mode_pitch()
    model = measure_grass(frame)
    assert model is not None
    top = model.band_for((100.0, 20.0, 140.0, 80.0), frame.shape)  # box in the shaded tiles
    bottom = model.band_for((100.0, 180.0, 140.0, 220.0), frame.shape)  # box in the sunlit tile
    assert top == tuple(int(v) for v in model.tiles[0, 0])
    assert bottom == tuple(int(v) for v in model.tiles[GRASS_TILES[1] - 1, 0])
    assert top != bottom


def test_a_tile_without_enough_grass_borrows_the_frame_wide_band():
    """A tile of stand and sky has no grass worth measuring; it must not invent a narrow band from a few pixels."""
    frame = _two_mode_pitch()
    frame[:80] = 200  # the whole top tile-row is a concrete stand
    model = measure_grass(frame)
    assert model is not None
    assert not model.tile_measured[0, 0]
    assert model.band_for((0.0, 0.0, 40.0, 40.0), frame.shape) == model.window


def test_a_greenish_kit_survives_the_local_band_that_the_frame_wide_band_would_have_eaten():
    """The concrete win of a tight band, at the size that actually matters: a small figure in a big tile.

    A green advertising hoarding elsewhere in the frame pushes the frame-wide band's upper edge to 80, so a teal-ish
    away kit at hue 70 reads as grass and is thrown away. The tile the player stands on holds ordinary turf, and the
    distant figure is far too small a part of that tile to move its percentiles, so its band stops near 55 and the
    kit is measured normally. This is the size a real far-side player is: ~0.1% of the tile, not a third of it.
    """
    frame = np.zeros((480, 480, 3), dtype=np.uint8)
    frame[:, :] = _grass(45, 480, 480)
    frame[0:160, 360:480] = _grass(80, 160, 120)  # hoarding in the top-right tile: 8% of the frame
    frame[200:212, 150:160] = _grass(70, 12, 10)  # a distant teal figure, 25 px of a 19,200 px tile

    model = measure_grass(frame)
    assert model is not None
    assert model.hue_hi > 70, "the hoarding should have stretched the frame-wide band past the kit hue"
    box = (150.0, 200.0, 160.0, 212.0)
    local = model.band_for(box, frame.shape)
    assert local[1] < 70, f"the local band still covers the kit hue: {local}"

    with_model = kit_descriptor(frame, box, model)
    flat = kit_descriptor(frame, box, model.window)
    assert with_model[0] > ALL_GRASS_KIT_FRACTION, "the local band threw the kit away"
    assert flat[0] == ALL_GRASS_KIT_FRACTION, "the frame-wide band was expected to eat this kit"
    assert kit_rgb(with_model) is not None and kit_rgb(flat) is not None
    # The fallback keeps the same pixels, so the colour is the same; what differs is how much of the crop was left
    # after masking, which is the confidence the tracker and the clustering actually weigh.
    assert with_model[0] > flat[0]


def test_bleached_turf_is_masked_even_though_it_fails_the_saturation_floor():
    """Worn, sun-bleached turf drops below MIN_SATURATION and used to leak into every kit colour on a bright day."""
    frame = _grass(45, 120, 120, saturation=WEAK_GRASS_SATURATION + 5)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    assert not np.any(_grass_mask(hsv)), "this turf is meant to sit below the saturation floor"

    model = measure_grass(frame)
    assert model is not None
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    assert model.mask(hsv, lab).mean() > 0.99, "bleached turf must still be masked"


def test_a_pale_nearly_neutral_shirt_is_not_mistaken_for_bleached_grass():
    """The faint-green test is proximity in Lab, not merely low saturation.

    A white shirt in shade picks up a slight green cast, so it passes the relaxed saturation floor and sits inside
    the hue band - but a white shirt lives near the neutral axis (a and b close to 128), nowhere near grass.
    """
    grass = _grass(45, 120, 120)
    model = measure_grass(grass)
    assert model is not None
    pale = np.zeros((40, 40, 3), dtype=np.uint8)
    pale[:, :] = cv2.cvtColor(
        np.full((40, 40, 3), (45, 25, 200), dtype=np.uint8), cv2.COLOR_HSV2BGR
    )
    hsv = cv2.cvtColor(pale, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(pale, cv2.COLOR_BGR2LAB)
    assert not model.mask(hsv, lab).any(), "a pale, nearly neutral shirt must survive the mask"


def test_an_all_green_crop_is_reported_weakly_rather_than_dropped():
    """A green kit and a box holding no kit are indistinguishable, so the kit must not simply disappear.

    Dropping it is not neutral: those players then land in whichever real team is nearer in grey space, or leave the
    team metrics entirely. The descriptor is kept at a confidence deliberately above the tracker's trust floor.
    """
    frame = _grass(45, 120, 120)
    model = measure_grass(frame)
    assert model is not None
    descriptor = kit_descriptor(frame, (10.0, 10.0, 110.0, 110.0), model)
    assert descriptor[0] == ALL_GRASS_KIT_FRACTION
    assert ALL_GRASS_KIT_FRACTION > MIN_KIT_FOR_COST, "the weak reading must stay usable for team clustering"
    assert np.any(descriptor[1:6] != 0.0), "the colour fields must carry the green, not stay at zero"


def test_a_plain_kit_reads_the_same_with_and_without_a_model():
    """No regression on the easy case: a red shirt filling the crop is a red shirt either way."""
    frame = np.zeros((120, 120, 3), dtype=np.uint8)
    frame[:, :] = RED_BGR
    flat = kit_descriptor(frame, (10.0, 10.0, 110.0, 110.0), None)
    model = measure_grass(_grass(45, 120, 120))
    assert model is not None
    built = kit_descriptor(frame, (10.0, 10.0, 110.0, 110.0), model)
    assert kit_rgb(flat) == kit_rgb(built)
