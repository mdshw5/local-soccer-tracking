"""Naming a team by the colour of its kit.

The clustering numbers the teams 0 and 1, and "the more red one" is not something a coach can picture. These are
the helpers that turn the stored kit descriptor back into a colour and into a word, and the word into a name a
team could go by - the anchor every view of a match (table, chart, replay, event tags) then uses.
"""

from __future__ import annotations

import numpy as np
import pytest

from soccer_analytics.analysis.kit import (
    colour_hex,
    colour_name,
    kit_descriptor,
    kit_rgb,
    suggest_team_name,
)


def _shirt(bgr: tuple[int, int, int], size: int = 40) -> np.ndarray:
    """A crop-sized BGR image of one flat colour, like the torso of a plain kit."""
    frame = np.zeros((size, size, 3), dtype=np.uint8)
    frame[:, :] = bgr
    return frame


def test_a_descriptor_reads_back_as_the_colour_it_was_taken_from() -> None:
    """The round trip that makes the swatch and the name trustworthy: descriptor -> RGB -> word.

    Descriptors are stored as OpenCV's 8-bit Lab (L 0-255, a/b offset by 128), which is easy to read back with the
    wrong offset - that mistake shifts every kit towards green and would have the page call a red team "grey".
    """
    cases = [
        ((30, 30, 200), "blue"),      # RGB for each case; the crop is built in BGR, as OpenCV reads it
        ((200, 30, 30), "red"),
        ((235, 235, 235), "white"),
        ((20, 20, 20), "black"),
        ((30, 160, 30), "green"),
    ]
    for rgb, expected in cases:
        descriptor = kit_descriptor(_shirt(rgb[::-1]), (0, 0, 40, 40), None)
        back = kit_rgb(descriptor)
        assert back is not None, f"no colour came back for {rgb}"
        # The mean of a flat crop is the crop's own colour, give or take Lab rounding.
        assert back == pytest.approx(rgb, abs=14), f"{rgb} read back as RGB {back}"
        assert colour_name(back) == expected, f"{rgb} was called {colour_name(back)!r}"


def test_a_descriptor_without_a_kit_has_no_colour() -> None:
    """A crop that was all grass leaves the colour fields at zero; that must not read back as black."""
    empty = np.zeros(12, dtype=np.float32)
    assert kit_rgb(empty) is None
    assert kit_rgb(None) is None
    assert colour_hex(None) == ""


def test_colour_name_keeps_the_lightness_that_tells_two_kits_apart() -> None:
    """A dark blue and a light blue kit are different teams, so the word has to carry the lightness."""
    assert colour_name((10, 10, 90)) == "dark blue"       # value below 110 is called dark
    assert colour_name((150, 200, 250)) == "light blue"   # bright and only lightly saturated is light
    assert colour_name((30, 30, 200)) == "blue"           # a strong mid blue stays plain "blue"
    assert colour_name((40, 40, 40)) == "black"
    assert colour_name((250, 250, 250)) == "white"
    assert colour_name((150, 150, 150)) == "grey"
    assert colour_name((200, 200, 30)) == "yellow"


def test_colour_hex_is_what_a_colour_picker_wants() -> None:
    assert colour_hex((255, 0, 128)) == "#ff0080"
    assert colour_hex((0, 0, 0)) == "#000000"
    # Out-of-range channels are clamped rather than producing "#-1..." or a nine-character string.
    assert colour_hex((300, -20, 10)) == "#ff000a"


def test_suggested_team_names_read_like_teams_not_like_colours() -> None:
    """The suggestion fills an editable field, so it aims at what someone on the touchline would say."""
    assert suggest_team_name((200, 30, 30)) == "Reds"
    assert suggest_team_name((10, 10, 90)) == "Dark blues"
    assert suggest_team_name((150, 200, 250)) == "Light blues"
    assert suggest_team_name((250, 250, 250)) == "Whites"
    assert suggest_team_name((40, 40, 40)) == "Blacks"
    # Two different kits must not be suggested the same name: the lightness is part of the name.
    dark = suggest_team_name((10, 10, 90))
    light = suggest_team_name((150, 200, 250))
    assert dark != light
