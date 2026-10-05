"""The 32-point pitch keypoint template, in metres, ported from the reference project.

The reference (`whisdev/soccer-video-detection-ai-agent`) detects 32 field markers with an HRNet heatmap model and
matches them to a fixed template image (`TEMPLATE_F0`). That template is a scaled diagram of a full-size pitch, so
the part of it worth keeping is the *geometry*: which marker each index is, and where it sits in metres. This module
is that geometry, expressed directly in metres rather than in template pixels, so the keypoints can be handed to
`solver.calibrate` as landmarks on a pitch of any format.

Index order is deliberately the reference's order. If a user ever drops the reference's `keypoint_detect.pt` into
`data/models/`, the channel map it ships with (`map_keypoints` in the reference's `agent.py`) already lines its
outputs up with these indices - renumbering them here would silently pair the wrong marker with the wrong pitch
point.

Geometry (a 105x68 pitch; `template_for` rescales to any format):

* 0-5    left goal line, top to bottom: corner, penalty-area corner, goal-area corner, goal-area corner,
         penalty-area corner, corner
* 6-7    left goal-area corners, 5.5 m out
* 8      left penalty spot
* 9-12   left penalty area: top corner, the two where the penalty arc crosses the area edge, bottom corner
* 13-16  halfway line: top touchline, centre-circle top, centre-circle bottom, bottom touchline
* 17-23  right-hand mirror of 9-8: penalty-area corners/arc and the right penalty spot
* 22-23  right goal-area corners
* 24-29  right goal line, top to bottom (mirror of 0-5)
* 30-31  the centre circle's west and east points, on the halfway line
"""

from __future__ import annotations

Tuple = tuple[float, float]

STANDARD_LENGTH_M = 105.0
STANDARD_WIDTH_M = 68.0

# FIFA markings, in metres, on the standard pitch. Kept named so the template reads as the pitch it describes
# rather than as a list of magic numbers.
_PENALTY_AREA_DEPTH = 16.5
_PENALTY_AREA_HALF_WIDTH = 20.16
_GOAL_AREA_DEPTH = 5.5
_GOAL_AREA_HALF_WIDTH = 9.16
_PENALTY_SPOT_FROM_LINE = 11.0
_CENTRE_CIRCLE_RADIUS = 9.15


def _normalised_template() -> tuple[Tuple, ...]:
    """The 32 markers as fractions of the standard pitch, so any format is one multiplication away."""
    length, width = STANDARD_LENGTH_M, STANDARD_WIDTH_M
    mid = 0.5
    ea_top = (width / 2 - _PENALTY_AREA_HALF_WIDTH) / width
    ea_bottom = (width / 2 + _PENALTY_AREA_HALF_WIDTH) / width
    ga_top = (width / 2 - _GOAL_AREA_HALF_WIDTH) / width
    ga_bottom = (width / 2 + _GOAL_AREA_HALF_WIDTH) / width
    # Where the penalty arc (radius = centre circle, centred on the spot) crosses the penalty-area edge.
    dx = _PENALTY_AREA_DEPTH - _PENALTY_SPOT_FROM_LINE
    arc_half = (_CENTRE_CIRCLE_RADIUS**2 - dx**2) ** 0.5
    arc_top = (width / 2 - arc_half) / width
    arc_bottom = (width / 2 + arc_half) / width
    penalty_x = _PENALTY_AREA_DEPTH / length
    goal_x = _GOAL_AREA_DEPTH / length
    spot_x = _PENALTY_SPOT_FROM_LINE / length
    circle_x = _CENTRE_CIRCLE_RADIUS / length
    circle_y = _CENTRE_CIRCLE_RADIUS / width
    cc_top = mid - circle_y
    cc_bottom = mid + circle_y
    return (
        (0.0, 0.0),  # 0  corner, top-left
        (0.0, ea_top),  # 1  penalty-area corner
        (0.0, ga_top),  # 2  goal-area corner
        (0.0, ga_bottom),  # 3
        (0.0, ea_bottom),  # 4
        (0.0, 1.0),  # 5  corner, bottom-left
        (goal_x, ga_top),  # 6
        (goal_x, ga_bottom),  # 7
        (spot_x, mid),  # 8  penalty spot
        (penalty_x, ea_top),  # 9
        (penalty_x, arc_top),  # 10
        (penalty_x, arc_bottom),  # 11
        (penalty_x, ea_bottom),  # 12
        (mid, 0.0),  # 13 halfway, top touchline
        (mid, cc_top),  # 14 centre circle, top
        (mid, cc_bottom),  # 15 centre circle, bottom
        (mid, 1.0),  # 16 halfway, bottom touchline
        (1.0 - penalty_x, ea_top),  # 17 right penalty-area corner
        (1.0 - penalty_x, arc_top),  # 18
        (1.0 - penalty_x, arc_bottom),  # 19
        (1.0 - penalty_x, ea_bottom),  # 20
        (1.0 - spot_x, mid),  # 21 right penalty spot
        (1.0 - goal_x, ga_top),  # 22
        (1.0 - goal_x, ga_bottom),  # 23
        (1.0, 0.0),  # 24 corner, top-right
        (1.0, ea_top),  # 25
        (1.0, ga_top),  # 26
        (1.0, ga_bottom),  # 27
        (1.0, ea_bottom),  # 28
        (1.0, 1.0),  # 29 corner, bottom-right
        (mid - circle_x, mid),  # 30 centre circle, west
        (mid + circle_x, mid),  # 31 centre circle, east
    )


NORMALISED_TEMPLATE: tuple[Tuple, ...] = _normalised_template()
KEYPOINT_COUNT = len(NORMALISED_TEMPLATE)


def template_for(length_m: float = STANDARD_LENGTH_M, width_m: float = STANDARD_WIDTH_M) -> list[Tuple]:
    """The 32 keypoints in metres for a pitch of this format.

    Independent scaling on each axis, which is how the project's other landmarks are derived too: a 9v9 pitch is a
    smaller pitch, not the same pitch cropped. It also means the template is correct at the standard 105x68 exactly,
    since the normalised points came from there.
    """
    if length_m <= 0 or width_m <= 0:
        raise ValueError("pitch length and width must be positive")
    return [(u * length_m, v * width_m) for u, v in NORMALISED_TEMPLATE]