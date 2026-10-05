"""Pitch-landmark clicking: where to click, and how a click on a zoomed crop maps back to the frame.

This is deliberately kept out of the Streamlit page. The coordinate maths is the part of landmark registration that
is easy to get subtly wrong (and impossible to unit-test inside a Streamlit script), so it lives here as pure
functions and the page only wires it to the component.

Three coordinate systems meet here, and mixing them up is silent rather than loud:

* **native pixels** - the frame as decoded, e.g. 3840x2160. `zoom_box` and the click crops work in these.
* **frame-normalised** - ``u = x / frame_width`` and ``v = y / frame_width``. Note ``v`` is divided by the *width*,
  not the height: that is the convention the camera model and the solver use, and it is not the same as the fraction
  of the frame you see on screen.
* **canvas pixels** - what the browser component reports, after the crop has been shrunk to at most `CLICK_MAX_WIDTH`.
  `canvas_scale` converts between these and native pixels, in both directions, so a click lands where it was made.

The zoom sliders take a fraction of the frame *width* and *height* respectively, because that is what a user reads
off the picture in front of them.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass, field

import cv2
import numpy as np

from soccer_analytics.geometry.pitch_calibration import Landmark, PitchCalibration, pitch_to_pixels

CLICK_MAX_WIDTH = 1600  # canvas pixels sent to the browser; small crops go at native size, which gives the accuracy

# Standard pitch markings (metres), the same for every format. Kept in one place so the clickable landmarks and the
# outline drawn back onto the frame cannot drift apart.
GOAL_BOX_DEPTH_M = 5.5
GOAL_BOX_HALF_WIDTH_M = 5.5
PENALTY_BOX_DEPTH_M = 16.5
PENALTY_BOX_HALF_WIDTH_M = 20.16
PENALTY_SPOT_DISTANCE_M = 11.0
CENTRE_CIRCLE_RADIUS_M = 9.15


def landmark_table(length_m: float, width_m: float) -> dict[str, tuple[float, float]]:
    """Pitch landmarks a user can click, in the order they should be clicked.

    Corners first because they spread the fit across the pitch, then the goal centres - which matter because the near
    corners are often out of frame, and the goal mouth is a far easier thing to pick out than a corner flag. The
    halfway line and centre spot then pin the middle down.

    The rest are the markings that are *usually* visible and add spread where the corners cannot: the goal box (the
    six-yard box) and penalty box corners at each end, the penalty spots, and the four cardinals of the centre
    circle. They are the same standard markings for every format, so the same list works once the dimensions are
    known - and because they sit at a range of distances from the camera they are exactly the near landmarks the fit
    is short of when the near corners are out of shot. The goal box and penalty box are drawn from the goal line
    (``x = 0`` and ``x = length_m``) and the centre circle from the halfway line, so a click on any of them is a
    measurement of the pitch, not of the camera.
    """
    half_length = length_m / 2
    half_width = width_m / 2
    return {
        "corner near-left": (0.0, 0.0),
        "corner near-right": (length_m, 0.0),
        "corner far-right": (length_m, width_m),
        "corner far-left": (0.0, width_m),
        "goal centre left": (0.0, half_width),
        "goal centre right": (length_m, half_width),
        "halfway near": (half_length, 0.0),
        "halfway far": (half_length, width_m),
        "centre spot": (half_length, half_width),
        # Goal box (six-yard box): 5.5 m out from the goal line, 5.5 m either side of the goal centre.
        "goal box near-left": (0.0, half_width - GOAL_BOX_HALF_WIDTH_M),
        "goal box near-right": (0.0, half_width + GOAL_BOX_HALF_WIDTH_M),
        "goal box far-left": (length_m, half_width - GOAL_BOX_HALF_WIDTH_M),
        "goal box far-right": (length_m, half_width + GOAL_BOX_HALF_WIDTH_M),
        # Penalty box (18-yard box): 16.5 m out, 20.16 m either side of the goal centre.
        "penalty box near-left": (0.0, half_width - PENALTY_BOX_HALF_WIDTH_M),
        "penalty box near-right": (0.0, half_width + PENALTY_BOX_HALF_WIDTH_M),
        "penalty box far-left": (length_m, half_width - PENALTY_BOX_HALF_WIDTH_M),
        "penalty box far-right": (length_m, half_width + PENALTY_BOX_HALF_WIDTH_M),
        # Penalty spots: 11 m out from the goal line, on the goal centre line.
        "penalty spot left": (PENALTY_SPOT_DISTANCE_M, half_width),
        "penalty spot right": (length_m - PENALTY_SPOT_DISTANCE_M, half_width),
        # Centre circle cardinals: 9.15 m from the centre spot, on the halfway line and the centre line.
        "centre circle near": (half_length, half_width - CENTRE_CIRCLE_RADIUS_M),
        "centre circle far": (half_length, half_width + CENTRE_CIRCLE_RADIUS_M),
        "centre circle left": (half_length - CENTRE_CIRCLE_RADIUS_M, half_width),
        "centre circle right": (half_length + CENTRE_CIRCLE_RADIUS_M, half_width),
    }


LANDMARK_HELP: dict[str, str] = {
    "corner near-left": "Corner flag at the left-hand end, on the touchline nearest the camera.",
    "corner near-right": "Corner flag at the right-hand end, on the touchline nearest the camera.",
    "corner far-right": "Corner flag at the right-hand end, on the far touchline - usually the hardest one to click.",
    "corner far-left": "Corner flag at the left-hand end, on the far touchline.",
    "goal centre left": (
        "Middle of the goal mouth at ground level - where the goal line crosses between the posts - at the same end "
        "as the *-left corners. Use this when that corner flag is out of frame."
    ),
    "goal centre right": (
        "Middle of the goal mouth at ground level, at the same end as the *-right corners. Use this when that "
        "corner flag is out of frame."
    ),
    "halfway near": "Where the halfway line meets the near touchline.",
    "halfway far": "Where the halfway line meets the far touchline.",
    "centre spot": "The centre spot (or the centre of the centre circle).",
    "goal box near-left": (
        "The goal box (six-yard box) corner on the goal line, on the near side of the goal - where the goal line "
        "meets the short line of the six-yard box. Usually visible even when the corner flag is not."
    ),
    "goal box near-right": (
        "The goal box (six-yard box) corner on the goal line, on the far side of the goal - where the goal line "
        "meets the short line of the six-yard box."
    ),
    "goal box far-left": (
        "The goal box (six-yard box) corner on the goal line at the far end, on the near side of the goal."
    ),
    "goal box far-right": (
        "The goal box (six-yard box) corner on the goal line at the far end, on the far side of the goal."
    ),
    "penalty box near-left": (
        "The penalty box (18-yard box) corner on the goal line, on the near side of the goal - where the goal line "
        "meets the long line of the penalty box. A good near landmark when the corner flag is out of shot."
    ),
    "penalty box near-right": (
        "The penalty box (18-yard box) corner on the goal line, on the far side of the goal."
    ),
    "penalty box far-left": (
        "The penalty box (18-yard box) corner on the goal line at the far end, on the near side of the goal."
    ),
    "penalty box far-right": (
        "The penalty box (18-yard box) corner on the goal line at the far end, on the far side of the goal."
    ),
    "penalty spot left": "The penalty spot at the left-hand end - 11 m out from the goal line, on the goal centre line.",
    "penalty spot right": "The penalty spot at the right-hand end - 11 m out from the goal line, on the goal centre line.",
    "centre circle near": "Where the centre circle crosses the centre line, on the near side of the halfway line.",
    "centre circle far": "Where the centre circle crosses the centre line, on the far side of the halfway line.",
    "centre circle left": "Where the centre circle crosses the halfway line, on the left-hand side of the centre spot.",
    "centre circle right": "Where the centre circle crosses the halfway line, on the right-hand side of the centre spot.",
}


def pitch_landmark_order(length_m: float, width_m: float) -> list[str]:
    """The click order, which is also the order the landmark dropdowns default to."""
    return list(landmark_table(length_m, width_m))


def zoom_box(width: int, height: int, zoom: float, centre_x: float, centre_y: float) -> tuple[int, int, int, int]:
    """The native-pixel crop the clicker shows: ``(x0, y0, w, h)``, always inside the frame.

    Zooming matters more than it sounds. A full 4K frame shown at 1600 px means one screen pixel is ~2.4 native
    pixels, and on the far side of the pitch that is roughly a metre of ground error per pixel. At zoom 8 a screen
    pixel is a native pixel, so the click is as good as the user's eye.
    """
    w = max(64, int(round(width / zoom)))
    h = max(64, int(round(w * height / width)))
    x0 = int(round(centre_x * width - w / 2))
    y0 = int(round(centre_y * height - h / 2))
    return max(0, min(width - w, x0)), max(0, min(height - h, y0)), w, h


def actual_centre(box: tuple[int, int, int, int], frame_size: tuple[int, int]) -> tuple[float, float]:
    """Where the crop really ended up, as fractions of frame width and height.

    This is the same as the requested centre unless the crop was clamped at a frame edge, and the difference is what
    the indicator in the component reports back to the user.
    """
    x0, y0, w, h = box
    frame_width, frame_height = frame_size
    return (x0 + w / 2) / frame_width, (y0 + h / 2) / frame_height


def clamp01(value: float) -> float:
    """Keep a 0-1 fraction inside its range, whatever the browser sends back."""
    return float(min(1.0, max(0.0, value)))


def point_centre(u: float, v: float, frame_size: tuple[int, int]) -> tuple[float, float]:
    """Where a stored landmark click sits for the view's centre: x by width, y by height.

    A click's ``v`` is normalised by the frame *width* (the solver's convention), while the view's centre is the
    fraction of the frame you see across and down. The vertical term is therefore converted exactly once, here -
    which is what makes "jump to this landmark" land on it instead of a fraction of an aspect ratio away.
    """
    frame_width, frame_height = frame_size
    return (float(u), float(v) * frame_width / frame_height)


def canvas_scale(crop_width: int) -> float:
    """How much a crop is shrunk before being sent to the browser.

    Used for both directions of the mapping, so a stored click lands exactly where it was clicked when it is drawn
    back onto the crop.
    """
    return min(1.0, CLICK_MAX_WIDTH / crop_width)


def frame_to_canvas(
    u: float, v: float, scale: float, box: tuple[int, int, int, int], frame_width: int
) -> tuple[float, float]:
    """Frame-normalised position -> canvas pixels of the current crop."""
    x0, y0, _w, _h = box
    return (u * frame_width - x0) * scale, (v * frame_width - y0) * scale


def canvas_to_frame(
    x: float, y: float, scale: float, box: tuple[int, int, int, int], frame_width: int
) -> tuple[float, float]:
    """Canvas pixels of the current crop -> frame-normalised position (both axes by width, as the solver expects)."""
    x0, y0, _w, _h = box
    return (x0 + x / scale) / frame_width, (y0 + y / scale) / frame_width


def inside_crop(u: float, v: float, box: tuple[int, int, int, int], scale: float, frame_width: int) -> bool:
    """Whether a frame-normalised position falls inside the current crop (so a marker there is draggable)."""
    x, y = frame_to_canvas(u, v, scale, box, frame_width)
    _x0, _y0, w, h = box
    return 0 <= x <= w * scale and 0 <= y <= h * scale


def marker_feature(label: str, u: float, v: float, box: tuple[int, int, int, int], scale: float, frame_width: int) -> dict:
    """One placed marker as a component feature: canvas position, the landmark it stands for, and an id.

    The id is what lets the component tell a *moved* marker from a stale copy of it: the positions Python sends
    only change when an Apply commits them, so anything the user has dragged since is a measurement in progress
    and must survive the reruns that other gestures (aiming, zooming) cause.
    """
    x, y = frame_to_canvas(u, v, scale, box, frame_width)
    return {"type": "point", "point": [x, y], "label": label, "id": f"seed:{label}"}


def merge_clicked(
    stored: list[dict],
    clicked: list[tuple[float, float] | tuple[float, float, str]],
    frame_index: int,
    box: tuple[int, int, int, int],
    scale: float,
    frame_width: int,
    next_pid: int,
    tolerance_px: float = 3.0,
) -> tuple[list[dict], int]:
    """Fold one Apply into the stored clicks.

    Apply commits what is on screen, so it replaces the clicks of this frame that lie inside the current crop and
    leaves the ones elsewhere alone. That is what lets a corner be clicked at zoom 8 and a different corner of the
    same frame be clicked at another zoom, without either wiping the other.

    A click that comes back where it was keeps its ``pid``. That matters well beyond tidiness: the landmark
    dropdowns are keyed by ``pid``, so re-creating the points on every Apply would silently throw away the landmark
    each click had been tagged as, and the tags would appear to revert on their own.

    A click may carry a third element: the landmark the marker picker had selected when it was made. That becomes
    the point's ``label``, the default the dropdown below the canvas opens on. A click without one keeps whatever
    its ``pid`` already had, so re-committing the crop cannot strip a tag off a point.
    """
    x0, y0, w, h = box

    def inside(point: dict) -> bool:
        x, y = point["u"] * frame_width, point["v"] * frame_width
        return x0 - 1 <= x <= x0 + w + 1 and y0 - 1 <= y <= y0 + h + 1

    kept = [point for point in stored if not (point["frame"] == frame_index and inside(point))]
    # The clicks this Apply is re-reporting, paired with where they sat on the canvas.
    reusable = [
        (frame_to_canvas(point["u"], point["v"], scale, box, frame_width), point)
        for point in stored
        if point["frame"] == frame_index and inside(point)
    ]
    merged = list(kept)
    for click in clicked:
        x, y = click[0], click[1]
        picked = click[2] if len(click) > 2 else ""
        same = next(
            (
                index
                for index, ((px, py), _point) in enumerate(reusable)
                if math.hypot(px - x, py - y) <= tolerance_px
            ),
            None,
        )
        if same is None:
            pid, previous = next_pid, ""
            next_pid += 1
        else:
            point = reusable.pop(same)[1]
            pid, previous = point["pid"], point.get("label", "")
        u, v = canvas_to_frame(x, y, scale, box, frame_width)
        entry = {"pid": pid, "frame": frame_index, "u": u, "v": v}
        if picked or previous:
            entry["label"] = picked or previous
        merged.append(entry)
    return merged, next_pid


def points_in_crop(
    stored: list[dict], frame_index: int, box: tuple[int, int, int, int], scale: float, frame_width: int
) -> list[dict]:
    """The stored clicks of one frame that are visible in the current crop, in the component's canvas coordinates."""
    x0, y0, w, h = box
    features = []
    for point in stored:
        if point["frame"] != frame_index:
            continue
        px, py = frame_to_canvas(point["u"], point["v"], scale, box, frame_width)
        if 0 <= px <= w * scale and 0 <= py <= h * scale:
            feature = {"type": "point", "point": [px, py], "id": f"click:{point['pid']}"}
            if point.get("label"):
                # The tag rides with the click, so an Apply cannot strip a landmark off a point.
                feature["label"] = point["label"]
            features.append(feature)
    return features


def landmarks_from_points(
    labelled: list[tuple[dict, str]], table: dict[str, tuple[float, float]]
) -> list[Landmark]:
    """Turn labelled clicks into solver landmarks, each carrying the frame it was clicked on."""
    return [
        Landmark(point["frame"], point["u"], point["v"], table[label][0], table[label][1], label)
        for point, label in labelled
    ]


def order_clicks(stored: list[dict]) -> list[dict]:
    """Clicks in a stable display order: by frame, then by the order they were clicked."""
    return sorted(stored, key=lambda point: (point["frame"], point["pid"]))


def frame_points(stored: list[dict], frame_index: int) -> list[tuple[float, float]]:
    """Every click on one frame as ``(u, v)``, so the whole-frame view can show the work in progress."""
    return [(point["u"], point["v"]) for point in order_clicks(stored) if point["frame"] == frame_index]


def projected_landmarks(
    calibration: PitchCalibration,
    labels: list[str],
    table: dict[str, tuple[float, float]],
    q: np.ndarray,
    focal: float,
) -> list[dict]:
    """Where the calibration puts the named landmarks on this frame, as frame-normalised ``(u, v)``.

    These are the *predicted* positions - where the current fit believes each marking is. Their whole use is the
    re-anchoring workflow: place them as markers, drag each onto the real marking, and the drags measure exactly
    how far the fit has slid by this moment in the video, which is what the drift correction is fitted from.
    """
    names = [name for name in labels if name in table]
    if not names:
        return []
    xy = np.array([table[name] for name in names], dtype=np.float64)
    uv, in_front = pitch_to_pixels(calibration, xy, q, focal)
    out: list[dict] = []
    for name, point, front in zip(names, uv, in_front):
        if not front or not np.isfinite(point).all():
            continue
        out.append({"label": name, "u": float(point[0]), "v": float(point[1])})
    return out


def restored_points(saved: list[dict], frame_count: int) -> list[dict] | None:
    """Session clicks rebuilt from the clicks saved with a calibration, or None when they cannot be used.

    A page refresh starts a new Streamlit session, so without this the committed clicks only exist in the match
    record while the editor shows an empty crop. Frames are indices into the segment that was clicked on, though -
    a set saved against the whole-game window means nothing on a five-minute one - so a set with any frame outside
    the current segment is refused rather than renumbered into nonsense. Pids are handed out in order, exactly as
    the editor would have.
    """
    points: list[dict] = []
    for index, click in enumerate(saved):
        try:
            frame, u, v = int(click["frame"]), float(click["u"]), float(click["v"])
        except (KeyError, TypeError, ValueError):
            return None
        if not 0 <= frame < max(1, frame_count):
            return None
        point = {"pid": index, "frame": frame, "u": u, "v": v}
        if click.get("label"):
            point["label"] = str(click["label"])
        points.append(point)
    return points or None


@dataclass
class ClickResult:
    """What the component last reported.

    ``action`` separates the two things a gesture in the component can mean: ``apply`` commits the points on screen,
    and ``navigate`` moved, zoomed or re-framed the view (a click, scroll or pan on the whole frame, which is also
    what loads a scrubbed-to frame into the crop). Neither may touch the stored clicks - that is what ``action`` is
    for. Scrubbing the timeline itself reports nothing at all: it is handled entirely in the browser.

    Each entry of ``points`` is ``(x, y, label)``: canvas pixels, plus the landmark the marker picker had selected
    when it was clicked (empty when the picker was left on automatic).
    """

    points: list[tuple[float, float, str]] = field(default_factory=list)
    centre: tuple[float, float] | None = None
    zoom: float | None = None
    action: str = ""
    frame: int | None = None


def frame_change(result_frame: int | None, reference: int, frame_count: int) -> int | None:
    """The clamped frame a gesture moved the timeline to, or ``None`` when it did not move it.

    ``None`` is also the answer for "no timeline at all", since the component sends ``null`` while the viewport is
    running on a still frame. Returning ``None`` for "no change" is what stops a rerun loop: a component's value is
    sticky and comes back on every later rerun, so a gesture may only be acted on while it still means a change.
    """
    if result_frame is None:
        return None
    clamped = int(np.clip(result_frame, 0, max(0, frame_count - 1)))
    current = int(np.clip(reference, 0, max(0, frame_count - 1)))
    return None if clamped == current else clamped


def parse_result(value: object) -> ClickResult:
    """Read the value a component gesture reports, as a :class:`ClickResult`.

    The component always reports the whole of its state - the points, the view and the frame - so this is the one
    place that knows the wire format. It is kept here, and pure, because the parsing is easy to get silently wrong:
    a missing ``features`` key reads as "no points", which is indistinguishable from the user having cleared the
    crop, and in Point mode the clicks live in ``features`` rather than ``polygon``.
    """
    if not isinstance(value, dict):
        return ClickResult()
    points: list[tuple[float, float, str]] = []
    for feature in value.get("features") or []:
        if not isinstance(feature, dict) or feature.get("type") != "point":
            continue
        point = feature.get("point") or []
        if len(point) == 2:
            label = feature.get("label")
            points.append((float(point[0]), float(point[1]), str(label) if label else ""))
    centre = None
    if isinstance(value.get("centre_x"), (int, float)) and isinstance(value.get("centre_y"), (int, float)):
        centre = (clamp01(value["centre_x"]), clamp01(value["centre_y"]))
    zoom = float(value["zoom"]) if isinstance(value.get("zoom"), (int, float)) else None
    frame = int(value["frame"]) if isinstance(value.get("frame"), (int, float)) else None
    return ClickResult(
        points=points, centre=centre, zoom=zoom, action=str(value.get("action") or ""), frame=frame
    )


def default_labels(count: int, length_m: float, width_m: float) -> list[str]:
    """Suggested label for the nth click: the landmarks in click order, and the last one repeats if there are more."""
    order = pitch_landmark_order(length_m, width_m)
    return [order[min(index, len(order) - 1)] for index in range(count)]


def per_anchor_labels(frames: list[int], length_m: float, width_m: float) -> list[str]:
    """Suggested label per click, restarting the landmark loop at every new frame.

    The suggestion restarts per moment because the workflow is per moment: click the landmarks you can see on one
    frame, then scrub on and click the ones you can see on the next. Continuing the loop across frames would suggest
    landmarks the user did not pick at that moment - and would read as advice to keep clicking *new* landmarks, when
    what anchors the drift is clicking the *same* landmark again at a later moment.
    """
    order = pitch_landmark_order(length_m, width_m)
    seen: dict[int, int] = {}
    labels: list[str] = []
    for frame in frames:
        rank = seen.get(frame, 0)
        labels.append(order[min(rank, len(order) - 1)])
        seen[frame] = rank + 1
    return labels


def repeated_labels_within_a_frame(pairs: Iterable[tuple[int, str]]) -> list[str]:
    """Landmarks clicked more than once *on one frame* - the clicks that contradict each other.

    Clicking one landmark again on a *different* frame is how the chain gets re-anchored across a long video, so it
    is not a duplicate and must not be blocked. Two clicks of one landmark on one picture, though, claim two
    different pitch positions for a single image point: whichever is wrong would drag the fit towards itself.
    """
    counts: dict[tuple[int, str], int] = {}
    for frame, label in pairs:
        counts[(frame, label)] = counts.get((frame, label), 0) + 1
    return sorted({label for (frame, label), count in counts.items() if count > 1})


def project_landmarks(
    calibration: PitchCalibration, table: dict[str, tuple[float, float]], q: np.ndarray, focal: float, frame_width: int
) -> dict[str, tuple[float, float]]:
    """Where each named landmark falls in one frame, in pixels.

    Landmarks behind the camera, or off the image, are left out. Drawing these is how the user sees where the corners
    they could not click on have ended up: the fit determines them, so they can be checked against the markings.
    """
    names = list(table)
    xy = np.array([table[name] for name in names], dtype=np.float64)
    uv, in_front = pitch_to_pixels(calibration, xy, q, focal)
    projected: dict[str, tuple[float, float]] = {}
    for name, point, front in zip(names, uv, in_front):
        if not front or not np.isfinite(point).all():
            continue
        x, y = point * frame_width
        if 0 <= x <= frame_width and 0 <= y <= frame_width:
            projected[name] = (float(x), float(y))
    return projected


def pitch_marking_polylines(length_m: float, width_m: float, *, arc_points: int = 48) -> list[np.ndarray]:
    """The pitch markings as polylines in pitch metres, for drawing the outline back onto a frame.

    Everything is a polyline - circles and arcs are sampled - so the caller only has to project points and draw
    segments. That is what keeps the overlay honest under the perspective projection, where a circle on the ground is
    not a circle in the image. The markings are the standard ones for every format, so the same shapes work once the
    dimensions are known.
    """
    half_length, half_width = length_m / 2, width_m / 2
    lines = [
        # Touchlines and goal lines, as one closed loop.
        np.array([[0.0, 0.0], [length_m, 0.0], [length_m, width_m], [0.0, width_m], [0.0, 0.0]]),
        # Halfway line.
        np.array([[half_length, 0.0], [half_length, width_m]]),
    ]
    # Goal box (six-yard) and penalty box (18-yard) at both ends.
    for goal_x, inward in ((0.0, 1.0), (length_m, -1.0)):
        lines.append(
            np.array(
                [
                    [goal_x, half_width - GOAL_BOX_HALF_WIDTH_M],
                    [goal_x + inward * GOAL_BOX_DEPTH_M, half_width - GOAL_BOX_HALF_WIDTH_M],
                    [goal_x + inward * GOAL_BOX_DEPTH_M, half_width + GOAL_BOX_HALF_WIDTH_M],
                    [goal_x, half_width + GOAL_BOX_HALF_WIDTH_M],
                ]
            )
        )
        lines.append(
            np.array(
                [
                    [goal_x, half_width - PENALTY_BOX_HALF_WIDTH_M],
                    [goal_x + inward * PENALTY_BOX_DEPTH_M, half_width - PENALTY_BOX_HALF_WIDTH_M],
                    [goal_x + inward * PENALTY_BOX_DEPTH_M, half_width + PENALTY_BOX_HALF_WIDTH_M],
                    [goal_x, half_width + PENALTY_BOX_HALF_WIDTH_M],
                ]
            )
        )
    # Centre circle.
    angles = np.linspace(0.0, 2.0 * np.pi, arc_points)
    lines.append(
        np.column_stack(
            [half_length + CENTRE_CIRCLE_RADIUS_M * np.cos(angles), half_width + CENTRE_CIRCLE_RADIUS_M * np.sin(angles)]
        )
    )
    # Penalty arcs: the part of the 9.15 m circle around each penalty spot that lies outside the penalty box.
    theta = np.arccos((PENALTY_BOX_DEPTH_M - PENALTY_SPOT_DISTANCE_M) / CENTRE_CIRCLE_RADIUS_M)
    for spot_x, facing in ((PENALTY_SPOT_DISTANCE_M, 0.0), (length_m - PENALTY_SPOT_DISTANCE_M, np.pi)):
        arc = np.linspace(facing - theta, facing + theta, arc_points)
        lines.append(
            np.column_stack(
                [spot_x + CENTRE_CIRCLE_RADIUS_M * np.cos(arc), half_width + CENTRE_CIRCLE_RADIUS_M * np.sin(arc)]
            )
        )
    # Corner arcs (1 m radius).
    for corner_x, corner_y, start in (
        (0.0, 0.0, 0.0),
        (length_m, 0.0, np.pi / 2),
        (length_m, width_m, np.pi),
        (0.0, width_m, 3 * np.pi / 2),
    ):
        arc = np.linspace(start, start + np.pi / 2, 16)
        lines.append(np.column_stack([corner_x + np.cos(arc), corner_y + np.sin(arc)]))
    return lines


def pitch_overlay(
    image: np.ndarray,
    calibration: PitchCalibration,
    q: np.ndarray,
    focal: float,
    length_m: float,
    width_m: float,
    table: dict[str, tuple[float, float]] | None = None,
) -> np.ndarray:
    """Draw the pitch markings back into the frame - the check that the landmarks were labelled the right way round.

    Without this the user has no way to tell a good registration from a mirrored one: the numbers alone look
    plausible either way. The whole set of standard markings is drawn - touchlines, halfway line, both boxes, the
    centre circle, the penalty arcs and spots and the corner arcs - so a fit can be checked against the markings
    that are actually visible rather than only the outline. Pass ``table`` to mark each named landmark as well, which
    shows the corners that were never clicked where the fit puts them.
    """
    drawn = image.copy()
    width = drawn.shape[1]
    for points in pitch_marking_polylines(length_m, width_m):
        uv, in_front = pitch_to_pixels(calibration, points, q, focal)
        pixels = uv * width
        for index in range(len(points) - 1):
            if not (in_front[index] and in_front[index + 1]):
                continue
            start, end = pixels[index], pixels[index + 1]
            if not (np.isfinite(start).all() and np.isfinite(end).all()):
                continue
            cv2.line(drawn, tuple(np.round(start).astype(int)), tuple(np.round(end).astype(int)), (0, 255, 255), 3)

    # The penalty spots are points, not lines, so they are drawn as small filled dots.
    spots = np.array([[PENALTY_SPOT_DISTANCE_M, width_m / 2], [length_m - PENALTY_SPOT_DISTANCE_M, width_m / 2]])
    uv, in_front = pitch_to_pixels(calibration, spots, q, focal)
    for point, front in zip(uv * width, in_front):
        if front and np.isfinite(point).all():
            cv2.circle(drawn, tuple(np.round(point).astype(int)), 5, (0, 255, 255), -1)

    if table:
        for name, (x, y) in project_landmarks(calibration, table, q, focal, width).items():
            spot = (int(round(x)), int(round(y)))
            cv2.drawMarker(drawn, spot, (255, 0, 255), cv2.MARKER_CROSS, 14, 3)
            # Outlined text so the labels stay readable over grass as well as over the pitch markings.
            for colour, thickness in (((0, 0, 0), 4), ((255, 255, 255), 1)):
                cv2.putText(drawn, name, (spot[0] + 10, spot[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, thickness)
    return drawn
