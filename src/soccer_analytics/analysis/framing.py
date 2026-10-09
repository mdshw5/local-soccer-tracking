"""Framing one player: where to point the crop, and the ffmpeg filter that follows them.

The pitch replay draws tracks in metres; a clip centred on one player needs the same player in *pixels* of the
source video. Those are exactly the boxes Stage A stored (bottom-centre boxes normalised by frame width), so no
calibration is involved: a centred clip can be cut for any track, registered pitch or not.

The crop does not chase the detector frame by frame - it would twitch with every box jitter, and a clip that
shudders is worse than one that lags. Instead the trajectory is interpolated onto a fixed command grid, smoothed
with a short centred window, and clamped to the frame; ffmpeg's ``sendcmd`` then drives a ``crop`` filter with the
result, which is how the cut follows the player without re-encoding the whole picture through Python.

Everything here is pure: the numbers go in, the crop commands and the filter string come out, and the ffmpeg call
that consumes them lives in ``analysis.highlights`` with the other clip cuts.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# The player's height as a fraction of the crop's height. A quarter of the frame is a "follow" shot - close enough
# to read a shirt number, wide enough that a run does not leave the frame between two commands.
DEFAULT_PLAYER_FRACTION = 0.25
# Crop-height floor, in source pixels. A far-side player is tens of pixels tall, and a crop computed purely from
# the fraction would ask for a postage stamp that then has to be blown up; below this floor the player simply
# occupies less of the frame.
MIN_CROP_HEIGHT_PX = 480.0
# How often ``sendcmd`` is told where to put the crop. At 6 Hz - the rate a broadcast operator would move a
# virtual camera at most - the motion reads as a pan rather than a teleport, and the filter string stays small.
COMMAND_RATE_HZ = 6.0
# Centred moving average over the crop centre. Long enough to ride out box jitter and detection flicker, short
# enough that a sprint is followed without visible lag.
SMOOTH_WINDOW_S = 0.7
# A gap longer than this inside the track means the player was not seen: the crop must not be asked to slide
# smoothly across a stretch where nobody knows where they went, so the window is cut short there instead.
MAX_GAP_S = 2.0
# How far past the last observation the window may run by holding the final position. A track that ends a fraction
# of a second before the requested window is a detection flicker, not a departure, and refusing to cut there would
# shave every clip; past this the player is genuinely gone and the clip stops where the evidence does.
TAIL_GRACE_S = 1.0


@dataclass(frozen=True)
class CropCommand:
    """Put the crop's top-left corner at ``(x, y)`` at ``time_s`` (seconds into the cut)."""

    time_s: float
    x: int
    y: int


@dataclass(frozen=True)
class FramingPlan:
    """Everything the cut needs: the crop rectangle and the commands that move it.

    ``duration_s`` is the window the plan actually covers, which is shorter than the window asked for when the
    track runs out or has a gap in it; ``truncated`` says which, so the page can tell the user why the clip stops.
    """

    crop_w: int
    crop_h: int
    commands: tuple[CropCommand, ...]
    duration_s: float
    truncated: str  # "" when the full window is covered, else "track ended" / "gap in the track"


def _pixel_track(times: np.ndarray, boxes: np.ndarray, source_width: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Track centres and heights in source pixels, from width-normalised boxes (both axes share the width scale)."""
    x1, y1, x2, y2 = (boxes[:, i].astype(np.float64) * source_width for i in range(4))
    centre_x = (x1 + x2) / 2.0
    centre_y = (y1 + y2) / 2.0
    height = np.abs(y2 - y1)
    return centre_x, centre_y, height


def _even(value: float) -> int:
    return max(2, int(round(value / 2.0)) * 2)


def crop_size(
    heights_px: np.ndarray,
    source_width: int,
    source_height: int,
    *,
    player_fraction: float = DEFAULT_PLAYER_FRACTION,
    min_crop_height_px: float = MIN_CROP_HEIGHT_PX,
) -> tuple[int, int]:
    """The crop rectangle for a track: the median player height scaled to ``player_fraction`` of the frame.

    The median, not the maximum: one close-up frame should not zoom the whole clip in on a face, and one distant
    frame should not leave the player a dot. Both sides are clamped to the source and forced even, and the width
    follows the source's aspect ratio so the cut needs no letterboxing.
    """
    median_h = float(np.median(heights_px)) if len(heights_px) else 0.0
    target = median_h / max(player_fraction, 1e-3) if median_h > 0 else min_crop_height_px
    crop_h = _even(min(max(target, min_crop_height_px), float(source_height)))
    crop_w = _even(min(crop_h * source_width / max(source_height, 1), float(source_width)))
    return crop_w, crop_h


def _smooth(values: np.ndarray, window: int) -> np.ndarray:
    """Centred moving average with edge-aware windows (no padding that would pull the ends inward)."""
    if window <= 1 or len(values) < 3:
        return values
    kernel = np.ones(window, dtype=np.float64) / window
    padded = np.pad(values, (window // 2, window - 1 - window // 2), mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def crop_times(*, duration_s: float, rate_hz: float = COMMAND_RATE_HZ) -> np.ndarray:
    """Command times on the cut's own clock (0 = first frame of the cut), inclusive of both ends.

    A uniform grid would drift a few milliseconds against the frame timestamps and draw imperceptible steps; the
    rounded grid is exact to the millisecond the filter string is written with. The last point is kept exactly at
    ``duration_s`` so the final position is stated rather than inherited from the previous command.
    """
    step = 1.0 / max(rate_hz, 1e-3)
    count = max(2, int(np.ceil(duration_s * rate_hz)) + 1)
    grid = np.unique(np.round(np.linspace(0.0, duration_s, count) / step) * step)
    # The final command sits exactly on the window's end: rounding must not leave the last stated position a grid
    # step past the cut (harmless to ffmpeg, but it makes the plan disagree with its own duration).
    trailing = grid >= duration_s - 1e-9
    grid = np.append(grid[~trailing], duration_s)
    return grid


def plan_framing(
    times: np.ndarray,
    boxes: np.ndarray,
    *,
    start_s: float,
    duration_s: float,
    source_width: int,
    source_height: int,
    player_fraction: float = DEFAULT_PLAYER_FRACTION,
    min_crop_height_px: float = MIN_CROP_HEIGHT_PX,
    rate_hz: float = COMMAND_RATE_HZ,
    smooth_s: float = SMOOTH_WINDOW_S,
    max_gap_s: float = MAX_GAP_S,
    tail_grace_s: float = TAIL_GRACE_S,
) -> FramingPlan | None:
    """Build the crop rectangle and the commands that follow this track's player-centre for ``duration_s``.

    ``times`` are source seconds (sorted) and ``boxes`` the matching ``(x1, y1, x2, y2)`` width-normalised
    rectangles. Returns ``None`` when the track has nothing to frame. The window is trimmed to the end of the
    track (with a short grace, so a detection flicker does not shave the clip), and further trimmed to just before
    the first gap longer than ``max_gap_s``, because across such a gap the tracker simply does not know where the
    player is.
    """
    times = np.asarray(times, dtype=np.float64)
    boxes = np.asarray(boxes, dtype=np.float64)
    if len(times) < 2 or len(boxes) != len(times):
        return None
    order = np.argsort(times)
    times, boxes = times[order], boxes[order]

    # The window the plan can honestly cover: from ``start_s`` to the earliest of (the requested end, the end of
    # the track, the first long gap).
    end_s = start_s + max(duration_s, 0.0)
    truncated = ""
    if times[-1] < end_s - tail_grace_s:
        end_s, truncated = float(times[-1]) + tail_grace_s, "track ended"
    gaps = np.diff(times)
    inside = np.where((gaps > max_gap_s) & (times[:-1] >= start_s) & (times[:-1] < end_s))[0]
    if len(inside):
        end_s, truncated = float(times[inside[0]]), "gap in the track"
    if end_s <= start_s:
        return None
    covered = (times >= start_s) & (times <= end_s)
    if covered.sum() < 2:
        # The track does not reach this window: hold its nearest end position rather than refusing.
        covered = np.zeros(len(times), dtype=bool)
        covered[np.argmin(np.abs(times - start_s))] = True
    frame = times[covered]
    centre_x, centre_y, heights = _pixel_track(frame, boxes[covered], source_width)

    crop_w, crop_h = crop_size(heights, source_width, source_height, player_fraction=player_fraction,
                               min_crop_height_px=min_crop_height_px)

    grid = crop_times(duration_s=end_s - start_s, rate_hz=rate_hz)
    window_times = start_s + grid
    xs = _smooth(np.interp(window_times, frame, centre_x), max(1, int(round(smooth_s * rate_hz))))
    ys = _smooth(np.interp(window_times, frame, centre_y), max(1, int(round(smooth_s * rate_hz))))

    half_w, half_h = crop_w / 2.0, crop_h / 2.0
    max_x = max(0.0, source_width - crop_w)
    max_y = max(0.0, source_height - crop_h)
    xs = np.clip(xs - half_w, 0.0, max_x)
    ys = np.clip(ys - half_h, 0.0, max_y)

    commands = tuple(
        CropCommand(float(t), int(round(x)), int(round(y)))
        for t, x, y in zip(grid, xs, ys)
    )
    # Runs of identical positions are one command, not forty: a player standing still must not fill the filter
    # string with repeats. The first command is always kept - it states where the crop starts, which the filter's
    # own ``x=``/``y=`` defaults only happen to match - and the last is kept so the final position holds to the
    # end of the window instead of being inherited from the last move.
    deduped: list[CropCommand] = [commands[0]]
    for command in commands[1:]:
        if (command.x, command.y) != (deduped[-1].x, deduped[-1].y):
            deduped.append(command)
    if deduped[-1].time_s < commands[-1].time_s:
        deduped.append(commands[-1])
    return FramingPlan(crop_w, crop_h, tuple(deduped), round(end_s - start_s, 3), truncated)


def sendcmd_filter(
    plan: FramingPlan,
    *,
    scale_width: int | None = None,
    fps: int | None = None,
) -> str:
    """The ffmpeg filter chain that cuts a moving window out of the frame and optionally rescales it.

    ``sendcmd`` sends each command at its own time (seconds from the first frame of the filter graph's input, which
    is the cut's start, because the cut seeks first); its targets are named filters, hence ``crop@follow``. The
    scale is a plain filter after the crop, so the moving window is always resampled from full resolution and the
    output size stays constant as the crop moves.
    """
    if not plan.commands:
        raise ValueError("the plan has no commands to send")
    # Each interval lists the x and y together: the pair is applied as one reposition, so the crop never moves
    # along one axis alone between two commands.
    spans = [f"{command.time_s:.3f} crop@follow x {command.x}, crop@follow y {command.y}" for command in plan.commands]
    schedule = "; ".join(spans)
    first = plan.commands[0]
    chain = (
        f"sendcmd=c='{schedule}',"
        f"crop@follow=w={plan.crop_w}:h={plan.crop_h}:x={first.x}:y={first.y}"
    )
    if scale_width:
        chain += f",scale={scale_width}:-2:flags=lanczos"
    if fps:
        chain += f",fps={fps}"
    return chain


def plan_covers(plan: FramingPlan, start_s: float, duration_s: float) -> bool:
    """Whether the plan covers the whole window that was asked for (used to tell the user a clip was trimmed)."""
    return plan.duration_s + 1e-6 >= duration_s


# The clip length the page offers before the user touches the slider: long enough to contain a moment, short
# enough to be watched. The slider's own range is wider (down to five seconds, up to a minute).
DEFAULT_CLIP_SECONDS = 24.0
SHORTEST_CLIP_SECONDS = 5.0


def default_clip_length(
    span_s: float,
    *,
    shortest_s: float = SHORTEST_CLIP_SECONDS,
    longest_s: float = DEFAULT_CLIP_SECONDS,
) -> float:
    """The clip length to offer for an appearance of this span: its own length, rounded and clamped.

    Always a ``float``. ``round(17.6)`` is an ``int`` in Python, and a widget whose value type disagrees with its
    float bounds is rejected outright - Streamlit raises ``StreamlitInvalidParameterTypeError`` and the page dies
    for every short appearance (measured live, 2026-10-09). Returning a float here is what keeps the rule in one
    tested place instead of at the call site.
    """
    return float(min(float(longest_s), max(float(shortest_s), float(round(float(span_s))))))
