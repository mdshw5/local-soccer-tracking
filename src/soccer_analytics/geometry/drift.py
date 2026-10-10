"""Time-varying correction for the camera-motion chain: what keeps a long game calibrated.

The chain in ``camera_motion`` is integrated step by step, and every step carries a little error: the gimbal's
rotation center is not the lens center - the classic cause of drift in rotating-camera self-calibration, analyzed by
Hayman & Murray (2003) - lens distortion is not modeled, and the per-step focal estimate is a bounded search on a
noisy homography. Each error is small, but they *accumulate over the game*. Measured on this footage the chain sits
3.7 px from a direct match after 15-30 s and 11.6 px after 150 s, and the landmark clicks that anchor the global fit
usually sit in the first minutes - so late in the game the projected pitch slides away from the markings.

One global camera pose (``pitch_calibration``) cannot describe that, because the error is a function of time. What
this module adds is the second half of the literature's answer to drift: sit the trajectory on a smooth model and
re-anchor it wherever the pitch is observed. Thomas (2007) tracks the camera against the pitch markings; Citraro et
al. (2020) calibrate every frame against the field model; Lu, Chen & Little (2019, PTZ-SLAM) correct a pan-tilt-zoom
trajectory with re-observed keyframes. Here the observations are the user's landmark clicks - exact ground control
points, already part of the workflow - spread over the video as far as the user has clicked.

The correction is a rotation ``D(t)`` and a focal multiplier inserted between the calibration's base rotation and the
chain's ``Q`` at every *anchor* frame (a frame that carries clicks), interpolated along the shortest great circle
between anchors and held flat outside them. That family is the shape of the error this is for: the ground points
are 20-90 m away, so a small translation of the camera center barely moves them while a small pointing error moves
them a lot, and the focal search's lag behind the true zoom is a radial, multiplicative error. It is a *similarity*
correction, not a full per-frame calibration: a chain error with a genuinely perspective component is left with a
residual a couple of meters wide at the far side of the pitch - per-anchor homographies (what Thomas 2007 and,
later, TVCalib-style per-frame calibrations solve for) would take that out too, at the price of threading a
per-frame matrix through every projection call.

A fitted correction is deliberately conservative: bounded to plausible drift, smoothed between anchors, and pulled
toward identity, so it can only ever describe the *difference* between the chain and the clicks - never a second
camera. Clicks that disagree with their own anchor-mates are left out of it entirely, because a knot carrying a
single mis-click could otherwise absorb almost any error and silence the outlier report.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import cv2
import numpy as np
from scipy.optimize import least_squares

if TYPE_CHECKING:  # the cycle: pitch_calibration imports this module
    from soccer_analytics.geometry.pitch_calibration import Landmark, PitchCalibration

IDENTITY = np.eye(3)
# Solver weights, in the units the clicks use (pixels divided by frame width). The robust scale sits at *drift*
# size, not click size: tens of pixels of error are exactly what this exists to correct, so they must stay in the
# loss's linear regime - only a genuinely wrong click (hundreds of pixels) is down-weighted as an outlier. The
# smoothness term is small enough that a knot's own clicks always dominate it, and the ridge only settles the
# directions nothing else constrains.
ROBUST_SCALE = 2.0e-2
SMOOTHNESS = 5.0e-2
RIDGE = 1.0e-3
# What a correction is allowed to be. Drift over a game is degrees of pointing and a few percent of zoom; a fit
# that wants more than this has found a different camera (a shrunken focal maps the whole pitch near the image
# center, which can look like a fit for clicks near the middle), not a correction.
MAX_ROTATION_RAD = 0.35
MIN_SCALE, MAX_SCALE = 0.6, 1.6


def _exp(vector: np.ndarray) -> np.ndarray:
    """Rotation matrix of a rotation vector."""
    return cv2.Rodrigues(np.asarray(vector, dtype=np.float64).reshape(3, 1))[0]


def _log(rotation: np.ndarray) -> np.ndarray:
    """Rotation vector of a rotation matrix."""
    return cv2.Rodrigues(np.asarray(rotation, dtype=np.float64))[0].ravel()


@dataclass(frozen=True)
class DriftCorrection:
    """The chain's error at each anchor frame - a rotation, and a scale on the focal - interpolated in between.

    ``frames`` are analysis-frame indices (the same numbering the clicks use); ``rotations`` are the pointing
    correction at exactly those frames and ``scales`` the multiplicative focal correction (1.0 for none). Between
    anchors the rotation follows the shortest great circle and the scale is interpolated linearly; before the first
    and after the last they are held, so the whole video is covered by construction even though only the clicked
    stretches are actually pinned.

    The scale is not decoration: the chain's focal is *searched* per step from a default, and on fast zooms it lags
    the true focal by a fraction of itself. That error changes the projection radially and, unlike the pointing
    error, no rotation can absorb it - it is the zoom half of Hayman & Murray's rotating-and-zooming calibration.
    """

    frames: tuple[int, ...]
    rotations: tuple[np.ndarray, ...]
    scales: tuple[float, ...] | None = None

    def __post_init__(self) -> None:
        if len(self.frames) != len(self.rotations):
            raise ValueError("a drift correction needs one rotation per anchor frame")
        if self.scales is not None and len(self.scales) != len(self.frames):
            raise ValueError("a drift correction needs one focal scale per anchor frame")
        if len(self.frames) == 0:
            raise ValueError("a drift correction needs at least one anchor frame")

    def rotation(self, frame: float) -> np.ndarray:
        """The rotation to apply to frame ``frame`` (flat outside the anchored range)."""
        frames = self.frames
        if frame <= frames[0]:
            return self.rotations[0].copy()
        if frame >= frames[-1]:
            return self.rotations[-1].copy()
        k = int(np.searchsorted(frames, frame, side="right") - 1)
        span = frames[k + 1] - frames[k]
        s = 0.0 if span <= 0 else (frame - frames[k]) / span
        if s <= 0.0:
            return self.rotations[k].copy()
        delta = _log(self.rotations[k].T @ self.rotations[k + 1])
        return self.rotations[k] @ _exp(s * delta)

    def scale(self, frame: float) -> float:
        """The focal multiplier to apply to frame ``frame`` (held outside the anchored range)."""
        if self.scales is None:
            return 1.0
        frames = self.frames
        if frame <= frames[0]:
            return float(self.scales[0])
        if frame >= frames[-1]:
            return float(self.scales[-1])
        k = int(np.searchsorted(frames, frame, side="right") - 1)
        span = frames[k + 1] - frames[k]
        s = 0.0 if span <= 0 else (frame - frames[k]) / span
        return float(self.scales[k] + s * (self.scales[k + 1] - self.scales[k]))

    def adjust(self, q: np.ndarray) -> np.ndarray:
        """The corrected chain: ``q[t] -> D(t) @ q[t]``.

        Pre-multiplying is the useful side: ``Q`` maps frame rays to *reference*-frame rays, and this is a rotation
        of that reference frame - which is exactly what a drifting chain is. Every consumer of the chain (projection,
        overlays, the replay) keeps working unchanged.
        """
        out = np.array(q, dtype=np.float64, copy=True)
        for t in range(len(out)):
            out[t] = self.rotation(t) @ out[t]
        return out

    def adjust_focal(self, focal: np.ndarray) -> np.ndarray:
        """The corrected focal track: one scale per frame, 1.0 when no scale was fitted."""
        out = np.array(focal, dtype=np.float64, copy=True)
        if self.scales is None:
            return out
        for t in range(len(out)):
            out[t] = out[t] * self.scale(t)
        return out

    def to_json(self) -> dict:
        payload = {
            "frames": [int(f) for f in self.frames],
            "rotations": [np.asarray(r, dtype=np.float64).tolist() for r in self.rotations],
        }
        if self.scales is not None:
            payload["scales"] = [float(s) for s in self.scales]
        return payload

    @classmethod
    def from_json(cls, data: dict) -> "DriftCorrection":
        scales = data.get("scales")
        return cls(
            tuple(int(f) for f in data["frames"]),
            tuple(np.asarray(r, dtype=np.float64) for r in data["rotations"]),
            tuple(float(s) for s in scales) if scales else None,
        )


def fit_drift(
    landmarks: list["Landmark"],
    calibration: "PitchCalibration",
    chain: dict[int, tuple[np.ndarray, float]],
    aspect: float,
    *,
    smoothness: float = SMOOTHNESS,
    ridge: float = RIDGE,
    robust_scale: float = ROBUST_SCALE,
    initial: "DriftCorrection | None" = None,
    max_rotation: float = MAX_ROTATION_RAD,
    max_scale: float = MAX_SCALE,
) -> DriftCorrection | None:
    """Solves the smooth rotation and focal correction that puts this calibration on its landmarks, everywhere.

    One anchor per distinct clicked frame. Each anchor is fitted to its own clicks - two equations per click for
    the three rotation unknowns and the focal scale - and the smoothness term decides the directions the clicks
    leave free (a single click cannot pin a rotation about its own ray; the neighbors' corrections decide it,
    which is what makes a lone landmark click useful rather than destabilizing). The robust loss keeps one wrong
    click from bending the trajectory toward itself; gross outliers are still caught by the caller's own outlier
    pass, which runs on the corrected poses.

    Landmarks carrying a ``direction`` (automatic line observations) contribute one residual each - the offset
    *across* the line, in the same normalized-pixel units as a click's two - so sliding along the line is free and
    only the registration the line actually measures is fitted.

    ``max_rotation`` and ``max_scale`` bound each anchor's correction. The defaults are sized for whole-game click
    drift; the automatic line re-anchor passes tighter ones, because a fit that wants more than a degree or two
    per anchor has misdetected a line, not corrected a camera.

    ``initial`` warm-starts the solver from a previous correction (same anchors). A cold start against a pose that
    has since moved can settle into a completely different basin - measured: a second round of fitting reached 131
    degrees of rotation from a zero start - and a warm start keeps a refinement a refinement.

    Returns ``None`` when the landmarks all sit on one frame: a constant correction there would be
    indistinguishable from the fit's own base rotation, and the drift this exists for is a function of time.

    The import of ``pitch_calibration`` is local because that module imports this one.
    """
    from soccer_analytics.geometry.pitch_calibration import pitch_to_pixels

    frames = sorted({lm.frame for lm in landmarks})
    if len(frames) < 2:
        return None
    index = {frame: k for k, frame in enumerate(frames)}

    # The Jacobian is block-sparse: a landmark's residual depends only on its own anchor's four parameters, and
    # the smoothness terms only on each anchor and its two neighbors. Declaring that (jac_sparsity) lets the
    # trust-region solver factor a banded system instead of a dense one - with two click anchors it changes
    # nothing, but with the automatic line re-anchor's dozens of anchors it is the difference between seconds
    # and minutes per iteration.
    n_params = 4 * len(frames)
    landmark_rows: list[tuple[int, int]] = []  # (anchor index, first residual row) per landmark
    row = 0
    for lm in landmarks:
        width = 1 if lm.direction is not None else 2
        landmark_rows.append((index[lm.frame], row, width))
        row += width
    n_landmark_rows = row
    n_smooth_rows = 4 * max(len(frames) - 2, 0)  # 3 rotation + 1 scale per interior anchor
    sparsity = np.zeros((n_landmark_rows + n_smooth_rows + n_params, n_params), dtype=int)
    for anchor, r, width in landmark_rows:
        sparsity[r : r + width, 4 * anchor : 4 * anchor + 4] = 1
    if len(frames) > 2:
        for k in range(1, len(frames) - 1):
            base_row = n_landmark_rows + 4 * (k - 1)
            for anchor in (k - 1, k, k + 1):
                sparsity[base_row : base_row + 4, 4 * anchor : 4 * anchor + 4] = 1
    sparsity[n_landmark_rows + n_smooth_rows :, :] = 1  # the ridge touches every parameter

    def unpack(params: np.ndarray) -> tuple[list[np.ndarray], list[float]]:
        rotations = [_exp(params[4 * k : 4 * k + 3]) for k in range(len(frames))]
        scales = [float(1.0 + params[4 * k + 3]) for k in range(len(frames))]
        return rotations, scales

    def residuals(params: np.ndarray) -> np.ndarray:
        rotations, scales = unpack(params)
        out: list[float] = []
        for lm in landmarks:
            q, focal = chain[lm.frame]
            k = index[lm.frame]
            uv, in_front = pitch_to_pixels(
                calibration,
                np.array([[lm.pitch_x, lm.pitch_y]]),
                rotations[k] @ q,
                focal * scales[k],
            )
            if not in_front[0]:
                # A correction that swings a landmark behind the camera is nonsense; a flat penalty walks the
                # solver back toward poses that project it.
                out.extend((0.5, 0.5) if lm.direction is None else (0.5,))
            elif lm.direction is not None:
                # A line observation: only the offset across the line is measured, so only it is residualized.
                # Same normalized-pixel units as a click's coordinates, so one robust loss serves both.
                perp = (float(uv[0, 0]) - lm.u) * lm.direction[0] + (float(uv[0, 1]) - lm.v) * lm.direction[1]
                out.append(perp)
            else:
                out.extend((float(uv[0, 0] - lm.u), float(uv[0, 1] - lm.v)))
        if len(rotations) > 2:
            for k in range(1, len(rotations) - 1):
                previous = params[4 * (k - 1) : 4 * (k - 1) + 4]
                here = params[4 * k : 4 * k + 4]
                following = params[4 * (k + 1) : 4 * (k + 1) + 4]
                second = following - 2.0 * here + previous
                # rotations and scales are different kinds of number - radians against percent - so they are
                # penalized apart; penalizing them together would let a large scale kink hide behind a small one
                out.extend(smoothness * second[:3])
                out.extend(0.5 * smoothness * second[3:])
        out.extend(ridge * params)
        return np.asarray(out, dtype=np.float64)

    start = np.zeros(4 * len(frames))
    if initial is not None and initial.frames == tuple(frames):
        for k, (rotation, scale) in enumerate(zip(initial.rotations, initial.scales or [1.0] * len(frames))):
            start[4 * k : 4 * k + 3] = _log(rotation)
            start[4 * k + 3] = float(scale) - 1.0
    # The tolerances and the iteration budget scale with the anchor count. The click fit's 1e-12 / 400-per-anchor
    # budget is right for a handful of knots, but the automatic line re-anchor produces hundreds - at that size
    # the same settings grind for hours on changes far below any pixel. 1e-6 is ~1e-4 px of pointing; the budget
    # is 40 sweeps per anchor, which the click fits never came close to needing.
    result = least_squares(
        residuals,
        start,
        jac_sparsity=sparsity,
        bounds=(
            np.tile([-max_rotation, -max_rotation, -max_rotation, MIN_SCALE - 1.0], len(frames)),
            np.tile([max_rotation, max_rotation, max_rotation, max_scale - 1.0], len(frames)),
        ),
        loss="soft_l1",
        f_scale=robust_scale,
        # A robust loss goes linear once a residual passes the scale, and in that regime the cost flattens out
        # long before the residual is gone - so the default stopping rules quit early and leave drifts of a dozen
        # pixels on the anchors that a few more steps would take out. The screen and the bounds carry the
        # robustness; the tolerances are loosened so the fit actually finishes.
        ftol=1e-6,
        xtol=1e-6,
        gtol=1e-6,
        max_nfev=40 * len(frames),
    )
    rotations, scales = unpack(result.x)
    return DriftCorrection(tuple(frames), tuple(rotations), tuple(scales))
