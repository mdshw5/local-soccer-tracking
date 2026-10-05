"""Calibrate a PTZ camera against the pitch from landmark clicks, then map any pixel of any frame to pitch metres.

Model
-----
World (pitch) frame: X along the touchline, Y across the pitch, Z up, metres. The camera sits at ``position`` (Z =
height above the pitch) and never translates. Frame ``t`` has orientation ``R_t = R_base @ Q_t`` (``Q_t`` from
``camera_motion`` maps frame-t rays to the reference frame's rays) and focal length ``f_t = f_ref * (chain zoom)``.

A pixel ``(u, v)`` of frame ``t`` is the world ray ``d = R_t K_t^-1 [u, v, 1]``; it meets the ground ``Z = 0`` at
``position + s d`` with ``s = -position_z / d_z`` for rays that point downward. Unlike a planar reference image this
is exact at *every* orientation, which matters because the camera pans ~90 degrees across a match.

Unknowns: camera position (3), base orientation (3 axis-angle), focal scale (1)  -> 7 parameters. Each landmark
(known pitch XY, clicked pixel in some frame) gives 2 equations, so 4+ landmarks spread over the pitch suffice.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from scipy.optimize import least_squares

from soccer_analytics.geometry.camera_motion import intrinsics

# Plausible rig: a tripod/pole 1-8 m above the pitch.
HEIGHT_RANGE = (0.8, 12.0)
FOCAL_SCALE_RANGE = (0.6, 1.8)

# A drift knot has four degrees of freedom (a rotation and a scale) while a click supplies two equations, so a knot
# fitted to a *single* click stays free in the two directions that click cannot see. On the whole-game clicks that
# freedom was enough to walk an anchor 21 deg off its neighbours with the residual still small - the anchors read as
# the solver's private opinion, not as measurements. Two agreeing clicks pin the direction of a correction and the
# smoothness term settles the rest, so a moment clicked once cannot anchor the drift (the UI asks for a second
# landmark there; the click is still checked and reported, it is just not fitted).
MIN_CLICKS_PER_ANCHOR = 2

# Registering the base pose on one frame needs the classic four landmarks. Doing so pins the metre scale of every
# reported distance on the moment the user trusts most, instead of pooling clicks across a span the chain had
# already drifted over. Measured on the whole-game clicks: the pooled fit parked the camera at the 12 m height
# bound, while the frame the user clicked five times registers the same clicks at 5.3 m - a real tripod.
MIN_REFERENCE_CLICKS = 4


from soccer_analytics.geometry.drift import DriftCorrection, fit_drift


@dataclass(frozen=True)
class Landmark:
    """A clicked pitch feature: pixel ``(u, v)`` (normalised by frame width) in frame ``frame`` and its pitch XY."""

    frame: int
    u: float
    v: float
    pitch_x: float
    pitch_y: float
    label: str = ""


@dataclass(frozen=True)
class PitchCalibration:
    position: np.ndarray  # camera (x, y, z) in pitch metres
    base_rotation: np.ndarray  # R_base (3x3): reference-frame rays -> pitch rays
    focal_scale: float  # multiplies the chain's focal length
    aspect: float  # frame height / width
    rms_error_m: float
    residuals_m: tuple[float, ...]  # per-landmark ground distance error, same order as the fit input
    excluded: tuple[int, ...] = ()  # landmarks dropped as gross outliers (their residuals are still reported)
    ambiguous: bool = False  # a very different camera fits the clicks almost as well: the set does not pin it down
    ill_conditioned: bool = False  # the clicks barely constrain the camera (near-singular residual Jacobian)
    # Time-varying correction of the motion chain (see geometry/drift.py). None when the clicks all sit on one
    # frame - a constant correction there is just the base rotation, and drift is a function of time.
    drift: DriftCorrection | None = None

    def to_json(self) -> dict:
        payload = {
            "position": self.position.tolist(),
            "base_rotation": self.base_rotation.tolist(),
            "focal_scale": self.focal_scale,
            "aspect": self.aspect,
            "rms_error_m": self.rms_error_m,
            "residuals_m": list(self.residuals_m),
            "excluded": list(self.excluded),
            "ambiguous": self.ambiguous,
            "ill_conditioned": self.ill_conditioned,
        }
        if self.drift is not None:
            payload["drift"] = self.drift.to_json()
        return payload

    @classmethod
    def from_json(cls, data: dict) -> "PitchCalibration":
        return cls(
            np.asarray(data["position"], dtype=np.float64),
            np.asarray(data["base_rotation"], dtype=np.float64),
            float(data["focal_scale"]),
            float(data["aspect"]),
            float(data["rms_error_m"]),
            tuple(float(x) for x in data["residuals_m"]),
            tuple(int(i) for i in data.get("excluded", [])),
            bool(data.get("ambiguous", False)),
            bool(data.get("ill_conditioned", False)),
            DriftCorrection.from_json(data["drift"]) if data.get("drift") else None,
        )

    def corrected_chain(self, q: np.ndarray) -> np.ndarray:
        """The chain with this calibration's drift correction applied (itself when there is none)."""
        return self.drift.adjust(q) if self.drift is not None else q

    def corrected_focal(self, focal: np.ndarray) -> np.ndarray:
        """The focal track with the drift correction's zoom term applied (itself when there is none)."""
        return self.drift.adjust_focal(focal) if self.drift is not None else focal

    def corrected_frame(self, q_frame: np.ndarray, focal_frame: float, frame: float) -> tuple[np.ndarray, float]:
        """One frame's chain state and focal with the drift correction applied - what a single-frame overlay needs."""
        if self.drift is None:
            return q_frame, float(focal_frame)
        return self.drift.rotation(frame) @ q_frame, float(focal_frame) * self.drift.scale(frame)


def pixel_rays(
    uv: np.ndarray, q: np.ndarray, focal: float, aspect: float, base_rotation: np.ndarray, focal_scale: float
) -> np.ndarray:
    """World-frame ray directions (N, 3, unit length) for normalised pixels ``uv`` (N, 2) of one frame.

    ``q`` maps frame rays to reference-frame rays (the convention of ``camera_motion.integrate_poses``), and
    ``base_rotation`` maps reference-frame rays to world rays.
    """
    k = intrinsics(focal * focal_scale, aspect)
    homogeneous = np.hstack([np.asarray(uv, dtype=np.float64).reshape(-1, 2), np.ones((len(uv), 1))])
    cam = homogeneous @ np.linalg.inv(k).T
    world = cam @ (base_rotation @ q).T
    return world / np.linalg.norm(world, axis=1, keepdims=True)


def intersect_ground(position: np.ndarray, rays: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Ground (Z=0) intersections for rays; returns ``(xy (N,2), valid (N,))``. Upward rays are invalid (NaN)."""
    dz = rays[:, 2]
    valid = dz < -1e-6
    s = np.full(len(rays), np.nan)
    s[valid] = -position[2] / dz[valid]
    hit = position[None, :2] + s[:, None] * rays[:, :2]
    return hit, valid


def _rodrigues(vector: np.ndarray) -> np.ndarray:
    return cv2.Rodrigues(np.asarray(vector, dtype=np.float64).reshape(3, 1))[0]


def _agreeing_per_anchor(landmarks: list[Landmark], errors: list[float]) -> list[Landmark]:
    """The clicks the drift fit may trust: at each anchor, the ones that agree with their anchor-mates.

    A *drifted* anchor is off as a group - its clicks still agree with each other, which is exactly what makes it
    a usable ground control point. A mis-click disagrees with the other clicks on the same frame, and a single
    such click must not bend the trajectory towards itself: given a knot of its own it could absorb almost any
    error, and the outlier report would go quiet. Anchors carrying one click are kept here (there is nothing to
    disagree with); whether their clicks can *anchor the drift* is a separate question, settled by
    `_drift_anchors`.
    """
    by_frame: dict[int, list[int]] = {}
    for index, landmark in enumerate(landmarks):
        by_frame.setdefault(landmark.frame, []).append(index)
    kept: list[Landmark] = []
    for indices in by_frame.values():
        if len(indices) < 2:
            kept.extend(landmarks[index] for index in indices)
            continue
        threshold = max(OUTLIER_FLOOR_M, OUTLIER_MEDIAN_FACTOR * float(np.median([errors[index] for index in indices])))
        kept.extend(landmarks[index] for index in indices if errors[index] <= threshold)
    return kept


def _reference_anchor(landmarks: list[Landmark], min_clicks: int = MIN_REFERENCE_CLICKS) -> tuple[int, list[Landmark]] | None:
    """The frame the base pose is registered on: the one carrying the most clicks, earliest first on ties.

    Four landmarks make a pose, so a frame clicked four or more times can stand alone as the registration; every
    other moment then only has to say how far the chain had drifted by its time. Pooling every click into one pose
    was tried first and is what a spread-click solve degenerates from: clicks either side of a drifted span cannot
    agree on one pose, and the solver answers with a corner of its search box.
    """
    by_frame: dict[int, list[Landmark]] = {}
    for landmark in landmarks:
        by_frame.setdefault(landmark.frame, []).append(landmark)
    if not by_frame:
        return None
    frame = min(sorted(by_frame), key=lambda f: (-len(by_frame[f]), f))
    if len(by_frame[frame]) < min_clicks:
        return None
    return frame, by_frame[frame]


def _drift_anchors(landmarks: list[Landmark], min_clicks: int = MIN_CLICKS_PER_ANCHOR) -> list[Landmark]:
    """The clicks offered to the drift fit: only frames whose knot a click count can identify (see MIN_CLICKS_PER_ANCHOR)."""
    counts: dict[int, int] = {}
    for landmark in landmarks:
        counts[landmark.frame] = counts.get(landmark.frame, 0) + 1
    return [landmark for landmark in landmarks if counts[landmark.frame] >= min_clicks]


def _plausible_registration(params: np.ndarray) -> bool:
    """A frame used to register the base pose must not have parked the camera against a height bound.

    One frame of clicks gives eight equations for seven unknowns, so with a bad click among them the solve can land
    anywhere the loss tolerates - measured: a four-click frame with one typo registers at the 0.8 m bound and
    sacrifices the wrong click doing it. Only the bounds say so plainly; the residual of such a solve looks fine.
    """
    return HEIGHT_RANGE[0] + 1e-3 < float(params[2]) < HEIGHT_RANGE[1] - 1e-3


def _residuals(params: np.ndarray, landmarks: list[Landmark], chain: dict[int, tuple[np.ndarray, float]], aspect: float):
    position, rvec, focal_scale = params[:3], params[3:6], params[6]
    base = _rodrigues(rvec)
    out = []
    for lm in landmarks:
        q, focal = chain[lm.frame]
        ray = pixel_rays(np.array([[lm.u, lm.v]]), q, focal, aspect, base, focal_scale)
        hit, valid = intersect_ground(position, ray)
        if not valid[0]:
            out += [50.0, 50.0]  # a clicked ground point whose ray points at the sky: heavily penalise
        else:
            out += [hit[0, 0] - lm.pitch_x, hit[0, 1] - lm.pitch_y]
    return np.asarray(out)


class CalibrationError(ValueError):
    pass


# A landmark is a gross outlier when it is both far off in absolute terms and far worse than its peers. The absolute
# floor matters: distant ground is compressed in the image, so a *correct* click 80 m away can legitimately be 3 m off.
OUTLIER_FLOOR_M = 4.0
OUTLIER_MEDIAN_FACTOR = 4.0
# A set this size or larger may have its worst click dropped and the fit retried. Six is the smallest useful set:
# dropping one leaves five, which is still more than the four the solve needs. Requiring more than that meant a
# six-click set - the common case - could never shed a bad click at all.
MIN_LANDMARKS_AFTER_REJECTION = 5

# When a second refined solution fits the clicks this well (relative cost) from this far away, the set is ambiguous:
# two very different cameras explain the clicks equally, and the residual cannot prefer one.
AMBIGUOUS_COST_FACTOR = 3.0
AMBIGUOUS_POSITION_M = 5.0

# Smallest singular value of the residual Jacobian below which the clicks barely constrain the camera (the camera
# can slide metres for millimetres of residual). Measured: four far landmarks come out at ~0.02; healthy sets with a
# near landmark or a wide pan spread sit at 0.7+.
ILL_CONDITIONED_SIGMA = 0.2


def _jacobian_conditioning(
    params: np.ndarray, landmarks: list[Landmark], chain: dict[int, tuple[np.ndarray, float]], aspect: float
) -> float | None:
    """Smallest singular value of the residual Jacobian at a solution; None when it cannot be evaluated stably."""
    r0 = _residuals(params, landmarks, chain, aspect)
    jacobian = np.zeros((len(r0), len(params)))
    for j in range(len(params)):
        step = 1e-5 * max(abs(params[j]), 1e-2)
        delta = np.zeros(len(params))
        delta[j] = step
        jacobian[:, j] = (_residuals(params + delta, landmarks, chain, aspect) - r0) / step
    if not np.all(np.isfinite(jacobian)):
        return None
    singular = np.linalg.svd(jacobian, compute_uv=False)
    return float(singular[-1])

# Multi-start search: every start gets a short run, then only the best few get a thorough one. Ranking the starts by
# their initial residual is not enough - a start can look hopeless and still be in the right valley, and one that
# looks good can be in a wrong one that fits the clicks plausibly. A coarse pass costs a fraction of a full
# refinement and judges that far better.
COARSE_ITERATIONS = 20
MAX_STARTS = 3
FINE_ITERATIONS = 250


def _frame_starts(
    landmarks: list[Landmark], chain: dict[int, tuple[np.ndarray, float]], aspect: float
) -> list[np.ndarray]:
    """Closed-form starting guesses: one per frame with 4+ landmarks, from its ground-plane -> image homography.

    For a ground point ``X = (x, y, 0)`` seen by a camera at ``C`` with world->camera rotation ``R``,
    ``pixel ~ K R (X - C)``, so the plane-to-image homography is ``K [r1 r2 -R C]``. Decomposing it gives ``R`` and
    ``C`` directly, and ``R_base = R_world<-camera(frame) @ Q_frame^T`` carries it back to the reference frame.
    """
    starts: list[np.ndarray] = []
    by_frame: dict[int, list[Landmark]] = {}
    for lm in landmarks:
        by_frame.setdefault(lm.frame, []).append(lm)
    for frame, group in by_frame.items():
        if len(group) < 4:
            continue
        pitch = np.array([[lm.pitch_x, lm.pitch_y] for lm in group], dtype=np.float64)
        if np.linalg.matrix_rank(pitch - pitch.mean(0), tol=0.5) < 2:
            continue
        pixels = np.array([[lm.u, lm.v] for lm in group], dtype=np.float64)
        homography, _ = cv2.findHomography(pitch, pixels, 0)
        if homography is None:
            continue
        q, focal = chain[frame]
        m = np.linalg.inv(intrinsics(focal, aspect)) @ homography
        scale = np.linalg.norm(m[:, 0])
        if scale < 1e-9:
            continue
        m = m / scale
        if m[2, 2] < 0:  # the pitch must be in front of the camera
            m = -m
        r1, r2, translation = m[:, 0], m[:, 1], m[:, 2]
        u, _, vt = np.linalg.svd(np.column_stack([r1, r2, np.cross(r1, r2)]))
        world_to_camera = u @ vt
        position = -world_to_camera.T @ translation
        if not (HEIGHT_RANGE[0] <= position[2] <= HEIGHT_RANGE[1] * 3):
            continue
        # `pixel_rays` gives world rays as R_base @ Q @ (frame ray), so frame -> world is R_base @ Q, and the
        # decomposed `world_to_camera.T` is that same frame -> world. Hence R_base = world_to_camera^T @ Q^T.
        # (Verified numerically: the Q form is wrong for every non-zero pan and only coincides at Q = I.)
        base = world_to_camera.T @ q.T
        x0 = np.concatenate([position, cv2.Rodrigues(base)[0].ravel(), [1.0]])
        starts.append(x0)
    return starts


def _grid_starts(landmarks: list[Landmark], initial_position: tuple[float, float, float] | None) -> list[np.ndarray]:
    """Fallback when no single frame has four landmarks: positions around the pitch with several headings."""
    pts = np.array([[lm.pitch_x, lm.pitch_y] for lm in landmarks])
    centre = pts.mean(0)
    z0 = initial_position[2] if initial_position else 3.0
    xy0 = np.array(initial_position[:2]) if initial_position else None
    spread = max(np.ptp(pts[:, 0]), np.ptp(pts[:, 1]), 10.0)
    starts = []
    for dx, dy in ([(0, -1), (0, 1), (-1, 0), (1, 0)] if xy0 is None else [(0, 0)]):
        start_xy = xy0 if xy0 is not None else centre + np.array([dx, dy]) * spread * 0.9
        heading = np.arctan2(*(centre - start_xy)[::-1]) if xy0 is None else 0.0
        for yaw_offset in (0.0, np.pi / 2, -np.pi / 2, np.pi):
            look = np.array([np.cos(heading + yaw_offset), np.sin(heading + yaw_offset), -0.15])
            z_axis = look / np.linalg.norm(look)
            x_axis = np.cross(np.array([0.0, 0.0, 1.0]), z_axis)
            x_axis /= np.linalg.norm(x_axis) + 1e-12
            r0 = np.column_stack([x_axis, np.cross(z_axis, x_axis), z_axis])  # camera -> world
            starts.append(np.concatenate([[start_xy[0], start_xy[1], z0], cv2.Rodrigues(r0)[0].ravel(), [1.0]]))
    return starts


def _ray_starts(
    landmarks: list[Landmark], chain: dict[int, tuple[np.ndarray, float]], aspect: float
) -> list[np.ndarray]:
    """Starts that guess only the camera *position*, and derive the rotation from the data in closed form.

    This is what makes a handful of landmarks work. Guessing a heading as well (as `_grid_starts` does) puts the
    solver in the wrong basin for exactly the sets that need help most: a few landmarks, all of them distant, where
    the cost surface has several troughs. But once a position is assumed, the rotation is not a guess at all - each
    click gives a ray in the reference frame and a known world direction from that position, and Kabsch aligns the
    two. So one position hypothesis buys a well-aimed start for the price of an SVD.
    """
    pts = np.array([[lm.pitch_x, lm.pitch_y] for lm in landmarks], dtype=np.float64)
    centre = pts.mean(0)
    spread = max(float(np.ptp(pts[:, 0])), float(np.ptp(pts[:, 1])), 10.0)

    # Observed rays in the reference frame, independent of where the camera turns out to be.
    reference_rays = []
    for lm in landmarks:
        q, focal = chain[lm.frame]
        camera = np.linalg.inv(intrinsics(focal, aspect)) @ np.array([lm.u, lm.v, 1.0])
        reference_rays.append(q @ (camera / np.linalg.norm(camera)))
    reference_rays = np.asarray(reference_rays)

    starts: list[np.ndarray] = []
    # Where the camera can plausibly be: standing beside the pitch, or among the clicks (halfway-line work from the
    # middle). The heading is not part of the hypothesis - Kabsch supplies it - so the positions need not be dense.
    x_min, x_max = float(pts[:, 0].min()), float(pts[:, 0].max())
    y_min, y_max = float(pts[:, 1].min()), float(pts[:, 1].max())
    mid_x, mid_y = (x_min + x_max) / 2, (y_min + y_max) / 2
    positions: list[tuple[float, float]] = [(mid_x, mid_y)]
    for outward in (12.0, 35.0):
        positions += [
            (x_min - outward, mid_y),
            (x_max + outward, mid_y),
            (mid_x, y_min - outward),
            (mid_x, y_max + outward),
        ]

    for px, py in positions:
        for height in (2.5, 6.0):
            position = np.array([px, py, height])
            to_landmark = pts - position[:2]
            distance = np.linalg.norm(to_landmark, axis=1)
            if (distance < 1e-6).any():
                continue
            world_dirs = np.column_stack(
                [to_landmark[:, 0] / distance, to_landmark[:, 1] / distance, -height / distance]
            )
            u, _s, vt = np.linalg.svd(world_dirs.T @ reference_rays)
            sign = np.sign(np.linalg.det(u @ vt))
            base = u @ np.diag([1.0, 1.0, sign]) @ vt  # reference rays -> world rays = R_base
            starts.append(np.concatenate([position, cv2.Rodrigues(base)[0].ravel(), [1.0]]))
    return starts


def _solve(
    landmarks: list[Landmark],
    chain: dict[int, tuple[np.ndarray, float]],
    aspect: float,
    initial_position: tuple[float, float, float] | None,
    robust: bool,
):
    starts = _frame_starts(landmarks, chain, aspect) + _ray_starts(landmarks, chain, aspect)
    if initial_position is not None:
        # A caller-supplied position (e.g. the previous solution) is tried first; the rest are the fallback.
        starts = _grid_starts(landmarks, initial_position)[:4] + starts
    if not starts:
        starts = _grid_starts(landmarks, None)

    lower = np.array([-500, -500, HEIGHT_RANGE[0], -np.pi * 2, -np.pi * 2, -np.pi * 2, FOCAL_SCALE_RANGE[0]])
    upper = np.array([500, 500, HEIGHT_RANGE[1], np.pi * 2, np.pi * 2, np.pi * 2, FOCAL_SCALE_RANGE[1]])
    loss = "soft_l1" if robust else "linear"

    def run(x0: np.ndarray, iterations: int):
        return least_squares(
            _residuals,
            np.clip(x0, lower + 1e-9, upper - 1e-9),
            args=(landmarks, chain, aspect),
            bounds=(lower, upper),
            loss=loss,
            f_scale=1.5,
            max_nfev=iterations,
        )

    coarse: list[tuple[float, np.ndarray]] = []
    for x0 in starts:
        try:
            fit = run(x0, COARSE_ITERATIONS)
        except ValueError:
            continue
        if np.isfinite(fit.cost):
            coarse.append((float(fit.cost), fit.x))
    if not coarse:
        raise CalibrationError("solver failed to start")
    coarse.sort(key=lambda item: item[0])

    best: least_squares | None = None
    refined: list[tuple[float, np.ndarray]] = []
    for _cost, x0 in coarse[:MAX_STARTS]:
        try:
            fit = run(x0, FINE_ITERATIONS)
        except ValueError:
            continue
        if not np.isfinite(fit.cost):
            continue
        refined.append((float(fit.cost), fit.x))
        if best is None or fit.cost < best.cost:
            best = fit
    if best is None:
        raise CalibrationError("solver failed to refine any starting guess")
    refined.sort(key=lambda item: item[0])
    best_x = refined[0][1]
    # A very different camera fitting the clicks almost as well is the honest answer to an under-constrained set
    # (e.g. every landmark far away): the residual alone cannot tell the two apart, so say so.
    ambiguous = any(
        cost <= refined[0][0] * AMBIGUOUS_COST_FACTOR + 1e-9 and np.linalg.norm(x[:3] - best_x[:3]) > AMBIGUOUS_POSITION_M
        for cost, x in refined[1:]
    )
    return best, ambiguous


def _errors(params: np.ndarray, landmarks: list[Landmark], chain: dict, aspect: float) -> np.ndarray:
    return np.linalg.norm(_residuals(params, landmarks, chain, aspect).reshape(-1, 2), axis=1)


# A click counts as agreeing with the fit when it is within this of the best-fitting click. Note this is anchored on
# the *best* click rather than the median: when most of the clicks are wrong the median moves with them, and a
# median-based rule cheerfully declares a fit where nothing agrees with anything.
AGREE_FLOOR_M = 2.0
AGREE_FACTOR = 3.0


@dataclass
class FitDiagnosis:
    """What can be said about a calibration without seeing the clicks.

    Kept separate from `CalibrationError` because a bad fit is not an exception - the numbers exist, they are just
    not to be trusted, and the useful thing is to say which clicks the fit could not reconcile.
    """

    reason: str | None = None  # a parameter parked on a bound, which means the fit is not a solution at all
    agreeing: tuple[int, ...] = ()
    disagreeing: tuple[int, ...] = ()
    tolerance_m: float = 0.0

    @property
    def too_few_agreeing(self) -> bool:
        """True when not even the minimum four clicks can be brought into agreement."""
        return len(self.agreeing) < 4


def diagnose_fit(calibration: PitchCalibration) -> FitDiagnosis:
    """Characterise a calibration: was it a real fit, and if not, which clicks disagree.

    A parameter on the edge of its search range is the solver saying it wanted to go further, so the answer is not a
    solution of the problem but the corner of the box. Otherwise the clicks are sorted by how well they fit: those
    close to the best one agree with each other, and the rest are the ones to look at.
    """
    residuals = np.asarray(calibration.residuals_m, dtype=np.float64)
    if residuals.size == 0:
        return FitDiagnosis(reason=suspect_fit_reason(calibration))
    tolerance = max(AGREE_FLOOR_M, AGREE_FACTOR * float(residuals.min()))
    agreeing = tuple(int(i) for i, r in enumerate(residuals) if r <= tolerance)
    disagreeing = tuple(int(i) for i, r in enumerate(residuals) if r > tolerance)
    return FitDiagnosis(
        reason=suspect_fit_reason(calibration),
        agreeing=agreeing,
        disagreeing=disagreeing,
        tolerance_m=tolerance,
    )


def suspect_fit_reason(calibration: PitchCalibration) -> str | None:
    """Why this fit should not be trusted, or None if nothing obviously wrong turned up.

    A parameter parked on the edge of its search range is the solver saying it wanted to go further: the answer is
    not a solution of the problem, it is the corner of the box it was allowed to search. Reporting that plainly is
    far more use than presenting a camera position and a residual that look like measurements.

    Worth knowing when reading the reason: this is *not* how a wrong match format shows up. Telling the solver the
    pitch is the wrong size leaves the residual small and moves the recovered camera height instead (see
    `format_scale_note`), so a pinned parameter means the clicks themselves could not be reconciled.
    """
    if calibration.ill_conditioned:
        return "the clicks barely constrain the camera - the residual can look small while the camera is tens of " \
               "metres out - add a landmark closer to the camera or spread them wider across frames"
    if calibration.ambiguous:
        return "two very different cameras fit the clicks almost equally well, so the landmarks do not pin the " \
               "camera down - add a landmark closer to the camera"
    if calibration.focal_scale >= FOCAL_SCALE_RANGE[1] - 1e-3:
        return "ran the lens scaling to the top of its range and stopped there - the clicks ask for a lens longer " \
               "than any real one for this footage"
    if calibration.focal_scale <= FOCAL_SCALE_RANGE[0] + 1e-3:
        return "ran the lens scaling to the bottom of its range and stopped there - the clicks ask for a lens " \
               "shorter than any real one for this footage"
    height = float(calibration.position[2])
    if height <= HEIGHT_RANGE[0] + 1e-3 or height >= HEIGHT_RANGE[1] - 1e-3:
        return f"put the camera at the edge of its height range ({height:.1f} m), which is not a real camera"
    return None


# A gimbal on a tripod is somewhere in here. Used only as a hint, and only for the scale of the pitch.
TRIPOD_HEIGHT_RANGE_M = (1.0, 5.0)


def format_scale_note(calibration: PitchCalibration) -> str | None:
    """A hint that the selected pitch size does not match the pitch that was filmed, or None.

    Measured on the simulated match: telling the solver the pitch is 0.6x, 1.0x or 1.67x its real size leaves the
    residuals small (0.5-1.6 m) and moves the recovered camera height with it instead - 2.3 m, 3.9 m, 6.9 m. So the
    residual is no guide to a wrong format and the height is, which is the opposite of what one would assume.
    """
    height = float(calibration.position[2])
    if TRIPOD_HEIGHT_RANGE_M[0] <= height <= TRIPOD_HEIGHT_RANGE_M[1]:
        return None
    return (
        f"the camera came out {height:.1f} m high, which is outside the usual tripod range. A match format that "
        "does not match the pitch you filmed shows up here rather than in the residual - the format sets the metre "
        "scale of everything reported - so it is worth double-checking."
    )

def calibrate(
    landmarks: list[Landmark],
    chain: dict[int, tuple[np.ndarray, float]],
    aspect: float,
    *,
    initial_position: tuple[float, float, float] | None = None,
    robust: bool = True,
    reject_outliers: bool = True,
    correct_drift: bool = True,
) -> PitchCalibration:
    """Solves camera position, base orientation and focal scale from landmark clicks.

    ``chain[frame] = (Q, focal)`` is the camera-motion state of every frame that has a click, where ``Q`` maps
    frame rays to reference-frame rays. Gross outliers (a mis-clicked or mislabelled landmark) are dropped and the
    fit repeated without them; they stay in ``residuals_m`` and are listed in ``excluded`` so a UI can point at them.

    When the clicks sit on more than one frame, a time-varying drift correction is fitted as well (see
    ``geometry.drift``) and everything after that - the outlier pass, the residuals, the conditioning check - runs
    on the corrected chain. That order matters in both directions: a good click late in the video would otherwise
    look like an outlier because the chain had drifted away from it, and a gross mis-click would otherwise be able
    to bend the trajectory towards itself. The base pose itself is registered on the frame clicked most, when one
    frame carries at least four clicks (see ``_reference_anchor``); the correction is bounded and only fitted to
    moments whose clicks can identify it (see ``MIN_CLICKS_PER_ANCHOR``), so a lone click cannot move the pose.
    """
    if len(landmarks) < 4:
        raise CalibrationError("need at least 4 landmarks (each gives 2 equations for 7 unknowns)")
    missing = {lm.frame for lm in landmarks} - set(chain)
    if missing:
        raise CalibrationError(f"no camera state for frames {sorted(missing)}")
    pts = np.array([[lm.pitch_x, lm.pitch_y] for lm in landmarks])
    if np.linalg.matrix_rank(pts - pts.mean(0), tol=0.5) < 2:
        raise CalibrationError("landmarks are collinear; spread them across the pitch")

    # Register the base pose on the moment clicked most, when there is one: four landmarks on a single frame make a
    # pose, and every other moment then only says how far the chain had drifted by its time. The alternative - one
    # pose pooled from every click - is what a spread-click solve degenerates from: clicks either side of a drifted
    # span cannot agree on one pose, and the solver answers with a corner of its search box (measured: with the
    # whole-game clicks it parked the camera at the 12 m height bound, where the frame clicked five times registers
    # the same clicks at 5.3 m). The reference frame is chosen by raw click count, then fitted on its own - judging
    # its clicks by a pose pooled from everywhere else would be circular, and would drop the very clicks that make
    # the frame a reference.
    reference = _reference_anchor(landmarks) if correct_drift else None
    fit: least_squares | None = None
    ambiguous = False
    if reference is not None:
        frame, group = reference
        try:
            fit, ambiguous = _solve(group, {frame: chain[frame]}, aspect, initial_position, robust)
        except CalibrationError:
            fit = None  # a degenerate run of clicks on one frame; fall back to pooling every click
        if fit is not None:
            # The frame must stand on its own clicks: at least four of them must survive a screen against the fit.
            # Four is the minimum that registers a pose, so a screen that keeps fewer is saying the frame cannot
            # carry the pose by itself - and a registration against a height bound is a solve that tolerated its
            # worst click, not a camera. Both cases fall back to pooling every click, the older behaviour.
            kept = _agreeing_per_anchor(group, list(_errors(fit.x, group, chain, aspect)))
            if len(kept) < MIN_REFERENCE_CLICKS or not _plausible_registration(fit.x):
                fit = None
            elif len(kept) < len(group):
                try:
                    refit, refit_ambiguous = _solve(kept, {frame: chain[frame]}, aspect, initial_position, robust)
                except CalibrationError:
                    fit = None
                else:
                    fit = refit if _plausible_registration(refit.x) else None
                    ambiguous = refit_ambiguous
    if fit is None:
        fit, ambiguous = _solve(landmarks, chain, aspect, initial_position, robust)
    drift: DriftCorrection | None = None
    if correct_drift:
        # The correction is fitted once, against the single registered pose, and the chain handed to the rest of the
        # function *is* the corrected chain - so the outlier pass, the residuals and the conditioning check all see
        # the poses the projection will use. Refitting the pose against the corrected chain and the correction
        # against *that* was tried and dropped: the rotation of the reference frame and a base rotation covering the
        # same angle are the same projection, and the pair walked that freedom off a cliff (a round took a 0.3 deg
        # correction to 17 m of click error, warm-started or not). One pass is stable and the pose it leaves behind
        # is the pose that describes the clicks.
        #
        # The clicks a knot may be fitted to are the ones that agree with their anchor-mates: a *drifted* anchor is
        # off as a group (they still agree with each other), while a mis-click disagrees with its own mates - and
        # given a knot of its own it could absorb almost any error, silencing the outlier report.
        trusted = _agreeing_per_anchor(landmarks, _errors(fit.x, landmarks, chain, aspect))
        base = PitchCalibration(
            position=fit.x[:3].copy(),
            base_rotation=_rodrigues(fit.x[3:6]),
            focal_scale=float(fit.x[6]),
            aspect=aspect,
            rms_error_m=0.0,
            residuals_m=(),
        )
        drift = fit_drift(_drift_anchors(trusted), base, chain, aspect)
        if drift is not None:
            chain = {
                frame: (drift.rotation(frame) @ q, focal * drift.scale(frame))
                for frame, (q, focal) in chain.items()
            }
    final_landmarks = landmarks
    excluded: list[int] = []
    if reject_outliers and len(landmarks) > MIN_LANDMARKS_AFTER_REJECTION:
        errors = _errors(fit.x, landmarks, chain, aspect)
        threshold = max(OUTLIER_FLOOR_M, OUTLIER_MEDIAN_FACTOR * float(np.median(errors)))
        bad = [i for i, e in enumerate(errors) if e > threshold]
        keep = [i for i in range(len(landmarks)) if i not in bad]
        # Never strip so many that the fit becomes under-determined or loses its spread across the pitch.
        kept_pts = pts[keep] if keep else pts
        if bad and len(keep) >= MIN_LANDMARKS_AFTER_REJECTION and np.linalg.matrix_rank(kept_pts - kept_pts.mean(0), tol=0.5) >= 2:
            final_landmarks = [landmarks[i] for i in keep]
            refit, ambiguous = _solve(final_landmarks, chain, aspect, initial_position, robust)
            fit, excluded = refit, bad

    errors = _errors(fit.x, landmarks, chain, aspect)  # every landmark, including excluded ones
    inliers = [e for i, e in enumerate(errors) if i not in excluded]
    sigma_min = _jacobian_conditioning(fit.x, final_landmarks, chain, aspect)
    ill_conditioned = sigma_min is None or sigma_min < ILL_CONDITIONED_SIGMA
    return PitchCalibration(
        position=fit.x[:3].copy(),
        base_rotation=_rodrigues(fit.x[3:6]),
        focal_scale=float(fit.x[6]),
        aspect=aspect,
        rms_error_m=float(np.sqrt(np.mean(np.square(inliers)))),
        residuals_m=tuple(float(e) for e in errors),
        excluded=tuple(excluded),
        ambiguous=ambiguous,
        ill_conditioned=ill_conditioned,
        drift=drift,
    )


def pixels_to_pitch(
    calibration: PitchCalibration, uv: np.ndarray, q: np.ndarray, focal: float
) -> tuple[np.ndarray, np.ndarray]:
    """Map normalised pixels of one frame to pitch metres; returns ``(xy (N,2), valid (N,))``."""
    rays = pixel_rays(uv, q, focal, calibration.aspect, calibration.base_rotation, calibration.focal_scale)
    return intersect_ground(calibration.position, rays)


def pitch_to_pixels(
    calibration: PitchCalibration, xy: np.ndarray, q: np.ndarray, focal: float
) -> tuple[np.ndarray, np.ndarray]:
    """Project pitch points (N, 2) into a frame; returns ``(uv (N,2), in_front (N,))`` (uv normalised by width)."""
    xy = np.asarray(xy, dtype=np.float64).reshape(-1, 2)
    world = np.column_stack([xy - calibration.position[None, :2], -np.full(len(xy), calibration.position[2])])
    cam = world @ (calibration.base_rotation @ q)  # = (R^-1) world  with R = base @ q
    k = intrinsics(focal * calibration.focal_scale, calibration.aspect)
    projected = cam @ k.T
    z = projected[:, 2]
    in_front = z > 1e-9
    uv = np.full((len(xy), 2), np.nan)
    uv[in_front] = projected[in_front, :2] / z[in_front, None]
    return uv, in_front


def recalibrate_orientation(
    base: PitchCalibration,
    landmarks: list[Landmark],
    chain: dict[int, tuple[np.ndarray, float]],
) -> PitchCalibration:
    """New segment, same tripod: keep camera position and focal scale, re-solve only the base orientation.

    Needs just two landmarks (each is a ray constraint). Initialised in closed form: each landmark's true world
    direction from the (known) camera is aligned to its observed reference-frame ray with a Kabsch rotation, then
    refined on the ground-plane residual.
    """
    if len(landmarks) < 2:
        raise CalibrationError("need at least 2 landmarks to re-orient a known camera")
    missing = {lm.frame for lm in landmarks} - set(chain)
    if missing:
        raise CalibrationError(f"no camera state for frames {sorted(missing)}")

    ref_rays, world_dirs = [], []
    for lm in landmarks:
        q, focal = chain[lm.frame]
        k = intrinsics(focal * base.focal_scale, base.aspect)
        cam = np.linalg.inv(k) @ np.array([lm.u, lm.v, 1.0])
        ref = q @ (cam / np.linalg.norm(cam))  # frame ray -> reference-frame ray
        ref_rays.append(ref)
        direction = np.array([lm.pitch_x - base.position[0], lm.pitch_y - base.position[1], -base.position[2]])
        world_dirs.append(direction / np.linalg.norm(direction))
    u, _, vt = np.linalg.svd(np.asarray(world_dirs).T @ np.asarray(ref_rays))
    sign = np.sign(np.linalg.det(u @ vt))
    r0 = u @ np.diag([1.0, 1.0, sign]) @ vt  # reference rays -> world rays

    def residual(rvec: np.ndarray) -> np.ndarray:
        rotation = _rodrigues(rvec)
        out = []
        for lm in landmarks:
            q, focal = chain[lm.frame]
            ray = pixel_rays(np.array([[lm.u, lm.v]]), q, focal, base.aspect, rotation, base.focal_scale)
            hit, valid = intersect_ground(base.position, ray)
            out += [50.0, 50.0] if not valid[0] else [hit[0, 0] - lm.pitch_x, hit[0, 1] - lm.pitch_y]
        return np.asarray(out)

    fit = least_squares(residual, cv2.Rodrigues(r0)[0].ravel(), loss="soft_l1", f_scale=1.5, max_nfev=200)
    errors = np.linalg.norm(fit.fun.reshape(-1, 2), axis=1)
    return PitchCalibration(
        position=base.position.copy(),
        base_rotation=_rodrigues(fit.x),
        focal_scale=base.focal_scale,
        aspect=base.aspect,
        rms_error_m=float(np.sqrt(np.mean(errors**2))),
        residuals_m=tuple(float(e) for e in errors),
    )
