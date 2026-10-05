"""Automatic pitch registration: turn detected pitch keypoints into a camera calibration.

The reference project detects 32 field markers with a heatmap model and matches them to a pitch template. This
project's solver (`geometry.pitch_calibration.calibrate`) already turns pixel<->pitch correspondences from a
*moving* camera into a camera pose; what it needs is correspondences. This module is the bridge: it takes keypoint
observations from any source (a heatmap model, a classical line detector, hand-placed points), screens them, and
hands the survivors to the solver as landmarks.

Why the extra screening, when `calibrate` already rejects gross outliers: automatic keypoints arrive with *wrong*
indices, not just noise. A detector that fires on a goalpost and calls it the centre spot contributes a confident,
plausible-looking correspondence that is simply wrong, and there can be several per frame. `calibrate`'s rejection
is a global pass anchored on the reference frame, so a bad frame can drag the whole pooled fit before it gets a
chance to reject anything.

The screen is the geometry of a single frame: on one frame the ground plane maps to the image by a homography, so
the true keypoints of that frame must all fit one. `cv2.findHomography` with RANSAC finds the largest subset that
does, and the rest of that frame is dropped before the solver sees it. That is a per-frame, model-free consistency
test - it needs no camera pose and rejects a wrong-index point even when it is the majority in its own frame, as
long as four agreeing points survive to define the frame's homography.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from soccer_analytics.geometry.pitch_calibration import (
    Landmark,
    PitchCalibration,
    calibrate,
    pitch_to_pixels,
    recalibrate_orientation,
)
from soccer_analytics.geometry.pitch_template import template_for

# A keypoint below this confidence is treated as a guess and not used. The reference detector's own threshold is
# 0.2; 0.3 is a little stricter because its false positives are expensive here (a wrong index is a wrong landmark).
DEFAULT_MIN_CONFIDENCE = 0.3
# Reprojection distance within which a keypoint counts as agreeing with its frame's homography, in pixels at the
# nominal analysis width. The reference refines with a 3 px RANSAC threshold on a *fixed* camera; a following gimbal
# at 4K is noisier, and the point of this test is to catch wrong indices (metres of error), not sub-pixel jitter.
DEFAULT_RANSAC_THRESHOLD_PX = 12.0
NOMINAL_FRAME_WIDTH = 1920.0
# A frame needs this many agreeing keypoints before it contributes landmarks at all. Four is exactly the number that
# defines its homography; a frame that cannot keep four is not evidence of anything and is skipped.
MIN_FRAME_KEYPOINTS = 4


@dataclass(frozen=True)
class KeypointObservation:
    """One detected pitch marker: template ``index`` seen at frame-normalised ``(u, v)``.

    ``u`` and ``v`` are both divided by the frame *width*, the convention the camera model and the solver use (``v``
    is not the fraction of the frame height). ``confidence`` is the detector's own score, 0-1.
    """

    frame: int
    index: int
    u: float
    v: float
    confidence: float = 1.0


@dataclass
class RegistrationResult:
    """A calibration plus what it was built from, so the page can say how much evidence there was."""

    calibration: PitchCalibration
    frames_used: tuple[int, ...]  # frames that contributed at least one surviving keypoint
    keypoints_total: int  # observations offered
    keypoints_kept: int  # observations that survived the per-frame screen
    frames: dict[int, dict]  # frame -> {"detected": n, "kept": m}
    notes: list[str] = field(default_factory=list)


def _as_arrays(points: list[KeypointObservation], template: list[tuple[float, float]]):
    source = np.array([template[p.index] for p in points], dtype=np.float64)
    target = np.array([[p.u, p.v] for p in points], dtype=np.float64)
    return source, target


def filter_frame(
    points: list[KeypointObservation],
    template: list[tuple[float, float]],
    *,
    ransac_threshold_px: float = DEFAULT_RANSAC_THRESHOLD_PX,
    frame_width: float = NOMINAL_FRAME_WIDTH,
) -> list[KeypointObservation]:
    """The subset of one frame's keypoints that agree on a single ground-plane homography.

    A frame with fewer than ``MIN_FRAME_KEYPOINTS`` observations is returned unchanged: there is nothing to test it
    against, and its points are still evidence - the global solver can weigh a lone correspondence, and rejecting it
    here would throw away a frame that might be the one carrying the reference pose.
    """
    if len(points) < MIN_FRAME_KEYPOINTS:
        return list(points)
    source, target = _as_arrays(points, template)
    threshold = float(ransac_threshold_px) / max(float(frame_width), 1.0)
    homography, mask = cv2.findHomography(source, target, cv2.RANSAC, threshold, maxIters=2000, confidence=0.995)
    if homography is None or mask is None:
        return list(points)
    return [point for point, keep in zip(points, mask.ravel().astype(bool)) if keep]


def observations_to_landmarks(
    observations: list[KeypointObservation], template: list[tuple[float, float]]
) -> list[Landmark]:
    """Landmarks for the solver, with the template index kept as the label for reporting."""
    out: list[Landmark] = []
    for observation in observations:
        if not 0 <= observation.index < len(template):
            continue
        x, y = template[observation.index]
        out.append(Landmark(observation.frame, observation.u, observation.v, x, y, f"kp{observation.index}"))
    return out


def auto_register(
    observations: list[KeypointObservation],
    chain: dict[int, tuple[np.ndarray, float]],
    aspect: float,
    *,
    length_m: float = 105.0,
    width_m: float = 68.0,
    template: list[tuple[float, float]] | None = None,
    min_confidence: float = DEFAULT_MIN_CONFIDENCE,
    ransac_threshold_px: float = DEFAULT_RANSAC_THRESHOLD_PX,
    frame_width: float = NOMINAL_FRAME_WIDTH,
    correct_drift: bool = True,
    **calibrate_kwargs,
) -> RegistrationResult:
    """Register the camera from pitch-keypoint detections, screening each frame before the solver fits.

    ``observations`` are indexed by the template order in :mod:`soccer_analytics.geometry.pitch_template` (the
    reference project's 32 field markers). ``chain`` and ``aspect`` are the same motion-chain input `calibrate`
    takes. Frames are screened independently, then every survivor is pooled into one robust fit, so keypoints from
    many moments across a panning camera still register one tripod.

    ``calibrate_kwargs`` are passed to :func:`~soccer_analytics.geometry.pitch_calibration.calibrate` (e.g.
    ``initial_position`` from a previous calibration).
    """
    template = template if template is not None else template_for(length_m, width_m)
    accepted = [o for o in observations if o.confidence >= min_confidence and 0 <= o.index < len(template)]

    by_frame: dict[int, list[KeypointObservation]] = {}
    for observation in accepted:
        by_frame.setdefault(int(observation.frame), []).append(observation)

    kept: list[KeypointObservation] = []
    per_frame: dict[int, dict] = {}
    for frame in sorted(by_frame):
        group = by_frame[frame]
        survivors = filter_frame(
            group, template, ransac_threshold_px=ransac_threshold_px, frame_width=frame_width
        )
        kept.extend(survivors)
        per_frame[frame] = {"detected": len(group), "kept": len(survivors)}

    notes: list[str] = []
    if len(accepted) < len(observations):
        notes.append(
            f"{len(observations) - len(accepted)} keypoint(s) below the confidence floor or off the template "
            "were ignored."
        )
    if len(kept) < MIN_FRAME_KEYPOINTS:
        raise ValueError(
            f"only {len(kept)} keypoint(s) survived screening; at least {MIN_FRAME_KEYPOINTS} agreeing markers are "
            "needed to register a camera"
        )

    landmarks = observations_to_landmarks(kept, template)
    calibration = calibrate(
        landmarks,
        chain,
        aspect,
        correct_drift=correct_drift,
        **calibrate_kwargs,
    )
    frames_used = tuple(sorted({landmark.frame for landmark in landmarks}))
    if len(kept) < len(accepted):
        notes.append(
            f"{len(accepted) - len(kept)} keypoint(s) disagreed with their frame's homography and were dropped "
            "before the fit."
        )
    return RegistrationResult(
        calibration=calibration,
        frames_used=frames_used,
        keypoints_total=len(observations),
        keypoints_kept=len(kept),
        frames=per_frame,
        notes=notes,
    )


def score_calibration(
    calibration: PitchCalibration,
    observations: list[KeypointObservation],
    chain: dict[int, tuple[np.ndarray, float]],
    template: list[tuple[float, float]],
    *,
    frame_width: float = NOMINAL_FRAME_WIDTH,
    tolerance_px: float = DEFAULT_RANSAC_THRESHOLD_PX,
) -> list[int]:
    """Indices of the observations this calibration reprojects onto, within a pixel tolerance.

    This is the inlier test the register-around-a-prior search maximises. A calibration explaining the main pitch
    will reproject *both* that pitch's markers and nothing else; the neighbouring goals and kickwalls are laid out
    on a different plane and are not explained by it, which is what separates the right answer from a confident fit
    to the wrong pitch.

    The tolerance should reflect the detector's real accuracy. A broadcast-trained model on this footage is good to
    a few tens of pixels, not the sub-pixel a hand click gives, and demanding more than it can deliver rejects the
    main pitch's own markers as outliers.
    """
    inliers: list[int] = []
    tolerance = tolerance_px / max(frame_width, 1.0)
    for index, observation in enumerate(observations):
        if observation.frame not in chain:
            continue
        q, focal = chain[observation.frame]
        hit, in_front = pitch_to_pixels(calibration, np.asarray([template[observation.index]]), q, focal)
        if not in_front[0] or not np.isfinite(hit[0]).all():
            continue
        if float(np.linalg.norm(hit[0] - np.array([observation.u, observation.v]))) <= tolerance:
            inliers.append(index)
    return inliers


def register_with_position_prior(
    observations: list[KeypointObservation],
    chain: dict[int, tuple[np.ndarray, float]],
    aspect: float,
    *,
    position_prior: tuple[float, float, float],
    focal_scale_prior: float = 1.0,
    length_m: float = 105.0,
    width_m: float = 68.0,
    template: list[tuple[float, float]] | None = None,
    min_confidence: float = DEFAULT_MIN_CONFIDENCE,
    ransac_threshold_px: float = DEFAULT_RANSAC_THRESHOLD_PX,
    frame_width: float = NOMINAL_FRAME_WIDTH,
    min_inliers: int = MIN_FRAME_KEYPOINTS,
    agree_px: float = DEFAULT_RANSAC_THRESHOLD_PX,
    correct_drift: bool = True,
    **calibrate_kwargs,
) -> RegistrationResult:
    """Register with the camera position known (a fixed PTZ on a tripod), searching orientations by RANSAC.

    This is the mode for footage where other pitches, goals or kickwalls share the frame. A free-pose fit has
    enough degrees of freedom to lock onto a neighbouring structure and still look confident - the real game here
    has a second goal in shot, and a full solve happily registered against it. Fixing the position removes that
    freedom: the only question left is which way the camera was pointing, and only the markers of *this* pitch can
    answer it consistently across a pan.

    The search is deliberately coarse. Each frame's screened keypoints are a candidate; re-orienting the known
    camera on them gives a pose, and that pose is scored by how many detections from *every* frame it reprojects.
    The candidate with the widest agreement wins, and the winner's inliers are then handed to the full solver, which
    is free to refine position and focal scale but starts from the prior.
    """
    template = template if template is not None else template_for(length_m, width_m)
    accepted = [o for o in observations if o.confidence >= min_confidence and 0 <= o.index < len(template)]
    by_frame: dict[int, list[KeypointObservation]] = {}
    for observation in accepted:
        by_frame.setdefault(int(observation.frame), []).append(observation)

    kept: list[KeypointObservation] = []
    per_frame: dict[int, dict] = {}
    for frame in sorted(by_frame):
        group = filter_frame(by_frame[frame], template, ransac_threshold_px=ransac_threshold_px, frame_width=frame_width)
        kept.extend(group)
        per_frame[frame] = {"detected": len(by_frame[frame]), "kept": len(group)}
    if len(kept) < MIN_FRAME_KEYPOINTS:
        raise ValueError(
            f"only {len(kept)} keypoint(s) survived screening; at least {MIN_FRAME_KEYPOINTS} agreeing markers are "
            "needed to register a camera"
        )

    prior = PitchCalibration(
        position=np.asarray(position_prior, dtype=np.float64),
        base_rotation=np.eye(3),
        focal_scale=float(focal_scale_prior),
        aspect=aspect,
        rms_error_m=0.0,
        residuals_m=(),
    )

    # One candidate per frame that carries enough screened keypoints to fix an orientation on its own.
    candidates: list[tuple[int, PitchCalibration]] = []
    for frame in sorted(by_frame):
        group = [o for o in kept if o.frame == frame]
        if len(group) < MIN_FRAME_KEYPOINTS:
            continue
        landmarks = observations_to_landmarks(group, template)
        try:
            candidate = recalibrate_orientation(prior, landmarks, chain)
        except Exception:  # a degenerate group just is not a candidate
            continue
        candidates.append((frame, candidate))

    if not candidates:
        raise ValueError("no frame carried enough keypoints to try an orientation with the known camera position")

    best_calibration, best_inliers, best_frame = candidates[0][1], [], candidates[0][0]
    for frame, candidate in candidates:
        inliers = score_calibration(
            candidate, kept, chain, template, frame_width=frame_width, tolerance_px=agree_px
        )
        if len(inliers) > len(best_inliers):
            best_calibration, best_inliers, best_frame = candidate, inliers, frame

    if len(best_inliers) < min_inliers:
        raise ValueError(
            f"the best orientation explains only {len(best_inliers)} keypoint(s); at least {min_inliers} agree on the "
            "main pitch"
        )

    inlier_observations = [kept[index] for index in best_inliers]
    landmarks = observations_to_landmarks(inlier_observations, template)
    notes: list[str] = []
    if len(kept) < len(accepted):
        notes.append(f"{len(accepted) - len(kept)} keypoint(s) disagreed with their frame's homography.")
    notes.append(
        f"orientation fixed on frame {best_frame}, then {len(best_inliers)} keypoint(s) across "
        f"{len({o.frame for o in inlier_observations})} frame(s) agreed on the main pitch."
    )
    try:
        calibration = calibrate(
            landmarks,
            chain,
            aspect,
            initial_position=tuple(float(v) for v in prior.position),
            correct_drift=correct_drift,
            **calibrate_kwargs,
        )
    except Exception as exc:  # a prior-constrained pose is still useful; report the refinement failure
        calibration = best_calibration
        notes.append(f"the full refinement failed ({exc}); the prior-constrained pose is reported instead.")

    return RegistrationResult(
        calibration=calibration,
        frames_used=tuple(sorted({o.frame for o in inlier_observations})),
        keypoints_total=len(observations),
        keypoints_kept=len(inlier_observations),
        frames=per_frame,
        notes=notes,
    )


def registration_note(result: RegistrationResult) -> str:
    """A one-line summary a person can act on, mirroring the page's tone: say what the evidence was."""
    kept = result.keypoints_kept
    total = result.keypoints_total
    frames = len(result.frames_used)
    share = kept / max(1, total)
    return (
        f"registered from {kept}/{total} keypoints ({share:.0%}) across {frames} frame(s); "
        f"rms error {result.calibration.rms_error_m:.2f} m"
    )