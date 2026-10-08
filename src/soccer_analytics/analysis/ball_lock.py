"""The gimbal's own ball lock: a second ball measurement, and possession that needs no homography.

The camera tracks the ball to aim itself, and its log records that tracking loop: whether it is holding the ball
(``Lock:a/b``), where the ball sits in the picture (``Ballx``, and the box's offset from the frame centre), how big
it looks (``BallSz``) and how fast it is moving in the image (``xv``/``yv``). That is a *hardware* ball measurement
beside the expensive ``analysis.ball`` scan, and it is available for the whole game at no extra cost.

Two things it gives that the pitch-space pipeline cannot:

* **Ball in play.** The lock is the camera's own answer to "is there a ball to follow". A long stretch with no lock
  is a stoppage, a substitution, or the ball out of shot; the locked stretches are the live play. This needs no
  calibration at all.
* **Possession in image space.** The ball's image position and the players' image positions are both measurements in
  the same frame, so "which player is nearest the ball" is a question about the picture, not about the pitch. It
  works even where the homography is poor or absent, which is exactly the case the pitch-space possession proxy
  struggles with.

The image coordinates are normalised by the log's own frame width (2560 px on this camera, where ``Crowdx1280`` is
the centre), so they are resolution-independent and line up with the detection boxes, which the pipeline also
normalises by frame width.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# The log's frame is 2560 px wide (its centre is Crowdx1280) and 1440 tall; the ball's image x is absolute in that
# frame and the box offset is from its centre. Normalising by the width matches the detection-box convention.
LOG_FRAME_WIDTH = 2560.0
LOG_FRAME_CENTRE_Y = 720.0

# A lock this many analysis frames long is a real spell of play, not a one-frame flicker of the detector.
MIN_PLAY_FRAMES = 5
# The ball is "near" a player within this image distance (normalised by frame width) for the possession proxy.
# Measured against the real game: a player on the ball is within ~0.05 of the ball's image position; a player a few
# metres away is 0.1-0.2, so the gate sits between them.
POSSESSION_IMAGE_RADIUS = 0.08


@dataclass
class BallLock:
    """Per-frame hardware ball state, aligned to the analysis frames of a segment.

    ``u``/``v`` are the ball's image position (normalised by frame width), NaN where the hardware had no position;
    ``locked`` is the hardware's own "I am holding the ball" flag; ``size`` is the apparent ball size and ``speed``
    the image-space speed (normalised units per frame). All arrays are length F (one per analysis frame).
    """

    locked: np.ndarray  # (F,) bool
    u: np.ndarray  # (F,) ball image x / width, NaN where unknown
    v: np.ndarray  # (F,) ball image y / width, NaN where unknown
    size: np.ndarray  # (F,) apparent ball size (BallSz), NaN where unknown
    speed: np.ndarray  # (F,) image-space speed, normalised units per frame

    @property
    def frames(self) -> int:
        return len(self.locked)

    def in_play(self) -> np.ndarray:
        """Frames that belong to a spell of live play: a locked run at least ``MIN_PLAY_FRAMES`` long."""
        return _long_runs(self.locked, MIN_PLAY_FRAMES)

    def to_json(self) -> dict:
        return {
            "locked": [bool(x) for x in self.locked],
            "u": [None if not np.isfinite(x) else round(float(x), 5) for x in self.u],
            "v": [None if not np.isfinite(x) else round(float(x), 5) for x in self.v],
            "size": [None if not np.isfinite(x) else float(x) for x in self.size],
            "speed": [round(float(x), 5) for x in self.speed],
        }


def _long_runs(flags: np.ndarray, minimum: int) -> np.ndarray:
    """A boolean mask keeping only the True runs at least ``minimum`` long."""
    out = np.zeros(len(flags), dtype=bool)
    start = None
    for i, flag in enumerate(flags):
        if flag and start is None:
            start = i
        elif not flag and start is not None:
            if i - start >= minimum:
                out[start:i] = True
            start = None
    if start is not None and len(flags) - start >= minimum:
        out[start:] = True
    return out


def ball_lock_from_aligned(aligned: dict[str, np.ndarray]) -> BallLock:
    """Build a :class:`BallLock` from the per-frame arrays ``gimbal_motion.align_log`` returns.

    The aligned arrays carry the lock flag and the ball's image x directly; the ball's image y comes from the box
    offset (``ball_ym``), which the alignment does not carry, so it is left NaN here and filled by
    :func:`ball_lock_from_logs` when the raw records are available.
    """
    locked = np.asarray(aligned["lock"], dtype=bool)
    u = np.asarray(aligned["ball_x"], dtype=np.float64) / LOG_FRAME_WIDTH
    frames = len(locked)
    return BallLock(
        locked=locked,
        u=u,
        v=np.full(frames, np.nan),
        size=np.full(frames, np.nan),
        speed=np.zeros(frames),
    )


def ball_lock_from_logs(logs, *, start_s: float, fps: float, frame_count: int) -> BallLock:
    """Build a :class:`BallLock` straight from parsed logs, with the ball's image y and size.

    ``logs`` is the ``(GimbalLog, clip_start_s)`` list the alignment uses. The ball's image y is the box's offset
    from the frame centre plus the centre, normalised by width; the speed is the magnitude of the logged image
    velocity (``xv``/``yv``), which is already in the log's pixels per frame.
    """
    from soccer_analytics.geometry.gimbal_motion import align_log

    aligned = align_log(logs, start_s=start_s, fps=fps, frame_count=frame_count)
    lock = ball_lock_from_aligned(aligned)
    # Fill v, size and speed from the raw records, sampled at the same analysis frames.
    times: list[float] = []
    v: list[float] = []
    size: list[float] = []
    speed: list[float] = []
    for log, clip_start in logs:
        for record in log.frames:
            if record.time_s is None:
                continue
            times.append(clip_start + record.time_s)
            if record.ball_ym is not None:
                v.append((LOG_FRAME_CENTRE_Y + record.ball_ym) / LOG_FRAME_WIDTH)
            else:
                v.append(np.nan)
            size.append(float(record.ball_sz) if record.ball_sz is not None else np.nan)
            if record.ball_xv is not None and record.ball_yv is not None:
                speed.append(float(np.hypot(record.ball_xv, record.ball_yv)) / LOG_FRAME_WIDTH)
            else:
                speed.append(0.0)
    if not times:
        return lock
    order = np.argsort(times)
    times_arr = np.asarray(times)[order]
    at = start_s + np.arange(frame_count) / fps
    index = np.searchsorted(times_arr, at).clip(1, len(times_arr) - 1)
    left = np.abs(times_arr[index - 1] - at) < np.abs(times_arr[index] - at)
    index = np.where(left, index - 1, index)
    lock.v = np.asarray(v)[order][index]
    lock.size = np.asarray(size)[order][index]
    lock.speed = np.asarray(speed)[order][index]
    return lock


def image_space_possession(
    lock: BallLock,
    det_frame: np.ndarray,
    det_box: np.ndarray,
    det_team: np.ndarray,
    *,
    radius: float = POSSESSION_IMAGE_RADIUS,
) -> dict:
    """Possession from the ball's image position and the players' image positions - no pitch, no homography.

    For each frame the ball is locked, the nearest player *in the picture* is found (by the distance from the ball's
    image position to the player's foot point) and, if within ``radius``, that player's team is credited with the
    touch. Returns per-team touch counts, the share, and how many locked frames were contested.

    ``det_frame``/``det_box`` are the segment's detection arrays (box normalised by frame width, ``x1,y1,x2,y2``);
    ``det_team`` is the team label per detection (``-1`` for referee/unknown), which the caller gets from Stage B's
    track assignment. This is deliberately a *picture* measure: it answers "who is on the ball" without ever
    projecting to the ground, so it survives a bad calibration.
    """
    foot_u = (det_box[:, 0] + det_box[:, 2]) / 2.0
    foot_v = det_box[:, 3]
    by_frame: dict[int, list[int]] = {}
    for row, frame in enumerate(det_frame):
        by_frame.setdefault(int(frame), []).append(row)

    touches = {0: 0, 1: 0}
    contested = 0
    for frame in range(lock.frames):
        if not lock.locked[frame] or not np.isfinite(lock.u[frame]):
            continue
        rows = by_frame.get(frame)
        if not rows:
            continue
        bu, bv = lock.u[frame], lock.v[frame]
        if not np.isfinite(bv):
            # Without the ball's image y, fall back to the horizontal distance only - still a picture measure.
            distance = np.abs(foot_u[rows] - bu)
        else:
            distance = np.hypot(foot_u[rows] - bu, foot_v[rows] - bv)
        nearest = int(np.argmin(distance))
        if distance[nearest] > radius:
            continue
        team = int(det_team[rows[nearest]])
        if team in touches:
            touches[team] += 1
        contested += 1
    total = touches[0] + touches[1]
    return {
        "touches": touches,
        "contested_frames": contested,
        "share": {0: (touches[0] / total if total else 0.0), 1: (touches[1] / total if total else 0.0)},
    }