"""Synthetic match with exact ground truth, producing the same arrays Stage A stores.

No video is rendered. We simulate players on a pitch, a gimbal that pans/tilts to follow the action, and the
detections a person detector would emit (with noise, misses, spectators and a coach standing outside the lines).
Because the truth is known, Stage B can be tested for what it recovers, not merely that it runs.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from soccer_analytics.analysis.kit import DESCRIPTOR_SIZE
from soccer_analytics.analysis.stage_a import SegmentData
from soccer_analytics.geometry.camera_motion import integrate_poses, intrinsics
from soccer_analytics.geometry.pitch_calibration import PitchCalibration, pitch_to_pixels

PITCH_LENGTH = 60.0  # a 9v9 field
PITCH_WIDTH = 40.0
ASPECT = 9 / 16
FOCAL = 0.82
CAMERA = np.array([30.0, -7.0, 4.5])  # behind the near touchline, mid-pitch
FOCAL_SCALE = 1.04
FPS = 5.0

TEAM_KITS = {  # mean LAB-ish and hue descriptors; only separation matters for the tests
    0: dict(L=0.55, a=0.62, b=0.58, sat=0.8, val=0.85, hue=[0.0, 0.0, 0.0, 0.1, 0.9, 0.0]),  # red-ish
    1: dict(L=0.60, a=0.46, b=0.36, sat=0.75, val=0.9, hue=[0.0, 0.0, 0.0, 0.8, 0.2, 0.0]),  # light blue
    2: dict(L=0.85, a=0.40, b=0.65, sat=0.9, val=0.95, hue=[0.0, 0.6, 0.4, 0.0, 0.0, 0.0]),  # neon referee
    3: dict(L=0.45, a=0.50, b=0.50, sat=0.1, val=0.8, hue=[0.17, 0.17, 0.17, 0.17, 0.16, 0.16]),  # spectator (neutral)
}


def kit_vector(role: int, rng: np.random.Generator, noise: float = 0.025) -> np.ndarray:
    k = TEAM_KITS[role]
    base = np.array([0.8, k["L"], k["a"], k["b"], k["sat"], k["val"], *k["hue"]], dtype=np.float32)
    out = base + rng.normal(0.0, noise, base.shape).astype(np.float32)
    return np.clip(out, 0.0, 1.0)


@dataclass
class Truth:
    positions: np.ndarray  # (F, P, 2) true pitch xy of every person each frame
    role: np.ndarray  # (P,) 0/1 team, 2 referee, 3 spectator/coach
    team_of_player: np.ndarray  # (P,) -1 for non-players
    visible: np.ndarray  # (F, P) in the camera's field of view
    detected: np.ndarray  # (F, P) a detection row exists
    det_person: np.ndarray  # (D,) which person each detection row is
    q: np.ndarray  # (F, 3, 3) true camera orientation per frame
    focal: np.ndarray  # (F,)
    calibration: PitchCalibration
    ball: np.ndarray  # (F, 2) the possession-weighted action centre the camera follows


def _look_rotation(heading: float, tilt_down: float) -> np.ndarray:
    z = np.array([np.cos(heading) * np.cos(tilt_down), np.sin(heading) * np.cos(tilt_down), -np.sin(tilt_down)])
    x = np.cross([0.0, 0.0, 1.0], z)
    x /= np.linalg.norm(x)
    return np.column_stack([x, np.cross(z, x), z])


def simulate_match(
    frames: int = 600,
    *,
    players_per_team: int = 9,
    seed: int = 0,
    detect_prob: float = 0.93,
    foot_noise_px: float = 3.0,
    attack_bias: float = 0.0,
) -> tuple[SegmentData, Truth]:
    """``attack_bias > 0`` makes team 0 spend more time in team 1's half (positive = team 0 dominates)."""
    rng = np.random.default_rng(seed)
    n_players = 2 * players_per_team
    roles = np.array([0] * players_per_team + [1] * players_per_team + [2] + [3] * 6)  # + referee + 6 sideline people
    p = len(roles)

    # --- player movement: each has a home position and wanders toward the action ---------------------------------
    home = np.zeros((p, 2))
    for team in (0, 1):
        for i in range(players_per_team):
            fx = (0.15 + 0.7 * (i / max(players_per_team - 1, 1))) if team == 0 else (0.85 - 0.7 * (i / max(players_per_team - 1, 1)))
            home[team * players_per_team + i] = [fx * PITCH_LENGTH, rng.uniform(5, PITCH_WIDTH - 5)]
    home[n_players] = [PITCH_LENGTH / 2, PITCH_WIDTH / 2]  # referee
    for k in range(6):  # coaches/parents: well outside the lines, behind the near touchline and at the far end
        home[n_players + 1 + k] = (
            [rng.uniform(5, PITCH_LENGTH - 5), -4.5 - rng.uniform(0, 1.0)]
            if k < 3
            else [PITCH_LENGTH + 7.0, rng.uniform(5, PITCH_WIDTH - 5)]
        )
    if attack_bias:
        # A team that is on top pushes up the pitch and pins the opponent back: shift both teams' shape, otherwise
        # the bias moves only the ball and both teams simply drift toward it equally (which measures nothing).
        home[:players_per_team, 0] += attack_bias * 0.15 * PITCH_LENGTH
        home[players_per_team : 2 * players_per_team, 0] -= attack_bias * 0.15 * PITCH_LENGTH
        home[: 2 * players_per_team, 0] = np.clip(home[: 2 * players_per_team, 0], 2.0, PITCH_LENGTH - 2.0)

    ball = np.zeros((frames, 2))
    ball[0] = [PITCH_LENGTH / 2, PITCH_WIDTH / 2]
    velocity = np.zeros(2)
    for t in range(1, frames):
        pull = (np.array([PITCH_LENGTH * (0.5 + 0.18 * attack_bias), PITCH_WIDTH / 2]) - ball[t - 1]) * 0.025
        velocity = 0.93 * velocity + pull + rng.normal(0.0, 0.9, 2)
        ball[t] = np.clip(ball[t - 1] + velocity, [3, 3], [PITCH_LENGTH - 3, PITCH_WIDTH - 3])

    positions = np.zeros((frames, p, 2))
    cur = home.copy()
    # Physically plausible motion: at 5 fps a player covers well under a metre per frame. Without this cap the
    # simulator teleports players ~2 m per frame (36 km/h sustained), which no real tracker could follow.
    max_step_m = 0.6
    for t in range(frames):
        for i in range(n_players):
            target = home[i] + 0.55 * (ball[t] - home[i])
            step = (target - cur[i]) * 0.10 + rng.normal(0.0, 0.12, 2)
            length = float(np.linalg.norm(step))
            if length > max_step_m:
                step = step * (max_step_m / length)
            cur[i] += step
        cur[n_players] += (ball[t] - cur[n_players]) * 0.05 + rng.normal(0.0, 0.1, 2)
        positions[t] = cur
        positions[t, n_players + 1 :] = home[n_players + 1 :] + rng.normal(0.0, 0.1, (6, 2))
    positions[..., :n_players, 0] = np.clip(positions[..., :n_players, 0], 0.5, PITCH_LENGTH - 0.5)
    positions[..., :n_players, 1] = np.clip(positions[..., :n_players, 1], 0.5, PITCH_WIDTH - 0.5)

    # --- gimbal: smoothly aims at the ball and zooms to keep the action filling the frame -------------------------
    aim = ball.copy()
    for t in range(1, frames):
        aim[t] = aim[t - 1] + (ball[t] - aim[t - 1]) * 0.22
    r_base = _look_rotation(np.arctan2(PITCH_WIDTH / 2 + 7, 0.0), np.radians(14))  # reference frame faces +Y
    calibration = PitchCalibration(CAMERA.copy(), r_base, FOCAL_SCALE, ASPECT, 0.0, ())
    q = np.zeros((frames, 3, 3))
    focal = np.zeros(frames)
    for t in range(frames):
        direction = np.array([aim[t, 0] - CAMERA[0], aim[t, 1] - CAMERA[1], -CAMERA[2]])
        world_look = direction / np.linalg.norm(direction)
        # solve Q so that the optical axis (frame +Z) maps to world_look: frame->world = r_base @ Q
        axis_in_ref = r_base.T @ world_look
        yaw = np.arctan2(axis_in_ref[0], axis_in_ref[2])
        pitch = np.arcsin(np.clip(-axis_in_ref[1], -1, 1))
        ry = cv2.Rodrigues(np.array([0.0, yaw, 0.0]))[0]
        rx = cv2.Rodrigues(np.array([pitch, 0.0, 0.0]))[0]
        q[t] = ry @ rx
        spread = 0.5 * np.linalg.norm(positions[t, :n_players].std(axis=0))
        focal[t] = FOCAL * float(np.clip(1.9 - spread / 14.0, 0.7, 1.5))

    # --- what the detector would see ---------------------------------------------------------------------------
    visible = np.zeros((frames, p), dtype=bool)
    det_frame, det_box, det_conf, det_kit, det_person = [], [], [], [], []
    for t in range(frames):
        uv, front = pitch_to_pixels(calibration, positions[t], q[t], focal[t])
        for i in range(p):
            if not front[i]:
                continue
            u, v = uv[i]
            if not (0.02 <= u <= 0.98 and 0.03 <= v <= ASPECT - 0.01):
                continue
            visible[t, i] = True
            if rng.random() > detect_prob:
                continue
            # height in pixels shrinks with distance: ~1.5 m person; head position from a point 1.5 m above the foot
            top_uv, top_front = _head(calibration, positions[t, i], q[t], focal[t])
            if not top_front:
                continue
            noise = rng.normal(0.0, foot_noise_px / 1920.0, 2)
            foot_u, foot_v = u + noise[0], v + noise[1]
            half_w = max(0.35 * (foot_v - top_uv[1]), 0.002)
            det_frame.append(t)
            det_box.append([foot_u - half_w, top_uv[1], foot_u + half_w, foot_v])
            det_conf.append(float(np.clip(rng.normal(0.75, 0.12), 0.3, 0.99)))
            det_kit.append(kit_vector(int(roles[i]), rng))
            det_person.append(i)

    # raw steps the tracker would measure: last-frame -> this-frame homography through K R K^-1
    steps = np.zeros((frames, 3, 3))
    steps[0] = np.eye(3)
    for t in range(1, frames):
        r_rel = q[t].T @ q[t - 1]  # previous-frame rays -> current-frame rays   (Q maps frame -> reference)
        steps[t] = intrinsics(focal[t], ASPECT) @ r_rel @ np.linalg.inv(intrinsics(focal[t - 1], ASPECT))
        steps[t] /= steps[t][2, 2]

    n_det = len(det_frame)
    meta = {"width": 1920, "height": 1080, "fps": FPS, "default_focal": FOCAL, "total_frames": frames, "video": "synthetic"}
    # Stage A stores the *chain's* focal per frame - what the tracker committed at analysis time - not the source
    # of truth. The simulator must do the same or the format lies: the dashboard's fast pose path reuses these
    # focals instead of re-searching them, and a segment whose column means something else would rebuild a
    # different chain.
    chain_steps = [None if t == 0 else steps[t] for t in range(frames)]
    chain_focal = integrate_poses(chain_steps, FOCAL, ASPECT)[1]
    segment = SegmentData(
        meta=meta,
        time=np.arange(frames) / FPS,
        ok=np.ones(frames, dtype=bool),
        inlier=np.full(frames, 0.9, dtype=np.float32),
        step=steps,
        focal=chain_focal.astype(np.float32),
        det_frame=np.asarray(det_frame, dtype=np.int32),
        det_box=np.asarray(det_box, dtype=np.float32).reshape(n_det, 4),
        det_conf=np.asarray(det_conf, dtype=np.float32),
        det_kit=np.asarray(det_kit, dtype=np.float32).reshape(n_det, DESCRIPTOR_SIZE),
        det_track=np.full(n_det, -1, dtype=np.int32),
    )
    detected = np.zeros((frames, p), dtype=bool)
    detected[np.asarray(det_frame, dtype=int), np.asarray(det_person, dtype=int)] = True
    team_of = np.where(roles < 2, roles, -1)
    truth = Truth(positions, roles, team_of, visible, detected, np.asarray(det_person, dtype=np.int32), q, focal, calibration, ball)
    return segment, truth


def _head(calibration: PitchCalibration, xy: np.ndarray, q: np.ndarray, focal: float, height_m: float = 1.45):
    """Pixel of a point ``height_m`` above the ground at ``xy`` (reuses the exact projection with a raised point)."""
    from soccer_analytics.geometry.pitch_calibration import intrinsics as _  # noqa: F401  (keep import local & explicit)

    world = np.array([xy[0] - calibration.position[0], xy[1] - calibration.position[1], height_m - calibration.position[2]])
    cam = world @ (calibration.base_rotation @ q)
    k = intrinsics(focal * calibration.focal_scale, calibration.aspect)
    projected = k @ cam
    if projected[2] <= 1e-9:
        return np.array([np.nan, np.nan]), False
    return projected[:2] / projected[2], True
