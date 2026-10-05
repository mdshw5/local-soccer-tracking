"""Ball tracking: turning per-frame candidates into one track, with the camera's own motion as the prediction.

Measured on the real whole-game footage (2026-10-05), the standing conclusion in this codebase - "the ball is not
detectable in this footage" - is wrong for the current toolchain. A COCO detector at *4K* resolution finds this
ball at 0.3-0.9 confidence on most frames where it is visible (the old test ran a nano model at 1920, where the
ball is ~8 px and the model has never seen a ball that small), and a second, open-vocabulary model that has no
COCO ball class to lean on agrees with it on the same spot on two thirds of sampled frames. Two model families
converging on the same pixels is not a coincidence of class priors: the detections are real.

What the detections are not is *continuous*, and two different things cause the gaps:

* the ball is briefly undetectable - small, motion-blurred, or behind a player - while it is still on screen;
* the ball is simply **not in the picture**: the gimbal follows it with a lag, so on a long pass or a shot it
  leaves the frame entirely, and the camera swings to catch up.

A tracker has to tell those apart, because a stale position reported as if it were current is worse than an
honest "out of view" - the whole point of the last stretch of work was that a measurement of nothing poisons
everything downstream. So this tracker keeps one identity across frames, predicts through the misses using the
camera's own rotation (the chain's per-step homography, which is a *measurement* of how the picture moved, not a
guess about the ball) *plus a learned ball velocity* - a pass moves half a frame between analysis frames, and the
camera step alone cannot explain that - and says out-of-view when the prediction itself leaves the picture.

Everything here is pure: the step homography goes in, associations and states come out. That is deliberate - this
is the logic where mistakes are expensive and invisible, so it is tested without a GPU or a video file, and the
detectors stay in the scanning script where they can be swapped.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# The video has a burned-in logo (bottom right) and timestamp (bottom left). Both are *fixed image content*: the
# logo scores 0.9+ as "soccer ball" on the open-vocabulary model, and its box passes the ball's size gate, so with
# it visible a scan can acquire the logo as the ball. They are blanked before any detection rather than reasoned
# about (regions as fractions of width/height; measured on the real 4K game footage).
LOGO_REGION = (0.74, 0.87)  # (x/w below which the logo starts, y/h below which it starts)
STAMP_REGION = (0.18, 0.93)


def blank_overlays(frame: np.ndarray) -> np.ndarray:
    """Blank the burned-in logo and timestamp in place; returns the same array for chaining."""
    h, w = frame.shape[:2]
    frame[int(h * LOGO_REGION[1]) :, int(w * LOGO_REGION[0]) :] = 0
    frame[int(h * STAMP_REGION[1]) :, : int(w * STAMP_REGION[0])] = 0
    return frame

# All coordinates are normalised by the frame width (the convention the rest of the pipeline uses, where a click's
# v is also divided by the width). At 4K one normalised unit is 3840 px; the comments give the 4K equivalents.
MIN_SIZE = 0.003  # ~12 px across; smaller is grass texture, a line speck, or detector noise
MAX_SIZE = 0.06  # ~230 px; larger is a player, a bib, a bottle - something the detector latched onto
GATE = 0.045  # ~170 px; how far from the prediction a detection may sit and still be the same ball
MAX_COAST = 12  # 2.4 s at the analysis rate; longer without a detection and the track is better declared lost
ACQUIRE_CONF = 0.25  # a detection this confident may start or restart a track from a full-frame scan
MARGIN = 0.02  # ~77 px; the band outside the frame within which a prediction still counts as "at the edge"

# The camera step explains only camera motion; a ball being passed moves relative to the world on top of it, and at
# the analysis rate a hard pass covers half a frame between frames - far past any gate that is tight enough to keep
# a white shoe out. So the tracker also learns the ball's own image velocity and predicts with it. Measured on the
# real game (t=690-694, a ball being passed around): without this the track re-acquires a *moving* ball by
# teleporting every second frame; with it, the window scan keeps the ball in view through the pass.
VELOCITY_GAIN = 0.5  # EMA gain on the innovation (the part of the motion the camera step did not explain)
SPEED_GATE = 0.35  # extra gate per unit of speed: a fast ball's next position is genuinely uncertain by more
MAX_SPEED = 0.2  # ~770 px/frame at 4K; faster than any pass the 5 fps window can follow, so cap rather than trust
REENTRY_VELOCITY_FRAMES = 8  # a re-entry within this many coasted frames is a moving ball, not a fresh sighting


@dataclass
class BallTrack:
    """One ball's track across frames: position, confidence, and how sure we are that it means anything.

    The states are deliberately distinct, because a caller wants to use them differently:

    ``tracking``    - a detection was accepted this frame; the position is a measurement.
    ``coasting``    - no detection, but the prediction (camera motion from the last position) is inside the frame;
                      the ball is probably still on screen, briefly undetected.
    ``out_of_view`` - the prediction left the picture; the ball is not visible and its position is a forecast, not
                      a sighting.
    ``lost``        - too long without a detection; no position is reported until a full-frame scan finds it again.
    """

    aspect: float  # frame height / width, to know where the picture's edges are
    gate: float = GATE
    max_coast: int = MAX_COAST
    acquire_conf: float = ACQUIRE_CONF
    velocity_gain: float = VELOCITY_GAIN
    speed_gate: float = SPEED_GATE
    max_speed: float = MAX_SPEED
    status: str = "lost"
    x: float | None = None
    y: float | None = None
    conf: float = 0.0
    coasted: int = 0
    # The ball's own image velocity per frame, learned from the part of each accepted detection's motion the camera
    # step did not explain. Zero until a detection has been seen (and again after a re-entry, which is a jump, not
    # a velocity).
    vx: float = 0.0
    vy: float = 0.0

    def predict(self, step: np.ndarray | None) -> tuple[float, float] | None:
        """Where the last position lands this frame with only the camera *and the learned velocity* at play.

        ``step`` is the chain's homography from the previous frame to this one - a measurement of the camera's own
        motion. ``None`` (a frame the chain lost) predicts "camera unchanged", which simply opens the gate to
        everything; the ball's own velocity still carries the prediction forward.
        """
        if self.x is None or self.y is None:
            return None
        if step is None:
            return (self.x + self.vx, self.y + self.vy)
        hom = np.asarray(step, dtype=np.float64) @ np.array([self.x, self.y, 1.0])
        if hom[2] <= 1e-12:
            # The prediction ran into the horizon (the scene plane the step describes is not the ball's world).
            # There is no useful forecast here; let the caller's full-frame scan take over.
            return None
        return (float(hom[0] / hom[2]) + self.vx, float(hom[1] / hom[2]) + self.vy)

    def inside_frame(self, u: float, v: float) -> bool:
        return -MARGIN <= u <= 1.0 + MARGIN and -MARGIN <= v <= self.aspect + MARGIN

    def update(self, detections, step: np.ndarray | None = None, full_frame: bool = False) -> dict:
        """Fold one frame's detections in; returns the state after this frame.

        ``detections`` are ``(conf, u, v, w, h)`` in normalised coordinates. ``full_frame`` says they came from a
        scan of the whole picture (so a detection far from the prediction may be a *re-entry*), rather than from a
        window around the prediction (where distance is evidence against, not for).
        """
        candidates = [
            (float(conf), float(u), float(v), float(w), float(h))
            for conf, u, v, w, h in detections
            if MIN_SIZE <= min(w, h) and max(w, h) <= MAX_SIZE
        ]
        if self.x is None or self.y is None:
            # No position to gate against: only a confident full-frame sighting starts (or restarts) the track.
            best = max(candidates, key=lambda det: det[0], default=None)
            if best is not None and full_frame and best[0] >= self.acquire_conf:
                self._accept(best, learn=False)
            else:
                self.status = "lost"
                self.x = self.y = None
                self.conf = 0.0
                self.vx = self.vy = 0.0
            return self.state()

        prediction = self.predict(step)
        best = None
        if prediction is not None:
            speed = float(np.hypot(self.vx, self.vy))
            gate = self.gate + self.speed_gate * speed  # a fast ball's next position is uncertain by more
            for det in candidates:
                distance = float(np.hypot(det[1] - prediction[0], det[2] - prediction[1]))
                if distance <= gate and (best is None or det[0] - distance / gate > best[0]):
                    # score = confidence minus the distance in gate-units: a confident detection 20 px away beats
                    # a marginal one on the prediction, but a coincidence at the gate's far edge loses to both.
                    best = (det[0] - distance / gate, det, distance)
            if best is not None:
                self._accept(best[1], learn=True, innovation=(best[1][1] - prediction[0], best[1][2] - prediction[1]))
                return self.state()
            if full_frame or self.coasted >= 1:
                # A confident sighting anywhere in the scan is the ball re-entering (a long pass the camera is
                # still catching up with), or the track having been wrong. Take it; the next frames judge it.
                #
                # The ``coasted >= 1`` half of the condition is the *onset* of a fast ball: the first frame of a
                # kick moves further than any gate that keeps a white shoe out, so the detection is rejected and
                # the track coasts once - and on the very next scan, a confident sighting anywhere in the window
                # is much more likely the ball that was kicked than a coincidence, because it was still on the
                # ball one frame ago. Without this, every pass loses 2-5 frames waiting for a full-frame scan,
                # which is exactly the tracking/coasting alternation measured on the real game (t=690-694).
                #
                # The onset re-lock from a *window* scan demands the stronger confidence: the window holds grass,
                # players' socks and line specks, and a weak sighting 200 px from a 170 px gate is more likely one
                # of those than a ball that was kicked. Seen from the full frame, a confident sighting is taken on
                # the normal threshold instead - the same rule as re-entry after a long absence.
                required = self.acquire_conf if full_frame else max(self.acquire_conf, 0.5)
                loose = [det for det in candidates if det[0] >= required]
                if loose:
                    best_det = max(loose, key=lambda det: det[0])
                    # A re-entry after a *short* coast is a sample of a moving ball: the displacement across the
                    # gap, per frame, is a velocity estimate. Over a longer gap it is a fresh sighting, not a
                    # measurement of motion - a jump is not a velocity.
                    if self.coasted + 1 <= REENTRY_VELOCITY_FRAMES:
                        gap = self.coasted + 1
                        innovation = (
                            (best_det[1] - prediction[0]) / gap,
                            (best_det[2] - prediction[1]) / gap,
                        )
                        self._accept(best_det, learn=True, innovation=innovation)
                    else:
                        self._accept(best_det, learn=False)
                    return self.state()
        self.coasted += 1
        if prediction is None or not self.inside_frame(*prediction):
            self.status = "out_of_view"
        else:
            self.status = "coasting"
        if prediction is not None:
            self.x, self.y = prediction
        if self.coasted > self.max_coast:
            self.status = "lost"
            self.x = self.y = None
            self.conf = 0.0
            self.vx = self.vy = 0.0
        return self.state()

    def _accept(
        self,
        det: tuple[float, float, float, float, float],
        *,
        learn: bool,
        innovation: tuple[float, float] | None = None,
    ) -> None:
        """Take a detection as the position; ``learn`` folds its unexplained motion into the velocity.

        The innovation is what the camera step could not explain about the detection's motion - the ball's own
        movement (or a detection that is not the ball, which the next frames will expose). An EMA of it is a good
        short-horizon velocity: a ball at rest in the world keeps it near zero, a ball being passed locks onto it,
        and a touched ball slows back to zero over a few frames.
        """
        self.conf, self.x, self.y, _w, _h = det
        if learn and innovation is not None:
            self.vx += self.velocity_gain * innovation[0]
            self.vy += self.velocity_gain * innovation[1]
            speed = float(np.hypot(self.vx, self.vy))
            if speed > self.max_speed:
                self.vx *= self.max_speed / speed
                self.vy *= self.max_speed / speed
        else:
            self.vx = self.vy = 0.0
        self.status = "tracking"
        self.coasted = 0

    def state(self) -> dict:
        return {
            "status": self.status,
            "u": None if self.x is None else round(self.x, 6),
            "v": None if self.y is None else round(self.y, 6),
            "conf": round(self.conf, 4),
            "vx": round(self.vx, 6),
            "vy": round(self.vy, 6),
        }

    def to_json(self) -> dict:
        """The whole tracker state, enough to resume a scan from a checkpoint mid-video."""
        return {
            "status": self.status,
            "x": self.x,
            "y": self.y,
            "conf": self.conf,
            "coasted": self.coasted,
            "vx": self.vx,
            "vy": self.vy,
        }

    @classmethod
    def from_json(cls, payload: dict, *, aspect: float) -> "BallTrack":
        track = cls(aspect=aspect)
        track.status = str(payload.get("status", "lost"))
        track.x = payload.get("x")
        track.y = payload.get("y")
        track.conf = float(payload.get("conf", 0.0))
        track.coasted = int(payload.get("coasted", 0))
        track.vx = float(payload.get("vx", 0.0))
        track.vy = float(payload.get("vy", 0.0))
        return track
