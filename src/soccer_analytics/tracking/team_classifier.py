"""Lightweight team classification via jersey-color clustering.

Cheap alternative to embedding-based classifiers (e.g. Roboflow's SigLIP
`TeamClassifier`, or the OSNet embeddings used by whisdev's
soccer-video-detection-ai-agent) chosen to fit the 5GB VRAM / 8GB RAM budget.
Falls back to an unassigned team label until enough samples have been collected
to fit clusters.

Ideas adapted from whisdev/soccer-video-detection-ai-agent (their HSV
kit-color fallback path — the OSNet path needs weights we don't ship):

  * estimate the pitch's grass color per frame and mask grass pixels out of
    each player crop, so the jersey — not the pitch behind it — drives the
    color feature;
  * aggregate samples per track and L2-normalise before clustering;
  * refuse to invent a split when the two clusters are near-identical;
  * order clusters deterministically so team ids are stable across clips.
"""

from __future__ import annotations

import cv2
import numpy as np
from sklearn.cluster import KMeans

UNASSIGNED_TEAM = -1

# OpenCV HSV window for "pitch surface" greens (same bounds as the reference project).
GRASS_HUE_MIN = 30
GRASS_HUE_MAX = 80
MIN_SATURATION = 40
MIN_VALUE = 40
GRASS_HUE_PADDING = 10


def grass_colour(frame: np.ndarray) -> tuple[float, float, float]:
    """Mean BGR of the green (pitch) pixels; (0, 0, 0) when the frame has none."""
    if frame is None or frame.size == 0:
        return (0.0, 0.0, 0.0)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(
        hsv,
        np.array([GRASS_HUE_MIN, MIN_SATURATION, MIN_VALUE]),
        np.array([GRASS_HUE_MAX, 255, 255]),
    )
    if not np.any(mask):
        return (0.0, 0.0, 0.0)
    mean = cv2.mean(frame, mask=mask)
    return (float(mean[0]), float(mean[1]), float(mean[2]))


def grass_hue_window(frame: np.ndarray) -> tuple[int, int] | None:
    """Hue window around the frame's dominant grass hue, or None when there is no grass."""
    colour = grass_colour(frame)
    if colour == (0.0, 0.0, 0.0):
        return None
    hue = int(cv2.cvtColor(np.uint8([[list(colour)]]), cv2.COLOR_BGR2HSV)[0, 0, 0])
    return (max(0, hue - GRASS_HUE_PADDING), min(179, hue + GRASS_HUE_PADDING))


def torso_crop(frame: np.ndarray, bbox: tuple[float, float, float, float]) -> np.ndarray | None:
    """Crop the upper-middle portion of a player's box, where the jersey dominates."""
    x1, y1, x2, y2 = (int(v) for v in bbox)
    height = y2 - y1
    width = x2 - x1
    if height <= 0 or width <= 0:
        return None
    torso_y1 = y1 + int(height * 0.15)
    torso_y2 = y1 + int(height * 0.55)
    torso_x1 = x1 + int(width * 0.25)
    torso_x2 = x1 + int(width * 0.75)
    crop = frame[torso_y1:torso_y2, torso_x1:torso_x2]
    return crop if crop.size > 0 else None


def kit_colour_histogram(
    crop: np.ndarray,
    grass_hues: tuple[int, int] | None = None,
    bins: int = 16,
) -> np.ndarray:
    """Normalised HSV histogram of a crop with grass pixels masked out.

    `grass_hues` comes from `grass_hue_window` on the full frame; pass None for
    crops with no pitch visible (nothing is masked then).
    """
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    mask = np.full(crop.shape[:2], 255, dtype=np.uint8)
    if grass_hues is not None:
        grass = cv2.inRange(
            hsv,
            np.array([grass_hues[0], MIN_SATURATION, MIN_VALUE]),
            np.array([grass_hues[1], 255, 255]),
        )
        mask = cv2.bitwise_and(mask, cv2.bitwise_not(grass))
        if not np.any(mask):
            mask = np.full(crop.shape[:2], 255, dtype=np.uint8)
    hist = cv2.calcHist([hsv], [0, 1], mask, [bins, bins], [0, 180, 0, 256])
    cv2.normalize(hist, hist)
    return hist.flatten()


class TeamClassifier:
    """Collects jersey color samples per track and clusters them into teams.

    Usage: call `observe(track_id, frame, bbox)` for every player detection,
    then `team_of(track_id)` once `is_fitted` to get a stable team label.
    Call `fit()` periodically (e.g. every N frames) once enough tracks have
    been observed.
    """

    def __init__(
        self,
        num_teams: int = 2,
        min_samples_before_fit: int = 10,
        similarity_threshold: float = 0.95,
    ):
        self.num_teams = num_teams
        self.min_samples_before_fit = min_samples_before_fit
        self.similarity_threshold = similarity_threshold
        self._samples: dict[int, list[np.ndarray]] = {}
        self._team_by_track: dict[int, int] = {}
        self.is_fitted = False

    def observe(self, track_id: int, frame: np.ndarray, bbox: tuple[float, float, float, float]) -> None:
        crop = torso_crop(frame, bbox)
        if crop is None:
            return
        hist = kit_colour_histogram(crop, grass_hue_window(frame))
        self._samples.setdefault(track_id, []).append(hist)

    def fit(self) -> None:
        if len(self._samples) < self.min_samples_before_fit:
            return

        track_ids = list(self._samples.keys())
        features = np.stack([np.mean(self._samples[t], axis=0) for t in track_ids])
        norms = np.linalg.norm(features, axis=1, keepdims=True)
        features = features / np.clip(norms, 1e-12, None)

        if len(track_ids) == 1 or self.num_teams < 2:
            self._team_by_track = {track_ids[0]: 0}
            self.is_fitted = True
            return

        if len(np.unique(features, axis=0)) == 1:
            # Every observed kit is identical — nothing to split.
            self._team_by_track = {track_id: 0 for track_id in track_ids}
            self.is_fitted = True
            return

        kmeans = KMeans(n_clusters=min(self.num_teams, len(track_ids)), n_init=10, random_state=0)
        labels = kmeans.fit_predict(features)
        centroids = kmeans.cluster_centers_

        if len(centroids) == 2:
            first, second = centroids
            similarity = float(
                np.dot(first, second) / (np.linalg.norm(first) * np.linalg.norm(second) + 1e-12)
            )
            if similarity > self.similarity_threshold:
                # Kits are indistinguishable (or only one team is in frame) — don't
                # invent a split; every track shares a single label.
                self._team_by_track = {track_id: 0 for track_id in track_ids}
                self.is_fitted = True
                return
            # Deterministic team ids across clips: order clusters by centroid norm.
            if np.linalg.norm(first) <= np.linalg.norm(second):
                labels = 1 - labels

        self._team_by_track = dict(zip(track_ids, labels.tolist()))
        self.is_fitted = True

    def team_of(self, track_id: int) -> int:
        return self._team_by_track.get(track_id, UNASSIGNED_TEAM)
