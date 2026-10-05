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

One measured deviation from the reference: their mask is a fixed +-10 band
around the mean grass hue, and on the real whole game that band covered as
little as 51% of the grass pixels (shaded and sunlit grass sit ~25 hue apart
with the mean in the gap, and the sunlit mode moves as the light changes late
in the afternoon). `grass_hue_window` therefore widens the band to the 2nd..98th
percentile of the measured grass population — never narrower than the reference
band, so frames whose grass is one tight mode behave exactly as before.

A frame-wide band, however wide, is the wrong shape for the job: it has to
cover every lighting mode in the picture, so a player standing on shaded turf
is masked with a band stretched to also cover the sunlit half. `measure_grass`
answers the same question once per tile of a coarse grid (`GRASS_TILES`), so
the band used on a torso is the band of the grass actually behind it - tight
where the frame-wide band has to be loose. It also measures the grass in Lab,
so sun-bleached turf that has fallen below the saturation floor (and used to
leak into every kit colour on a bright afternoon) is still masked. `kit.py`
uses the tile band; the frame-wide band stays as the documented fallback and
as the number the probe script's historical comparison is measured against.
"""

from __future__ import annotations

from dataclasses import dataclass

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

# --------------------------------------------------------------------------------------------------------------
# The spatial grass model
# --------------------------------------------------------------------------------------------------------------
# A frame-wide hue band has to be wide enough to cover *every* mode of grass in the picture, because it is measured
# over the whole frame: shaded turf under the stand, sunlit turf in the open, and both of them again after the sun
# moves. Width is therefore paid for twice - as grass that leaks into a kit estimate, and as a pitch that survives
# the mask because the band no longer describes the grass right behind that particular player.
#
# Measuring a small band *per tile* removes the trade-off: a tile of shaded turf gets a tight band around shaded
# turf, and only the tiles that actually straddle a shadow line get a wide one. It is the same measurement as
# `grass_hue_window`, done where the grass actually is.
GRASS_TILES = (4, 3)  # (columns, rows)
GRASS_TILE_MIN_PIXELS = 200  # a tile with less grass than this borrows the frame-wide band

# Grass that has been bleached by sun, worn by feet or flattened by the rain does not stop being grass; it stops
# being *saturated*. Those pixels fail the MIN_SATURATION test above and used to leak into every kit colour on a
# bright afternoon, so they get a second chance: still green by hue, but close to the measured grass in Lab.
WEAK_GRASS_SATURATION = 15
GRASS_LAB_PERCENTILE = 85.0  # share of the measured grass's chroma spread taken as "still the same grass"
GRASS_CHROMA_MIN = 10.0
GRASS_CHROMA_MAX = 35.0
GRASS_LIGHT_TOL_MIN = 8.0
GRASS_LIGHT_TOL_MAX = 45.0  # OpenCV Lab L units; shaded and sunlit grass differ in lightness by this much


def grass_colour(frame: np.ndarray) -> tuple[float, float, float]:
    """Mean BGR of the green (pitch) pixels; (0, 0, 0) when the frame has none."""
    if frame is None or frame.size == 0:
        return (0.0, 0.0, 0.0)
    mask = _grass_mask(cv2.cvtColor(frame, cv2.COLOR_BGR2HSV))
    if not np.any(mask):
        return (0.0, 0.0, 0.0)
    mean = cv2.mean(frame, mask=mask)
    return (float(mean[0]), float(mean[1]), float(mean[2]))


def _grass_mask(hsv: np.ndarray) -> np.ndarray:
    """Pixels inside the coarse pitch-green window (the reference project's bounds)."""
    return cv2.inRange(
        hsv,
        np.array([GRASS_HUE_MIN, MIN_SATURATION, MIN_VALUE]),
        np.array([GRASS_HUE_MAX, 255, 255]),
    )


def _grass_population(hsv: np.ndarray) -> np.ndarray:
    """Every green pixel, including the bleached ones, as the *measurement* population for the model.

    Deliberately looser than :func:`_grass_mask`: the saturation floor is lowered so that sun-bleached turf is
    measured (and can therefore be masked) instead of being invisible to the model and leaking into kit colour. The
    stricter floor still applies when the mask is actually cut - see :meth:`GrassModel.mask`.
    """
    return cv2.inRange(
        hsv,
        np.array([GRASS_HUE_MIN, WEAK_GRASS_SATURATION, MIN_VALUE]),
        np.array([GRASS_HUE_MAX, 255, 255]),
    )


@dataclass(frozen=True, eq=False)
class GrassModel:
    """Everything one frame says about its own grass: where it is, how wide it is, and how far it spreads in Lab.

    ``hue_lo``/``hue_hi`` are the frame-wide band (kept, so the single-band behaviour stays available and
    comparable). ``tiles`` holds a band per tile of a ``GRASS_TILES`` grid and ``tile_measured`` says which tiles
    had enough grass to earn their own answer - the rest borrow the frame-wide one rather than inventing a narrow
    band from a handful of pixels. ``coverage`` is the honest self-check: the share of the frame's saturated grass
    that the frame-wide band actually masks, which is the number the probe script reports.
    """

    hue_lo: int
    hue_hi: int
    coverage: float
    ab_centre: tuple[float, float]
    l_centre: float
    chroma_radius: float
    light_tol: float
    tiles: np.ndarray  # (rows, cols, 2) int32 hue bands
    tile_measured: np.ndarray  # (rows, cols) bool

    @property
    def window(self) -> tuple[int, int]:
        """The frame-wide band, as ``(lo, hi)`` - the shape older callers expect."""
        return (self.hue_lo, self.hue_hi)

    def band_for(self, bbox: tuple[float, float, float, float], frame_shape: tuple[int, ...]) -> tuple[int, int]:
        """The grass band for the tile the box sits in, falling back to the frame-wide band.

        The torso is what gets masked, so it is the tile under the box that decides the band: masking a kit against
        the grass of some other part of the pitch is what the wide frame-wide band has to do, and what this avoids.
        """
        rows, cols = self.tiles.shape[:2]
        height, width = float(frame_shape[0]), float(frame_shape[1])
        if rows == 0 or cols == 0 or height <= 0 or width <= 0:
            return self.window
        cx = 0.5 * (float(bbox[0]) + float(bbox[2]))
        cy = 0.5 * (float(bbox[1]) + float(bbox[3]))
        col = int(np.clip(cx / width * cols, 0, cols - 1))
        row = int(np.clip(cy / height * rows, 0, rows - 1))
        if not self.tile_measured[row, col]:
            return self.window
        return (int(self.tiles[row, col, 0]), int(self.tiles[row, col, 1]))

    def mask(self, hsv: np.ndarray, lab: np.ndarray, band: tuple[int, int] | None = None) -> np.ndarray:
        """True for the grass pixels of an HSV/Lab pair, using ``band`` or the frame-wide one.

        Strongly saturated green inside the band is grass, as before. Faint green inside the band is only grass if
        it also sits close to the measured grass in Lab - which keeps a pale, nearly neutral shirt out of the mask
        (it is faint, but nowhere near grass on the a/b axes) while catching bleached turf, which is.
        """
        lo, hi = band if band is not None else self.window
        hue, sat, val = hsv[..., 0], hsv[..., 1], hsv[..., 2]
        in_band = (hue >= lo) & (hue <= hi) & (val >= MIN_VALUE)
        grass = in_band & (sat >= MIN_SATURATION)
        faint = in_band & (sat >= WEAK_GRASS_SATURATION)
        if np.any(faint):
            axes = lab.astype(np.float32)
            chroma = np.hypot(axes[..., 1] - self.ab_centre[0], axes[..., 2] - self.ab_centre[1])
            lightness = np.abs(axes[..., 0] - self.l_centre)
            grass |= faint & (chroma <= self.chroma_radius) & (lightness <= self.light_tol)
        return grass


def _measure_tiles(hsv: np.ndarray, population: np.ndarray, frame_shape: tuple[int, ...], fallback: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    """A per-tile hue band over a ``GRASS_TILES`` grid, plus which tiles were measured rather than borrowed.

    A tile is centred on the *median* hue of its own grass and widened to its own 2nd..98th percentile span - the
    same rule as the frame-wide band, applied to a patch of pitch small enough to be one lighting mode. A tile that
    is mostly stand, sky or players has no grass worth measuring, so it is marked unmeasured and uses ``fallback``.
    """
    height, width = frame_shape[:2]
    rows, cols = GRASS_TILES[1], GRASS_TILES[0]
    bands = np.tile(np.asarray(fallback, dtype=np.int32), (rows, cols, 1))
    measured = np.zeros((rows, cols), dtype=bool)
    ys = np.linspace(0, height, rows + 1).astype(int)
    xs = np.linspace(0, width, cols + 1).astype(int)
    for row in range(rows):
        for col in range(cols):
            y0, y1 = ys[row], ys[row + 1]
            x0, x1 = xs[col], xs[col + 1]
            patch = population[y0:y1, x0:x1] > 0
            if int(patch.sum()) < GRASS_TILE_MIN_PIXELS:
                continue
            hues = hsv[y0:y1, x0:x1, 0][patch].astype(np.float32)
            centre = int(np.median(hues))
            p2, p98 = (int(value) for value in np.percentile(hues, [2, 98]))
            bands[row, col] = (
                max(0, min(centre - GRASS_HUE_PADDING, p2)),
                min(179, max(centre + GRASS_HUE_PADDING, p98)),
            )
            measured[row, col] = True
    return bands, measured


def measure_grass(frame: np.ndarray) -> GrassModel | None:
    """Measure this frame's grass: a frame-wide band, a band per tile, and its spread in Lab. None without grass.

    Three answers rather than one, because a frame's grass is not one thing. The frame-wide band is what older
    callers and the stored-behaviour comparisons use. The per-tile bands are what the kit descriptor masks with, so
    a shaded pitch is masked tightly instead of with a band stretched to also cover the sunlit half. The Lab spread
    is what catches bleached turf, which is green by hue but too faint for the saturation test.
    """
    if frame is None or frame.size == 0:
        return None
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    population = _grass_population(hsv)
    if not np.any(population):
        return None
    hue_plane = hsv[..., 0]
    hues = hue_plane[population > 0]
    mean_bgr = cv2.mean(frame, mask=population)
    centre = int(cv2.cvtColor(np.uint8([[list(mean_bgr[:3])]]), cv2.COLOR_BGR2HSV)[0, 0, 0])
    p2, p98 = (int(value) for value in np.percentile(hues, [2, 98]))
    lo = max(0, min(centre - GRASS_HUE_PADDING, p2))
    hi = min(179, max(centre + GRASS_HUE_PADDING, p98))

    strong = _grass_mask(hsv) > 0
    coverage = float(np.count_nonzero((hue_plane[strong] >= lo) & (hue_plane[strong] <= hi)) / max(1, int(strong.sum())))

    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB).astype(np.float32)
    axes = lab[population > 0]
    ab_centre = (float(np.median(axes[:, 1])), float(np.median(axes[:, 2])))
    l_centre = float(np.median(axes[:, 0]))
    chroma = np.hypot(axes[:, 1] - ab_centre[0], axes[:, 2] - ab_centre[1])
    radius = float(np.clip(np.percentile(chroma, GRASS_LAB_PERCENTILE), GRASS_CHROMA_MIN, GRASS_CHROMA_MAX))
    light_tol = float(np.clip(np.percentile(np.abs(axes[:, 0] - l_centre), 95.0), GRASS_LIGHT_TOL_MIN, GRASS_LIGHT_TOL_MAX))

    bands, measured = _measure_tiles(hsv, population, frame.shape, (lo, hi))
    return GrassModel(
        hue_lo=lo,
        hue_hi=hi,
        coverage=coverage,
        ab_centre=ab_centre,
        l_centre=l_centre,
        chroma_radius=radius,
        light_tol=light_tol,
        tiles=bands,
        tile_measured=measured,
    )


def grass_hue_window(frame: np.ndarray) -> tuple[int, int] | None:
    """Hue band that covers the frame's measured grass, or None when the frame has no grass.

    Centred on the mean grass colour +- ``GRASS_HUE_PADDING``, then widened to the 2nd..98th percentile of the
    grass pixels' hues - never narrower than the fixed band, and never outside the coarse green window. The
    widening is the point, and it was measured on the real whole game: grass has two hue modes (shaded ~34,
    sunlit ~60+), the mean sits in the gap between them, and a fixed +-10 band around it covered as little as
    51% of the grass pixels at dusk while the sunlit mode moved. With this window, coverage was 98%+ on every
    frame checked - what leaks into kit estimates is fringe pixels, not a whole mode.

    The frame-wide band is the *fallback* half of :class:`GrassModel`; this function stays as the single-band
    answer so its measured behaviour remains comparable and existing callers keep working.
    """
    model = measure_grass(frame)
    return model.window if model is not None else None


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
    grass: GrassModel | tuple[int, int] | None = None,
    bins: int = 16,
    band: tuple[int, int] | None = None,
) -> np.ndarray:
    """Normalised HSV histogram of a crop with grass pixels masked out.

    `grass` is a :class:`GrassModel` from :func:`measure_grass` (the kit descriptor passes the band of the tile
    the player is standing on) or a plain ``(lo, hi)`` hue pair; None for crops with no pitch visible, where
    nothing is masked. A crop that is entirely grass falls back to the unmasked crop, so the feature is always
    defined - an all-zero histogram would otherwise silently look like a distinct kit.
    """
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    mask = np.full(crop.shape[:2], 255, dtype=np.uint8)
    if grass is not None:
        if isinstance(grass, GrassModel):
            lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB)
            keep = ~grass.mask(hsv, lab, band)
        else:
            keep = ~(
                cv2.inRange(
                    hsv, np.array([grass[0], MIN_SATURATION, MIN_VALUE]), np.array([grass[1], 255, 255])
                ).astype(bool)
            )
        mask = np.where(keep, 255, 0).astype(np.uint8)
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
        model = measure_grass(frame)
        hist = kit_colour_histogram(crop, model, band=model.band_for(bbox, frame.shape) if model else None)
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
