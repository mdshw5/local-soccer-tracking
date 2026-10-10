"""Roles: which tracked player is the referee, and which is a goalkeeper, from how they use the pitch.

Measured on the real game (2026-10-04), because none of this is inferable from a single frame:

* the referee is an unlabeled track that ranges widely across the pitch - the real one appeared as three
  fragments (t9023/t7981/t11362) with spans of 14-30 m between their 5th and 95th x percentile. Its kit reads
  green, orange or dark red depending on which crop the median landed on, so color can gate the search (a
  color-evidence check rules out stationary tracker junk like t11055, 746 observations of no kit) but cannot
  identify it;
* a goalkeeper is the deepest player at one goal end: at least ``GK_DEEP_FRACTION`` of a track's observations
  within ``GK_ZONE_M`` of the goal line. Fragmentation means each keeper arrives as several tracks too (the
  left-goal figure split into t9253/t8242/t19647-style fragments);
* net-area people (ball boys, photographers) also spend their time at a goal, so this labels the best candidate
  per end rather than claiming a person - the same honesty rule as shirt numbers, and the same staleness: roles
  are attached to track ids and must be recomputed when a rebuild moves them.

Roles are deliberately few and honest: at most one referee and at most one goalkeeper per goal end. Everything
else stays unlabeled.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

# A goalkeeper stays within this pocket of one goal line; below the bar the player is a deep defender, not a keeper.
GK_ZONE_M = 12.0
GK_DEEP_FRACTION = 0.9
GK_MIN_OBSERVATIONS = 60
# The referee's fragments carry this many observations and this much x-range; a stationary person at one end has
# the observations but not the range.
REF_MIN_OBSERVATIONS = 150
REF_MIN_SPAN_M = 18.0
# Percentiles for the x-span: 95-5 rather than max-min so one bad projection cannot qualify a track.
SPAN_LOW, SPAN_HIGH = 5.0, 95.0


def _deep_fraction(x: np.ndarray, pitch_length_m: float) -> tuple[float, str]:
    """The share of x-samples inside each goal pocket, and which side it belongs to."""
    left = float(np.mean(x < GK_ZONE_M))
    right = float(np.mean(x > pitch_length_m - GK_ZONE_M))
    if left >= right:
        return left, "left"
    return right, "right"


def classify_roles(
    players,
    *,
    pitch_length_m: float,
    kit_evidence: Callable[[int], object | None] | None = None,
) -> dict[int, dict]:
    """Roles per track id: ``{"role": "goalkeeper", "side": ...}`` and at most one ``{"role": "referee"}``.

    ``players`` are the report's players (already bystander-filtered); ``kit_evidence(track_id)`` returns
    non-``None`` when the track has trustworthy kit crops, used to gate the referee search - pass ``None`` to skip
    that gate (tests). Every track deep enough in a goal pocket is labeled goalkeeper; the longest
    kit-evidenced unlabeled wide-ranging track is the referee.
    """
    roles: dict[int, dict] = {}
    best_ref: tuple[int, int] | None = None  # (observations, track_id)
    for player in players:
        track_id = int(player.track_id)
        xy = np.asarray(player.xy, dtype=np.float64)
        observations = len(xy)
        if observations >= GK_MIN_OBSERVATIONS:
            deep, side = _deep_fraction(xy[:, 0], pitch_length_m)
            if deep >= GK_DEEP_FRACTION:
                roles[track_id] = {"role": "goalkeeper", "side": side}
                continue  # a keeper is never the referee, whatever its range
        if (
            observations >= REF_MIN_OBSERVATIONS
            and int(player.team) == -1
            and (kit_evidence is None or kit_evidence(track_id) is not None)
        ):
            x = xy[:, 0]
            span = float(np.percentile(x, SPAN_HIGH) - np.percentile(x, SPAN_LOW))
            if span >= REF_MIN_SPAN_M and (best_ref is None or observations > best_ref[0]):
                best_ref = (observations, track_id)
    if best_ref is not None:
        roles[best_ref[1]] = {"role": "referee"}
    return roles
