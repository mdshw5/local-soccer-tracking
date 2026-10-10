"""Offline track stitching and the distance rule it exposed.

On the real footage the online tracker fragments constantly (the gimbal whips between the ends of the pitch and
players step out of tight shots), so fragments are reconnected afterwards. These pin the rules: who may stitch to
whom, and that movement across a long unobserved gap is not invented as distance.

The online pass already reconnects gaps up to its buffer (~3 s), so the scenarios here deliberately use gaps
beyond it - that is where stitching earns its keep.
"""

from __future__ import annotations

import numpy as np

from soccer_analytics.analysis.projection import PitchDetections
from soccer_analytics.analysis.stage_b import TrackAssignment, _stitch_tracks, _track_people, build_tracks

KIT_A = [0.5, 1.0, 0.0, 0.0, 0.2, 0.9]  # weight + (L, a, b, sat, val)
KIT_B = [0.5, -1.0, 0.5, -0.5, -0.8, 0.3]


def _detections(rows: list[tuple[int, float, float, list[float]]]) -> PitchDetections:
    """One detection per (frame, x, y, kit) row; everything else is filler the tracker does not need here."""
    frame = np.array([r[0] for r in rows], dtype=np.int32)
    xy = np.array([[r[1], r[2]] for r in rows], dtype=np.float64)
    kit = np.array([r[3] for r in rows], dtype=np.float64)
    n = len(rows)
    return PitchDetections(
        frame=frame,
        time=frame / 5.0,
        xy=xy,
        sigma_m=np.full(n, 0.3),
        valid=np.ones(n, dtype=bool),
        height_px=np.full(n, 200.0),
        box=np.full((n, 4), 0.25),
        conf=np.full(n, 0.9),
        kit=kit,
        det_track=np.full(n, -1, dtype=np.int32),
        det_index=np.arange(n),
        aim_xy=np.full((int(frame.max()) + 1, 2), np.nan),
        camera_xy=np.zeros(2),
    )


def _chain(x0: float, kit: list[float], frames: range, step: float = 0.4) -> list[tuple[int, float, float, list[float]]]:
    return [(f, x0 + step * i, 10.0, kit) for i, f in enumerate(frames)]


def _two_fragments(kit_a: list[float], kit_b: list[float], start_b: int, x_b: float):
    rows = _chain(10.0, kit_a, range(0, 8)) + _chain(x_b, kit_b, range(start_b, start_b + 8))
    detections = _detections(rows)
    keep = np.ones(len(rows), dtype=bool)
    assignment = _track_people(detections, keep)
    return detections, keep, assignment


def test_a_reentry_beyond_the_online_buffer_is_stitched() -> None:
    """16 frames (3.2 s) without a detection: the online tracker must let the track die, stitching may revive it."""
    detections, _, assignment = _two_fragments(KIT_A, KIT_A, start_b=24, x_b=12.0)
    assert len(assignment.tracks) == 2, "the raw pass sees two fragments"

    stitched = _stitch_tracks(detections, assignment)
    assert len(stitched.tracks) == 1
    (merged_rows,) = stitched.tracks.values()
    assert len(merged_rows) == 16
    assert (stitched.track_id == list(stitched.tracks)[0]).all()


def test_a_different_kit_is_not_stitched() -> None:
    detections, _, assignment = _two_fragments(KIT_A, KIT_B, start_b=24, x_b=12.0)
    assert len(assignment.tracks) == 2
    stitched = _stitch_tracks(detections, assignment)
    assert len(stitched.tracks) == 2, "color evidence must veto a spatial coincidence"


def test_overlapping_tracks_are_never_stitched() -> None:
    """A fragment cannot continue a track that is still running - that would be two people at once."""
    rows = _chain(10.0, KIT_A, range(0, 8)) + _chain(11.0, KIT_A, range(4, 12))
    detections = _detections(rows)
    assignment = _track_people(detections, np.ones(len(rows), dtype=bool))
    stitched = _stitch_tracks(detections, assignment)
    assert len(stitched.tracks) == 2


def test_a_reentry_far_away_is_not_stitched() -> None:
    detections, _, assignment = _two_fragments(KIT_A, KIT_A, start_b=24, x_b=45.0)
    assert len(assignment.tracks) == 2
    stitched = _stitch_tracks(detections, assignment)
    assert len(stitched.tracks) == 2, "35 m within 3 s is not plausibly the same run"


def test_movement_across_a_long_gap_is_not_counted_as_distance() -> None:
    """Two runs of four observations with a 7.4 s gap: only the observed steps are distance."""
    rows = _chain(10.0, KIT_A, range(0, 4), step=1.0) + _chain(40.0, KIT_A, range(38, 42), step=1.0)
    detections = _detections(rows)
    keep = np.ones(len(rows), dtype=bool)
    # one manual track across the gap, so the rule is tested without the tracker in the way
    assignment = TrackAssignment(
        track_id=np.zeros(len(rows), dtype=np.int32), team={}, kit_quality={}, tracks={0: np.arange(len(rows))}
    )
    tracks = build_tracks(detections, keep, assignment=assignment, teams={0: 0})
    assert len(tracks) == 1
    # observed steps: 1 m + 1 m + 1 m within each run = 6 m; the 28 m jump is not movement
    assert tracks[0].distance_m == 6.0
    assert float(np.max(tracks[0].speed_kmh)) <= 18.0 + 1e-6  # 1 m per 0.2 s = 18 km/h, never a teleport


def test_a_crowd_contest_does_not_strand_the_other_pair() -> None:
    """Two near-identical continuations, each contested: matching must join both pairs, not one.

    Two players run side by side; both fragments end and two continuations start after the online buffer. The
    nearer continuation is within reach of *both* endpoints, so under "join only mutual best" the loser's
    preferred link is consumed and its pair stays split - on the real game that stranded 14,617 of 24,273
    fragments' preferred links, leaving crowd players split in two. Both pairs pass the gates, so a global
    matching must take both.
    """
    rows = (
        _chain(10.0, KIT_A, range(0, 8))  # player 1: ends at (12.8, 10)
        + _chain(10.6, KIT_A, range(0, 8))  # player 2: ends at (13.4, 10)
        + _chain(12.5, KIT_A, range(24, 32))  # continuation 1 (reachable from both)
        + _chain(13.05, KIT_A, range(24, 32))  # continuation 2 (nearer to both endpoints)
    )
    detections = _detections(rows)
    assignment = _track_people(detections, np.ones(len(rows), dtype=bool))
    assert len(assignment.tracks) == 4, "the online pass must have left four fragments"

    stitched = _stitch_tracks(detections, assignment)
    assert len(stitched.tracks) == 2, "both gate-valid pairs must be joined despite the contested bests"
