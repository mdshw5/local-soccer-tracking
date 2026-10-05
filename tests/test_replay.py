"""The replay payload: what the animated pitch view can and cannot show.

Built against the simulated match because it is the only place with known ground truth for tracks, positions and
the camera aim the ball proxy comes from.
"""

from __future__ import annotations

import numpy as np
import pytest

from soccer_analytics.analysis import stage_b
from soccer_analytics.analysis.projection import project_segment, segment_poses
from soccer_analytics.dashboard.replay import build_replay, player_table_rows
from soccer_analytics.geometry.pitch_calibration import PitchCalibration
from synthetic_match import PITCH_LENGTH, PITCH_WIDTH, simulate_match


@pytest.fixture(scope="module")
def replay_case():
    segment, truth = simulate_match(frames=200, seed=3)
    q, focal = segment_poses(segment, focal0=float(truth.focal[0]))
    calibration = PitchCalibration(
        truth.calibration.position,
        truth.calibration.base_rotation @ truth.q[0],
        truth.calibration.focal_scale,
        truth.calibration.aspect,
        0.0,
        (),
    )
    detections = project_segment(segment, calibration, poses=(q, focal))
    report, _ = stage_b.build_report(
        detections, pitch_length_m=PITCH_LENGTH, pitch_width_m=PITCH_WIDTH, match_frames=len(segment.time)
    )
    replay = build_replay(
        (PITCH_LENGTH, PITCH_WIDTH),
        float(segment.meta["fps"]),
        len(segment.time),
        detections.aim_xy,
        report.players,
        ["Team A", "Team B"],
    )
    return replay, report, segment, detections


def test_every_reported_track_is_in_the_replay(replay_case) -> None:
    replay, report, segment, _ = replay_case
    assert len(replay["players"]) == len(report.players)
    ids = {player["track_id"] for player in replay["players"]}
    assert ids == {player.track_id for player in report.players}
    assert replay["pitch"] == [PITCH_LENGTH, PITCH_WIDTH]
    assert replay["fps"] == float(segment.meta["fps"])
    assert replay["duration_s"] > 0


def test_observations_are_ordered_and_complete(replay_case) -> None:
    replay, _, segment, _ = replay_case
    for player in replay["players"]:
        frames = player["frames"]
        assert len(frames) == len(player["xy"]) == len(player["speed"])
        assert frames == sorted(frames), "the component interpolates between consecutive observations"
        assert frames[0] >= 0 and frames[-1] < len(segment.time)
        assert 0.0 <= player["stats"]["top_speed_kmh"] <= stage_b.MAX_PLAUSIBLE_SPEED_KMH
        assert player["stats"]["distance_m"] >= 0.0


def test_positions_stay_on_the_pitch(replay_case) -> None:
    """A replay dot outside the pitch would mean the projection, not the player, is on show."""
    replay, _, _, _ = replay_case
    for player in replay["players"]:
        xy = np.asarray(player["xy"], dtype=np.float64)
        assert xy[:, 0].min() > -3.0 and xy[:, 0].max() < PITCH_LENGTH + 3.0
        assert xy[:, 1].min() > -3.0 and xy[:, 1].max() < PITCH_WIDTH + 3.0


def test_the_aim_trail_has_one_entry_per_frame(replay_case) -> None:
    replay, _, segment, _ = replay_case
    assert len(replay["aim"]) == len(segment.time)
    for entry in replay["aim"]:
        assert entry is None or (len(entry) == 2 and all(isinstance(value, float) for value in entry))
    assert any(entry is not None for entry in replay["aim"]), "the camera looked at the pitch at least sometimes"


def test_the_touch_proxy_is_bounded_by_frames_with_an_aim(replay_case) -> None:
    replay, _, _, _ = replay_case
    touches = sum(player["stats"]["touches"] for player in replay["players"])
    aimed = sum(1 for entry in replay["aim"] if entry is not None)
    assert 0 < touches <= aimed, "at most one player touches the ball proxy per frame"
    assert all(player["stats"]["touches"] >= 0 for player in replay["players"])


def test_the_player_table_merges_shirt_numbers(replay_case) -> None:
    replay, _, _, _ = replay_case
    track = replay["players"][0]["track_id"]
    numbers = {track: {"number": 7, "name": "Sam", "source": "manual", "confidence": 1.0}}
    table = player_table_rows(replay, numbers)
    assert len(table) == len(replay["players"])
    row = table[table["track"] == track].iloc[0]
    assert row["number"] == 7 and row["name"] == "Sam" and row["source"] == "manual"
    others = table[table["track"] != track]
    assert (others["source"] == "unassigned").all()
    assert (others["number"].isna()).all()
