"""End-to-end: synthetic footage -> report -> archive -> highlights.

Each module is tested on its own elsewhere; this ties them together in the order the dashboard runs them, using the
synthetic oracle so the answers can be checked against known truth. It is the test that would catch a wiring mistake
between stages, which unit tests on either side cannot.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from soccer_analytics.analysis import stage_b
from soccer_analytics.analysis.events import Event, EventLog
from soccer_analytics.analysis.highlights import build_moments, select_reel
from soccer_analytics.analysis.library import MatchLibrary
from soccer_analytics.analysis.projection import project_segment
from soccer_analytics.geometry.pitch_calibration import PitchCalibration
from synthetic_match import PITCH_LENGTH, PITCH_WIDTH, simulate_match


@pytest.fixture(scope="module")
def played_match():
    segment, truth = simulate_match(frames=400, players_per_team=7, seed=11)
    return segment, truth


def test_stage_b_report_matches_the_simulated_match(played_match) -> None:
    segment, truth = played_match
    # The oracle's own camera model (the chain is relative to frame 0, so fold in the true frame-0 orientation),
    # so this isolates Stage B wiring from calibration error.
    from soccer_analytics.analysis.projection import segment_poses

    _q, _focal = segment_poses(segment, focal0=float(truth.focal[0]))
    cal = PitchCalibration(
        truth.calibration.position,
        truth.calibration.base_rotation @ truth.q[0],
        truth.calibration.focal_scale,
        truth.calibration.aspect,
        0.0,
        (),
    )
    detections = project_segment(segment, cal, poses=(_q, _focal))
    report, assignment = stage_b.build_report(
        detections, pitch_length_m=PITCH_LENGTH, pitch_width_m=PITCH_WIDTH, match_frames=len(segment.time)
    )

    assert report.frames_analysed == len(segment.time)
    assert report.detections_used > 0
    assert len(report.teams) == 2, "the two kit colours should surface as two teams"
    assert report.teams[0].possession_share + report.teams[1].possession_share <= 1.0 + 1e-6

    tracked = [player for player in report.players if len(player.frame) >= 8]
    assert tracked, "no track lasted long enough to be reported"

    # Team assignment should mostly agree with the simulator's own labels. A permutation is allowed because which
    # cluster is called "team 0" is arbitrary.
    agreement = max(
        np.mean([player.team == _team_of(truth, assignment, player) for player in tracked]),
        np.mean([player.team != _team_of(truth, assignment, player) for player in tracked]),
    )
    assert agreement > 0.85, f"team agreement only {agreement:.2f}"

    # Everyone on the pitch is a person, so speeds must stay in the human range.
    assert all(0.0 <= float(player.speed_kmh.max()) <= stage_b.MAX_PLAUSIBLE_SPEED_KMH for player in tracked)
    assert sum(player.distance_m for player in report.players) > 0.0


def _team_of(truth, assignment, player) -> int:
    """The true team of the person a track followed most often."""
    persons = truth.det_person[assignment.tracks[player.track_id]]
    teams = truth.team_of_player[persons]
    teams = teams[teams >= 0]
    return int(round(float(np.mean(teams)))) if len(teams) else -1


def test_report_survives_the_archive_and_feeds_the_highlights(tmp_path: Path) -> None:
    """The dashboard's exact chain: build report -> save -> reload -> moments -> reels."""
    segment, truth = simulate_match(frames=300, players_per_team=7, seed=5)
    from soccer_analytics.analysis.projection import segment_poses

    q, focal = segment_poses(segment, focal0=float(truth.focal[0]))
    cal = PitchCalibration(
        truth.calibration.position,
        truth.calibration.base_rotation @ truth.q[0],
        truth.calibration.focal_scale,
        truth.calibration.aspect,
        0.0,
        (),
    )
    detections = project_segment(segment, cal, poses=(q, focal))
    report, _ = stage_b.build_report(
        detections, pitch_length_m=PITCH_LENGTH, pitch_width_m=PITCH_WIDTH, match_frames=len(segment.time)
    )

    library = MatchLibrary(tmp_path / "matches")
    record = library.create("/videos/match.MP4", format="9v9", pitch_length_m=PITCH_LENGTH, pitch_width_m=PITCH_WIDTH)
    library.save_calibration(record.match_id, truth.calibration)
    library.save_report(
        record.match_id,
        {
            "teams": [vars(team) for team in report.teams],
            "players": [{"track_id": p.track_id, "team": p.team, "distance_m": p.distance_m} for p in report.players],
            "momentum": report.momentum,
            "notes": report.notes,
            "pitch": [PITCH_LENGTH, PITCH_WIDTH],
            "frames_analysed": report.frames_analysed,
            "detections_used": report.detections_used,
        },
    )

    payload = library.load_report(record.match_id)
    assert payload["momentum"], "the simulated match should produce momentum"
    restored = {int(key): value for key, value in payload["momentum"].items()}
    assert sorted(restored) == sorted(report.momentum), "momentum minutes changed on the way through JSON"

    loaded_calibration = library.load_calibration(record.match_id)
    assert loaded_calibration is not None
    assert np.allclose(loaded_calibration.position, truth.calibration.position)

    # Now the highlight tiers, as the dashboard offers them.
    events = EventLog([Event(time_s=20.0, type="goal", team=0), Event(time_s=90.0, type="shot", team=1)])
    library.save_events(record.match_id, events)
    assert len(library.events(record.match_id).events) == 2

    match_seconds = len(segment.time) / max(1e-6, float(np.mean(np.diff(segment.time))))
    moments = build_moments(events.events, restored, match_duration_s=None)
    assert moments, "no highlight moments were produced"

    for tier in ("clip", "goals", "match"):
        reel = select_reel(tier, moments, match_duration_s=match_seconds)
        assert reel.tier == tier
        assert all(0.0 <= moment.start_s < moment.end_s <= match_seconds + 1e-6 for moment in reel.moments)

    goals_reel = select_reel("goals", moments, match_duration_s=match_seconds)
    assert any(moment.event_type == "goal" for moment in goals_reel.moments), "the tagged goal was not used"

    summary = library.summaries()[0]
    assert summary["has_report"] and summary["has_calibration"] and summary["events"] == 2
