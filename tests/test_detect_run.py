"""``detect_run.run_detection``: the wiring from a match's saved artefacts to its event log.

The detectors themselves are exercised against the simulated match elsewhere; this file is about the *glue* the
dashboard's button and the background run share - which artefacts are required before anything may run, that the
candidates keep the segment's own time base, and that a re-run reconciles rows (replacing its own, keeping a
human's tags and the whistle candidates) instead of duplicating or stranding them.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from soccer_analytics.analysis import detect_run, stage_b
from soccer_analytics.analysis.events import Event, EventLog
from soccer_analytics.analysis.library import MatchLibrary
from soccer_analytics.analysis.projection import project_segment, segment_poses
from soccer_analytics.dashboard.replay import build_replay
from soccer_analytics.geometry.pitch_calibration import PitchCalibration, pitch_to_pixels
from synthetic_match import PITCH_LENGTH, PITCH_WIDTH, simulate_match


@pytest.fixture(scope="module")
def detection_case(tmp_path_factory):
    """A simulated match saved the way the pipeline saves one: replay, calibration and a ball scan.

    The ball scan is synthesised by projecting the simulator's true ball path back into the image and storing the
    per-frame normalised pixels a scan would have written, so the projection under test has the same ground to
    stand on the real one does. The segment itself is passed in directly (as the dashboard's cached loader does)
    - it never needs to be written to disk for detection.
    """
    workspace = tmp_path_factory.mktemp("detect-run")
    segment, truth = simulate_match(frames=300, players_per_team=7, seed=7)
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
    library = MatchLibrary(workspace / "matches")
    record = library.create(
        str(workspace / "synthetic.mp4"), format="9v9", pitch_length_m=PITCH_LENGTH, pitch_width_m=PITCH_WIDTH
    )
    library.save_calibration(record.match_id, calibration)
    library.save_replay(
        record.match_id,
        build_replay(
            (PITCH_LENGTH, PITCH_WIDTH),
            float(segment.meta["fps"]),
            len(segment.time),
            detections.aim_xy,
            report.players,
            ["Team A", "Team B"],
            camera_xy=detections.camera_xy,
        ),
    )

    # The "scan": the true ball path seen through the recovered camera chain, as normalised pixels per frame.
    aspect = float(segment.meta["height"]) / float(segment.meta["width"])
    records = []
    for frame in range(len(segment.time)):
        uv, front = pitch_to_pixels(calibration, truth.ball[frame][None, :], q[frame], focal[frame])
        if not front[0]:
            continue
        u, v = float(uv[0][0]), float(uv[0][1])
        if 0.0 <= u <= 1.0 and 0.0 <= v <= aspect:
            records.append({"i": frame, "status": "tracking", "u": u, "v": v})
    segment_dir = workspace / "segments" / "synthetic"
    segment_dir.mkdir(parents=True)
    (segment_dir / "ball_track.json").write_text(json.dumps({"frames": records, "complete": True}))
    return library, record.match_id, segment, segment_dir, (q, focal)


def _clear_events(library: MatchLibrary, match_id: str) -> None:
    """Each test starts from an empty log so the counts it asserts are its own."""
    library.save_events(match_id, EventLog())


def test_missing_inputs_are_named_with_what_to_do(tmp_path: Path) -> None:
    library = MatchLibrary(tmp_path / "matches")
    record = library.create(
        str(tmp_path / "x.mp4"), format="9v9", pitch_length_m=PITCH_LENGTH, pitch_width_m=PITCH_WIDTH
    )
    segment_dir = tmp_path / "seg"
    segment_dir.mkdir()
    with pytest.raises(detect_run.MissingInput, match="report"):
        detect_run.run_detection(library, record.match_id, segment_dir)
    library.save_replay(record.match_id, {"players": []})
    with pytest.raises(detect_run.MissingInput, match="ball scan"):
        detect_run.run_detection(library, record.match_id, segment_dir)
    (segment_dir / "ball_track.json").write_text(
        json.dumps({"frames": [{"i": 0, "status": "tracking", "u": 0.5, "v": 0.5}]})
    )
    with pytest.raises(detect_run.MissingInput, match="pitch"):
        detect_run.run_detection(library, record.match_id, segment_dir)


def test_detection_runs_against_the_saved_artefacts(detection_case) -> None:
    library, match_id, segment, segment_dir, poses = detection_case
    _clear_events(library, match_id)
    result = detect_run.run_detection(
        library, match_id, segment_dir, video="synthetic.mp4", half_bounds=None, segment=segment, poses=poses
    )
    assert result["detected"] == result["added"]
    assert result["dropped"] == 0
    log = library.events(match_id)
    assert len(log.events) == result["added"]
    times = np.asarray(segment.time)
    for event in log.events:
        assert event.source == "ball", "the detectors' rows carry their own source"
        assert event.video == "synthetic.mp4"
        # The candidates must live on the segment's own time base - an off-by-`start_s` shift is the classic bug
        # here, and it would land every event outside the analysed window.
        assert times.min() - 1.0 <= event.time_s <= times.max() + 1.0


def test_a_rerun_reconciles_instead_of_duplicating(detection_case) -> None:
    library, match_id, segment, segment_dir, poses = detection_case
    _clear_events(library, match_id)
    first = detect_run.run_detection(
        library, match_id, segment_dir, video="synthetic.mp4", segment=segment, poses=poses
    )
    second = detect_run.run_detection(
        library, match_id, segment_dir, video="synthetic.mp4", segment=segment, poses=poses
    )
    assert second["detected"] == first["detected"]
    assert second["added"] == 0, "the same candidates must not be stored a second time"
    assert second["dropped"] == 0, "rows the detector still reports must survive the reconcile"
    assert len(library.events(match_id).events) == first["added"]


def test_manual_tags_and_whistles_survive_the_detectors(detection_case) -> None:
    """The detectors replace their own rows only: a human's tag and an audio candidate are not theirs to drop."""
    library, match_id, segment, segment_dir, poses = detection_case
    log = EventLog(
        [
            Event(time_s=30.0, type="goal", note="hand tagged", video="synthetic.mp4"),
            Event(time_s=31.0, type="other", note="whistle", source="audio", video="synthetic.mp4"),
        ]
    )
    library.save_events(match_id, log)
    detect_run.run_detection(library, match_id, segment_dir, video="synthetic.mp4", segment=segment, poses=poses)
    kept = library.events(match_id).events
    assert any(event.source == "manual" and event.note == "hand tagged" for event in kept)
    assert any(event.source == "audio" and event.note == "whistle" for event in kept)
