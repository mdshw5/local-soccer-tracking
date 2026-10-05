"""Tests for the ported HRNet keypoint model.

The reference's weights are not distributed, so detection quality cannot be tested here. What *can* be tested is
everything that would be a porting mistake: that the architecture builds and forwards to the right shape, that the
channel->template map is the reference's and lands on real template indices, and that a heatmap peak becomes a
solver observation in the right coordinate convention. A forward pass with random weights catches the shape and
config errors; it does not pretend to catch a bad detector.
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")
import torch.nn as nn  # noqa: E402

from soccer_analytics.geometry.pitch_keypoint_model import (  # noqa: E402
    HRNET_W48_CONFIG,
    TEMPLATE_CHANNEL_MAP,
    build_keypoint_model,
    keypoint_observations,
    keypoint_peaks,
    load_keypoint_model,
    resolve_keypoint_weights,
)
from soccer_analytics.geometry.pitch_template import KEYPOINT_COUNT  # noqa: E402


def test_hrnet_builds_and_forwards_to_half_resolution_joint_maps() -> None:
    """A forward pass with random weights: the realistic porting bug is a shape or config error, not a bad detector."""
    model = build_keypoint_model()
    model.eval()
    with torch.no_grad():
        heatmaps = model(torch.zeros(1, 3, 64, 96))
    assert heatmaps.shape == (1, HRNET_W48_CONFIG["MODEL"]["NUM_JOINTS"], 32, 48)


def test_channel_map_is_the_reference_mapping_onto_the_template() -> None:
    """Each heatmap channel names a distinct template marker, and every marker is reachable.

    The map is the reference's own `map_keypoints`, re-based to 0. If a transcription were off, either two channels
    would claim the same marker or one of the 32 would be unreachable - both silent at runtime, both caught here.
    """
    assert set(TEMPLATE_CHANNEL_MAP.values()) == set(range(KEYPOINT_COUNT))
    assert len(set(TEMPLATE_CHANNEL_MAP.values())) == KEYPOINT_COUNT
    assert all(0 <= channel < HRNET_W48_CONFIG["MODEL"]["NUM_JOINTS"] - 1 for channel in TEMPLATE_CHANNEL_MAP)


class _StubHeatmapModel(nn.Module):
    """Returns a fixed heatmap regardless of input, with one peak on a chosen channel."""

    def __init__(self, channel: int, row: int, col: int):
        super().__init__()
        self.dummy = nn.Parameter(torch.zeros(1))
        self.channel, self.row, self.col = channel, row, col

    def forward(self, x):
        heat = torch.zeros(x.shape[0], 58, 4, 6)
        heat[:, self.channel, self.row, self.col] = 0.9
        return heat


def test_keypoint_peaks_returns_input_pixel_coordinates() -> None:
    """The heatmap is half the model input, so a peak at heatmap (col=2) is input pixel 2*2=4... times the resize."""
    # Input is resized to 960 wide; 6 heatmap columns -> 960/6 = 160 input pixels per heatmap pixel.
    stub = _StubHeatmapModel(channel=0, row=1, col=2)
    frame = np.zeros((40, 60, 3), dtype=np.uint8)
    peaks = keypoint_peaks(stub, frame, threshold=0.2)
    assert set(peaks) == {0}
    assert peaks[0]["x"] == pytest.approx(2 * 160.0)
    assert peaks[0]["y"] == pytest.approx(1 * 160.0)
    assert peaks[0]["p"] == pytest.approx(0.9)


def test_keypoint_observations_use_the_solver_coordinate_convention() -> None:
    """Both axes are divided by the input *width*, and the channel is mapped to its template index."""
    stub = _StubHeatmapModel(channel=0, row=1, col=2)
    frame = np.zeros((40, 60, 3), dtype=np.uint8)
    observations = keypoint_observations(stub, frame, frame_index=7, threshold=0.2)
    assert len(observations) == 1
    observation = observations[0]
    assert observation.frame == 7
    assert observation.index == TEMPLATE_CHANNEL_MAP[0]
    assert observation.u == pytest.approx(320.0 / 960.0)
    assert observation.v == pytest.approx(160.0 / 960.0)  # by width, not height
    assert observation.confidence == pytest.approx(0.9)


def test_keypoint_observations_ignore_channels_with_no_template_marker() -> None:
    """Channel 57 is the last real channel but not in the map; it must not invent a marker."""
    unmapped = next(c for c in range(56, -1, -1) if c not in TEMPLATE_CHANNEL_MAP)
    assert unmapped not in TEMPLATE_CHANNEL_MAP
    stub = _StubHeatmapModel(channel=unmapped, row=0, col=0)
    frame = np.zeros((40, 60, 3), dtype=np.uint8)
    assert keypoint_observations(stub, frame, frame_index=0, threshold=0.2) == []


def test_resolve_keypoint_weights_prefers_data_models(tmp_path, monkeypatch) -> None:
    import soccer_analytics.geometry.pitch_keypoint_model as module

    models = tmp_path / "models"
    models.mkdir()
    monkeypatch.setattr(module, "MODELS_DIR", models)
    monkeypatch.setattr(module, "DEFAULT_WEIGHTS_NAME", "keypoint_detect.pt")
    assert resolve_keypoint_weights(None) is None

    older = models / "older.pt"
    older.write_bytes(b"x")
    assert resolve_keypoint_weights(older) == str(older)

    # The named checkpoint wins over other *.pt files when no explicit path is given.
    named = models / "keypoint_detect.pt"
    named.write_bytes(b"x")
    assert resolve_keypoint_weights(None) == str(named)


def test_load_keypoint_model_says_what_to_do_when_weights_are_missing(tmp_path, monkeypatch) -> None:
    import soccer_analytics.geometry.pitch_keypoint_model as module

    monkeypatch.setattr(module, "MODELS_DIR", tmp_path / "empty-models")
    monkeypatch.setattr(module, "DEFAULT_WEIGHTS_NAME", "keypoint_detect.pt")
    with pytest.raises(FileNotFoundError, match="not distributed"):
        load_keypoint_model()


# ----------------------------------------------------------------------------------------------------------------
# The YOLO pitch source (rustyneuron01/Real-Time-Football-Detection)
# ----------------------------------------------------------------------------------------------------------------


def _yolo_stub(points):
    """A stand-in for the Ultralytics result: ``points`` is ``[(index, x_px, y_px, conf), ...]``."""
    from soccer_analytics.geometry.pitch_keypoint_yolo import frame_observations

    class _Keypoints:
        def __init__(self, xy, conf):
            self.xy = torch.tensor(xy, dtype=torch.float32)
            self.conf = torch.tensor(conf, dtype=torch.float32)

    class _Result:
        def __init__(self):
            xy = [[0.0, 0.0] for _ in range(32)]
            conf = [0.0] * 32
            for index, x, y, c in points:
                xy[index] = [x, y]
                conf[index] = c
            self.keypoints = _Keypoints([xy], [conf])

    class _Model:
        def __call__(self, frame, imgsz=640, verbose=False):
            return [_Result()]

    frame = np.zeros((270, 960, 3), dtype=np.uint8)
    return frame_observations(_Model(), frame, frame_index=5, image_sizes=(640,), threshold=0.3)


def test_yolo_observations_use_template_indices_and_by_width_coordinates() -> None:
    """Output index i is template marker i (Roboflow's vertex order), off the half-res heatmap."""
    observations = _yolo_stub([(0, 100.0, 200.0, 0.9), (31, 480.0, 135.0, 0.8)])
    assert [o.index for o in observations] == [0, 31]
    first = observations[0]
    assert first.frame == 5
    assert first.u == pytest.approx(100.0 / 960.0)
    assert first.v == pytest.approx(200.0 / 960.0)  # v is by width, not the 270-px height
    assert first.confidence == pytest.approx(0.9)


def test_yolo_observations_drop_weak_and_out_of_range_keypoints() -> None:
    observations = _yolo_stub([(0, 100.0, 200.0, 0.1), (4, 50.0, 60.0, 0.7)])
    assert [o.index for o in observations] == [4]