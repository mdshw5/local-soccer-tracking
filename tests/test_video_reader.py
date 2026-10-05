"""Smoke tests for frame-streaming video I/O (Phase 0)."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from soccer_analytics.ingest.video_reader import VideoReader, VideoWriter


@pytest.fixture()
def sample_video(tmp_path):
    path = tmp_path / "sample.mp4"
    width, height, fps, num_frames = 64, 48, 10.0, 20
    with VideoWriter(path, fps=fps, width=width, height=height) as writer:
        for i in range(num_frames):
            frame = np.full((height, width, 3), fill_value=i % 256, dtype=np.uint8)
            writer.write(frame)
    return path, width, height, fps, num_frames


def test_video_reader_reads_all_frames(sample_video):
    path, width, height, fps, num_frames = sample_video
    with VideoReader(path) as reader:
        info = reader.info
        assert info.width == width
        assert info.height == height
        frames = list(reader.frames())
    assert len(frames) == num_frames
    assert frames[0][0] == 0
    assert frames[0][1].shape == (height, width, 3)


def test_video_reader_frame_stride_subsamples(sample_video):
    path, *_ = sample_video
    with VideoReader(path, frame_stride=3) as reader:
        indices = [idx for idx, _frame in reader.frames()]
    assert indices == [0, 3, 6, 9, 12, 15, 18]


def test_video_reader_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        VideoReader(tmp_path / "missing.mp4")
