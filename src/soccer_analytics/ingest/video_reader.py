"""Frame-by-frame video I/O that never buffers a full video in memory."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass(frozen=True)
class VideoInfo:
    fps: float
    width: int
    height: int
    frame_count: int


class VideoReader:
    """Streams frames from a video file one at a time via OpenCV."""

    def __init__(self, path: str | Path, frame_stride: int = 1, start_time_s: float = 0.0):
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(f"Video not found: {self.path}")
        if frame_stride < 1:
            raise ValueError("frame_stride must be >= 1")
        if start_time_s < 0:
            raise ValueError("start_time_s must be >= 0")
        self.frame_stride = frame_stride
        self.start_time_s = start_time_s
        self._cap = cv2.VideoCapture(str(self.path))
        if not self._cap.isOpened():
            raise OSError(f"Could not open video: {self.path}")
        if start_time_s > 0:
            self._cap.set(cv2.CAP_PROP_POS_MSEC, start_time_s * 1000.0)

    @property
    def info(self) -> VideoInfo:
        return VideoInfo(
            fps=self._cap.get(cv2.CAP_PROP_FPS),
            width=int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            height=int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            frame_count=int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        )

    def frames(self) -> Iterator[tuple[int, np.ndarray]]:
        """Yield (frame_index, frame) pairs, skipping according to frame_stride."""
        index = 0
        while True:
            ok, frame = self._cap.read()
            if not ok:
                break
            if index % self.frame_stride == 0:
                yield index, frame
            index += 1

    def close(self) -> None:
        self._cap.release()

    def __enter__(self) -> "VideoReader":
        return self

    def __exit__(self, *_exc_info: object) -> None:
        self.close()


class VideoWriter:
    """Writes frames to a video file incrementally."""

    def __init__(self, path: str | Path, fps: float, width: int, height: int):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self._writer = cv2.VideoWriter(str(self.path), fourcc, fps, (width, height))
        if not self._writer.isOpened():
            raise OSError(f"Could not open video writer for: {self.path}")

    def write(self, frame: np.ndarray) -> None:
        self._writer.write(frame)

    def close(self) -> None:
        self._writer.release()

    def __enter__(self) -> "VideoWriter":
        return self

    def __exit__(self, *_exc_info: object) -> None:
        self.close()
