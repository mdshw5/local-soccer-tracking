"""Parse the gimbal camera's own telemetry log: the camera telling us where it pointed and whether it held the ball.

The Chameleon/Falcon writes a text log beside each video clip. It is the *hardware's* record of the tracking loop,
not an estimate from the picture, and it carries three things the pipeline has had to reconstruct by hand:

* the camera's own **yaw and pitch** every frame (the pan/tilt the gimbal actually commanded), plus its **zoom**
  step - the ground truth that ``geometry.camera_motion`` estimates from the image and that ``geometry.drift``
  exists to correct;
* the **ball-lock state** (``Lock:a/b``) and the hardware ball tracker's own box and velocity (``T0``/``BT0``),
  which is a second, independent ball measurement beside the expensive ``analysis.ball`` scan;
* the **frame counter**, which runs continuously across the clips of one game, so a log frame maps onto the
  combined game video's frame index without any image matching.

Format
------
One record per timestamp: a ``YYYY-MM-DD HH:MM:SS.mmm`` line, then one or more content lines, then a blank line.
A frame is described by two records - a motion line (``FrameN fpsX SZx Yaw.. Pit..``) and a ``FIN`` line
(``FrameN FIN:x.. y.. V.. YawErr.. out.. Ctrl:X.. Y.. zoomSz.. In.. Lock:a/b``) - which this parser merges by
frame number. When the hardware has the ball, the motion record also carries ``T0``/``BT0`` lines with the ball's
box, velocity and image position.

Everything here is pure text in, records out: no video, no numpy, no GPU. That is deliberate - the mapping from
yaw/pitch/zoom to a camera pose is the part where mistakes are expensive and invisible, so it is tested on
synthetic log text and against the estimated chain, never guessed.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

# One record: a timestamp line, then content lines, then a blank line. The timestamp is the only line that starts
# with a date, so it is the record separator.
_TIMESTAMP = re.compile(r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d+)$")
# Motion line: the gimbal's commanded orientation and its zoom step for this frame.
_MOTION = re.compile(r"^Frame(\d+) fps(\d+) SZ(\d+) Yaw(-?[\d.]+) Pit(-?[\d.]+)$")
# FIN line: the tracking loop's own state - where the target sits, the yaw error, the control output, the zoom
# step, and the ball-lock counter. ``+Zoom:1.0`` appears only while a zoom is in progress.
_FIN = re.compile(
    r"^Frame(\d+) FIN:x(-?\d+) y(-?\d+) V(-?\d+) YawErr(-?\d+) out(-?\d+) "
    r"Ctrl:X(-?\d+) Y(-?\d+) (?:\+Zoom:[\d.]+ )?zoomSz(\d+) In:(\d+) Lock:(\d+)/(\d+)$"
)
# Ball size line: the ball's apparent size and the target size, and whether the detector failed this frame. When
# the hardware has a position it also carries the ball's image x (``Ballx``) and the control output (``OutX``).
# ``TSz`` is absent on some frames, so it is optional.
_BALL_SIZE = re.compile(r"^BallSz(\d+)(?: TSz(\d+))?(?: StaX(-?\d+) Ballx(\d+) OutX(\d+))?( BallFail)?$")
# Hardware ball track: the ball's velocity in the image and its offset from the frame center. The index after T is
# the track id (T0, T1, ...).
_BALL_TRACK = re.compile(r"^T(\d+) RB(\d+) xv(-?\d+) yv(-?\d+) xm(-?\d+) ym(-?\d+)$")
# Hardware ball box: the tracked box, its offset, which side of the frame it is on, and its image x. The velocity
# carries an ``F`` suffix on some frames (a fast-motion flag), so it is optional.
_BALL_BOX = re.compile(
    r"^BT(\d+)(?:maySta)?(RB|PB)\((-?\d+),(-?\d+),(\d+),(\d+)\)V(-?\d+)F?to([LR]) Ballx(\d+) OutX(\d+)$"
)
# The motion filter's state: the filtered period, the motion magnitude, and the recent (velocity, x) deque.
_AFILTER = re.compile(r"^AFilterP(\d+) mot(\d+) Deque(\d+)((?:\(V(-?\d+),x(-?\d+)\))+)$")
_DEQUE_ENTRY = re.compile(r"\(V(-?\d+),x(-?\d+)\)")
# Fast-motion counter and the crowd's image x.
_FAST = re.compile(r"^Fast(\d+) Crowdx(\d+)$")
# The gimbal hit a pan limit and auto-centered; not needed for the pose but kept so the parser is complete.
_BOUNDARY = re.compile(r"^boundary(\d+) [LR]X(\d+) autoCenterX(\d+)$")
_AUTO_CENTER = re.compile(r"^autoCenterX(\d+)$")
# Header lines: SDK version, the sport/config line, and the start/restart markers.
_SDK = re.compile(r"^SDK(V[\d.]+-[\w-]+)$")
_CONFIG = re.compile(r"^Soccer (.+)$")


@dataclass
class GimbalFrame:
    """One frame of gimbal telemetry, merged from its motion and FIN records.

    Every field is optional because the two records arrive separately and a log can be truncated mid-frame; a
    consumer must treat a missing field as "the hardware did not say", not as zero.
    """

    frame: int
    time_s: float | None = None  # seconds since the log's first timestamp (monotonic within a file)
    yaw_deg: float | None = None  # commanded pan, degrees
    pitch_deg: float | None = None  # commanded tilt, degrees
    zoom_sz: int | None = None  # zoom step from the motion line
    fin_zoom_sz: int | None = None  # zoom step from the FIN line (the loop's own view)
    lock: tuple[int, int] | None = None  # (a, b) from Lock:a/b; a > 0 means the hardware is holding the ball
    ball_sz: int | None = None  # apparent ball size
    target_sz: int | None = None  # the size the tracker is aiming for
    ball_fail: bool = False  # the detector reported no ball this frame
    # Motion filter state (AFilter/Fast/Crowd): the filtered period, the motion magnitude, the recent
    # (velocity, x) deque, the fast-motion counter and the crowd's image x. Kept because they are the loop's own
    # view of how much the picture is moving - useful for cross-checking the pose and for image-space speed.
    a_filter_p: int | None = None
    motion: int | None = None
    deque: tuple[tuple[float, float], ...] = ()
    fast: int | None = None
    crowd_x: float | None = None
    # Hardware ball track (T0): image velocity and offset from the frame center.
    ball_track_id: int | None = None
    ball_xv: float | None = None
    ball_yv: float | None = None
    ball_xm: float | None = None
    ball_ym: float | None = None
    # Hardware ball box (BT0): the tracked box, its offset, side and image x.
    ball_box: tuple[float, float, float, float] | None = None
    ball_dir: str | None = None  # "L" or "R": which side of the frame the ball is on
    ball_x: float | None = None  # Ballx: the ball's image x, pixels
    out_x: float | None = None  # OutX: the control output x
    # FIN-line state, kept for completeness and for the alignment cross-check.
    fin_x: float | None = None
    fin_y: float | None = None
    yaw_err: float | None = None
    ctrl_x: float | None = None
    ctrl_y: float | None = None

    @property
    def locked(self) -> bool:
        """Whether the hardware is holding the ball this frame (``Lock:a/b`` with ``a > 0``)."""
        return self.lock is not None and self.lock[0] > 0

    def to_json(self) -> dict:
        return {
            "frame": self.frame,
            "time_s": self.time_s,
            "yaw_deg": self.yaw_deg,
            "pitch_deg": self.pitch_deg,
            "zoom_sz": self.zoom_sz,
            "fin_zoom_sz": self.fin_zoom_sz,
            "lock": list(self.lock) if self.lock else None,
            "ball_sz": self.ball_sz,
            "target_sz": self.target_sz,
            "ball_fail": self.ball_fail,
            "a_filter_p": self.a_filter_p,
            "motion": self.motion,
            "deque": [list(entry) for entry in self.deque],
            "fast": self.fast,
            "crowd_x": self.crowd_x,
            "ball_track_id": self.ball_track_id,
            "ball_xv": self.ball_xv,
            "ball_yv": self.ball_yv,
            "ball_xm": self.ball_xm,
            "ball_ym": self.ball_ym,
            "ball_box": list(self.ball_box) if self.ball_box else None,
            "ball_dir": self.ball_dir,
            "ball_x": self.ball_x,
            "out_x": self.out_x,
            "fin_x": self.fin_x,
            "fin_y": self.fin_y,
            "yaw_err": self.yaw_err,
            "ctrl_x": self.ctrl_x,
            "ctrl_y": self.ctrl_y,
        }


@dataclass
class GimbalLog:
    """One log file's frames, in frame order, plus the header the camera wrote at the top."""

    path: str
    frames: list[GimbalFrame] = field(default_factory=list)
    sdk: str = ""
    config: str = ""
    unknown_lines: int = 0  # lines the parser did not recognize; a non-zero count is a format change to look at

    def by_frame(self) -> dict[int, GimbalFrame]:
        return {record.frame: record for record in self.frames}

    @property
    def first_frame(self) -> int | None:
        return self.frames[0].frame if self.frames else None

    @property
    def last_frame(self) -> int | None:
        return self.frames[-1].frame if self.frames else None


def _parse_timestamp(text: str) -> float:
    """Wall-clock seconds since the epoch for a log timestamp (local time, as the camera writes it)."""
    return datetime.strptime(text, "%Y-%m-%d %H:%M:%S.%f").timestamp()


def parse_gimbal_log(text: str, *, path: str = "") -> GimbalLog:
    """Parse one log file's text into frames, merging each frame's motion and FIN records.

    The parser is line-oriented and tolerant: a record whose content lines are split across timestamps (which is
    how the camera writes them) is merged by frame number, and any line it does not recognize is counted rather
    than raised on, so a firmware that adds a field does not break the whole log.

    The camera occasionally splits a content line mid-token across two lines (a logging race), so a line that does
    not parse on its own is held and retried joined to the next one before being counted as unknown.
    """
    log = GimbalLog(path=path)
    records: dict[int, GimbalFrame] = {}
    order: list[int] = []
    first_time: float | None = None
    current_time: float | None = None
    pending = ""

    def frame_record(number: int) -> GimbalFrame:
        record = records.get(number)
        if record is None:
            record = GimbalFrame(frame=number)
            records[number] = record
            order.append(number)
        return record

    def latest() -> GimbalFrame | None:
        return records[order[-1]] if order else None

    def consume(line: str) -> bool:
        """Handle one content line; returns whether it was recognized."""
        motion = _MOTION.match(line)
        if motion:
            record = frame_record(int(motion.group(1)))
            record.yaw_deg = float(motion.group(4))
            record.pitch_deg = float(motion.group(5))
            record.zoom_sz = int(motion.group(3))
            if record.time_s is None and current_time is not None and first_time is not None:
                record.time_s = current_time - first_time
            return True
        fin = _FIN.match(line)
        if fin:
            record = frame_record(int(fin.group(1)))
            record.fin_x = float(fin.group(2))
            record.fin_y = float(fin.group(3))
            record.yaw_err = float(fin.group(5))
            record.ctrl_x = float(fin.group(7))
            record.ctrl_y = float(fin.group(8))
            record.fin_zoom_sz = int(fin.group(9))
            record.lock = (int(fin.group(11)), int(fin.group(12)))
            if record.time_s is None and current_time is not None and first_time is not None:
                record.time_s = current_time - first_time
            return True
        ball_size = _BALL_SIZE.match(line)
        if ball_size:
            # The ball-size line belongs to the most recent frame seen; it is written inside the motion record.
            record = latest()
            if record is not None:
                record.ball_sz = int(ball_size.group(1))
                record.target_sz = int(ball_size.group(2)) if ball_size.group(2) is not None else None
                if ball_size.group(4) is not None:
                    record.ball_x = float(ball_size.group(4))
                    record.out_x = float(ball_size.group(5))
                record.ball_fail = ball_size.group(6) is not None
            return True
        track = _BALL_TRACK.match(line)
        if track:
            record = latest()
            if record is not None:
                record.ball_track_id = int(track.group(1))
                record.ball_xv = float(track.group(3))
                record.ball_yv = float(track.group(4))
                record.ball_xm = float(track.group(5))
                record.ball_ym = float(track.group(6))
            return True
        box = _BALL_BOX.match(line)
        if box:
            record = latest()
            if record is not None:
                record.ball_box = (float(box.group(3)), float(box.group(4)), float(box.group(5)), float(box.group(6)))
                record.ball_dir = box.group(8)
                record.ball_x = float(box.group(9))
                record.out_x = float(box.group(10))
            return True
        a_filter = _AFILTER.match(line)
        if a_filter:
            record = latest()
            if record is not None:
                record.a_filter_p = int(a_filter.group(1))
                record.motion = int(a_filter.group(2))
                record.deque = tuple((float(v), float(x)) for v, x in _DEQUE_ENTRY.findall(a_filter.group(4)))
            return True
        fast = _FAST.match(line)
        if fast:
            record = latest()
            if record is not None:
                record.fast = int(fast.group(1))
                record.crowd_x = float(fast.group(2))
            return True
        return bool(_BOUNDARY.match(line) or _AUTO_CENTER.match(line))

    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        stamp = _TIMESTAMP.match(line)
        if stamp:
            # A split line's two halves are separated by the frame's own timestamp, so the pending buffer is kept
            # across timestamps rather than flushed here.
            current_time = _parse_timestamp(stamp.group(1))
            if first_time is None:
                first_time = current_time
            continue
        sdk = _SDK.match(line)
        if sdk:
            log.sdk = sdk.group(1)
            continue
        config = _CONFIG.match(line)
        if config:
            log.config = config.group(1)
            continue
        if line in ("SDKStart", "SDK~", "SDKStop") or line.startswith("SetPitchInitialAngle"):
            continue
        if consume(line):
            if pending:
                log.unknown_lines += 1  # a held fragment that never completed
                pending = ""
            continue
        # Not recognized on its own: it may be the first half of a line the camera split mid-token. Hold it and
        # retry joined to the next line - with and without a separator, because the split sometimes lands on a
        # space and consumes it. A genuinely unknown line is counted once it is too long to be a prefix.
        if pending:
            for joined in (pending + line, pending + " " + line):
                if consume(joined):
                    pending = ""
                    break
            else:
                pending += line
        else:
            pending = line
        if pending and len(pending) > 200:
            log.unknown_lines += 1
            pending = ""
    if pending:
        log.unknown_lines += 1

    log.frames = [records[number] for number in sorted(order)]
    return log


def load_gimbal_log(path: str | Path) -> GimbalLog:
    """Read and parse one log file."""
    path = Path(path)
    return parse_gimbal_log(path.read_text(errors="replace"), path=str(path))


def load_gimbal_logs(paths: list[str | Path]) -> GimbalLog:
    """Merge several log files into one frame-ordered log.

    The clips of a game each get their own log, and the frame counter runs continuously across them, so the merge
    is a sort by frame number. Where two files describe the same frame (a clip boundary), the record with more
    fields wins, so a truncated tail does not overwrite a complete record.
    """
    merged = GimbalLog(path="")
    records: dict[int, GimbalFrame] = {}
    for path in paths:
        log = load_gimbal_log(path)
        if not merged.sdk:
            merged.sdk = log.sdk
        if not merged.config:
            merged.config = log.config
        merged.unknown_lines += log.unknown_lines
        for record in log.frames:
            existing = records.get(record.frame)
            if existing is None or _completeness(record) > _completeness(existing):
                records[record.frame] = record
    merged.frames = [records[number] for number in sorted(records)]
    return merged


def _completeness(record: GimbalFrame) -> int:
    """How many fields a record actually carries, to break ties when merging overlapping logs."""
    return sum(
        value is not None
        for value in (
            record.yaw_deg,
            record.pitch_deg,
            record.zoom_sz,
            record.fin_zoom_sz,
            record.lock,
            record.ball_sz,
            record.ball_x,
        )
    )