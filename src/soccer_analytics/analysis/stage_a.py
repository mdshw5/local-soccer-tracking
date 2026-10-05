"""Stage A: the heavy, run-once pass over one video segment.

Decodes on the GPU, tracks the gimbal's motion, detects people, and writes everything *raw* to disk in checkpointed
chunks. Nothing here depends on the pitch or on tracking, so it never has to be repeated when the user re-draws
pitch landmarks or tunes tracking: those are Stage B and take seconds.

Output directory layout (one per segment)::

    meta.json          probe info, analysis fps/width, status
    status.json        live progress for the dashboard (written atomically)
    chunk_00000.npz    frames [0, chunk_frames)    camera state + detections
    ...

Each chunk holds, per analysed frame ``i``: time, ok flag, inlier ratio, the *raw* camera step (so the focal length
can be recalibrated without re-reading video), and the detection arrays with a ``det_frame`` index.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from soccer_analytics.analysis.kit import DESCRIPTOR_SIZE, frame_grass_model, kit_descriptor
from soccer_analytics.geometry.camera_motion import (
    DEFAULT_FOCAL,
    CameraMotionTracker,
    RotationChain,
    overlay_mask,
)
from soccer_analytics.ingest.ffmpeg_reader import FFmpegFrameReader, probe_video

ANALYSIS_FPS = 5.0
MOTION_WIDTH = 960  # motion is estimated on a small copy; detection uses the full analysis frame
DETECT_WIDTH = 1920
CHUNK_FRAMES = 300  # 60 s at 5 fps
MIN_PERSON_HEIGHT_PX = 14  # at 1920 wide; smaller boxes are far-side noise
PERSON_CONF = 0.25
SCHEMA_VERSION = 1
REPO_ROOT = Path(__file__).resolve().parents[3]  # src/soccer_analytics/analysis/stage_a.py -> repo root
MODELS_DIR = REPO_ROOT / "data" / "models"  # where a locally-supplied checkpoint is looked for


@dataclass(frozen=True)
class SegmentConfig:
    fps: float = ANALYSIS_FPS
    motion_width: int = MOTION_WIDTH
    detect_width: int = DETECT_WIDTH
    chunk_frames: int = CHUNK_FRAMES
    confidence: float = PERSON_CONF
    weights: str = "yolov8n.pt"  # stock COCO name: Ultralytics fetches it on first use, so a fresh clone runs
    device: str | int = 0


def resolve_weights(weights: str | None = None) -> str:
    """The weights to actually load, preferring a locally-supplied model over the stock download.

    The default used to be ``data/models/yolov8n-coco-baseline.pt``, a file that is gitignored (weights are
    fetched again rather than stored) and that Ultralytics' downloader does not recognise by name - so a fresh
    clone died with a bare ``FileNotFoundError`` at ``YOLO(...)`` and no way to recover. The default is now the
    stock ``yolov8n.pt``, which Ultralytics fetches itself. With no explicit ``--weights``, a checkpoint under
    ``data/models/`` is still honoured - the most recently modified ``*.pt`` there wins, because that is where a
    fine-tuned model for this sideline camera would be dropped. Person detection is the only class Stage A needs,
    so stock COCO works - but a real fine-tuned checkpoint should detect far-side players better.
    """
    name = weights or SegmentConfig.weights
    if name and Path(name).exists():
        return name
    # Only the *default* falls back to a local checkpoint: an explicit path is the caller's choice, and a stock
    # name is a deliberate "give me exactly this model" - silently swapping either for whatever sits in
    # data/models would make a run's weights depend on what happens to be on disk.
    if weights is None:
        local = sorted(MODELS_DIR.glob("*.pt"), key=lambda p: p.stat().st_mtime) if MODELS_DIR.exists() else []
        if local:
            return str(local[-1])
    return name  # a stock name Ultralytics can download, or the caller's own (missing) path, reported as such


def _write_json_atomic(path: Path, payload: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2))
    os.replace(tmp, path)


def segment_dir_for(video_path: str | Path, root: str | Path, *, window_label: str | None = None) -> Path:
    """Stable per-video output directory; keyed by name + size so a replaced file is not mistaken for a done one.

    ``window_label`` separates runs over different parts of the same video - the two halves of a game, say. Stored
    meta refuses a different window in the same directory (correctly), so each window needs its own directory; the
    label must be filename-safe, and :meth:`soccer_analytics.analysis.game.GameRecord.window_label` produces one.
    """
    video_path = Path(video_path)
    size = video_path.stat().st_size
    safe = "".join(c if c.isalnum() or c in "-_." else "_" for c in video_path.stem)
    name = f"{safe}_{size}"
    if window_label:
        name = f"{name}__{window_label}"
    return Path(root) / name


def chunk_path(directory: Path, index: int) -> Path:
    return directory / f"chunk_{index:05d}.npz"


def completed_chunks(directory: Path) -> int:
    """Number of leading, fully-written, readable chunks (a torn last file is not counted)."""
    count = 0
    while chunk_path(directory, count).exists():
        try:
            with np.load(chunk_path(directory, count)) as data:
                _ = data["time"]
        except Exception:
            break
        count += 1
    return count


def read_status(directory: Path) -> dict | None:
    path = Path(directory) / "status.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None


def resolve_window(start_s: float, duration_s: float | None, source_duration_s: float) -> tuple[float, float]:
    """The analysed ``[start, end)`` window in source seconds.

    ``duration_s`` of ``None`` or ``0`` means "to the end of the video from the offset" - the dashboard's default,
    so picking a video and pressing run analyses everything unless a shorter length is typed in.
    """
    start = max(0.0, float(start_s))
    source_end = max(0.0, float(source_duration_s))
    if duration_s is None or float(duration_s) <= 0:
        return start, max(start, source_end)
    return start, max(start, min(source_end, start + float(duration_s)))


def _save_chunk(directory: Path, index: int, rows: dict) -> None:
    path = chunk_path(directory, index)
    tmp = path.with_name(path.name + ".tmp.npz")
    np.savez_compressed(tmp, **rows)
    os.replace(tmp, path)


def _empty_rows() -> dict[str, list]:
    return {k: [] for k in ("time", "ok", "inlier", "step", "focal", "det_frame", "det_box", "det_conf", "det_kit")}


def _pack_rows(rows: dict[str, list]) -> dict[str, np.ndarray]:
    n_det = len(rows["det_frame"])
    return {
        "time": np.asarray(rows["time"], dtype=np.float64),
        "ok": np.asarray(rows["ok"], dtype=bool),
        "inlier": np.asarray(rows["inlier"], dtype=np.float32),
        "step": np.asarray(rows["step"], dtype=np.float64).reshape(-1, 3, 3),
        "focal": np.asarray(rows["focal"], dtype=np.float32),
        "det_frame": np.asarray(rows["det_frame"], dtype=np.int32),
        "det_box": np.asarray(rows["det_box"], dtype=np.float32).reshape(n_det, 4),
        "det_conf": np.asarray(rows["det_conf"], dtype=np.float32),
        "det_kit": np.asarray(rows["det_kit"], dtype=np.float32).reshape(n_det, DESCRIPTOR_SIZE),
    }


def detect_people(model, frame: np.ndarray, config: SegmentConfig, ignore: np.ndarray) -> list[tuple]:
    """COCO person boxes ``(x1, y1, x2, y2, conf)`` in ``frame`` pixels, minus burned-in overlays and tiny boxes."""
    result = model.predict(
        frame, imgsz=config.detect_width, conf=config.confidence, classes=[0], device=config.device, verbose=False
    )[0]
    boxes = result.boxes.xyxy.cpu().numpy()
    confs = result.boxes.conf.cpu().numpy()
    out = []
    height, width = frame.shape[:2]
    for (x1, y1, x2, y2), conf in zip(boxes, confs):
        if y2 - y1 < MIN_PERSON_HEIGHT_PX * width / DETECT_WIDTH:
            continue
        foot_x, foot_y = int(np.clip((x1 + x2) / 2, 0, width - 1)), int(np.clip(y2 - 1, 0, height - 1))
        if ignore[foot_y, foot_x] == 0:  # the box's foot sits on the logo/clock overlay
            continue
        out.append((x1 / width, y1 / width, x2 / width, y2 / width, float(conf)))  # normalised by frame width
    return out


def analyse_segment(
    video_path: str | Path,
    out_dir: str | Path,
    *,
    config: SegmentConfig | None = None,
    start_s: float = 0.0,
    duration_s: float | None = None,
    model=None,
    should_stop=lambda: False,
    on_progress=lambda status: None,
) -> dict:
    """Runs (or resumes) Stage A for one segment. Returns the final status dict."""
    config = config or SegmentConfig()
    video_path, out_dir = Path(video_path), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    probe = probe_video(video_path)
    start_s, end_s = resolve_window(start_s, duration_s, probe.duration_s)
    total_frames = max(1, int((end_s - start_s) * config.fps))
    total_chunks = -(-total_frames // config.chunk_frames)

    meta = {
        "schema": SCHEMA_VERSION,
        "video": str(video_path),
        "width": probe.width,
        "height": probe.height,
        "source_fps": probe.fps,
        "duration_s": probe.duration_s,
        "start_s": start_s,
        "end_s": end_s,
        "fps": config.fps,
        "motion_width": config.motion_width,
        "detect_width": config.detect_width,
        "chunk_frames": config.chunk_frames,
        "total_frames": total_frames,
        "total_chunks": total_chunks,
        "weights": config.weights,
        "default_focal": DEFAULT_FOCAL,
    }
    existing = out_dir / "meta.json"
    if existing.exists():
        previous = json.loads(existing.read_text())
        compatible = all(previous.get(k) == meta[k] for k in ("schema", "fps", "motion_width", "detect_width", "chunk_frames", "start_s", "end_s"))
        if not compatible:
            raise ValueError(f"{out_dir} holds results with different settings; choose a new output directory")
    _write_json_atomic(existing, meta)

    if model is None:
        from ultralytics import YOLO

        model = YOLO(resolve_weights(config.weights))

    first_chunk = completed_chunks(out_dir)
    started = time.time()
    status = {"state": "running", "chunk": first_chunk, "total_chunks": total_chunks, "frames_done": first_chunk * config.chunk_frames,
              "total_frames": total_frames, "fps": 0.0, "eta_s": None, "pid": os.getpid(), "updated": started, "lost": 0, "error": None}

    def publish(**changes) -> dict:
        status.update(changes, updated=time.time())
        _write_json_atomic(out_dir / "status.json", status)
        on_progress(dict(status))
        return status

    if first_chunk >= total_chunks:
        return publish(state="done", chunk=total_chunks, frames_done=total_frames, eta_s=0.0)
    publish()

    chain = RotationChain(DEFAULT_FOCAL, probe.height / probe.width)
    motion_height = int(round(config.motion_width * probe.height / probe.width)) // 2 * 2
    resume_overlap = 1 if first_chunk > 0 else 0
    if first_chunk > 0:
        chain = _restore_chain(out_dir, first_chunk, probe)
    tracker = CameraMotionTracker((motion_height, config.motion_width), chain=chain)

    # On resume, decode one extra frame *before* the resume point. It is the frame the restored chain belongs to
    # (the last frame of the previous chunk), so seeding the tracker with it keeps the motion across the chunk
    # boundary instead of measuring the first new frame against itself.
    frame_step = 1.0 / config.fps
    reader_start = start_s + first_chunk * config.chunk_frames / config.fps - resume_overlap * frame_step
    reader = FFmpegFrameReader(
        video_path, fps=config.fps, width=config.detect_width,
        start_s=max(0.0, reader_start), duration_s=end_s - max(0.0, reader_start),
    )
    ignore_mask = None
    rows = _empty_rows()
    chunk_index, frames_in_chunk = first_chunk, 0
    lost_total = 0
    to_skip = resume_overlap

    try:
        for source_time, frame in reader.frames():
            if should_stop():
                publish(state="stopped")
                return status
            if ignore_mask is None:
                ignore_mask = overlay_mask(frame.shape[:2])
            small = cv2.resize(frame, (config.motion_width, motion_height), interpolation=cv2.INTER_AREA)
            gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
            if to_skip:
                tracker.seed_good_frame(gray)
                to_skip -= 1
                continue
            state = tracker.update(gray)
            lost_total += 0 if state.ok else 1

            frame_idx = frames_in_chunk
            rows["time"].append(source_time)
            rows["ok"].append(state.ok)
            rows["inlier"].append(state.inlier_ratio)
            rows["step"].append(state.step if state.step is not None else np.eye(3))
            rows["focal"].append(state.focal)

            grass = frame_grass_model(frame)
            for x1, y1, x2, y2, conf in detect_people(model, frame, config, ignore_mask):
                width = frame.shape[1]
                rows["det_frame"].append(frame_idx)
                rows["det_box"].append((x1, y1, x2, y2))
                rows["det_conf"].append(conf)
                rows["det_kit"].append(kit_descriptor(frame, (x1 * width, y1 * width, x2 * width, y2 * width), grass))

            frames_in_chunk += 1
            if frames_in_chunk >= config.chunk_frames:
                _save_chunk(out_dir, chunk_index, _pack_rows(rows))
                chunk_index += 1
                done = min(chunk_index * config.chunk_frames, total_frames)
                elapsed = max(time.time() - started, 1e-6)
                rate = (done - first_chunk * config.chunk_frames) / elapsed
                publish(chunk=chunk_index, frames_done=done, fps=rate, eta_s=(total_frames - done) / max(rate, 1e-6), lost=lost_total)
                rows, frames_in_chunk = _empty_rows(), 0
        if frames_in_chunk:
            _save_chunk(out_dir, chunk_index, _pack_rows(rows))
            chunk_index += 1
        publish(state="done", chunk=chunk_index, frames_done=total_frames, eta_s=0.0, lost=lost_total)
    except Exception as exc:  # leave a readable status for the dashboard instead of a silently dead job
        publish(state="error", error=f"{type(exc).__name__}: {exc}")
        raise
    return status


def _restore_chain(directory: Path, chunks: int, probe) -> RotationChain:
    """Rebuild the camera chain by replaying every stored accepted step up to the last completed chunk.

    Lost frames and the very first frame store ``step = None`` semantics as an identity step *with ok=False / init*;
    those carry no motion, so only frames flagged ``ok`` that actually had a measured step are replayed. Frame 0 of the
    segment is the only ok frame without a measured step, and it is identified by being the first row.
    """
    chain = RotationChain(DEFAULT_FOCAL, probe.height / probe.width)
    for index in range(chunks):
        with np.load(chunk_path(directory, index)) as data:
            for row, (ok, step) in enumerate(zip(data["ok"], data["step"])):
                if not ok or (index == 0 and row == 0):
                    continue
                q, focal, _ = chain.preview(step)
                chain.commit(q, focal)
    return chain


# --------------------------------------------------------------------------------------------------------------
# Reading results back
# --------------------------------------------------------------------------------------------------------------


@dataclass
class SegmentData:
    meta: dict
    time: np.ndarray  # (F,) source seconds
    ok: np.ndarray  # (F,)
    inlier: np.ndarray  # (F,)
    step: np.ndarray  # (F, 3, 3) raw normalised step (last good frame -> this frame); identity if none
    focal: np.ndarray  # (F,) chain focal at analysis time
    det_frame: np.ndarray  # (D,) global frame index
    det_box: np.ndarray  # (D, 4) x1,y1,x2,y2 normalised by frame width
    det_conf: np.ndarray  # (D,)
    det_kit: np.ndarray  # (D, DESCRIPTOR_SIZE)

    @property
    def aspect(self) -> float:
        return self.meta["height"] / self.meta["width"]


def load_segment(directory: str | Path) -> SegmentData:
    directory = Path(directory)
    meta = json.loads((directory / "meta.json").read_text())
    chunks = completed_chunks(directory)
    if chunks == 0:
        raise FileNotFoundError(f"no completed chunks in {directory}")
    parts: dict[str, list] = {k: [] for k in ("time", "ok", "inlier", "step", "focal", "det_frame", "det_box", "det_conf", "det_kit")}
    offset = 0
    for index in range(chunks):
        with np.load(chunk_path(directory, index)) as data:
            for key in parts:
                value = data[key]
                parts[key].append(value + offset if key == "det_frame" else value)
            offset += len(data["time"])
    joined = {k: np.concatenate(v) if v else np.empty(0) for k, v in parts.items()}
    return SegmentData(meta=meta, **joined)
