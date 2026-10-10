"""The Step 1 marking route: MJPEG frames straight from the clips, no merged video and no encode.

The route is exercised against the real HTTP server with two tiny real clips. What matters is the routing
contract: a bad id never reaches the filesystem, a missing game is refused, a still request (``frames=1``) ends
after one JPEG, a bounded stream ends after that many, and a seek lands in the clip containing the game time -
the frame's tint names the clip. Nothing is built and nothing is cached; frames decode on demand.
"""

from __future__ import annotations

import http.client
import threading
from pathlib import Path

import cv2
import numpy as np
import pytest

from soccer_analytics.analysis import game as game_lib
from soccer_analytics.dashboard.stream import MatchStreamServer
from soccer_analytics.ingest import ffmpeg_reader as fr
from soccer_analytics.ingest.video_reader import VideoWriter

GAME_COMPONENT = (
    Path(__file__).resolve().parents[1]
    / "src"
    / "soccer_analytics"
    / "dashboard"
    / "game_timeline_component"
    / "index.html"
)

W, H = 96, 64


def _clip(path: Path, *, seconds: float, tint: str, fps: float = 10.0) -> None:
    """A short clip whose tint names it (red or blue), so a decoded frame says which clip it came from."""
    rng = np.random.default_rng(5)
    base = (rng.random((H, W)).astype(np.float32) * 200 + 30).astype(np.uint8)
    with VideoWriter(path, fps=fps, width=W, height=H) as writer:
        for _ in range(int(seconds * fps)):
            frame = cv2.cvtColor(base, cv2.COLOR_GRAY2BGR)
            if tint == "red":
                frame[:, :, 0] //= 4
                frame[:, :, 1] //= 4
            else:
                frame[:, :, 2] //= 4
            writer.write(frame)


def _make_game(root: Path, game_id: str = "game_test_1") -> Path:
    """Two small clips and their manifest in the legacy games root the server was told about."""
    root.mkdir(parents=True, exist_ok=True)
    first, second = root / f"{game_id}_a.mp4", root / f"{game_id}_b.mp4"
    _clip(first, seconds=1.0, tint="red")
    _clip(second, seconds=1.0, tint="blue")
    d1 = float(fr.probe_video(first).duration_s)
    d2 = float(fr.probe_video(second).duration_s)
    directory = game_lib.game_dir(root, game_id)
    directory.mkdir(parents=True)
    record = game_lib.GameRecord(
        game_id=game_id,
        output=str(root / f"{game_id}_merged.mp4"),  # the merge that is deliberately never created
        duration_s=d1 + d2,
        clips=[
            game_lib.Clip(path=str(first), start_s=0.0, duration_s=d1, bytes=first.stat().st_size),
            game_lib.Clip(path=str(second), start_s=d1, duration_s=d2, bytes=second.stat().st_size),
        ],
    )
    record.save(directory)
    return directory


@pytest.fixture()
def game_server(tmp_path):
    games_root = tmp_path / "games"
    matches_root = tmp_path / "matches"
    matches_root.mkdir()
    server = MatchStreamServer(("127.0.0.1", 0), root=matches_root, games_root=games_root)
    server.pace = False  # no real-time pacing: tests read as fast as frames decode
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host, port = server.server_address
    try:
        yield host, port, games_root
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def _request(host: str, port: int, path: str, headers: dict | None = None) -> tuple[int, bytes]:
    connection = http.client.HTTPConnection(host, port, timeout=10)
    connection.request("GET", path, headers=headers or {})
    response = connection.getresponse()
    body = response.read()
    connection.close()
    return response.status, body


def _jpegs(body: bytes) -> list[np.ndarray]:
    """The JPEG frames of a multipart MJPEG response, decoded."""
    frames: list[np.ndarray] = []
    for part in body.split(b"--frame"):
        start = part.find(b"\xff\xd8")
        end = part.find(b"\xff\xd9", start)
        if start == -1 or end == -1:
            continue
        image = cv2.imdecode(np.frombuffer(part[start : end + 2], dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is not None:
            frames.append(image)
    return frames


def test_a_still_request_returns_exactly_one_frame(game_server) -> None:
    host, port, games_root = game_server
    _make_game(games_root)

    status, body = _request(host, port, "/game/game_test_1.mjpg?frames=1&width=320&fps=8&t=0.2")

    frames = _jpegs(body)
    assert status == 200
    assert len(frames) == 1, "frames=1 is a still preview: one JPEG, then the stream closes"
    assert frames[0].shape[1] == 320


def test_a_seek_lands_in_the_clip_that_contains_the_game_time(game_server) -> None:
    host, port, games_root = game_server
    directory = _make_game(games_root)
    joint = float(game_lib.GameRecord.load(directory).clips[1].start_s)

    before = _jpegs(_request(host, port, f"/game/game_test_1.mjpg?frames=1&width=320&t={joint - 0.4:.3f}")[1])
    after = _jpegs(_request(host, port, f"/game/game_test_1.mjpg?frames=1&width=320&t={joint + 0.4:.3f}")[1])

    assert before and after
    assert before[0][:, :, 2].mean() > before[0][:, :, 0].mean() + 20, "before the join is the red clip"
    assert after[0][:, :, 0].mean() > after[0][:, :, 2].mean() + 20, "after the join is the blue clip"


def test_a_bounded_stream_ends_after_the_requested_frames(game_server) -> None:
    host, port, games_root = game_server
    _make_game(games_root)

    status, body = _request(host, port, "/game/game_test_1.mjpg?frames=3&width=320&fps=10&t=0")

    assert status == 200
    assert len(_jpegs(body)) == 3


def test_unknown_games_are_refused(game_server) -> None:
    host, port, _games_root = game_server
    status, _body = _request(host, port, "/game/no_such_game.mjpg")
    assert status == 409, "the route reports a missing game through the StreamError path"


def test_bad_game_ids_never_reach_the_filesystem(game_server) -> None:
    host, port, _games_root = game_server
    for bad in ("..", ".hidden", "nested/name"):
        status, _body = _request(host, port, f"/game/{bad}.mjpg")
        assert status == 404, f"{bad!r} must be refused before any directory is joined"


def test_skip_frame_is_an_input_option(tmp_path, monkeypatch) -> None:
    video = tmp_path / "v.mp4"
    video.write_bytes(b"x")
    probe = fr.VideoProbe(width=64, height=48, fps=30.0, duration_s=1.0, codec="h264", has_audio=False)
    monkeypatch.setattr(fr, "probe_video", lambda _path: probe)

    command = fr.FFmpegFrameReader(video, skip_frame="nokey")._command(use_gpu=False)
    assert command[command.index("-skip_frame") + 1] == "nokey"
    assert command.index("-skip_frame") < command.index("-i"), "a decoder option belongs before its input"

    assert "-skip_frame" not in fr.FFmpegFrameReader(video)._command(use_gpu=False)


def test_the_component_composes_the_url_from_the_pages_own_host() -> None:
    """The marking stream URL must not be a hard-coded localhost: a forwarded or LAN dashboard reaches the
    stream server on its own host, the same convention the replay pane's footage pane uses."""
    html = GAME_COMPONENT.read_text()
    assert "args.game_id" in html and "/game/" in html and ".mjpg" in html
    assert "location.hostname" in html and "args.stream_port" in html
    assert "//localhost" not in html, "the stream host must follow the page's own host, not a literal localhost"



