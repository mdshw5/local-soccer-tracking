"""The Step 1 marking video route: on-demand game encodes, cached at the old proxy path.

The route is exercised against the real HTTP server with an injected encoder that writes bytes - what the encoder
*produces* is video.py's job (tested there, and it needs a real video); what matters here is the routing contract:
which games are refused, that a long game is handed the keyframe-only decoder flag, that the encode happens once
and its file is then served with byte ranges, and that an already-built proxy is never re-encoded.
"""

from __future__ import annotations

import http.client
import threading
from pathlib import Path

import numpy as np
import pytest

from soccer_analytics.analysis import game as game_lib
from soccer_analytics.dashboard.stream import MatchStreamServer, StreamError
from soccer_analytics.ingest import ffmpeg_reader as fr

GAME_COMPONENT = (
    Path(__file__).resolve().parents[1]
    / "src"
    / "soccer_analytics"
    / "dashboard"
    / "game_timeline_component"
    / "index.html"
)


def _make_game(root: Path, game_id: str = "game_test_1", duration_s: float = 4000.0) -> Path:
    directory = game_lib.game_dir(root, game_id)
    directory.mkdir(parents=True)
    video = root / f"{game_id}_combined.mp4"
    video.write_bytes(b"not really a video")
    game_lib.GameRecord(game_id=game_id, output=str(video), duration_s=duration_s, clips=[]).save(directory)
    return directory


class _FakeEncoder:
    """Stands in for ``encode_clip``: records the call and writes the bytes a serve would hand out."""

    def __init__(self) -> None:
        self.calls: list[dict] = []

    def __call__(self, match, *, start_s, duration_s, fps, width, overlays, output, encoder, audio, skip_frame):
        self.calls.append(
            {
                "video": match.video,
                "start_s": start_s,
                "duration_s": duration_s,
                "fps": fps,
                "width": width,
                "overlays": overlays,
                "audio": audio,
                "skip_frame": skip_frame,
            }
        )
        Path(output).write_bytes(b"fake-marking-video")
        return 7


@pytest.fixture()
def game_server(tmp_path):
    games_root = tmp_path / "games"
    matches_root = tmp_path / "matches"
    matches_root.mkdir()
    encoder = _FakeEncoder()
    server = MatchStreamServer(
        ("127.0.0.1", 0), root=matches_root, clip_encoder=encoder, games_root=games_root
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host, port = server.server_address
    try:
        yield host, port, games_root, encoder
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


def test_a_long_game_is_encoded_keyframes_only_then_served_from_cache(game_server) -> None:
    host, port, games_root, encoder = game_server
    directory = _make_game(games_root)

    status, body = _request(host, port, "/game/game_test_1.mp4")
    assert status == 200 and body == b"fake-marking-video"
    assert len(encoder.calls) == 1
    call = encoder.calls[0]
    assert call["skip_frame"] == "nokey", "an hour-long game must not decode every frame"
    assert call["duration_s"] == 4000.0 and call["start_s"] == 0.0
    assert call["audio"] is False and call["overlays"] == {}
    assert (directory / "scrubber" / "proxy.mp4").read_bytes() == b"fake-marking-video"

    status, body = _request(host, port, "/game/game_test_1.mp4")
    assert status == 200 and body == b"fake-marking-video"
    assert len(encoder.calls) == 1, "the second request is served from the cache"

    status, body = _request(host, port, "/game/game_test_1.mp4", headers={"Range": "bytes=0-3"})
    assert status == 206 and body == b"fake"


def test_a_short_game_decodes_every_frame(game_server) -> None:
    host, port, games_root, encoder = game_server
    _make_game(games_root, game_id="game_short", duration_s=120.0)
    status, _body = _request(host, port, "/game/game_short.mp4")
    assert status == 200
    assert encoder.calls[0]["skip_frame"] is None, "a couple of minutes of footage is cheap to decode in full"


def test_a_prebuilt_proxy_is_served_without_any_encode(game_server) -> None:
    """Games marked before the on-demand route existed keep working: their file is the cache."""
    host, port, games_root, encoder = game_server
    directory = _make_game(games_root, game_id="game_old")
    proxy = game_lib.proxy_path(directory)
    proxy.parent.mkdir(parents=True)
    proxy.write_bytes(b"old-proxy-bytes")
    status, body = _request(host, port, "/game/game_old.mp4")
    assert status == 200 and body == b"old-proxy-bytes"
    assert encoder.calls == []


def test_unknown_games_are_refused(game_server) -> None:
    host, port, _games_root, encoder = game_server
    status, _body = _request(host, port, "/game/no_such_game.mp4")
    assert status == 409, "the route reports a missing game through the StreamError path"
    assert encoder.calls == []


def test_bad_game_ids_never_reach_the_filesystem(tmp_path) -> None:
    server = MatchStreamServer(("127.0.0.1", 0), root=tmp_path, games_root=tmp_path / "games")
    try:
        for bad in ("..", "../escape", "nested/name", ".hidden"):
            with pytest.raises(StreamError):
                server.game_source(bad)
    finally:
        server.server_close()


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
    """The marking video URL must not be a hard-coded localhost: a forwarded or LAN dashboard reaches the
    stream server on its own host, the same convention the replay pane's footage pane uses."""
    html = GAME_COMPONENT.read_text()
    assert "args.game_id" in html and "/game/" in html
    assert "location.hostname" in html and "args.stream_port" in html
    assert "//localhost" not in html, "the stream host must follow the page's own host, not a literal localhost"


def test_a_real_encode_produces_a_playable_marking_video(tmp_path) -> None:
    """One real encode through the same call the route makes: GameVideo + encode_clip + the real decoder.

    This is the integration seam the fake-encoder tests cannot see - a raw-video pipe that disagrees with the
    decoded frames would corrupt the file, and a bad skip_frame flag would fail the decode outright.
    """
    from soccer_analytics.ingest.ffmpeg_reader import probe_video
    from soccer_analytics.ingest.video_reader import VideoWriter

    games_root = tmp_path / "games"
    matches_root = tmp_path / "matches"
    matches_root.mkdir()
    video = tmp_path / "combined.mp4"
    with VideoWriter(video, fps=10.0, width=96, height=64) as writer:
        for i in range(40):
            writer.write(np.full((64, 96, 3), (i * 5) % 256, dtype=np.uint8))
    directory = game_lib.game_dir(games_root, "game_real")
    directory.mkdir(parents=True)
    game_lib.GameRecord(game_id="game_real", output=str(video), duration_s=4.0, clips=[]).save(directory)

    server = MatchStreamServer(("127.0.0.1", 0), root=matches_root, games_root=games_root, encoder="libx264")
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host, port = server.server_address
    try:
        status, body = _request(host, port, "/game/game_real.mp4?fps=10&width=320")
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)

    assert status == 200 and len(body) > 0
    probe = probe_video(game_lib.proxy_path(directory))
    assert probe.width == 320, "the width came back through the route's query"
    assert probe.fps == pytest.approx(10.0, abs=0.5)
    assert probe.duration_s == pytest.approx(4.0, abs=0.5)
