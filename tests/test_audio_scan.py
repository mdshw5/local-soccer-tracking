"""The whistle scan is a background process: it must report progress, and it must land in the match.

It used to run inside the Streamlit script, which froze the page for minutes on a full game. What these tests pin
down is the contract that makes moving it out safe: the decode reports a monotone fraction, the wav cache is only
renamed into place once it is complete, the status file is what the dashboard reads, and the candidates end up in
the match's events - once, even if the scan is run twice.
"""

from __future__ import annotations

import importlib.util
import subprocess
import wave
from pathlib import Path

import numpy as np
import pytest

from soccer_analytics.analysis import game as game_lib
from soccer_analytics.analysis.library import MatchLibrary, audio_cache_path
from soccer_analytics.ingest.ffmpeg_reader import extract_audio

REPO_ROOT = Path(__file__).resolve().parents[1]
SR = 16000


def _script():  # noqa: ANN202 - the scan module, loaded without making scripts/ a package
    spec = importlib.util.spec_from_file_location("run_audio_scan", REPO_ROOT / "scripts" / "run_audio_scan.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_audio_cache_lives_inside_the_match_keyed_by_the_source(tmp_path: Path) -> None:
    """Everything derived from a match lives beside it: clearing or moving the analysis resets or carries its
    cached audio with the rest of the state. The name is keyed by what the video is - a clip set by its game id
    (every manifest is called game.json), a plain file by its stem - so two recordings cannot share a cache."""
    video = tmp_path / "2026-10-03" / "game_16-28-37.784.mp4"
    video.parent.mkdir(parents=True)
    video.write_bytes(b"x" * 8)
    match = tmp_path / "2026-10-03" / "analysis" / "2026-10-03_game_16-28-37.784"
    assert audio_cache_path(match, video) == match / "audio" / "game_16-28-37.784.wav"

    clips = []
    for index, (start, length) in enumerate(((0.0, 5.0), (5.0, 5.0))):
        clip = tmp_path / "2026-10-03" / f"clip{index}.MP4"
        clip.write_bytes(b"c" * 4)
        clips.append(game_lib.Clip(path=str(clip), start_s=start, duration_s=length, bytes=0))
    game_dir = tmp_path / "2026-10-03" / "analysis" / "2026-10-03_game_x"
    game_dir.mkdir(parents=True)
    game_lib.GameRecord(
        game_id=game_dir.name, output=str(tmp_path / "2026-10-03" / "merged.mp4"), duration_s=10.0, clips=clips
    ).save(game_dir)
    manifest = game_dir / "game.json"
    assert audio_cache_path(match, manifest) == match / "audio" / f"{game_dir.name}.wav"


def _tone(freq: float, duration: float, amplitude: float = 0.35) -> np.ndarray:
    time = np.arange(int(duration * SR)) / SR
    return (amplitude * np.sin(2 * np.pi * freq * time)).astype(np.float32)


def _speech_like(duration: float, rng: np.random.Generator) -> np.ndarray:
    """Harmonic stacks with formant-ish peaks: what the real match audio mostly is."""
    time = np.arange(int(duration * SR)) / SR
    out = np.zeros_like(time)
    for harmonic in range(1, 9):
        out += (0.25 / harmonic) * np.sin(2 * np.pi * 120 * harmonic * time + rng.uniform(0, 6))
    out *= 0.5 + 0.5 * np.sin(2 * np.pi * 3.0 * time)
    out += rng.normal(0.0, 0.02, len(time))
    return (0.3 * out).astype(np.float32)


def _make_video(path: Path, samples: np.ndarray) -> Path:
    """A small real video carrying the given audio, so ffmpeg does the extraction - nothing here is stubbed."""
    source = path.with_suffix(".source.wav")
    frames = np.clip(samples, -1.0, 1.0)
    with wave.open(str(source), "wb") as out:
        out.setnchannels(1)
        out.setsampwidth(2)
        out.setframerate(SR)
        out.writeframes((frames * 32767.0).astype(np.int16).tobytes())
    subprocess.run(
        [
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
            "-f", "lavfi", "-i", f"color=c=green:s=320x240:r=5:d={samples.size / SR:.3f}",
            "-i", str(source), "-shortest",
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac",
            str(path),
        ],
        check=True,
        capture_output=True,
    )
    return path


def _ramp(calls: list[float], label: str, *, at_least: int = 2) -> None:
    assert len(calls) >= at_least, f"{label}: a long task that never reports is a spinner in disguise"
    assert all(0.0 <= call <= 1.0 for call in calls), f"{label}: fractions must be bounded"
    assert calls == sorted(calls), f"{label}: a progress bar must not jump backwards"
    assert calls[-1] == 1.0, f"{label}: the last report must be the finished state"


def test_extracting_the_audio_reports_a_monotone_progress_ramp(tmp_path: Path) -> None:
    """The decode is the phase that touches the whole file; it has to be able to drive a bar.

    A clip this small decodes inside ffmpeg's half-second stats period, so the finished tick is the only one it
    emits - what the test holds to is the shape of what does arrive. A game's audio takes tens of seconds and ticks
    all the way up.
    """
    rng = np.random.default_rng(3)
    video = _make_video(tmp_path / "clip.mp4", _speech_like(6.0, rng))
    calls: list[float] = []
    wav = extract_audio(video, tmp_path / "audio.wav", on_progress=lambda fraction: calls.append(float(fraction)))
    _ramp(calls, "extract_audio", at_least=1)
    assert wav.exists() and wav.stat().st_size > 1024


def test_the_background_scan_reports_progress_and_lands_in_the_events(tmp_path: Path) -> None:
    module = _script()
    rng = np.random.default_rng(11)
    audio = _speech_like(9.0, rng)
    blast = _tone(3800.0, 0.5, amplitude=0.2)
    start = int(4.0 * SR)
    audio[start : start + blast.size] += blast
    video = _make_video(tmp_path / "background-scan.mp4", audio)
    library = MatchLibrary(tmp_path / "matches")
    match_id = library.create(video).match_id
    wav = tmp_path / "audio.wav"

    seen: list[dict] = []

    class Recording(module.Status):
        """The real status file, plus a transcript of every update, so the ramp can be checked."""

        def update(self, *, force: bool = False, **changes) -> None:
            seen.append(dict(changes))
            super().update(force=force, **changes)

    status = Recording(library.path(match_id) / module.STATUS_FILE)
    payload = module.scan(
        library, match_id, str(video), strictness=50.0, wav_path=wav, status=status
    )

    assert payload["state"] == "done", payload
    assert payload["found"] >= 1, "the synthetic blast clears the same gates the real ones do"
    assert payload["minutes"] == pytest.approx(9.0 / 60.0, abs=0.01)

    extraction = [float(c["progress"]) for c in seen if c.get("stage") == "extract" and "progress" in c]
    detection = [float(c["progress"]) for c in seen if c.get("stage") == "detect" and "progress" in c]
    assert extraction and extraction[0] == 0.0, "the decode reports from nothing done"
    assert detection, "the transform reports too"
    assert detection == sorted(detection) and detection[-1] == 1.0
    assert detection[0] >= module.EXTRACT_SHARE - 1e-9, "the transform starts where the decode left off"

    # The cache is what a repeat scan reuses; a half-written extraction must never take that name.
    assert wav.exists() and not list(tmp_path.glob("*part*")), "the audio cache is renamed into place, not written in place"
    on_disk = library.load_audio_scan_status(match_id)
    assert on_disk["state"] == "done" and on_disk["progress"] == 1.0

    events = library.events(match_id)
    detected = events.detected()
    assert detected, "the candidates belong to the match once the scan finishes"

    # A second scan with the same settings re-reads the cache and adds nothing: the page says "already recorded".
    again = module.scan(library, match_id, str(video), strictness=50.0, wav_path=wav)
    assert again["state"] == "done" and again["found"] == payload["found"] and again["added"] == 0
    assert len(library.events(match_id).detected()) == len(detected)
