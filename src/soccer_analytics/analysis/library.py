"""Match archive: one directory per match holding everything produced for it.

Layout::

    data/matches/<match_id>/
        match.json            this index (sources, segments, calibration, counts)
        events.json           manual tags + audio candidates
        report.json           Stage B metrics
        calibration.json      pitch calibration
        highlights/*.mp4      reels, each with a .json manifest

Keeping artefacts together (rather than one flat output directory) is what makes the archive browsable later, which
is the point of keeping a season's worth of matches.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

from soccer_analytics.analysis.events import EventLog
from soccer_analytics.geometry.pitch_calibration import PitchCalibration

MATCHES_ROOT = Path("data/matches")


@dataclass
class MatchRecord:
    match_id: str
    sources: list[str] = field(default_factory=list)
    segments: list[str] = field(default_factory=list)
    pitch_length_m: float = 100.0
    pitch_width_m: float = 64.0
    format: str = "11v11"
    created: float = field(default_factory=time.time)
    team_names: list[str] = field(default_factory=lambda: ["Team A", "Team B"])
    notes: list[str] = field(default_factory=list)

    def directory(self, root: str | Path = MATCHES_ROOT) -> Path:
        return Path(root) / self.match_id


def _atomic_write(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2))
    tmp.replace(path)


def new_match_id(source: str | Path) -> str:
    """Date-stamped id from the source file, so matches sort chronologically and stay recognisable."""
    stem = Path(source).stem
    safe = "".join(c if c.isalnum() or c in "-_" else "-" for c in stem)[:60]
    return f"{time.strftime('%Y-%m-%d')}_{safe}"


class MatchLibrary:
    def __init__(self, root: str | Path = MATCHES_ROOT):
        self.root = Path(root)

    def create(self, source: str | Path, **kwargs) -> MatchRecord:
        """The match for this footage, creating it only if it is not already archived.

        The id is derived from the video's name and the date, so creating twice from the same footage lands on the
        same id. Rewriting the record there would throw away the format, the team names and the segment list already
        saved against it - and the page reads that record back on every run, so the loss would be silent.
        """
        match_id = new_match_id(source)
        if (self.path(match_id) / "match.json").exists():
            return self.load(match_id)
        record = MatchRecord(match_id=match_id, sources=[str(source)], **kwargs)
        self.save(record)
        return record

    def path(self, match_id: str) -> Path:
        return self.root / match_id

    def save(self, record: MatchRecord) -> Path:
        path = self.path(record.match_id) / "match.json"
        _atomic_write(path, asdict(record))
        return path

    def load(self, match_id: str) -> MatchRecord:
        data = json.loads((self.path(match_id) / "match.json").read_text())
        return MatchRecord(**data)

    def add_segment(self, match_id: str, segment_dir: str | Path) -> MatchRecord:
        """Remember that a segment directory belongs to this match, so the archive says what has been produced."""
        record = self.load(match_id)
        entry = str(segment_dir)
        if entry not in record.segments:
            record.segments.append(entry)
            self.save(record)
        return record

    def list_ids(self) -> list[str]:
        if not self.root.exists():
            return []
        return sorted(p.name for p in self.root.iterdir() if (p / "match.json").exists())

    def summaries(self) -> list[dict]:
        """One row per match for the dashboard's library list, with what has been produced so far."""
        rows = []
        for match_id in self.list_ids():
            directory = self.path(match_id)
            record = self.load(match_id)
            rows.append(
                {
                    "match_id": match_id,
                    "sources": record.sources,
                    "format": record.format,
                    "segments": len(record.segments),
                    "has_calibration": (directory / "calibration.json").exists(),
                    "has_report": (directory / "report.json").exists(),
                    "events": len(EventLog.load(directory / "events.json").events),
                    "highlights": len(list((directory / "highlights").glob("*.mp4"))) if (directory / "highlights").exists() else 0,
                    "created": record.created,
                }
            )
        return rows

    # --- artefacts -------------------------------------------------------------------------------------------
    def save_calibration(self, match_id: str, calibration: PitchCalibration) -> Path:
        path = self.path(match_id) / "calibration.json"
        _atomic_write(path, calibration.to_json())
        return path

    def load_calibration(self, match_id: str) -> PitchCalibration | None:
        path = self.path(match_id) / "calibration.json"
        if not path.exists():
            return None
        return PitchCalibration.from_json(json.loads(path.read_text()))

    def clear_calibration(self, match_id: str) -> bool:
        """Forget the saved calibration, for when the fit was no good. Returns whether there was one to remove."""
        path = self.path(match_id) / "calibration.json"
        if not path.exists():
            return False
        path.unlink()
        return True

    def save_clicks(self, match_id: str, clicks: list[dict], pitch: tuple[float, float] | None = None) -> Path:
        """Keep the landmark clicks next to the calibration they produced.

        Without this a bad fit cannot be looked at afterwards: the residual table in the page can only name a click
        by its landmark while the browser session that made it is still alive.
        """
        path = self.path(match_id) / "clicks.json"
        _atomic_write(path, {"clicks": clicks, "pitch": list(pitch) if pitch else None})
        return path

    def load_clicks(self, match_id: str) -> list[dict]:
        path = self.path(match_id) / "clicks.json"
        if not path.exists():
            return []
        return json.loads(path.read_text()).get("clicks", [])

    def save_report(self, match_id: str, payload: dict) -> Path:
        path = self.path(match_id) / "report.json"
        _atomic_write(path, payload)
        return path

    def load_report(self, match_id: str) -> dict | None:
        path = self.path(match_id) / "report.json"
        return json.loads(path.read_text()) if path.exists() else None

    def save_replay(self, match_id: str, payload: dict) -> Path:
        """Per-frame track data for the animated pitch view (fetched by the browser as a media file)."""
        path = self.path(match_id) / "replay.json"
        _atomic_write(path, payload)
        return path

    def load_replay(self, match_id: str) -> dict | None:
        path = self.path(match_id) / "replay.json"
        return json.loads(path.read_text()) if path.exists() else None

    def save_jerseys(self, match_id: str, payload: dict) -> Path:
        """The background scanner's raw readings and per-track suggestions."""
        path = self.path(match_id) / "jerseys.json"
        _atomic_write(path, payload)
        return path

    def load_jerseys(self, match_id: str) -> dict:
        path = self.path(match_id) / "jerseys.json"
        return json.loads(path.read_text()) if path.exists() else {}

    def save_roster(self, match_id: str, roster: dict[int, dict]) -> Path:
        """Manual identity per track: the user's own reads of shirt numbers and names, which win over the scanner."""
        path = self.path(match_id) / "roster.json"
        _atomic_write(path, {"players": {str(track): entry for track, entry in roster.items()}})
        return path

    def load_roster(self, match_id: str) -> dict[int, dict]:
        path = self.path(match_id) / "roster.json"
        if not path.exists():
            return {}
        players = json.loads(path.read_text()).get("players", {})
        return {int(track): entry for track, entry in players.items()}

    def load_jerseys_status(self, match_id: str) -> dict:
        """Progress of the background shirt-number scan: {} until it has ever run."""
        path = self.path(match_id) / "jerseys_status.json"
        return json.loads(path.read_text()) if path.exists() else {}

    def load_audio_scan_status(self, match_id: str) -> dict:
        """Progress of the background whistle scan: {} until it has ever run."""
        path = self.path(match_id) / "audio_scan.json"
        return json.loads(path.read_text()) if path.exists() else {}

    def events(self, match_id: str) -> EventLog:
        return EventLog.load(self.path(match_id) / "events.json")

    def save_events(self, match_id: str, log: EventLog) -> Path:
        path = self.path(match_id) / "events.json"
        log.save(path)
        return path

    def highlights_dir(self, match_id: str) -> Path:
        path = self.path(match_id) / "highlights"
        path.mkdir(parents=True, exist_ok=True)
        return path

    def artifacts(self, match_id: str) -> list[dict]:
        """Everything on disk for a match, with sizes, for a 'what did we produce?' view."""
        directory = self.path(match_id)
        out = []
        for path in sorted(directory.rglob("*")):
            if path.is_file() and path.suffix != ".tmp":
                out.append({"path": str(path.relative_to(directory)), "size_mb": round(path.stat().st_size / 1e6, 2)})
        return out
