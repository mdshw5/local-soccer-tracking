"""Small helpers that turn what is on disk into what the page shows."""

from __future__ import annotations

from soccer_analytics.analysis.library import MatchLibrary


def report_from_library(library: MatchLibrary, match_id: str | None) -> dict | None:
    """Load a saved report, restoring the momentum keys to integers.

    JSON turns the minute keys into strings, and strings sort lexicographically ("10" before "2"), which would both
    reorder the momentum chart and misplace the momentum-based highlight moments.
    """
    if match_id is None:
        return None
    payload = library.load_report(match_id)
    if payload is None:
        return None
    momentum = {}
    for key, value in (payload.get("momentum") or {}).items():
        try:
            momentum[int(key)] = value
        except (TypeError, ValueError):
            continue
    payload["momentum"] = momentum
    return payload
