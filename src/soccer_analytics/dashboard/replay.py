"""Build the animated pitch replay: the payload the pitch-view component fetches and draws.

Pure maths, no Streamlit. The dashboard saves the result beside the match (``replay.json``) and hands the component
a URL, so the tens of thousands of points are fetched once by the browser instead of being serialised onto the
Streamlit websocket on every rerun.

One entry per player holds their frame indices, pitch positions (metres) and speeds; upstream positions stay in
``PlayerTrack`` order, which is sorted by time - the component interpolates between consecutive observations.
"""

from __future__ import annotations

import numpy as np

from soccer_analytics.analysis.stage_b import PlayerTrack

# A player is "on the ball" when they are the nearest player to the camera's aim point and within this distance.
# Same gate as stage_b's momentum: the aim point is a ball proxy, not a detection.
POSSESSION_RADIUS_M = 12.0
ROUND_M = 0.1


def _round(value: float) -> float:
    return float(round(float(value), 1))


def build_replay(
    pitch: tuple[float, float],
    fps: float,
    frame_count: int,
    aim_xy: np.ndarray,
    players: list[PlayerTrack],
    team_names: list[str],
) -> dict:
    """Assemble the replay payload from tracks and the per-frame camera aim (ball proxy).

    Shirt numbers are deliberately *not* baked in here: they come from the roster and the scanner and are passed to
    the component as a small argument, so editing a roster updates the view without rebuilding the replay. Players
    keep every observation the tracker produced - the client decides what to draw.
    """
    aim = [
        [_round(x), _round(y)] if np.isfinite(x) and np.isfinite(y) else None
        for x, y in np.asarray(aim_xy, dtype=np.float64)
    ]

    # Who was on the ball each frame, by proximity to the camera's aim point (the same proxy stage_b uses).
    by_frame: dict[int, list[tuple[int, float, float]]] = {}
    for track in players:
        for frame, position in zip(track.frame, track.xy):
            by_frame.setdefault(int(frame), []).append((int(track.track_id), float(position[0]), float(position[1])))
    touches: dict[int, int] = {}
    for frame, entries in by_frame.items():
        if frame >= len(aim) or aim[frame] is None:
            continue
        ax, ay = aim[frame]
        best_track, best_distance = None, POSSESSION_RADIUS_M
        for track_id, x, y in entries:
            distance = float(np.hypot(x - ax, y - ay))
            if distance <= best_distance:
                best_track, best_distance = track_id, distance
        if best_track is not None:
            touches[best_track] = touches.get(best_track, 0) + 1

    out_players = []
    for track in sorted(players, key=lambda item: int(item.track_id)):
        speed = np.asarray(track.speed_kmh, dtype=np.float64)
        moving = speed[speed > 0.5]
        out_players.append(
            {
                "track_id": int(track.track_id),
                "team": int(track.team),
                "frames": [int(f) for f in track.frame],
                "xy": [[_round(x), _round(y)] for x, y in track.xy],
                "speed": [_round(s) for s in speed],
                "stats": {
                    "observations": int(len(track.frame)),
                    "time_s": _round(len(track.frame) / max(fps, 1e-6)),
                    "first_t": _round(track.time[0]) if len(track.time) else 0.0,
                    "last_t": _round(track.time[-1]) if len(track.time) else 0.0,
                    "distance_m": _round(track.distance_m),
                    "top_speed_kmh": _round(float(speed.max())) if len(speed) else 0.0,
                    "mean_speed_kmh": _round(float(moving.mean())) if len(moving) else 0.0,
                    "mean_x": _round(float(np.mean(track.xy[:, 0]))),
                    "mean_y": _round(float(np.mean(track.xy[:, 1]))),
                    "touches": int(touches.get(int(track.track_id), 0)),
                },
            }
        )

    return {
        "pitch": [_round(pitch[0]), _round(pitch[1])],
        "fps": float(fps),
        "frame_count": int(frame_count),
        "duration_s": _round(frame_count / max(fps, 1e-6)),
        "team_names": list(team_names),
        "aim": aim,
        "players": out_players,
    }


def player_table_rows(replay: dict, numbers: dict[int, dict] | None = None):  # noqa: ANN201
    """One row per player for the dashboard table; ``numbers`` merges shirt numbers/names in.

    Sorted by time on screen descending: with fragmented real-footage tracks there are hundreds of rows, and the
    players who were actually followed belong at the top.
    """
    import pandas as pd

    numbers = numbers or {}
    rows = []
    for player in replay["players"]:
        stats = player["stats"]
        identity = numbers.get(int(player["track_id"]), {})
        rows.append(
            {
                "track": player["track_id"],
                "team": "referee/other" if player["team"] < 0 else f"Team {player['team'] + 1}",
                "number": identity.get("number"),
                "name": identity.get("name") or "",
                "source": identity.get("source") or "unassigned",
                "seen (s)": stats["time_s"],
                "distance (m)": stats["distance_m"],
                "top speed (km/h)": stats["top_speed_kmh"],
                "mean speed (km/h)": stats["mean_speed_kmh"],
                "touches (proxy)": stats["touches"],
                "mean x (m)": stats["mean_x"],
                "mean y (m)": stats["mean_y"],
            }
        )
    table = pd.DataFrame(rows)
    return table.sort_values("seen (s)", ascending=False).reset_index(drop=True)
