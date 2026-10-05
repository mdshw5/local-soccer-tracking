"""Stage B: turn raw Stage A detections into pitch-space tracks, teams, metrics, events and momentum.

Everything here is cheap and re-runnable (seconds for a match), because it consumes Stage A's stored output. That is
the point of the split: re-drawing pitch landmarks, re-tuning the tracker, or changing the match format never
requires decoding the video again.

Honesty about what is and is not measurable with one following camera:
* positions/teams/distances/territory/momentum are derived from players we can see on the pitch;
* the ball has its own dedicated scan (``analysis.ball``, run from the dashboard); this stage does not consume it yet,
  so goals, shots, saves and blocks remain *manual* tags or candidates for review, never claimed automatically;
  this stage's ball *proxy* is the aim point - the pitch position the camera was pointed at each frame;
* the camera sees part of the pitch at a time, so no full-pitch formation is reported.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import linear_sum_assignment

from soccer_analytics.analysis.kit import kit_rgb
from soccer_analytics.analysis.projection import PitchDetections, on_pitch_mask

# Physics-limited gating: at 5 fps a player covers < ~1 m per frame, so anything beyond this is not the same person.
# The gate cannot be generous: in packed play two players can be a metre apart, and the simulation showed that a 6 m
# gate mixes 33% of tracks. Position alone is not enough there, so appearance (kit colour) is part of the cost.
MAX_STEP_M = 1.4
SIGMA_STEP_FACTOR = 1.5  # extra slack where the position itself is uncertain (far side of the pitch)
MAX_GATE_M = 3.0
MAX_AGE_GATE_FACTOR = 1.5  # a track lost for a while may be re-acquired a little further away, but not far
KIT_COST_WEIGHT = 3.0  # metres of equivalent position error per unit of kit-colour distance
MIN_KIT_FOR_COST = 0.2  # kit fraction below which the colour descriptor is not trusted
TRACK_BUFFER = 15  # frames a track survives without a detection (~3 s at 5 fps)
MIN_TRACK_OBSERVATIONS = 8
MIN_KIT_OBSERVATIONS = 6
SPEED_PERCENTILE = 95
MAX_PLAUSIBLE_SPEED_KMH = 36.0  # faster than any human sprint: a metre of far-side error, not a real speed
MAX_GAP_FOR_DISTANCE_S = 1.0  # movement across a longer unobserved gap is unknown; do not invent it as distance
_UNREACHABLE = 1e6  # cost above the gate: used to forbid an assignment rather than to rank it


@dataclass
class TrackAssignment:
    track_id: np.ndarray  # (D,) -1 where unassigned
    team: dict[int, int]  # track id -> team index (0/1), absent if unknown
    kit_quality: dict[int, float]  # track id -> how separable its kit was (0..1); low means "guess"
    tracks: dict[int, np.ndarray]  # track id -> detection row indices


@dataclass
class PlayerTrack:
    """One tracked person, summarised. Positions are pitch metres with a per-point uncertainty."""

    track_id: int
    team: int
    frame: np.ndarray
    time: np.ndarray
    xy: np.ndarray
    sigma_m: np.ndarray
    speed_kmh: np.ndarray
    distance_m: float


@dataclass
class TeamMetrics:
    team: int
    players_observed: int
    distance_m: float
    top_speed_kmh: float
    mean_speed_kmh: float
    mean_x_fraction: float  # mean pitch x of the team's players, 0 = own goal line, 1 = opponent's
    possession_share: float  # share of frames the team had the nearest player to the action
    # The team's mean kit colour as RGB, or None when the kits could not be separated. It is what makes "team 0"
    # recognisable - it anchors the name the user gives the team to the colour they can see on the pitch.
    kit_rgb: tuple[int, int, int] | None = None


@dataclass
class MatchReport:
    teams: list[TeamMetrics]
    players: list[PlayerTrack]
    momentum: dict  # minute -> {"team_0": float, "team_1": float, "action_x": float}
    pitch_length_m: float
    pitch_width_m: float
    frames_analysed: int
    detections_used: int
    notes: list[str] = field(default_factory=list)


def _kit_cost(
    detections: PitchDetections, here: np.ndarray, ids: list[int], states: dict[int, dict]
) -> np.ndarray:
    """Colour distance between each track's running kit estimate and each candidate detection, in kit units.

    Break-even: a track keeps its own colour (distance ~0) unless another detection is more than
    ``KIT_COST_WEIGHT``-times closer in colour, which is what happens when two players cross.
    """
    colour = detections.kit[here, 1:6]
    valid = detections.kit[here, 0] > MIN_KIT_FOR_COST
    cost = np.zeros((len(ids), len(here)))
    for i, tid in enumerate(ids):
        track_colour = states[tid].get("kit")
        if track_colour is None:
            continue
        distance = np.linalg.norm(colour - track_colour[None, :], axis=1)
        cost[i] = np.where(valid, distance, 0.0)
    return cost


def _track_people(detections: PitchDetections, keep: np.ndarray, on_progress=None) -> TrackAssignment:
    """Greedy-by-cost assignment of detections to tracks with constant-velocity prediction, in pitch metres.

    Association is gated by each detection's own uncertainty, so a far-side player (sigma of metres) is allowed a
    larger jump than a near-side one, and the gate tightens rather than loosens when the geometry is good.
    """
    rows = np.where(keep)[0]
    track_id = np.full(len(detections.xy), -1, dtype=np.int32)
    tracks: dict[int, np.ndarray] = {}
    if len(rows) == 0:
        return TrackAssignment(track_id, {}, {}, tracks)

    frame_of = detections.frame
    states: dict[int, dict] = {}  # id -> last xy, velocity, last frame
    next_id = 0
    unique_frames = np.unique(frame_of[rows])
    progress_every = max(1, len(unique_frames) // 25)
    for index, frame in enumerate(unique_frames):
        if on_progress is not None and (index % progress_every == 0 or index == len(unique_frames) - 1):
            on_progress((index + 1) / len(unique_frames))
        here = rows[frame_of[rows] == frame]
        age_out = [tid for tid, st in states.items() if frame - st["frame"] > TRACK_BUFFER]
        for tid in age_out:
            del states[tid]
        if not states or len(here) == 0:
            for row in here:
                states[next_id] = {
                    "xy": detections.xy[row],
                    "vel": np.zeros(2),
                    "frame": int(frame),
                    "kit": detections.kit[row, 1:6].copy() if detections.kit[row, 0] > MIN_KIT_FOR_COST else None,
                }
                track_id[row] = next_id
                next_id += 1
            continue

        ids = sorted(states)
        predicted = np.array([states[tid]["xy"] + states[tid]["vel"] * (frame - states[tid]["frame"]) for tid in ids])
        measured = detections.xy[here]
        sigma = np.maximum(detections.sigma_m[here], 0.35)
        age = np.array([frame - states[tid]["frame"] for tid in ids])
        position_cost = np.linalg.norm(predicted[:, None, :] - measured[None, :, :], axis=2)
        gate = np.minimum(MAX_STEP_M + SIGMA_STEP_FACTOR * sigma[None, :], MAX_GATE_M) * np.minimum(
            1.0 + 0.1 * age, MAX_AGE_GATE_FACTOR
        )[:, None]
        cost = position_cost + KIT_COST_WEIGHT * _kit_cost(detections, here, ids, states)
        cost = np.where(position_cost <= gate, cost, _UNREACHABLE)
        assigned_rows, assigned_cols = linear_sum_assignment(cost)
        matched: set[int] = set()
        for i, j in zip(assigned_rows, assigned_cols):
            if cost[i, j] >= _UNREACHABLE:
                continue
            tid, row = ids[i], here[j]
            step = (detections.xy[row] - states[tid]["xy"]) / max(1, frame - states[tid]["frame"])
            states[tid]["vel"] = 0.5 * states[tid]["vel"] + 0.5 * step
            states[tid]["xy"] = detections.xy[row]
            states[tid]["frame"] = int(frame)
            if detections.kit[row, 0] > MIN_KIT_FOR_COST:
                # Running kit estimate: slow enough that one bad crop cannot change a track's identity.
                previous = states[tid].get("kit")
                current = detections.kit[row, 1:6]
                states[tid]["kit"] = current.copy() if previous is None else 0.85 * previous + 0.15 * current
            track_id[row] = tid
            matched.add(int(row))
        for row in here:  # unmatched detections start new tracks (a player entering the camera's view)
            if int(row) in matched:
                continue
            states[next_id] = {
                "xy": detections.xy[row],
                "vel": np.zeros(2),
                "frame": int(frame),
                "kit": detections.kit[row, 1:6].copy() if detections.kit[row, 0] > MIN_KIT_FOR_COST else None,
            }
            track_id[row] = next_id
            next_id += 1

    for row in rows:
        if track_id[row] >= 0:
            tracks.setdefault(int(track_id[row]), np.array([], dtype=int))
    for row in rows:
        if track_id[row] >= 0:
            tracks[int(track_id[row])] = np.append(tracks[int(track_id[row])], row)
    return TrackAssignment(track_id, {}, {}, tracks)


# Offline stitching. The online tracker cannot follow a player through a big camera move (the gimbal whips between
# the ends of the pitch), and a player who leaves a tight shot and returns gets a new track. Measured on the real
# sample: 847 raw tracks, 302 of which start within 6 frames and 4 m of where an earlier track ended - the same
# players, redetected. Those fragments are reconnected after the fact.
STITCH_MAX_GAP_FRAMES = 30  # 6 s at 5 fps
STITCH_BASE_M = 2.0  # slack at the join, on top of the distance a sprint could cover
STITCH_SPEED_M_PER_FRAME = 1.4  # a full sprint between two observations (7 m/s at 5 fps)
STITCH_MAX_KIT_DISTANCE = 0.45  # kit colours (L, a, b, saturation, value) further apart than this never stitch
STITCH_NO_KIT_FACTOR = 0.6  # with no colour evidence the spatial window tightens


def _track_kit_colours(detections: PitchDetections, tracks: dict[int, np.ndarray]) -> dict[int, np.ndarray | None]:
    """Median kit colour per track (None when too few usable crops), used only to gate stitching."""
    medians: dict[int, np.ndarray | None] = {}
    for track_id, rows in tracks.items():
        usable = rows[detections.kit[rows, 0] > MIN_KIT_FOR_COST]
        medians[track_id] = np.median(detections.kit[usable, 1:6], axis=0) if len(usable) >= MIN_KIT_OBSERVATIONS else None
    return medians


def _stitch_tracks(detections: PitchDetections, assignment: TrackAssignment) -> TrackAssignment:
    """Reconnect track fragments that the online pass had to break.

    A fragment is a stitch candidate for a later fragment when it starts where the earlier one plausibly could have
    moved to - within a sprint plus slack - and the two wear compatible kit colours (when both have colour
    evidence). Only *mutual best* pairs join, so two nearby candidates cannot both claim the same continuation.
    Chains are resolved afterwards: a player who left and returned twice becomes one track.
    """
    tracks = {track_id: np.sort(rows) for track_id, rows in assignment.tracks.items()}
    if len(tracks) < 2:
        return assignment
    last = {track_id: (int(detections.frame[rows[-1]]), detections.xy[rows[-1]]) for track_id, rows in tracks.items()}
    first = {track_id: (int(detections.frame[rows[0]]), detections.xy[rows[0]]) for track_id, rows in tracks.items()}
    kits = _track_kit_colours(detections, tracks)

    cost: dict[tuple[int, int], float] = {}
    for before, (end_frame, end_xy) in last.items():
        for after, (start_frame, start_xy) in first.items():
            if before == after:
                continue
            gap = start_frame - end_frame
            if gap <= 0 or gap > STITCH_MAX_GAP_FRAMES:
                continue
            distance = float(np.linalg.norm(start_xy - end_xy))
            kit_before, kit_after = kits[before], kits[after]
            if kit_before is not None and kit_after is not None:
                kit_distance = float(np.linalg.norm(kit_before - kit_after))
                if kit_distance > STITCH_MAX_KIT_DISTANCE:
                    continue
                if distance > STITCH_BASE_M + STITCH_SPEED_M_PER_FRAME * gap:
                    continue
                pair_cost = distance / gap + 2.0 * kit_distance
            else:
                # No colour evidence on one side: only short, small jumps are believable.
                if distance > STITCH_BASE_M + STITCH_NO_KIT_FACTOR * STITCH_SPEED_M_PER_FRAME * gap:
                    continue
                pair_cost = distance / gap + 0.5
            cost[(before, after)] = pair_cost

    best_successor: dict[int, int] = {}
    best_predecessor: dict[int, int] = {}
    for (before, after), value in cost.items():
        if before not in best_successor or value < cost[(before, best_successor[before])]:
            best_successor[before] = after
        if after not in best_predecessor or value < cost[(best_predecessor[after], after)]:
            best_predecessor[after] = before

    successor: dict[int, int] = {}
    for before, after in best_successor.items():
        if best_predecessor.get(after) == before:
            successor[before] = after

    # Walk the chains from their heads (members with no predecessor) so a player who left and returned twice
    # becomes a single track under its first id.
    has_predecessor = set(successor.values())
    chains: dict[int, list[int]] = {}
    for head in tracks:
        if head in has_predecessor:
            continue
        members, cursor = [head], head
        while cursor in successor:
            cursor = successor[cursor]
            if cursor in members:  # cannot happen with strictly increasing frames; cheap insurance
                break
            members.append(cursor)
        chains[head] = members

    track_id = assignment.track_id.copy()
    merged_tracks: dict[int, np.ndarray] = {}
    for head, members in chains.items():
        merged_tracks[head] = np.concatenate([tracks[member] for member in members]) if len(members) > 1 else tracks[head]
        for member in members[1:]:
            track_id[assignment.track_id == member] = head
    return TrackAssignment(track_id, assignment.team, assignment.kit_quality, merged_tracks)


def _team_assignment(
    detections: PitchDetections, assignment: TrackAssignment, *, num_teams: int = 2
) -> tuple[dict[int, int], dict[int, float], dict[int, np.ndarray]]:
    """Cluster per-track kit descriptors into teams, plus a possible "other" cluster for the referee.

    Returns the team label per track (``-1`` for a cluster identified as neither team), how separable the basis was,
    and each team's mean kit descriptor - the colour that tells the two teams apart, read back out for display.
    Two things make this honest rather than confident:
    * a third cluster is only believed when there are enough tracks, and it is only treated as "other" when it is
      clearly smaller than the two team clusters - that is the referee or a stray track, who would otherwise be
      counted as a team member and would inflate that team's numbers (a referee follows the ball everywhere);
    * separation is a between/within ratio, so identical kits cannot score as a confident split.
    """
    from sklearn.cluster import KMeans

    colours = detections.kit[:, 1:6]  # L, a, b, saturation, value: the part that identifies a kit
    weight = detections.kit[:, 0]  # fraction of the torso that was kit rather than grass
    per_track: dict[int, np.ndarray] = {}
    for tid, rows in assignment.tracks.items():
        usable = rows[(weight[rows] > MIN_KIT_FOR_COST)]
        if len(usable) >= MIN_KIT_OBSERVATIONS:
            per_track[tid] = np.median(colours[usable], axis=0)
    if len(per_track) < num_teams * 2:
        return {}, {tid: 0.0 for tid in assignment.tracks}, {}

    ids = sorted(per_track)
    features = np.stack([per_track[tid] for tid in ids])
    features = (features - features.mean(0)) / (features.std(0) + 1e-6)
    clusters = 3 if len(ids) >= 3 * num_teams else num_teams
    kmeans = KMeans(n_clusters=clusters, n_init=10, random_state=0).fit(features)
    centres, labels = kmeans.cluster_centers_, kmeans.labels_

    sizes = np.bincount(labels, minlength=clusters)
    team_clusters = list(np.argsort(-sizes)[:num_teams])
    other_cluster = None
    if clusters == 3:
        smallest = int(np.argmin(sizes))
        reference = float(np.median(sizes[team_clusters])) or 1.0
        if sizes[smallest] < 0.4 * reference:
            other_cluster = smallest
            team_clusters = [c for c in range(clusters) if c != other_cluster]

    # Deterministic team ids across runs: the more red kit (standardised 'a' axis) is team 0.
    team_clusters = sorted(team_clusters[:num_teams], key=lambda c: -centres[c, 1])
    remap = {int(cluster): team for team, cluster in enumerate(team_clusters)}
    teams = {tid: remap.get(int(label), -1) for tid, label in zip(ids, labels)}

    # Each team's colour, averaged over the kit descriptors of the tracks that wear it (not the cluster centre,
    # which lives in standardised space and no longer means a colour).
    members: dict[int, list[np.ndarray]] = {}
    for tid, team in teams.items():
        if team >= 0:
            members.setdefault(team, []).append(per_track[tid])
    colours = {team: np.mean(np.stack(rows), axis=0) for team, rows in members.items()}

    within = float(np.mean(np.linalg.norm(features - centres[labels], axis=1)))
    between = float(np.linalg.norm(centres[team_clusters[0]] - centres[team_clusters[1]])) if len(team_clusters) >= 2 else 0.0
    separation = float(np.clip((between - within) / (between + within + 1e-9), 0.0, 1.0))
    quality: dict[int, float] = {tid: separation for tid in ids}
    for tid in assignment.tracks:
        quality.setdefault(tid, 0.0)
    return teams, quality, colours


def _speeds(xy: np.ndarray, time: np.ndarray, sigma_m: np.ndarray) -> np.ndarray:
    """Per-observation speed in km/h, using each step's own displacement uncertainty and a physical cap.

    A step across a gap longer than ``MAX_GAP_FOR_DISTANCE_S`` is not reported as movement at all: the player was
    not seen in between, so whatever they did is unknown - counting the straight-line jump as distance would invent
    a sprint. The same rule governs the distance sums in ``build_tracks``.
    """
    if len(time) < 2:
        return np.zeros(len(time))
    step = np.linalg.norm(np.diff(xy, axis=0), axis=1)
    gap = np.diff(time)
    dt = np.clip(gap, 1e-3, None)
    observed = gap <= MAX_GAP_FOR_DISTANCE_S
    noise = np.hypot(sigma_m[:-1], sigma_m[1:])
    speed = (step / dt) * 3.6
    # Only report a speed when the displacement is meaningfully larger than the measurement noise of the two points.
    speed = np.where(observed & (step > np.maximum(0.35, noise)), speed, 0.0)
    return np.concatenate([[0.0], np.clip(speed, 0.0, MAX_PLAUSIBLE_SPEED_KMH)])


def build_tracks(
    detections: PitchDetections,
    keep: np.ndarray,
    assignment: TrackAssignment | None = None,
    teams: dict[int, int] | None = None,
) -> list[PlayerTrack]:
    """Summarise assigned detections into per-player tracks (distance, speed profile)."""
    if assignment is None:
        assignment = _stitch_tracks(detections, _track_people(detections, keep))
    teams = teams if teams is not None else _team_assignment(detections, assignment)[0]
    out: list[PlayerTrack] = []
    for tid, rows in assignment.tracks.items():
        rows = np.sort(rows)
        if len(rows) < MIN_TRACK_OBSERVATIONS:
            continue
        order = np.argsort(detections.time[rows])
        rows = rows[order]
        xy, time, sigma = detections.xy[rows], detections.time[rows], detections.sigma_m[rows]
        speed = _speeds(xy, time, sigma)
        steps = np.linalg.norm(np.diff(xy, axis=0), axis=1)
        observed = np.diff(time) <= MAX_GAP_FOR_DISTANCE_S
        distance = float(np.sum(steps * observed * (speed[1:] > 0)))
        out.append(PlayerTrack(tid, teams.get(tid, -1), detections.frame[rows], time, xy, sigma, speed, distance))
    return out


def _momentum(
    detections: PitchDetections,
    keep: np.ndarray,
    teams_by_track: dict[int, int],
    track_id: np.ndarray,
    bucket_s: float = 60.0,
    possession_radius_m: float = 12.0,
    on_progress=None,
) -> dict:
    """Per-minute action share per team, and the mean x of the action (territory).

    Possession is decided by *proximity to the ball proxy*, not by counting players: the gimbal aims at the ball, so
    the pitch point it is aimed at each frame is the best available ball position. Whoever has the nearest player to
    that point is treated as being on the ball, and the team share is the fraction of frames won.

    Counting detections instead would let the referee decide the result: a referee follows the ball all match long, so
    they are the single nearest person very often, and exposure to them depends on how the kit clustering fell - this
    was measurable in the simulator and produced a 10-point swing.
    """
    frames = np.unique(detections.frame[keep])
    if len(frames) == 0:
        return {}
    buckets: dict[int, dict] = {}
    progress_every = max(1, len(frames) // 25)
    for index, frame in enumerate(frames):
        if on_progress is not None and (index % progress_every == 0 or index == len(frames) - 1):
            on_progress((index + 1) / len(frames))
        aim = detections.aim_xy[frame] if frame < len(detections.aim_xy) else np.full(2, np.nan)
        if not np.all(np.isfinite(aim)):
            continue
        rows = np.where(keep & (detections.frame == frame))[0]
        teams = np.array([teams_by_track.get(int(track_id[row]), -1) for row in rows])
        players = teams >= 0
        minute = int(detections.time[rows[0]] // bucket_s)
        entry = buckets.setdefault(minute, {"won": [0, 0], "x": [], "uncontested": 0})
        entry["x"].append(float(aim[0]))
        if not players.any():
            entry["uncontested"] += 1
            continue
        distance = np.linalg.norm(detections.xy[rows] - aim[None, :], axis=1)
        distance = np.where(players, distance, np.inf)
        nearest = int(np.argmin(distance))
        if distance[nearest] <= possession_radius_m:
            entry["won"][int(teams[nearest])] += 1
        else:
            entry["uncontested"] += 1
    out = {}
    for minute, entry in buckets.items():
        total = entry["won"][0] + entry["won"][1]
        if total == 0:
            continue
        share = entry["won"][0] / total
        out[minute] = {
            "team_0": round(share, 3),
            "team_1": round(1.0 - share, 3),
            "action_x": round(float(np.mean(entry["x"])), 1),
            "contested_frames": int(total),
            "uncontested_frames": int(entry["uncontested"]),
        }
    return dict(sorted(out.items()))


def build_report(
    detections: PitchDetections,
    *,
    pitch_length_m: float,
    pitch_width_m: float,
    match_frames: int | None = None,
    on_progress=None,
) -> tuple[MatchReport, TrackAssignment]:
    """Full Stage B pipeline: mask -> track -> stitch fragments -> team -> metrics -> momentum.

    ``on_progress(fraction)`` covers the whole pipeline (tracking is the long part; teams and metrics are quick).
    """
    def report(fraction: float) -> None:
        if on_progress is not None:
            on_progress(min(1.0, max(0.0, fraction)))

    keep = on_pitch_mask(detections, pitch_length_m, pitch_width_m)
    assignment = _stitch_tracks(detections, _track_people(detections, keep, on_progress=lambda f: report(0.55 * f)))
    report(0.6)
    teams, quality, kit_colours = _team_assignment(detections, assignment)
    assignment.team = teams
    assignment.kit_quality = quality
    players = build_tracks(detections, keep, assignment, teams)
    report(0.7)

    momentum = _momentum(detections, keep, teams, assignment.track_id, on_progress=lambda f: report(0.7 + 0.25 * f))
    summary: list[TeamMetrics] = []
    shares = [momentum[m]["team_0"] for m in momentum] if momentum else []
    mean_share = float(np.mean(shares)) if shares else 0.0
    for team in (0, 1):
        members = [p for p in players if p.team == team]
        if members:
            distance = float(sum(p.distance_m for p in members))
            speeds = np.concatenate([p.speed_kmh for p in members])
            moving = speeds[speeds > 0.5]
            # Mean pitch x per team, normalised by pitch length: a direct "where does this team play" measure.
            mean_x = float(np.mean(np.concatenate([p.xy[:, 0] for p in members])) / pitch_length_m)
        else:
            distance = mean_x = 0.0
            moving = np.zeros(1)
        possession = mean_share if team == 0 else (1.0 - mean_share if shares else 0.0)
        summary.append(
            TeamMetrics(
                team=team,
                players_observed=len(members),
                distance_m=round(distance, 1),
                top_speed_kmh=round(float(moving.max()) if len(moving) else 0.0, 1),
                mean_speed_kmh=round(float(moving.mean()) if len(moving) else 0.0, 1),
                mean_x_fraction=round(mean_x, 3),
                possession_share=round(possession, 3),
                kit_rgb=kit_rgb(kit_colours.get(team)),
            )
        )
    notes = []
    if not any(p.team >= 0 for p in players):
        notes.append("Kit colours were not separable: team metrics are unavailable for this match.")
    weak = [tid for tid, q in quality.items() if q < 0.35]
    if weak:
        notes.append(f"{len(weak)} track(s) had weakly separated kit colours; their team label is a best guess.")
    notes.append(
        "Goals, shots, saves and blocks are manual tags: the ball scan can follow the ball, but no event is inferred "
        "from it yet."
    )
    notes.append("Only part of the pitch is in view at once, so no full-pitch formation is reported.")
    report(1.0)
    return (
        MatchReport(
            teams=summary,
            players=players,
            momentum=momentum,
            pitch_length_m=pitch_length_m,
            pitch_width_m=pitch_width_m,
            frames_analysed=int(match_frames or (int(detections.frame.max()) + 1 if len(detections.frame) else 0)),
            detections_used=int(keep.sum()),
            notes=notes,
        ),
        assignment,
    )
