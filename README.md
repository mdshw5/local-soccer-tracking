# Soccer Video Analytics

Analyses single-camera (PTZ gimbal) soccer footage: recovers the camera's own motion, places players on the pitch,
and produces a one-page tactical report, a browsable match archive and three tiers of highlight reels.

The gimbal camera pans, tilts and zooms to follow the ball, so nothing downstream can assume a fixed view. The
pipeline therefore recovers the camera pose first and works in pitch metres afterwards.

## Hardware target

Quadro P2200 (5GB VRAM), 12GB system RAM, 6 CPU cores — the pipeline decodes on the GPU (NVDEC) and streams frames
rather than buffering the file.

## Setup

```bash
export PATH="$HOME/.local/bin:$PATH"   # uv
uv venv --python 3.12 .venv
source .venv/bin/activate
uv pip install -e ".[dev]"
```

Python 3.12 is used deliberately: PyTorch/ultralytics wheels are not available for the system's default Python 3.14.
The Quadro P2200 is sm_61, so install torch from the cu126 index
(`uv pip install torch --index-url https://download.pytorch.org/whl/cu126`).

`ffmpeg` must be present with NVDEC and NVENC support. OpenCV decoding is only ~1x realtime on 4K/60 HEVC, which is
why `ingest/ffmpeg_reader.py` exists.

## Running it

```bash
.venv/bin/streamlit run src/soccer_analytics/dashboard/app.py --server.port 8505
```

The dashboard walks through four steps in order, and every step stores its result so you never repeat work:

1. **Choose footage.** Pick a video (found under `data/videos` and `/srv/storage/home_video/Xbot`, newest first),
   choose a start offset and length, and press *Run analysis* to launch the heavy pass in the background. Progress,
   analysed frames per second and lost frames come from `status.json` in the segment directory. Re-running resumes
   from the last completed chunk.
2. **Register the pitch.** Click pitch landmarks on any frame (the four corners first, then the goal centres, halfway
   line and centre spot). The magnified view you click in and the whole frame sit **side by side**: **click a point on
   the whole frame** to bring it into the middle of the magnified view, and **scroll** (or use +/-) to zoom. The
   yellow box on the whole frame shows what the magnified view covers. Click landmarks on the magnified view
   (**Point** mode) and press **Apply**, because nothing reaches the app until you do. Landmarks clicked on different
   frames are combined, so a corner that is out of view in one frame can be clicked in another.

   **If a corner is not in shot, use the goal centre** (`goal centre left` / `goal centre right`) - the middle of the
   goal line says as much about that end of the pitch as the corner flag does, and the goal mouth is far easier to
   pick out. What you cannot substitute is *spread*: landmark clicks are the only thing tying the video to the pitch,
   and a set that is all far away or all along one line leaves the fit badly undetermined. Measured against a
   simulated match with 4 px of click noise, four distant landmarks were out by more than 50 m; eight spread across
   the frame were within 2.4 m every time. Six is usually enough, four is the bare minimum.

   Because the camera motion is already known, the calibration solves for the camera position, its rotation and a
   focal-length scale. **The app says when a fit cannot be trusted**: if the solver runs a parameter to the edge of
   its search range it reports that rather than presenting the number as a measurement, and it names the clicks that
   disagree with the rest, by landmark and frame. A yellow pitch outline is drawn back onto the frame, with a magenta
   cross on every landmark the fit implies - including the corners you never clicked - so you can see where the
   missing ones have landed. If it sits on the real markings the registration is good, and if it is mirrored a label
   is on the wrong corner. A fit that is no good can be discarded from the same panel and the clicks redone.

   Two things that are easy to assume wrongly, both measured. A **wrong match format does not show up in the fit
   residual** - it moves the recovered camera height instead (2.3, 3.9 and 6.9 m for the same pitch described as 60%,
   100% and 167% of its size), which is what the app checks. And the **remaining risk is the venue, not the maths**:
   with several goals in view it is easy to click a corner belonging to the next pitch along, and the only thing that
   reveals it is the clicks disagreeing with each other.
3. **Build the report.** Project detections to the pitch, track players across frames, split them into two teams from
   their kit colours, and compute distances, speeds, territory and a momentum chart. This step takes seconds, so you
   can re-run it after re-clicking landmarks without touching the video again.
4. **Tag events and cut highlights.** Whistles are detected in the audio track and offered as candidates; goals,
   shots, saves and blocks are tagged by hand. Reels are then cut in three tiers: `clip` (15-30 s), `goals`
   (1-2 min) and `match` (up to 5 min).

   Whistle detection is deliberately strict, and the strictness is adjustable. A referee's whistle is a loud,
   sustained, tonal blast, and that is all three things the detector requires: a narrow band peak that dominates the
   2.2-4.6 kHz band, held for at least 0.2 s, and *loud relative to the rest of the match* - the gate is in
   multiples of the match's own median level in that band, so it does not depend on the recording level. The default
   of 50x came from measurement: on the reference footage the tonal blips that are nothing to do with this match
   (shouts, kicks, whistles from the pitches next door) sit around 17x, while blasts from the pitch being filmed run
   from 100x to over 1000x. At the original settings the detector reported 191 candidates in five minutes; it now
   reports 16 in the same window, each of them a real blast. Blasts within a second of each other are one candidate,
   because a whistle and the shouting that follows it are one thing happening.

   The gate is not pushed higher than that on purpose. The camera is on the sideline, so the same whistle varies by
   roughly 30x depending on whether it was blown five metres away or in the far half - and a far-half whistle is
   very much part of this match. The trade-off therefore leans slightly toward recall, and the level of each
   candidate is recorded in its note so the loud ones can be trusted first. Raise the strictness in the dashboard if
   the neighbouring pitches dominate the list.

## Project layout

```
data/videos/         # local input videos
data/segments/       # per-segment Stage A output (one directory per video + size)
data/matches/        # the archive: match.json, calibration.json, report.json, events.json, highlights/
src/soccer_analytics/
    ingest/          # GPU-accelerated frame and audio I/O (ffmpeg)
    geometry/        # camera motion recovery, pitch calibration
    analysis/        # staged analysis (see below)
    dashboard/       # Streamlit app
tests/               # including a synthetic-match oracle with known ground truth
scripts/             # run_stage_a.py: the background heavy pass
```

## The two-stage split

This is the central design decision, and it is what makes the dashboard usable interactively.

**Stage A** (`analysis/stage_a.py`) is heavy, GPU-bound and run once per segment. It decodes the video, recovers
camera motion frame by frame, detects people and records a compact kit-colour descriptor for each one. It writes
chunk files that let it resume, and a `status.json` the dashboard polls. Measured on the real footage: **2.11x
realtime** (10.55 analysed frames per second), 36 detections per frame, ~0.9 MB per 150 analysed frames.

**Stage B** (`analysis/stage_b.py`) is cheap and re-runnable in seconds. It projects stored detections to the pitch
using the current calibration, tracks players, assigns teams and computes the metrics — no video decoding. Re-clicking
a landmark therefore costs seconds, not minutes: pitch calibration deliberately happens *after* Stage A.

## Camera motion model

`geometry/camera_motion.py` models each step as a **rotation plus focal length**, $H = K_c R K_g^{-1}$, and
integrates orientation in SO(3) rather than chaining free homographies. Free chained homographies random-walk in
their perspective terms and produced a degenerate "674x zoom" within ~70 s of real footage; the rotation model stays
stable across the real pan range of −87° to +29°.

Focal length is self-calibrated (`DEFAULT_FOCAL = 0.82` frame widths, measured from 189 real large-rotation steps) and
the zoom deadband, `0.0006`, is set from the measured per-step noise (σ = 0.00008 on static steps) rather than
guessed. Logo and clock overlays are masked before estimation.

## What the footage supports, and what it does not

Honest limits, because the report is only useful if its numbers can be trusted:

- **No ball detection.** The ball is a few pixels across at typical gimbal zoom and is not reliably detectable, so
  goals, shots, saves and blocks are manual tags. The camera's own aim point is used as a ball *proxy* for
  possession, and the report says so.
- **Possession is a proximity estimate.** Possession is attributed to the team whose player is nearest the camera's
  aim point. Counting detections per team instead lets the referee decide possession (they follow the ball all
  match); measured as a 10-point swing, which is why the aim proxy is used.
- **Partial pitch coverage.** The camera follows play, so only part of the pitch is visible at any moment. Team shape
  and full-team formation cannot be measured, and the report does not claim them.
- **Far-side landmarks are uncertain.** At 40-90 m, four pixels of click noise costs 0.6-5 m of ground error in the
  depth direction, so calibration reports a residual per click and rejects outliers.
- **Distances are per visible run.** A player's distance is summed over the stretches where they were actually
  tracked, not extrapolated across the gaps.

The abandoned two-camera workflow (`video_merge.py`, `roi_mask.py`, the stitching dashboard and the fixed-camera
detector/tracker path) has been removed; the gimbal pipeline above is the only supported path.

## Tests

```bash
.venv/bin/python -m pytest -q
```

`tests/synthetic_match.py` is the oracle: it simulates a gimbal camera aiming at a ball, with known player
trajectories, kit colours, detection noise and misses. The stage tests assert against that ground truth — including a
mutation-tested resume equivalence check for Stage A, and a distance check measured only over *observable* stretches.
