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

Extra footage directories are picked up from `SOCCER_VIDEO_ROOTS` (colon-separated; default
`/srv/storage/home_video/Xbot`), searched after the repo's own `data/videos`. Stage A's detection weights default
to stock `yolov8n.pt` (fetched by Ultralytics on first use); a checkpoint dropped in `data/models/` is preferred
over it, and `scripts/run_stage_a.py --weights` overrides both.

The dashboard walks through four steps in order, and every step stores its result so you never repeat work:

1. **Choose footage.** Pick a video (found under `data/videos` and the `SOCCER_VIDEO_ROOTS` directories, newest
   first),
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

   **Automatic registration is built, and its detector is downloadable.** `geometry/auto_register.py` takes
   pitch-marker detections from any source, screens each frame against its own ground-plane homography (which is
   what catches a detector that fires on a goalpost and calls it the centre spot), and hands the survivors to the
   same robust solver as the clicks. The standard 32-marker template lives in `geometry/pitch_template.py`. The
   keypoint source is the YOLO-pose model from `rustyneuron01/Real-Time-Football-Detection`, hosted at
   `tmoklc/scorevisionv1` and fetched on demand by `geometry/pitch_keypoint_yolo.py` - unlike the reference
   project's HRNet checkpoint, whose LFS object was never pushed, this one actually downloads. Because the camera
   is a fixed tripod, `register_with_position_prior` searches orientations around the known position rather than
   solving a free pose, which is what stops it locking onto a structure that is not the main pitch.

   **On this footage the detector is not yet good enough to trust unattended.** The recording is a small-sided game
   filmed from a 4 m tripod at midfield, and other goals and kickwalls share the frame. The model - trained on
   broadcast views of one full-size pitch - regularly locks onto the neighbouring goal/half of the pitch instead:
   on the reference segment its confident markers clustered on a goal that is not the one being filmed, and a
   position-constrained solve could only reconcile one or two of them with the saved calibration. The model's
   weights and the registration pipeline are here and tested against the synthetic oracle; the missing piece is a
   detector that does not confuse one pitch with the next, which means fine-tuning on this camera's own footage.
   The test server (`scripts/pitch_keypoint_demo.py`) exists to show that evidence directly rather than hide it.
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
scripts/             # run_stage_a.py: the background heavy pass; run_ball_scan.py: the ball scan;
                    # refresh_kit_descriptors.py: re-derive kit colours without re-analysing
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

**Refreshing kit colours without Stage A.** The kit descriptor is a pure function of one decoded frame and one
stored detection box, so improving how grass is masked out of a torso does not require redoing detection or camera
motion. `scripts/refresh_kit_descriptors.py` re-reads the frames the boxes were found in and rewrites only
`det_kit` in place (atomically, resumable, everything else copied through untouched) — minutes instead of the hour a
Stage A re-run costs. Used after the grass-window fix below.

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

- **The ball is tracked by its own scan, not by the report.** The dashboard's ball scan
  (`scripts/run_ball_scan.py`) re-reads a segment at 4K with two detectors and follows the ball frame by frame,
  keeping measurements, short forecasts and "out of the picture" distinct; the replay draws it when a scan has
  run. It is expensive (~70 min for a whole game) and checkpointed, so it can be stopped and resumed. No event is
  inferred from it yet: goals, shots, saves and blocks are manual tags. Where no scan has run, the camera's own
  aim point is used as the ball *proxy* for possession, and the report says so. On the whole 2026-10-03 game the
  scan held the ball on 70% of frames (15,071 detected, 5,380 forecast across a one-frame miss, 742 out of view,
  359 lost); of the frames it *saw* the ball, 61% also project onto the ground, the rest being a ball in the air
  (a ray with no ground intersection is left undrawn rather than guessed at a point on the pitch).
- **Kit colour is grass-aware, and that matters more than it sounds.** A player's torso crop is mostly pitch, so
  the grass is measured per frame and masked out before the colour is taken — otherwise the reported "kit" is the
  pitch. The reference approach (a fixed ±10 hue band around the mean grass) covered only 51-94% of the grass
  pixels on the real game: grass has two hue modes, shaded and sunlit, with the mean in the gap between them.
  Masking the measured 2nd-98th percentile span instead covers 98%+, and halved the green contamination in the
  large, camera-followed torsos. That band is still *frame-wide*, so it has to be wide enough to cover every
  lighting mode in the picture at once; the mask now measures a band per tile of a 4x3 grid (`measure_grass`), so a
  player on shaded turf is masked against shaded turf instead of against a band stretched to also cover the sunlit
  half. It also measures the grass in Lab, so sun-bleached turf that has fallen below the saturation floor - which
  used to leak into every kit colour on a bright afternoon - is still removed. Re-running the descriptors over the
  first 120 frames of the real 17:28 segment (4,261 detections) cut the green hue mass left in the kit by a further
  11.2%, with the usable-crop fraction unchanged at 0.92 (no kit was discarded to get it). A crop that is entirely
  grass is reported at low confidence rather than dropped, because a genuinely green kit and a box holding no kit
  look the same and dropping it sends those players into whichever team is nearer in grey space.
- **…but the reported team colour is still washed out, and the reason is size, not masking.** The descriptors
  themselves are good — a near player's torso crop reads as pure saturated red or blue. What the clustering is
  fed is dominated by *tiny* far-side and touchline figures (median box height 2.2% of the frame width, ~85 px),
  where a torso crop is a handful of pixels and no kit colour is recoverable; those average to grey. Measured over
  the whole game, only a fifth of the usable descriptors are clearly red-ish or blue-ish, and the split is
  strongly time-skewed. So the swatch and suggested team name are a *weak* signal on this footage, and the page
  says the colours are measured rather than claiming them as ground truth.
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

## Fine-tuning the pitch detector

The YOLO pitch-keypoint model (`football-pitch-detection.pt`, from `rustyneuron01/Real-Time-Football-Detection`)
is trained on broadcast views of one full-size pitch. On this camera's footage - a small-sided game from a 4 m
tripod at midfield, with other goals and kickwalls in frame - it locks onto neighbouring structures, and the
registration built on top of it is only as good as the detector. Fine-tuning it on this camera's own games is the
fix. Three scripts run the loop; all three are tested end to end on the reference match.

**Step 0, the one that matters: a good calibration.** The labels are only as accurate as the camera pose they are
projected through, and a handful of clicks on one moment is not accurate enough - the reference match's own clicks
reproject 4-63 px away. The calibration studio (`scripts/pitch_calibration_studio.py`) is what fixes that. It runs
the model on sampled frames, drops its suggestions into the same click editor the dashboard uses, and lets a person
drag each marker onto the real marking. Corrections accumulate across many frames, and the fit solves a
drift-corrected camera path from all of them - much more accurate than the same number of clicks on one frame. It
writes `calibration.json` and `clicks.json` for the match.

```bash
.venv/bin/streamlit run scripts/pitch_calibration_studio.py --server.port 8507
```

Then the loop:

```bash
# 1. Labels, from a match that was calibrated by clicking landmarks. The template is projected through the
#    calibration and only the main pitch's markers are written, so the neighbouring goals are never taught.
python scripts/build_pitch_dataset.py \
    --segment data/segments/game_...__whole_game_541_4851 \
    --calibration data/matches/2026-10-04_17-28-37-430/calibration.json \
    --out data/pitch_keypoints --max-frames 1200

# 2. Fine-tune from the downloaded checkpoint (horizontal flip and mosaic off - see the script's docstring).
python scripts/train_pitch_keypoints.py --data data/pitch_keypoints/data.yaml --epochs 60 --imgsz 1280

# 3. Score it by the thing that matters - a correct camera pose - not keypoint mAP.
python scripts/evaluate_pitch_registration.py \
    --segment data/segments/game_...__whole_game_541_4851 \
    --calibration data/matches/2026-10-04_17-28-37-430/calibration.json \
    --weights runs/pose/pitch_keypoints_finetune/weights/best.pt
```

What the reference run showed, honestly: the dataset builder works (`249` frames, median `10` visible markers
each, correct labels - Ultralytics' own label plot has every keypoint on the main pitch), and even a short 5-epoch
fine-tune visibly moves the model's predictions off the neighbouring goal and onto the main pitch. It is **not yet
reliable enough to register unattended** - the pose loss is still falling and keypoints are imprecise, so the
registration check fails. The reason is label quality, and it was measured: the reference calibration's own clicks
reproject 4-63 px away, recomputing it with drift correction tightens the fit (rms 1.47 -> 0.55 m) but leaves those
click errors, and the model cannot learn finer than its labels (pose mAP stayed 0 while box mAP reached 0.49). The
fix is Step 0 above - correct the markers on many frames in the studio, rebuild, retrain. Training at a higher
resolution (`--imgsz 1280`) and over several games rather than one 30-minute slice will help too.

## Tests

```bash
.venv/bin/python -m pytest -q
```

`tests/synthetic_match.py` is the oracle: it simulates a gimbal camera aiming at a ball, with known player
trajectories, kit colours, detection noise and misses. The stage tests assert against that ground truth — including a
mutation-tested resume equivalence check for Stage A, and a distance check measured only over *observable* stretches.
