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

`easyocr` (the shirt-number scan's OCR) is a declared dependency, so `uv pip install -e ".[dev]"` brings it in. It
pulls its own torch/torchvision, so on a CUDA box install it after the cu126 torch above or it will drag in the CPU
wheel. The scan is the only thing that imports it, and it does so lazily - a missing install reports
`easyocr is not installed in this environment` in the scan's status rather than failing the page.

## Running it

```bash
.venv/bin/streamlit run src/soccer_analytics/dashboard/app.py --server.port 8505
```

Extra footage directories are picked up from `SOCCER_VIDEO_ROOTS` (colon-separated; default
`/srv/storage/home_video/Xbot`), searched after the repo's own `data/videos`. Stage A's detection weights default
to stock `yolov8n.pt` (fetched by Ultralytics on first use); a checkpoint dropped in `data/models/` is preferred
over it, and `scripts/run_stage_a.py --weights` overrides both.

The dashboard walks through four steps in order, and every step stores its result so you never repeat work:

0. **Where things are saved.** Every analysis is *self-contained*: it lives in an `analysis/` folder beside the
   footage it was made from, named by the recording date and the video
   (`<footage>/analysis/2026-10-03_game_16-28-37-784/`). The match record, the pitch clicks, the report, the
   replay, the events, the reels, the Stage A segment directories and the game manifest are all inside it, so a
   match directory can be copied or moved as a unit - recorded paths are stored relative to the folder and
   re-resolved on load. Match recordings are expected to sit one per directory (one directory per match, for as
   many teams as you like); the picker browses them by folder and a video that already has an analysis opens it
   when it is picked. `scripts/migrate_analysis.py` moves archives from the old repository-era `data/matches`,
   `data/segments` and `data/games` roots into the same shape.

1. **Choose footage.** Pick a folder (found under `data/videos` and the `SOCCER_VIDEO_ROOTS` directories, newest
   first),
   choose a start offset and length, and press *Run analysis* to launch the heavy pass in the background. Progress,
   analysed frames per second and lost frames come from `status.json` in the segment directory. Re-running resumes
   from the last completed chunk.
2. **Register the pitch.** Click pitch landmarks on any frame (the four corners first, then the goalposts, halfway
   line and centre spot). The magnified view you click in and the whole frame sit **side by side**: **click a point on
   the whole frame** to bring it into the middle of the magnified view, and **scroll** (or use +/-) to zoom. The
   yellow box on the whole frame shows what the magnified view covers. Click a landmark on the magnified view and
   pick which one it is in the list that appears at the click - the click is committed the moment you pick, and the
   calibration refits itself automatically from the clicks so far. Landmarks clicked on different frames are
   combined, so a corner that is out of view in one frame can be clicked in another.

   **If a corner is not in shot, use the goalposts** (`goal post left-near` / `goal post left-far`, and the `right`
   pair) - the base of a post is a hard, high-contrast edge that can be clicked to a pixel, and the two posts of a
   goal are 7.32 m apart, so together they say as much about that end of the pitch as the corner flag does. The
   **penalty spots and the four cardinals of the centre circle** are usually visible too, and they are the same
   standard markings for every format: they sit at a range of distances from the camera, which is exactly the
   spread the fit is short of when the near corners are out of shot. The boxes are deliberately not clickable -
   the six-yard box is small and lost against the netting, and an eighteen-yard corner is a bare junction of two
   lines with nothing to focus on. What you cannot substitute is *spread*: landmark clicks are the only thing tying the video to the pitch, and a set that is all far
   away or all along one line leaves the fit badly undetermined. Measured against a simulated match with 4 px of
   click noise, four distant landmarks were out by more than 50 m; eight spread across the frame were within 2.4 m
   every time. Six is usually enough, four is the bare minimum.

   Because the camera motion is already known, the calibration solves for the camera position, its rotation and a
   focal-length scale. **The app says when a fit cannot be trusted**: if the solver runs a parameter to the edge of
   its search range it reports that rather than presenting the number as a measurement, and it names the clicks that
   disagree with the rest, by landmark and frame. The whole set of pitch markings is drawn back onto the frame, with
   a magenta cross on every landmark the fit implies - including the corners you never clicked - so you can see where
   the missing ones have landed. If the markings sit on the real ones the registration is good, and if they are
   mirrored a label is on the wrong corner. A fit that is no good can be discarded from the same panel and the
   clicks redone.

   Two things that are easy to assume wrongly, both measured. A **wrong match format does not show up in the fit
   residual** - it moves the recovered camera height instead (2.3, 3.9 and 6.9 m for the same pitch described as 60%,
   100% and 167% of its size), which is what the app checks. And the **remaining risk is the venue, not the maths**:
   with several goals in view it is easy to click a corner belonging to the next pitch along, and the only thing that
   reveals it is the clicks disagreeing with each other.
3. **Build the report.** Project detections to the pitch, track players across frames, split them into two teams from
   their kit colours, and compute distances, speeds, territory and a momentum chart. This step takes seconds, so you
   can re-run it after re-clicking landmarks without touching the video again. **Build report + run all detections**
   does the same and then everything else in the background: the ball scan, the whistle scan and the shirt-number
   scan, the event detectors over what they find, and a final report + replay rebuild - one press for a full report.
   Every stage skips or resumes finished work, so a long ball scan can be left running.
4. **Watch, tag and cut highlights.** Tag buttons stack under the pitch animation, filling the space its column
   leaves (Goal, Shot, Save, Tackle, Foul, Corner, Penalty, Block, Clearance, Substitution, Other, with a team
   picker and an optional note): a press
   stamps the event on the playback's own second, so tagging happens while the moment is on screen. Whistles are
   detected in the audio track and offered as candidates; goals, shots, corners, penalties, clearances and tackles
   are inferred from the ball scan and the player tracks (*Detect events*, or the one-press build above); saves and
   blocks are tagged by hand. The review queue, the scan controls and the reels sit directly under the playback,
   so watching, tagging, reviewing and exporting never leave the game. Reels are cut in three tiers: `clip`
   (15-30 s), `goals` (1-2 min) and `match` (up to 5 min).

   The event detectors are conservative and every one of them says in its note what it measured, because a wrong
   event on the timeline is worse than a missing one. A **goal** needs the ball to reach a goal mouth moving in
   *and* to be reset to the centre spot within a minute - the reset is what separates a goal from a shot into the
   side netting, and it has to be a *measured* position, not the tracker's forecast across a missed frame (the scan
   loses the ball against the net, so the crossing itself may be a forecast; only a sighting can confirm a restart).
   A **shot** is a hard kick aimed at a goal that travels toward it and did not produce a reset; a save and a shot
   wide look the same to a ball track, so both are reported as a shot and the note says where it was aimed. A
   **corner** is a ball at rest near a corner flag that is then kicked (or a fresh detection entering from a
   corner). A **penalty** is a whistle, then the ball still on the penalty spot, then a hard kick - without the
   whistle the same geometry is left to the shot detector. A **clearance** is a hard, long kick away from the goal
   the team is defending, from its own defensive third; which goal that is comes from the per-half orientation
   below. A **tackle** is a player who was moving coming to a near stop right beside the ball as the ball's own
   velocity changes - a *motion* proxy, not pose, and the note says so.

   Measuring the ball's motion is the hard part, and the numbers come from the real whole-game scan. A per-frame
   difference of the projected positions reads 57 m/s at p90 and 96,000 m/s at worst: the projection turns a pixel
   of jitter into metres when the ball is far away or near the horizon, and one bad frame then looks like a 200 m/s
   kick. So positions more than 3 m off the pitch are dropped (18% of the scan's finite positions), a position far
   from its neighbours' median is dropped as a spike, and the speed is the net displacement over a short *look-back*
   window paired with a straightness ratio - jitter is fast but not straight, a struck ball is both. That brings the
   real scan to a p50 of 2.4 m/s and a p90 of 15 m/s, and the physically impossible readings are gone (nothing on
   this pitch travels faster than ~45 m/s). On the reference game the detectors report 8 events in 72 minutes: five
   challenges, a clearance, a goal and a shot.

   Which end each team defends is read per half from where its players spend the half (a team defends the goal its
   players are nearer to), which is also the team's attack vector. It is what tells a clearance from a shot: the
   same fast kick is one or the other depending on which goal it is heading away from. The detected events are
   review candidates exactly like the whistle scan's - the same verdict buttons and the same *discard detected*
   button apply - and each one names the player it is attributed to (track id and shirt number) when a player was
   near enough to the ball to be named.

   The animated pitch replay carries a **timeline strip** above it: the momentum curve (each team's share of
   contested frames, drawn above and below the centre line) with every tagged and detected event marked on it -
   filled dots for manual tags, hollow rings for detected candidates, one colour per event type. Clicking the strip
   seeks the replay to that moment, so the timeline doubles as the animation's scrubber; hovering a marker names
   the event (type, time, team, tagged or detected, note) in a popover, and the arrows beside play step to the
   previous/next event.

   Beside the animation sits the **annotated footage** of the match itself (the stream described further down), kept
   in step with it: a still of the exact second while the animation is paused - so scrubbing the animation scrubs
   the footage - and a live stream opened at that second and speed while it plays (H.264 with sound where the
   browser can play it; the MJPEG stream, without sound, in Safari and other WebKit browsers, whose media stack
   will not play an endless fragmented MP4). The footage holds the larger half of the row, and a play press waits
   for the stream's first frame before the animation runs - the encoder's start-up is not baked in as a lag - with
   a cap for a stream that never starts. Above it, **Footage moment to
   jump to** lists the same tagged and detected events as the strip; picking one and pressing *Jump to this moment*
   moves the animation there and starts both together, so the events are how a review picks its start times. The
   pane needs the stream server (`scripts/run_match_stream.py`); the page says so - and offers to start it - when
   it is not running.

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

## The annotated match stream

The footage pane beside the pitch is served by a small stream server of its own - MJPEG, plus the same frames
encoded as H.264 for the pane's `<video>` - which can also be opened directly:

```bash
.venv/bin/python scripts/run_match_stream.py --port 8510
```

`http://localhost:8510/` is an index of streamable matches; `/play/<match_id>` plays one (a page that embeds the
stream in an `<img>` - browsers download a bare multipart URL instead of playing it), and the stream itself is
`/stream/<match_id>.mjpg?start=<source seconds>&rate=<speed>&width=<px>`. A single annotated still is at
`/frame/<match_id>.jpg?t=<seconds>`, and `/matches` lists what is available as JSON. Every frame carries the pitch
markings projected back through the same corrected camera chain the report uses (the homography drawn on the
field), a box on each tracked player in their measured team colour with a chip showing their shirt number - or
`#track_id` while nobody has named them - the ball from the segment's scan (a filled dot when a detector saw it, a
hollow ring where the scan coasted across a miss), and a clock/legend HUD.

The encoded form of the same stream is what the dashboard pane plays where the browser can: `/live/<match_id>.mp4`
(an endless fragmented MP4: `fps=`, `rate=`, `width=`, `overlays=`, `audio=`), `/video/<match_id>.mp4` (a bounded,
seekable clip encoded on demand and cached; `duration=` bounds it; `codec=h264|hevc`), and `/game/<game_id>.mp4`
(the whole combined game, for Step 1 marking). Safari will not play an open-ended fragmented MP4 - Apple's media
stack wants byte-range support or HLS, and an endless encode has neither - so WebKit-family browsers get the MJPEG
stream instead; the pane makes the choice itself, and falls back the same way in any browser whose video loads keep
failing.

The stream draws what the archive already knows and invents nothing: boxes come from the report build (the
`boxes.npz` beside the replay - a replay without them is refused with the `scripts/rebuild_match.py` command
instead of streaming silently empty boxes), and shirt numbers come from the roster and the number scan, with a
stale scan called out in the corner rather than mapping numbers onto the wrong tracks. Each viewer gets their own
ffmpeg decode, so streams can start at different times and speeds; `rate` is capped by how fast the footage
decodes, and when that is slower than real time the pane restarts the stream at the animation's second rather than
drifting away from it (switch *sync footage* off to let it play on its own).

## Project layout

```
data/videos/         # local input videos (other roots via SOCCER_VIDEO_ROOTS)
<footage root>/      # e.g. /srv/storage/home_video/Xbot - one directory per match, per team as you like
    <date>/          #   the recording folder (the date names the analysis)
        *.MP4        #   original clips, untouched
        game_*.mp4   #   the combined game video, a stream copy of the clips
        analysis/
            <id>/           # everything computed for this match: match.json, calibration.json, report.json,
                            # replay.json, events.json, highlights/, identities/, segments/ (Stage A), game.json
src/soccer_analytics/
    ingest/          # GPU-accelerated frame and audio I/O (ffmpeg)
    geometry/        # camera motion recovery (estimated chain + gimbal log), pitch calibration
    analysis/        # staged analysis (see below)
    dashboard/       # Streamlit app
tests/               # including a synthetic-match oracle with known ground truth
scripts/             # run_stage_a.py: the background heavy pass; run_ball_scan.py: the ball scan;
                    # refresh_kit_descriptors.py: re-derive kit colours without re-analysing;
                    # rebuild_match.py: headless report+replay rebuild; run_match_stream.py: annotated MJPEG;
                    # migrate_analysis.py: move repository-era archives beside their footage
```

## The two-stage split

This is the central design decision, and it is what makes the dashboard usable interactively.

**Stage A** (`analysis/stage_a.py`) is heavy, GPU-bound and run once per segment. It decodes the video, recovers
camera motion frame by frame, detects people and records a compact kit-colour descriptor for each one. It writes
chunk files that let it resume, and a `status.json` the dashboard polls. It samples at **15 fps by default** (both
60 and 30 fps camera files divide by it), and every frame-count window downstream — BoT-SORT's identity buffer,
the tracker's gates, the ball scan's coast windows, the offline stitcher — is derived from seconds at the rate the
segment was built with, so an older 5 fps segment keeps its tuned behaviour. Measured on the real footage: ~12
analysed frames per second of processing (~90 minutes for a 72-minute game at 15 fps), 36 detections per frame,
and chunk files that grow linearly with the rate.

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

## The gimbal's own log (ground-truth camera motion)

The camera writes a text log beside each clip recording the tracking loop: the **yaw and pitch it commanded** every
frame, its **zoom step**, and its **ball-lock state** (`Lock:a/b`, the ball's image position and velocity). This is a
*hardware measurement* of where the camera pointed, not an estimate from the picture, and it does not drift.

`geometry/gimbal_log.py` parses the log (tolerant of the camera's habit of splitting a line mid-token) and
`geometry/gimbal_motion.py` turns it into a camera pose. The gimbal pans about the **world vertical**, so the pan
axis is the world-up direction in the camera's frame - an angle of `90 - tilt` from the optical axis (78.4° on this
footage, where the tilt is held at 11.6°). The yaw *scale* is fitted from the estimated chain's large per-step
rotations, which are accurate even though the chain's accumulated orientation is not.

`analysis/projection.py` prefers the log-backed orientation wherever a log exists and falls back to the estimated
chain otherwise. Measured on the whole 2026-10-03 game: a model fitted on the first minutes predicts landmark clicks
40 minutes later to a **median 16 m, against 48 m for the chain** - the drift the chain accumulates over a game is
exactly what the log removes. Re-fitting the existing clicks against the log-backed motion drops the calibration RMS
from 2.55 m (25 clicks rejected as outliers) to **1.14 m (2 rejected)**, because the late-game clicks the chain had
drifted away from now fit.

A calibration is fitted against one motion source's reference frame, so it records which one (`pose_source`); the
dashboard warns when a saved fit was built against the other and asks for a refit rather than projecting through a
stale pose.

`analysis/ball_lock.py` uses the hardware lock for two things that need no homography at all: **ball in play** (a
sustained lock is live play; a long gap is a stoppage) and **image-space possession** (the nearest player to the
ball *in the picture*, which survives a bad calibration).

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
  large, camera-followed torsos.
- **Team colour is read from player-sized, well-evidenced tracks only.** Three measured failure modes shaped the
  rule, and all three were found on the real 2026-10-03 game:
  1. *The largest boxes are not players.* Boxes over ~200 px at 1920 width are near-sideline bystanders — coaches,
     photographers, spectators — whose random clothing lands in whichever cluster is nearest. The trusted band is
     60–200 px: big enough for a torso crop to carry colour, small enough to be a player on the pitch.
  2. *Most tracks have no colour evidence at all.* Of ~2,800 stitched tracks, half have no player-sized
     observation and their grey medians would outvote the real kits. Only tracks with enough player-sized
     observations (scaled to segment length) and crops that were mostly kit rather than grass may vote — that
     cuts the electorate to ~150 tracks, and the red/blue split becomes clean.
  3. *The colour vector was decoded one field off.* The team colour is stored as `[L, a, b, sat, val]` but
     `kit_rgb` expects the full 12-float descriptor layout starting with `kit_fraction`; passing the bare
     5-float vector read `a/b/sat` as `L/a/b` and reported a red team as "light grey". With the layout fixed and
     the electorate fixed, the same game reads team 0 = red (172,102,117), team 1 = blue (103,140,198) — matching
     what is visible in the footage.
- **Jersey numbers were read from the wrong frames.** `scripts/extract_jerseys.py` converted the reader's source
  timestamps with `round(time_s * fps)`, ignoring the segment's `start_s` — on the whole game every crop came from
  ~2,704 frames (9 minutes) after the one analysed, which is why 5,455 crops yielded 27 readings and 0 suggestions.
  The index is now `(time_s - start_s) * fps`. Crop selection also stays inside the player band (130–200 px) and
  rejects motion-blurred torsos (variance of Laplacian) before spending OCR on them.
- **Possession is a proximity estimate.** Possession is attributed to the team whose player is nearest the camera's
  aim point. Counting detections per team instead lets the referee decide possession (they follow the ball all
  match); measured as a 10-point swing, which is why the aim proxy is used.
- **A pan-locked apparent field rotation exists, but it is not a fixable pose error.** The user's insight — that a
  camera pan should not make every player's direction change coherently — was tested directly: during active pans
  (|dyaw| > 0.5 deg/frame) the projected field rotates about the camera by ~0.0022 rad per deg/frame of pan
  (bootstrap 95% CI 0.0016-0.0021 on the median ratio, zero-lag peak, still-frame control ~0.000, and the rotation
  accumulates over a pan burst rather than cancelling). Three checks say it is not a correctable camera-pose error:
  (1) it grows with range (near band -0.0009, mid +0.0017, far +0.0035 rad/deg) instead of being uniform, which a
  yaw-scale error cannot do; (2) applying the implied yaw-scale correction (effective scale 0.86 vs the fitted
  0.70) zeroes the field-rotation slope but *worsens* the landmark-click residuals (rms 6.4 -> 8.6 m) and shortens
  tracks; (3) subtracting the rotation from the displacements does not reduce the pan-speed inflation (5.4 -> 5.5
  km/h), so the inflation mostly comes from elsewhere (detection lag during fast image motion, not projection).
  The signal is real and pan-locked, but it is a small (sub-metre) image-row-dependent artifact of fast pans, not
  a homography error the pipeline can correct; the honest use of the user's fact is as a *diagnostic* — the
  collective-motion check is implemented in the analysis scripts and can flag segments whose pans are fast enough
  to distrust.
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
