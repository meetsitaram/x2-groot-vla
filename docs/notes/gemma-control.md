# Gemma → X2 command & control — exploration (2026-10-23)

Status: EXPLORATION ONLY, parked (active focus stays robot-fight). No code yet.

## Thesis

Gemma never touches joints. It is a **slow (0.2–1 Hz) task-level planner
that calls tools**; every tool maps onto a command surface that already
works in sim and on the robot. SONIC (50 Hz planner → ONNX WBC) stays the
only thing that moves the body.

```
voice/text ─► Gemma 4 (Ollama, local 5090 or PC2 Jetson)
                 │ native function calling
                 ▼
           x2_agent.py  (tool executor + safety gate + world state)
     ┌───────────┼───────────────┬──────────────────┐
 planner_cmd   motion_clip_cmd   nav_policy_bridge   VLA --prompt
 ZMQ :5563     ZMQ :5568         (walk-to-waypoint)  (learned pick)
 (sim :6563)
     └──────── kplanner (50 Hz) ─► SONIC ONNX ─► MuJoCo / IsaacLab / X2
```

## Decision (user, 2026-10-23): Gemma → kplanner for loco AND manip

Target daemon = **`gear_sonic/scripts/pc2_kplanner_onnx.py`** (not the
laptop torch `x2_kplanner.py`, which has no arm ingest and no clip
playback). It runs on PC2 *and* on the laptop (`wb_wrist_kplanner_test.py`
launches it with env_isaaclab python on isolated ports), so sim and robot
share one daemon. Its three input channels:

| Channel | Port | Payload | Gemma use |
|---|---|---|---|
| `planner_cmd` (JSON) | 5563 | intent/magnitude/sticks; also `arm_release`, `wrist_targets`, `head_targets`, `estop` | locomotion, head look, release arms |
| `arm_targets` + `hand_finger_cmd` (msgpack) | 5572 (`--arm-port`) | `left_q_rad[7]`, `right_q_rad[7]`; `left_hand_q[10]`, `right_hand_q[10]`; `passthrough_arm_targets` clears | arm reach / grasp, overlaid on the legs while walking or standing (`ArmTargetIngest`, `:2057`) |
| `motion_clip_cmd` (JSON) | 5568 | `{"action":"play","motion_key":k,"kind":"locomotion"}` from `--dances-dir/*.x2m2`; `stop` | whole-body pick_up/put_down clips + gestures (`DancePlayback`); crouch via `PrimitivePlayback` |

Overlay semantics that matter:
- Arms slew at 0.06 rad/frame (3 rad/s), so a target jump never snaps.
- If the link drops, the arms **hold** their last pose.
- A playing clip suspends the arm overlay, so the clip owns the whole body.
- `hip_h` intent is dead in the deployed graphs (measured 2026-08-04).
  Crouch = primitive clip only.

**Gemma never emits joint angles.** The executor turns its symbolic calls
into 50 Hz streams:
- `reach(hand, target)`: target = an object from `look()`, or a named
  surface spot. The executor runs DLS IK (`gear_sonic/utils/teleop/solver/arm/x2_arm_ik.py`)
  each tick and streams `arm_targets`.
- `grasp(hand, open|close)` → `hand_finger_cmd` presets.
- `pick(object, height_tier)` → a stance from kitchen_surfaces.json, then
  `go_to` that stance, then a matching `*pick_up_{low,medium,high}` clip
  on `motion_clip_cmd` (70/73 of these clips executed clean in MuJoCo).
  Alternatively, `pick` hands off to the VLA `--prompt`.
- `release_arms()` → `planner_cmd` `arm_release` / passthrough.

Two manipulation tiers, in order:
1. **Clip-based (fast):** these are proven assets, and Gemma only picks the
   clip and the stance.
2. **IK-overlay reach:** more general, but it needs a per-tick IK loop, a
   target pose from perception, and testing for balance under the arm
   overlay while the legs idle.

## Existing command surfaces (verified 2026-10-23)

| Surface | Where | Message | Sim | Real |
|---|---|---|---|---|
| `planner_cmd` | `GR00T-WholeBodyControl/gear_sonic/scripts/x2_kplanner.py:931` (laptop), `pc2_kplanner_onnx.py:654` (Jetson) | multipart `[b"planner_cmd", json]`: `intent` (idle/walk/fwd_step/back_step/side_*/turn_*/crouch/lean_*/torso_*/locomotion/shutdown), `magnitude` (deg_15..deg_90, one_ft.., continuous), `stick_*` ∈[-1,1], `target_velocity`, `waist_*_deg`, `hip_height_m`, `source` | ✓ | ✓ |
| `motion_clip_cmd` | `gear_sonic/utils/teleop/motion_clip_session.py:127` port 5568 | `{"action":"play","name":<catalog>,"kind":"gesture"}` / `{"action":"stop"}` | ✓ | ✓ |
| Nav to waypoint | `gear_sonic/scripts/x2-navigation/nav_policy_bridge.py --goal sink` | publishes sticks on :5563 from pose on :6570 | IsaacLab | (needs localisation) |
| Learned pick | `gear_sonic/scripts/vla/live_vla_publish_motion_token.py --prompt "..."` | free-text task description | – | ✓ |
| Named places | `x2-kitchen-sim/configs/waypoints.json` (8 places, xy/yaw/radius), `analysis/kitchen_surfaces/kitchen_surfaces.json` (35 surfaces) | | | |

Commands are latched (no fixed rate needed). Pad/VR ownership arbitration
exists — the agent should publish as `source:"pad"` (or a new `"llm"` source
below pad priority, see Safety).

No LLM client code exists in any repo yet.

## Model choice — Gemma 4 (Apache-2.0, Apr 2026)

| Variant | Params | Inputs | Fit |
|---|---|---|---|
| E2B / E4B | ~2.3B / 4.5B effective | text+image+**audio** | on-robot (PC2 Jetson), speech-in without a separate ASR |
| 26B A4B MoE | 4B active | text+image | **default on the 5090** (fast, strong tool use) |
| 31B dense | 31B | text+image | quality ceiling; heavy next to Isaac |

All have native function calling. Serve via Ollama (`gemma4:e4b`,
`gemma4:26b`; ≥0.22, set `num_ctx` > 4K default) or vLLM (OpenAI-compatible
`tools` API). Not installed locally yet (no ollama, no transformers in the
WBC venv; Docker 29 present; 115 GB disk free — 26B Q4 ≈ 17 GB).

GPU note: the 5090 is shared with Isaac. 26B-A4B Q4 ≈ 18–20 GB VRAM — may not
coexist with an IsaacLab kitchen run; E4B (~5 GB) will. MuJoCo runs are fine.

## Tool schema (v0)

Keep tools coarse, closed-vocabulary, and blocking (return when done/failed),
so a small model cannot compose unsafe motion:

- `stop()` — idle immediately (also bound to a non-LLM hotkey/pad button).
- `step(direction: fwd|back|left|right, distance: quarter_ft|half_ft|one_ft)`
- `turn(direction: left|right, degrees: 15|30|45|60|90)`
- `walk(duration_s ≤ 5, speed: slow|default)` — capped, re-planned each call.
- `go_to(place: enum from waypoints.json)` — wraps nav bridge; returns arrival error.
- `posture(action: crouch|stand|lean_*, …)` — bounded waist/hip.
- `gesture(name: enum from gesture catalog)` — motion_clip_cmd.
- `pick(object_description: str)` — hands off to VLA `--prompt` (real) / pick clip (sim).
- `look()` — returns head-cam frame to Gemma (vision) + pose + current place.
- `get_state()` — pose, nearest waypoint, standing/fallen, active source.

The executor translates each to `planner_cmd` JSON; the model only sees enums.

## Safety gate (non-negotiable before PC2)

1. LLM source has LOWEST priority — pad and VR always pre-empt; stick move = abort plan.
2. Executor-side whitelist + clamps (no `target_velocity`, no `shutdown` from LLM).
3. Watchdog: no tool completes in its timeout → `idle`.
4. Every real-robot session needs the per-action approval rule from CLAUDE.md;
   start with "dry-run" mode that prints the JSON instead of publishing.
5. Fall detector from `get_state()` halts the agent loop.

## Milestones

- **G0** (≈1 h): install Ollama + `gemma4:e4b` & `gemma4:26b`; tool-call
  accuracy bench on ~50 canned utterances ("turn around and walk to the
  fridge") in dry-run → JSON. Measure latency + correct-tool rate per size.
- **G1**: `x2_agent.py` → `pc2_kplanner_onnx.py` (laptop, isolated ports)
  → MuJoCo: step/turn/walk/stop + gesture/pick clips via motion_clip_cmd.
  Film it.
- **G1b**: IK-overlay `reach`/`grasp` on `arm_targets` :5572 while
  standing; check balance + slew behaviour.
- **G2**: IsaacLab kitchen — `go_to(place)` via nav bridge + `look()` with
  head-cam frames ("what's on the counter?" → go_to + look).
- **G3**: speech in (E4B native audio, or whisper) + E4B on PC2 Jetson.
- **G4** (real robot, per-action approval): dry-run on PC2, then gated
  step/turn only, pad in hand.

## Open questions

- **RULING (user, 2026-10-23): NO Gemma on the robot.** SONIC already uses
  all of the Orin NX. Gemma runs off-board:
  - the laptop now, and an external **Thor** later;
  - it PUB-connects to PC2's kplanner ports over the network (same
    topology as the Quest manager);
  - the E2B on-robot track is dropped.
- (superseded) **Compute:** the robot has only an **Orin NX**. If Gemma
  doesn't fit there, an external **Thor** rides along.
  - Thor modules: T5000 = 128 GB, T4000 = 64 GB.
  - Ollama 4-bit downloads: e4b = 5.5 GB (measured), e2b ~3 GB, 26b ~17 GB.
  - So E2B, or maybe E4B, on a 16 GB NX; 26B and up need Thor.
  - The G0 bench decides whether E2B/E4B are good enough.
  - PC2 free memory is still unmeasured; reading it needs user approval.
- Gemini Robotics On-Device / ER 2 (July 2026) is the Google-native
  alternative for the embodied-reasoning layer, but early-access only —
  Gemma 4 is the open, local path.

## Results (2026-10-23)

**G0 bench** (`python -m gemma_ctl.bench`, 50 utterances incl. 8 must-not-move):
gemma4:26b 50/50, 8/8, 0.27 s median; gemma4:e4b 48/50, 8/8, 0.07 s.
Runs: `runs/gemma/g0_bench_*`.

**Robot-side safety:** `pc2_kplanner_onnx.py` has a new `llm` command source,
saved as `patches/pc2_kplanner_llm_source.patch` and applied UNCOMMITTED.
- Lowest priority: any pad/VR message preempts it, with a 3 s lockout.
- A 1 s silence failsafe idles the robot and exits primitives.
- `llm_keepalive` heartbeat.
- `python -m gemma_ctl.test_llm_source`: 6/6 PASS. On wifi loss the robot
  is stopped about 2.5 s after the last heartbeat (1 s detect + ~1.5 s
  planner decel).
- NOT deployed to PC2.

**G2 kitchen demo PASSED** (`runs/gemma/kitchen/demo1`, artifact
https://claude.ai/artifact/DwMxEZxApTKEqhZR3MiUz6). Gemma 26B, 4 requests
(look / mug→sink / can→stove counter / go to fridge), all done, 1 pick
retry, palm error 1.3–6.6 cm, no falls, 451 s sim.
- Launch: `gemma_ctl/launch_kitchen_gemma.sh` (rig + `KP_GEMMA=1` hooks,
  headless, frames recorded).
- Drive: `python -m gemma_ctl.agent --kitchen --live --model gemma4:26b --say ...`
- Skills without an LLM: `python -m gemma_ctl.skill_check pick:mug place:sink_counter`.
- Rig changes (env-gated, UNCOMMITTED): `patches/kitchen_rig_gemma_hooks.patch`.
  - `KP_GEMMA` hook import, `_write_row` arm/waist override, `gemma.tick()`.
  - `KP_RING_S`: ring length knob.

**Findings / pitfalls (each cost a debug cycle):**
- **Episode resets:** the rig resets every 120 s sim when playback wraps
  the 6000-row ring, re-spawning the robot. Long sessions need `KP_RING_S`
  (the launcher sets 1500).
- **Coasting:** the rig's planner commits ~0.9 s per replan, so the robot
  coasts (90° turn → 119°, 0.5 m → 0.84 m). Stop predictively on
  progress + v·latency (turn 0.4 s, walk 0.9 s), then re-correct.
  - Positioning quantum is ~0.3 m, so arrival is reach-ENVELOPE based,
    not exact stand points.
- **nav_policy_bridge** misfires STUCK because its stuck timer is wall-clock
  and the rig runs at 0.23× with recording. Replaced by `gemma_ctl/nav_plan.py`
  (A* on the nav grid + esdf line-of-sight smoothing) with closed-loop legs.
- **Arm targets reach the robot late:** reference rows are written ~1 s ahead
  of playback, so the palm moves ~1.3 s after the overlay finishes.
- **SONIC (softland_4800) tracks hand keypoints, not arm joints.**
  - Unregularised IK gave contorted arms (shoulder roll -2.3) and 10–20 cm
    palm error.
  - Natural-posture nullspace IK (`gemma_ctl/ik.py`) plus a 0.25 rad waist
    lean below z 1.0 brings it to 4–8 cm at counter height.
  - Bounded closed-loop correction gets 1–3 cm.
  - Palm floor ≈ 0.9 m; the 0.73 m side table is marginal.
- **Shortcuts, stated in the artifact:**
  - Kinematic grasp: the object is attached to the wrist; objects are
    visual-only.
  - Ground-truth `look()`.

**Next:** Gemma vision on head-cam frames for `look()`; whole-body pick clips
instead of the arm overlay; deploy the `llm` source to PC2 (needs user
approval).

## Branches (user, 2026-10-23): isolate until the stack is proven

- **x2-groot-vla:** branch `gemma-control`, in place. All `gemma_ctl/` files
  are uncommitted.
- **Runtime repo:** worktree `~/Projects/GR00T-WBC-gemma` on branch
  `gemma-control`, branched from `stability` @379040d.
  - It holds the llm-source and kitchen-rig patches. Both are applied
    there and UNCOMMITTED.
  - The main checkout `~/Projects/GR00T-WholeBodyControl` is back to
    clean `stability` for those files.
  - The worktree was created with GIT_LFS_SKIP_SMUDGE, then
    `git lfs pull --include "gear_sonic/data/**"`.
  - `x2_sonic_feasible_stand_single.pkl` (untracked in main) is symlinked in.
- `gemma_ctl` resolves the runtime via `GEMMA_WBC` (default = the worktree).
- The llm-source test passes against the worktree.

## Target architecture (user, 2026-10-23)

Gemma must perceive, not be handed a list:
1. A live camera feed: room cams for the holistic scene view, plus the
   head cam for close checks. Gemma returns object/place BOXES itself.
   NO separate detector (user ruling).
2. A world model built from Gemma's boxes, lifted to 3D with depth and
   camera pose.
3. A live chat session.
4. The stack executes, with camera feedback to Gemma for occasional
   corrections.

Milestones:
- V1: perception, with 3D accuracy graded against sim truth.
- V2: world model replaces the sim list and the catalog.
- V3: chat UI.
- V4: head-cam checkpoints.
- V5: robot.

Probe result: on a clean 1080p frame Gemma 26B boxed the mug, robot and
fridge correctly (`runs/gemma/kitchen/demo2/review/gemma_boxes.jpg`). The
earlier failure came from 800-px captioned stills.

## V1 results (2026-10-23): Gemma perception, graded vs sim truth

Code: `gemma_ctl/perceive.py`, `v1_batch.py`; rig `survey` and `move` ops.
Runs: `runs/gemma/perception/{survey_*,v1_batch_*,close_*}`.

- **Depth:** the splat writes NO depth (only meshes do). Fuse the rendered
  depth with Open3D ray-cast depth of `kitchen_collision.obj`
  (0.05 s/frame at 1080p), keeping the nearer value per pixel.
- **Camera pose:** the tiled camera's reported pose is STALE after
  `set_world_poses_from_view`. Rebuild it from the commanded eye/target
  (ROS look-at, up=+z).
- **Object lift:** support surface = histogram mode inside the padded box;
  the object = points above it, clustered around their median.
- **Localization when Gemma boxes the right thing: 3–5 cm, consistently**
  (12 hits, all 3.0–5.7 cm). Places (counters 0.87, table 0.73, sink)
  land at the right heights.
- **Recall is the gap:**
  - Room cams at 2–3 m, exhaustive survey: 6/12.
  - Targeted "find the X" on tiles: 4/12.
  - Point-then-zoom: 2/12.
  - Close cams at ~1.4 m: 3/6.
  - Clutter-free side table: always found. Busy counters: missed.
- **The mug prop (Mug023, black) reads as a black disc from above,** with
  no handle visible. Gemma doesn't call it a mug, even at 1.4 m.
- **Gemma box JSON is inconsistent on small objects:** it says "yes, a dark
  cup on the sink edge", then returns no box at 400 px and a good box at
  200 px.

## Robot cameras in sim (user correction, 2026-10-23)

My first "head cam" was the render camera's 53° lens at the head pose. That
was WRONG: no X2 camera has that view. The survey now uses the calibrated
URDF (`x2_ultra_sphere_feet_cams_calibrated.urdf`) and the
robot-camera-calibration memory:
- `camera:"head_wide"` = `stereo_head_front_left`.
  - Mount: head_pitch_link (0.068, 0.0298, 0.05), level.
  - The real lens is a ~170° equidistant fisheye (fx≈685 @2064). Rendered
    as the centre 120° pinhole.
  - Role: SEARCH.
- `camera:"orbbec"` = `rgbd_head_front`, the Gemini 335.
  - Mount: head_pitch_link (0.0576, -0.0112, -0.0484), 40° down; 86° HFOV;
    1280×720 + depth.
  - Role: MANIPULATION close-up. From across the room it sees floor.
- The render camera's lens is swapped per capture (USD focalLength) and
  restored afterwards. K is computed from the HFOV (the reported K goes stale).

Port clash: another local stack (`run_x2_quest3_planner_stack.sh --pad-only`,
GR00T-WBC-X2-supermimic) took 5563 and likely took down a Gemma rig.
- The Gemma rig now uses `KP_PAD_PORT=7563`. The kitchen-rig knob defaults
  to 5563.
- Only ONE driving KitchenExecutor may exist: observers use `drive=False`.

## Coarse-to-fine results, run headcam_20261023T120016 (gated, noun search, full robot-cam labels)

Errors are horizontal; truth = bbox centre.

| prop | room | wide head | Orbbec | fused | label |
|---|---|---|---|---|---|
| big mug | 1.7 cm | 2.0 cm | 1.5 cm | 1.6 cm, ±1.3 | mug |
| kettle | 6.5 cm | 2.5 cm | 1.5 cm | 2.1 cm, ±1.5 | kettle |
| pitcher | 3.5 cm | — | 4 cm xy | z off 16 cm | "white mug" |
| pot | not matched | | | ~16 cm xy | black pot |
| can | not seen by room cams | | | | |

- **Pitcher:** Gemma's box was correct. The z error is because my scene
  placed it hovering at sink-rim height over the basin, and the support
  surface came out as the basin floor (0.70 m). Grade and gate horizontal
  position for reaching.
- **Pot:** all three cameras agree on a position ~16–18 cm from the "truth"
  centre. Suspect the truth: pot.usd may have an internal transform the
  bbox-centre calc misses. OPEN.
- **World-model fixes:**
  - semantic association gate (`labels_compatible`);
  - target gate;
  - noun-only targeted search;
  - quality-weighted label votes.
- **3D viewer:** `gemma_ctl/scene3d.py` builds a run into the viewer: mesh
  coloured from the splat (transform world = (x, -z, y+1.371)), objects as
  boxes, places, cams, truth. Artifact YaqUnRvg3bP2v7UsV8DuTM.

## V2 milestone (2026-10-23): perception-driven pick-and-place PASSED

Gemma drove the full task on its OWN perception (`gemma_ctl/executor_perception.py`,
`agent.py --perception`):
- `look()` = room cams + wide head cam -> Gemma boxes -> world model
  (16 objects).
- "put the mug on the white table":
  - `pick(obj4)` FAILED honestly. The "black mug" was the rice cooker's dark
    side; nothing was within the 12 cm attach radius.
  - `pick(obj5)` OK: the real mug, ±2.2 cm.
  - `place(white_table)` OK: palm 2.5 cm; Gemma's own place and height.
- A fresh session's `look` then found the mug on the table (7.1 cm).

Sim truth confirms: mug at (2.08, -1.06, 0.73), 23 cm clear of the kettle.

- Pages:
  - demo 8k9f4qaJHsDBN4fThYHqBf;
  - 3D after-task VnELH45DA8FQz3cjFRtEYy;
  - 3D earlier YaqUnRvg3bP2v7UsV8DuTM;
  - camera views 7P8Nx3NTrWxibAXCqHfeXh.
- Run: `runs/gemma/kitchen/live1`, `runs/gemma/live/live_*`.
- Next:
  - NEGATIVE EVIDENCE: a close look that finds nothing should demote or
    remove the object;
  - label stability across views;
  - search for unseen objects with the wide head cam;
  - Orbbec verify should search by the held object's noun.

## Run bundles + overlay video + negative evidence (2026-10-23)

- **Bundle** `runs/gemma/bundles/pickplace_mug_20261023T1330/`:
  - `frames/`, `cam_poses.jsonl` (per-frame camera pose);
  - `trajectory.jsonl` (25 Hz robot root + 31 joints + props);
  - `live/` (`world_timeline.jsonl`, `conversation.jsonl`, survey images);
  - transcripts;
  - `run_bundle.json` (manifest, both repos' revs; `gemma_ctl/bundle.py`).
- **Re-render without the sim:**
  - `overlay_video.py <b> <b>/live` draws Gemma 3D boxes, labels ±σ,
    "What Gemma sees" insets and the bottom-left conversation HUD;
  - then `REVIEW_FRAMES=frames_ov make_review.py <b>`.
- **New camera angles:** rig `KP_GEMMA_REPLAY=<traj>` with `_T0/_T1`
  (kinematic replay). A replayed frame matches the original exactly.
  NEXT: camera modes for replays (fixed/orbit/follow); a replay frame-time
  fix is done.
- **Negative evidence:** a close look with the target in frame and no
  matching box sets `misses`; pick refuses and lists other candidates;
  `look` tags "NOT there when checked up close". In this run it rejected
  the rice-cooker "mug" without reaching.
- **Gemma asks to disambiguate** ("Which one would you like me to move?")
  when two mugs are seen.
- **Gemma 4 sometimes leaks "thought <channel|>..." as its reply:** strip it
  (agent + HUD).
- **Self-kill trap AGAIN:** pgrep/pkill patterns matching your own command
  line. List PIDs in one command, kill by number in another.
- Demo page 8k9f4qaJHsDBN4fThYHqBf v2 (overlay + HUD video).
