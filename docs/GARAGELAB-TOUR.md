# Garagelab drone tour — how to reproduce and tweak

`scripts/garagelab_tour_shot.py` renders the cinematic tour of the garagelab
splat (`assets/garagelab/`) headless in IsaacLab, with posed/animated X2
mannequins and a waypoint drone camera. `scripts/garagelab_orbit_shot.py` is
the simpler scene-only orbit that preceded it.

## The exact command for the shipped video (2026-09-18)

```bash
cd ~/projects/x2-groot-vla
/home/stickbot/miniconda3/envs/env_isaaclab/bin/python -u \
  scripts/garagelab_tour_shot.py \
  --scene-usdz assets/garagelab/garagelab_splat.usdz \
  --collision-usd assets/garagelab/garagelab_collision.usd \
  --chores-pkl ~/Projects/GR00T-WholeBodyControl/data_local/eval_sets/x2_pico_chores8_20260904.pkl \
  --x2m2 middle=~/Projects/GR00T-WholeBodyControl/gear_sonic_deploy/data/motions_x2m2/demo_bank/victory_dance_macarena_R_001__A306.x2m2 \
  --x2m2 south=~/Projects/GR00T-WholeBodyControl/gear_sonic_deploy/data/motions_x2m2/demo_bank/shadow_boxing_R_001__A359.x2m2 \
  --sofa-wave-x2m2 ~/Projects/GR00T-WholeBodyControl/gear_sonic_deploy/data/motions_x2m2/demo_bank/right_wave_001.x2m2 \
  --grid 16 --grid-spacing 2.5 \
  --out-dir /tmp/tour_frames --accum 12 --headless --enable_cameras
ffmpeg -framerate 24 -i /tmp/tour_frames/frame_%04d.png -c:v libx264 \
  -preset slow -crf 28 -pix_fmt yuv420p -movflags +faststart garagelab_tour.mp4
```

Splat binaries are gitignored — regenerate `garagelab_splat.usdz` /
`garagelab_collision.usd` from `assets/garagelab/garagelab1.spz|.glb` with
`x2-kitchen-sim/scripts/01a_spz_to_usdz.py` and `01c_mesh_align_to_usd.py`
(run with `~/projects/g1-kitchen-sim/.venv/bin/python`).

## Iterating cheaply

- `--probe` renders ~2 frames per waypoint instead of the full path (~4 min
  incl. Isaac startup). Inspect the PNGs, adjust, repeat. Full 58 s render
  with the 1073-robot farm is ~35 min on the 5090.
- `TOUR_DIAG=1` renders three fixed side/top/front views of the couch area —
  use these (not path frames) to debug robot placements; perspective lies.
- Robot placements and the camera path are the `ROBOTS` and `WAYPOINTS`
  tables at the top of the script — world-frame coords, waypoints are
  `(t_seconds, eye_xyz, look_xyz)` on a Catmull-Rom spline.

## What the pieces do

- **Mannequins**: roots pinned every physics step (cannot fall), collision
  USD via TerrainImporter so limbs settle on real geometry, MJCF colours
  (approved garagelab look: grey 0.03 / white 0.7 / orange feet).
- **Chores replay** (`--chores-pkl`): the `executed` clips are SONIC-executed
  trajectories, replayed with full root motion (`anchor_root` re-anchors each
  clip's root track at the placement) — visually equals SONIC driving.
- **Demo-bank gestures** (`--x2m2 name=path`, repeatable): loop any
  deploy-bank `.x2m2` on a named robot (macarena, shadow boxing, waves…).
  X2M2 = `<III` magic/frames/31 + `<d` fps + per-frame 31+4 float64
  (MuJoCo dof order; map with `X2_ULTRA_ISAACLAB_TO_MUJOCO_DOF`).
- **Sofa wave** (`--sofa-wave-x2m2`): overlays only the gesture's right-arm
  channels on the held sit pose; the loop phase is set so the crest lands on
  the camera pass (see the `off` comment in the script).
- **Training farm** (`--grid N`, `--grid-spacing`): (2N+1)²−house cells of
  fixed-base chore-replaying X2s on the visible grid plane + the crowd
  fly-through/scale-reveal/roof-dive intro (15.5 s, tour shifts after it).
- **Camera sanity**: gaze direction is precomputed, tilt-capped (40° tour /
  60° intro) and box-filtered 0.5 s — do NOT reintroduce per-frame look
  clamps; they cause whip-pan spikes when the look spline crosses the eye
  path.
- **`--free-roots` (experimental)**: releases the standers (floating base,
  stiff PD, real-time stepping, auto-reset on fall). Finding 2026-09-18:
  chores clips → all 6 fell in 2–33 s; `--free-idle-x2m2
  gear_sonic_deploy/data/idle_stand.x2m2` → 4/6 survived the full tour, the
  2 fallers were release transients caught by auto-reset. True released
  replay needs live SONIC (manager-env route — the bridge's in-proc
  `--sonic-onnx` is a known shelved failure, see project ledger).

## Gotchas that will bite you

- **Disk**: `UrdfFileCfg` writes a ~140 MB `/tmp/IsaacLab/usd_*` conversion
  dir per spawn — the script converts once (robot #1) and reuses the USD for
  the rest, but still leaves one dir per launch. `rm -rf` today's
  `usd_<date>_*` dirs after rendering (verify no Isaac running first).
- Never launch a second Isaac instance (`pgrep -af "isaac|omni|kit"` first).
- Ground plane sits at z = −0.02; at 0.0 it z-fights the splat floor (white
  shimmer on the dark mats).
- Camera above ~2.4 m inside the room is in ceiling-splat fuzz; the intro
  dive crosses it deliberately.
- `close()` hangs with cameras enabled — the script watchdogs and hard-exits;
  the `[tour] DONE` line is the success signal.
