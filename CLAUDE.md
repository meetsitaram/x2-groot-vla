# x2-groot-vla — working context

Read [README.md](README.md) first, then [docs/PLAN.md](docs/PLAN.md) (the
canonical plan, including the two feasibility audits). This file carries the
operational context a fresh session needs.

## Sibling repos & key paths

- Runtime repo: `/home/stickbot/Projects/GR00T-WholeBodyControl` (capital P).
  Its venv: `.venv/bin/python` (activate per shell). IsaacLab work uses conda
  env `env_isaaclab` instead.
- World repo: `/home/stickbot/projects/x2-kitchen-sim` — splat kitchen assets
  in `assets/kitchen/` (kitchen_splat.usdz, kitchen_collision.usd,
  kitchen_params.json, nav_grid.npz), waypoints in `configs/waypoints.json`.
- Boneseed corpus: `GR00T-WholeBodyControl/gear_sonic/data/motions/x2_ultra_bones_seed.pkl`
  — 2.6 GB, zlib — load with `joblib.load`, NOT pickle. 37,968 clips; dict per
  clip: `dof (T,31)`, `root_trans_offset`, `root_rot`, `pose_aa (T,32,3)`
  (REAL data, verified), `fps` (30). Manipulation clips: keys matching
  `locomanip__*pick_up*` (1,098) and `*put_down*` (1,108).
- SONIC model cache: `~/.cache/sonic/x2/sonic_policy/x2_sonic_policy.onnx`
  (auto-resolved by all eval scripts).
- Kitchen surface analysis: `analysis/kitchen_surfaces/kitchen_surfaces.json`
  (35 surfaces: z, area, placement spots + clearance radii, walkable standing
  bands, nearest waypoint). Frame: kitchen world, floor z=0 — same frame as
  nav_grid and collision USD.

## State as of 2026-08-03 (where we left off)

- MuJoCo feasibility PROVEN: 70/73 stratified pick_up clips execute clean
  through the deployed X2 ONNX; determinism is bit-exact (max qpos diff 0.0).
  Per-clip results: `analysis/pickup_sweep_reach_2026-08-03.csv`; raw
  executed trajectories: `analysis/qpos_dumps/*.npz` (t, qpos[T,38], fps,
  reason).
- Reach-height ↔ kitchen-surface matching done (PLAN.md "Kitchen edition"):
  `*_medium` → counters 0.876/0.936 m, `*_high` → pantry shelf 1.29 m,
  `*_low` → bench/sofa/console or floor stands.
- PENDING PATCH: `patches/eval_x2_mujoco_onnx_qpos_dump.patch` adds
  `--qpos-dump` to `gear_sonic/scripts/eval_x2_mujoco_onnx.py`. It was
  applied but UNCOMMITTED in GR00T-WholeBodyControl — check `git status`
  there; re-apply from the patch if it's gone.
- NEXT STEP = M0-kitchen smoke (blocked 2026-08-03 on a user G1 im_eval
  occupying the 5090):
  1. Bridge route sanity: `x2_isaaclab_bridge.py --sonic-onnx <fused.onnx>
     --motion <clip.pkl> --scene-usdz <kitchen_splat.usdz>
     --collision-usd <kitchen_collision.usd> --spawn X Y Z YAW --headless`
     — spawn at the dishwasher standing spot from the surfaces JSON, play a
     front_low clip, screenshot.
  2. PhysX determinism A/B (same clip twice, measure qpos divergence —
     MuJoCo was bit-exact; PhysX is NOT guaranteed; reach tier tolerates cm).
  3. Manager-env route with trajectory + render recorders (exact Hydra
     overrides in PLAN.md "Kitchen edition"); object/stand baking via
     `config.add_object` / `config.add_table`.
  4. IsaacLab→LeRobot offline converter (trajectory.pkl + ego frames →
     `Gr00tDataExporter`, schema in `gear_sonic/data/features_x2_vla.py`).

## Regenerating scratch artifacts

Single-clip pkls for eval: load the corpus with joblib, `pickle.dump({key:
clip}, f)` per clip. Sweep: `eval_x2_mujoco_onnx.py --motion <single.pkl>
--no-viewer --max-episode 15 --qpos-dump out.npz` (needs the patch above).

## Hard rules (carried over from the main project)

- Isaac launches: NEVER start a second instance — check
  `pgrep -af "isaac|omni|kit"` + `nvidia-smi` first; kill prior instances by
  PID only with user approval if they're not ours.
- Anything >15–20 s runs in background; stay responsive.
- No `git add`/commit/push without user-approved diff; never force-push.
- NO commands to the real robot (PC2/PC3) without explicit per-action
  approval.
- Proceed autonomously otherwise; pause only for rm / git ops / irreversible.
