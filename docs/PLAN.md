# Boneseed → GR00T VLA bootstrap plan

**Idea**: skip the 100s-of-teleop-sessions phase. Take boneseed mocap clips that
contain a pure manipulation motion with a dummy hand and no object
(e.g. `locomanip__small_light_one_hand_pick_up_front_low_R_002__A504`), build a
scene around the robot with an object (apple / orange / ball / cube) placed
**exactly where the hand reaches**, execute the clip through SONIC in sim,
and record the rollout as a LeRobot episode GR00T can train on. Result: an
X2-native VLA base checkpoint people extend with real recorded datasets.

Scope guard: **object manipulation, in-place motions only**. v0 task is
**"reach the object"** (no contact required); "grasp/pick up" is a later tier.

## What we already have (audited 2026-08-03)

| piece | where | status |
|---|---|---|
| Boneseed corpus | `gear_sonic/data/motions/x2_ultra_bones_seed.pkl` (zlib, load via joblib) | 37,968 clips; **1,098 pick_up + 1,108 put_down** under `locomanip__*`; per-clip `dof (T,31)`, `root_trans_offset`, `root_rot`, `smpl_joints (24,3)`, fps=30. Pick_up clips are in-place (≤0.2 m root excursion, 3–8 s). |
| Name grammar | clip keys | `{size}_{weight}_{one|two}_hand(s)_pick_up_{front|right_side|behind}_{low|medium|high}_R` → free language prompts + placement priors + hand choice |
| Scene grafting | `gear_sonic/scripts/build_x2_robocasa_scene_xml.py` | grafts robocasa table+objects into the canonical `compose_x2_with_omnihand` 31-DOF model; SONIC drives the robot normally (robosuite env never stepped) |
| Envs + assets | `decoupled_wbc/dexmg/gr00trobocasa/.../x2_tabletop_pnp.py`; assets in `robocasa/models/assets/objects/omniverse/locomanip/` | X2PickPlaceApple/Bowl/Cube working; apple_0, rubix_cube, plate, jug, clock meshes; primitives for cube/bowl. No ball yet (sphere primitive = trivial) |
| OmniHand in sim | `compose_x2_with_omnihand.py` | 10 actuated DOF/hand, position-controlled (kp 20, 3 Nm), mimic joints for the rest — finger channel is independent of the 31-DOF policy, same as teleop |
| Recorder / schema | `gear_sonic/utils/teleop/x2_dataset_recorder.py`, `gear_sonic/data/features_x2_vla.py` | LeRobot @50 Hz: `ego_view` 640×480 + `front_cam`, 51-D state (31 body + 2×10 hand), actions `motion_token(64)` + `body_q_mj(31)` + hand joints, `task.success` per frame |
| Synthetic-episode precedent | `gear_sonic/scripts/record_synthetic_smoketest_dataset.py` | already builds LeRobot v2.1 episodes from a scripted trajectory with mujoco-rendered cameras (flat floor, no scene) |
| Clip playback | `run_x2_pkl_direct_stack.sh` + `play_gesture.py --pkl` | plays arbitrary pkl clips through the SONIC deploy loop in MuJoCo |
| Executed-feasibility recipe | `x2_sonic_executed_feasible/`, `mine_x2_motion_candidates.py` | prior art for batch-executing clips through the policy and gating on tracking |
| Train / eval / replay | `vla/train_groot_vla.sh`, `vla/run_x2_vla_runtime.sh --robocasa-env`, `view_x2_recorded_dataset.sh` | unchanged — synthetic datasets are schema-identical to teleop ones |
| Splat worlds | `~/projects/x2-kitchen-sim` + `x2_isaaclab_bridge.py` | IsaacLab/NuRec only (no MuJoCo splat path); robot-camera-over-splat composite proven (gate G1); **zero object-spawning code yet** |

The only genuinely new machinery is: (1) the **reach-point extractor**, (2) the
**scene synthesizer** (object + stand at the extracted point), (3) the
**batch episode driver** (headless: play clip → record → success-gate), and
(4) for the grasp tier, a **finger-overlay generator**.

## Core mechanism: placing the object exactly where the hand reaches

Two-pass, per clip:

**Pass A — measure.** Execute the clip through the SONIC ONNX in the composed
MuJoCo model (headless, deploy tuning, fixed seed, *no objects*). FK the
**executed** (not reference) palm frame each step — tracking error shifts the
hand by cm, and the object must match what the robot actually does, not what
mocap says. Detect the grasp event `t*`: reach extremum with a low-velocity
dwell of the active wrist, sanity-checked against the name tags
(`front|side` × `low|medium|high` give a coarse expected region). Emit
`T_palm(t*)` + approach direction.

**Pass B — build & record.** Synthesize a scene XML with the object centered
at `T_palm(t*) ⊕ per-object grasp offset` and a support fixture underneath
(top surface = object bottom), then re-execute the *same* clip with the same
seed and record the episode. Sim determinism makes the hand land on the
object; a post-hoc check (palm-to-object distance at `t*`) gates the episode.

Support fixture = a parametric **stand/pedestal** (box prim, randomizable
texture like the existing `_randomize_table_texture`), height set exactly so
the object sits at the extracted z. Low picks → stool-height stand; high picks
→ tall stand/shelf. This is simpler and more general than reusing `lab_table`
(which is fixed-height), but the lab_table stays an option whenever the
extracted z happens to be near table height.

Extend `build_x2_robocasa_scene_xml.py` with an "object + stand at pose" spec
instead of hand-writing new env classes per task.

## Episode tiers

**Tier R — reach (v0)**. No contact. Fingers hold a neutral open pose.
Success = palm (or fingertip centroid) within ε (~4 cm) of the object center
at `t*` with a short dwell, object undisturbed. Language: "reach for the
apple on the stand", "touch the orange", … Immune to contact-physics risk;
every feasible clip converts.

**Tier G — grasp/pick-up (v1)**. Overlay a synthesized finger trajectory on
the 10-DOF hands: open pre-shape during approach → close at `t*` with a
per-object aperture → hold closed through the clip's lift phase (pick_up
clips lift after the reach, so a successful grasp raises the object).
Success = object z-rise + sustained grasp contacts (the bridge already
publishes grasp contacts + fingertip positions in `scene_state`). Yield will
be <100% — that's fine: generate at scale, keep survivors, log the yield per
(clip-bucket × object). 3 Nm fingertips → start with light objects (apple,
ball, cube), skip `heavy`-tagged clips for tier G.

**Put-down / place (v1.5)**. Mirror of tier G using the 1,108 `put_down`
clips: episode starts with the object welded/held in hand, fingers open at
the extracted release point, target fixture (bowl/box/stand) placed at the
release location.

## Scaling axes (cheap multipliers)

1. **Object swap at the same location** — apple → orange → ball → cube → can
   at identical pose; each is a new episode with a new prompt for free.
   Curate ~10 graspable assets (existing omniverse set + sphere/cylinder
   prims + a few new meshes).
2. **Clip variety** — 6 visible buckets (front/side × low/medium/high) ×
   many takes/actors × `_M` mirrors. `behind_*` clips: drop in v0 (object
   invisible to `ego_view`); revisit later with a wrist/rear camera or keep
   only for proprio-heavy ablations.
3. **Visual DR** — table/stand texture (exists), lighting, floor/background
   materials, distractor objects placed *away* from the reach point (also
   forces vision to pick the right object).
4. **Prompt paraphrases** — templated from the name grammar + object noun,
   k paraphrases per episode.
5. **Spatial micro-jitter** — ±1–2 cm object offsets within grasp tolerance,
   so vision isn't pixel-locked (motion is open-loop, so jitter must stay
   inside the tolerance the success gate proves).

## Measured feasibility (2026-08-03 sweep)

Concern raised: most pick_up motions are hard (behind reaches, tip-toe highs,
floor-level lows) and may not be trackable. Two data points say otherwise:

1. **G1 prior art**: `g1_onnx_executed/corpus2k/corpus_feasibility_report.csv`
   contains 45 pick_up clips executed through the G1 SONIC ONNX — 44/45
   feasible, 43 CLEAN/prime (upper local err mostly 15–45 mm), incl.
   behind_low and front_low.
2. **X2 sweep (this repo, deployed ONNX + walk_101 tuning)**: 73 clips
   stratified 8-per-bucket through `eval_x2_mujoco_onnx.py` headless:

   | bucket | clean | fell |
   |---|---|---|
   | behind low/med/high | 24 | 0 |
   | front low/med/high | 25 | 0 |
   | right_side med/high | 16 | 0 |
   | right_side_low | 5 | 3 |
   | **total** | **70 (96%)** | 3 |

   All 3 falls are `two_hands_*_right_side_low` (deep asymmetric sideways
   squat). Sweep artifacts: `analysis/sweep_reasons.txt`, per-clip `analysis/qpos_dumps/`, summary `analysis/pickup_sweep_reach_2026-08-03.csv`.

Note "clean" = stable execution, not mocap-faithful tracking — but the
object-follows-hand design only *needs* stability: if the policy squats
shallower than mocap or skips a toe-stand, pass A measures where the executed
hand actually went and the stand height absorbs the difference. Sloppy
tracking relocates the object; only falls (≈4%) kill episodes. Tracking-
fidelity distribution (executed reach extents per bucket) still gets measured
in pass A and folds into per-bucket stand-height stats.

Rough volume: 1,098 pick_up clips × ~90% stable × 5 objects ≈
**4–5k tier-R episodes** before DR or put_down — minus the `behind_*` camera-
visibility cut (~35%) still leaves ~3k.

## Kitchen edition (IsaacLab splat world) — feasibility audit 2026-08-03

Direction change (user): skip RoboCasa scenes; bake object+stand into the
x2-kitchen-sim gaussian-splat kitchen (IsaacLab/NuRec). Prefer real surfaces:
countertop where the executed reach height fits, floor stand below that.

**Executed reach z vs kitchen surfaces** (73-clip sweep FK × collision-mesh
surface extraction, `analysis/kitchen_surfaces/kitchen_surfaces.json`):

| reach band (executed) | buckets | matching real surface (z, floor=0) |
|---|---|---|
| 0.28–0.45 m | *_low | sofa 0.21/0.28, bench 0.445 — else floor stand |
| 0.59–0.73 m | front_low hi-end | east console table 0.732 |
| 0.85–1.11 m | *_medium | main L-counter 0.876, sink counter 0.936, pantry 0.95 |
| 1.22–1.44 m | *_high | pantry shelf 1.29 (walkable-adjacent); 1.48 bookcase |

Main counter has 1.42 m² of walkable standing band (nav_grid-verified);
placement spots with clearance radii are in the JSON. Spawn selection per
clip: pick (x,y,yaw) from the standing band so the body-frame reach offset
lands on the chosen surface at matching z.

**Execution routes in IsaacLab** (updated 2026-08-03 after live testing):
- ~~Bridge route~~ **DEAD for clip playback**: `x2_isaaclab_bridge.py
  --sonic-onnx` crash-loops instantly (physics explosion in the post-RSI
  settle steps; falls within 0.3 s) — in the kitchen, on flat ground, and
  even with `--motion default`. This is the *known* shelved failure from the
  nav M1 session (commit 6be7b3a, ledger #1: "unresolved assembly gap");
  the original audit missed that. Bridge stays useful only for its
  standalone statue gate (PASSes: pelvis drift 0.2 cm) and future DDS
  deploy-parity checks.
- Manager-env route **WORKS — the only route**: `eval_x2_isaacsim_onnx
  --run-dir <demo_v2 run> --onnx ~/.cache/sonic/x2/sonic_policy/x2_sonic_policy.onnx`
  (deploy ONNX is fused obs[1670]→action[31]; shim attaches clean) with
  `+headless=True +run_once=True`,
  `++manager_env.config.world_usd/world_collision_usd`,
  `++manager_env.commands.motion.motion_lib_cfg.motion_file=<single-clip pkl>`,
  `++manager_env.commands.motion.start_from_first_frame=true`. Verified: the
  dishwasher-placed front_low clip executes cleanly, 10+ consecutive
  episodes, zero falls. Gotchas: `filter_motion_keys` must stay null (`[]`
  matches nothing → "Loaded 0 motions"); without `+run_once=True` the eval
  loop replays forever; video needs `manager_env/recorders=render` +
  `++manager_env.config.render_results=true` (creates eval_camera);
  trajectory dump via `++manager_env.recorders.trajectory._target_=
  gear_sonic.envs.manager_env.mdp.recorders.TrajectoryRecorderCfg` +
  `.save_path=...` (dof_pos + root pose at 25 Hz). Object/stand baking via
  `config.add_object` / `config.add_table` still to be exercised.
- World placement: `world_pos` translates the world but has **no yaw** —
  instead rotate/translate the clip pkl itself (`root_trans_offset`,
  `root_rot`, and `pose_aa[:,0]` together) so frame 0 sits at the chosen
  spawn. Done for the dishwasher spot (`front_low_002_A506_dishwasher.pkl`
  recipe in scratch; kitchen frame == world frame with world_pos 0).
- ~~splat-in-camera mystery~~ **SOLVED (user caught it)**: the splat renders
  fine through camera render products — sensor-path rendering was formally
  proven in nav gate G1 (`gear_sonic/envs/nav_house/poc_validation.md`,
  proof `x2-kitchen-sim/media/poc_g1_sensor_frame.png`; the recorder
  already carries G1's stale-buffer fix `cam.update(force_recompute=True)`).
  The "splat-less" frames were a **world placement bug**: the manager env's
  env_0 origin sits at ≈(-19.99, -75.96), NOT (0,0) — the kitchen must be
  placed with `'++manager_env.config.world_pos=[-19.99,-75.96,0.0]'` (the
  measured offset; full working command in
  `x2_upgraded_demo/kitchen-sim-training.md`). Without it the robot spawns
  ~78 m from the kitchen on bare terrain. All references to kitchen frame
  == world frame in this plan mean the kitchen frame — env-relative
  coordinates are unchanged; only add world_pos to every kitchen run.
  Also: default `eval_camera_offset=[2,2,1]` from the dishwasher spot puts
  the camera inside wall/ceiling splat fuzz — use an interior viewpoint
  (e.g. `[1.2,-2.2,0.6]`) and `++manager_env.commands.motion.debug_vis=false`
  to hide the yellow tracking markers for VLA frames (marker-hygiene note in
  x2-kitchen-navigation.md §4). The `eval_camera_type` knob added earlier is
  moot for this bug but kept (harmless, default unchanged).

**Deltas vs the MuJoCo plan**:
1. Pass A must run in IsaacLab (PhysX ≠ MuJoCo trajectories). Trajectory
   recorder ≡ qpos dump; extraction code unchanged.
2. ~~GPU PhysX is NOT proven bit-exact run-to-run~~ **GATE PASSED
   2026-08-03**: same placed clip run twice through the manager-env route,
   trajectory dumps compared — max |diff| = 0.0 over all 180 frames
   (dof_pos, root_pos_w, root_quat_w). PhysX is bit-exact run-to-run here,
   same as MuJoCo. Pass A → pass B placement holds exactly.
3. LeRobot writer doesn't exist for IsaacLab — build offline converter:
   trajectory.pkl + rendered ego frames → `Gr00tDataExporter` episodes
   (state 51-D, motion tokens computed offline from the reference clip).
4. Splat has no physics; collision USD has known multi-robot sink-through
   (issue #8) — single-robot eval is the supported case.

**M0-kitchen status (2026-08-03)**:
1. ✅ Clip playback in IsaacLab: dishwasher-placed front_low clip executes
   cleanly via manager-env route (10+ episodes, zero falls; 1080p mp4 in
   scratch `render_m0/000000.mp4`). Bridge route abandoned (see above).
2. ✅ PhysX determinism A/B: bit-exact (0.0 over 180 frames).
3. ✅ Spawn placement: robot spawns exactly at (-0.311, 2.071) yaw 2.004
   (dishwasher standing spot), root drift ≤6 cm over the episode.
4. ✅ Trajectory recorder: dof_pos + root pose @25 Hz, and it already
   records `object_pos_w`/`table_pos_w` — the demo_v2 env instantiates
   object+table rigid bodies, so pass-B baking hooks are live.
5. ✅ Robot-in-kitchen render (after the world_pos fix, see above): X2 at
   the sink/dishwasher counter executing the reach, photoreal splat all
   around, headless sensor-path render — see
   `analysis/m0_kitchen/` (m0_dishwasher_reach_kitchen.mp4, frames,
   determinism A/B trajectory dumps, placed clip pkl). Working full
   command (add to every kitchen run):
   `'++manager_env.config.world_pos=[-19.99,-75.96,0.0]'` +
   `'++manager_env.config.eval_camera_offset=[1.2,-2.2,0.6]'`.
6. ✅ **Pass-B PoC on the countertop (2026-08-03)**: object baked at the
   extracted reach point on the sink counter; executed hand lands on it.
   Clip `big_light_one_hand_pick_up_front_medium_R_001__A510` (right hand),
   trimmed 0.6 s (drops a low fwd-swing that would clip the counter lip),
   spawn (-0.536, 2.246) yaw 104°, object = Lightwheel "mug.usd" (actually
   a wooden mug-tree, 0.30 m) from
   `g1-kitchen-sim/assets/nvidia_kitchen/kitchen_assets_usd/` at kitchen
   (-0.563, 2.733, 0.876). Artifacts: `analysis/m0_kitchen/passB_mug/`
   (mp4, frames, trajectories, placed clip, FK tools + reference-arc FK
   cache). Key machinery/lessons:
   - Static prop via new config keys `++manager_env.config.prop_usd` +
     `prop_pos` (kitchen frame; code composes world_pos — patch:
     `patches/modular_tracking_eval_camera_and_prop.patch`). Avoids the
     add_object/motion-object-channel machinery (which requires object
     trajectories in the motion pkl and crashes without them).
   - **Screen clips on REFERENCE arcs, not executed MuJoCo dumps** (the
     A525_M detour: MuJoCo execution slop looked like a counter-height
     reach; the reference was a 1.55 m shelf pick — IsaacLab tracks the
     reference faithfully). Reference FK cache for the 38 stable one-hand
     sweep clips: `passB_mug/ref_wrist_cache.npz`.
   - IsaacLab under-extends reaches by ~0.2 m vs reference; SONIC is
     proprioceptive-blind + deterministic, so compensate by shifting the
     spawn along facing (measure executed reach in pass A, shift, rerun).
   - Eval needs: termination nulls (ee_body_pos, foot_pos_xyz,
     anchor_ori_full) + DR event nulls (5 events) for nominal deploy-like
     dynamics.
   - Counter-compatible clips are SCARCE: only ~2 front-facing one-hand
     clips (of 38 stable) have reference arcs that stay above counter
     height with usable over-edge reach. Options to scale: screen the full
     1,098-clip corpus; accept counter-edge placements; use *_high clips
     for the pantry shelf (references reach 1.3-1.55 m); or teleop for
     counter-dense coverage.
   Remaining polish: un-trimmed approach variant (smaller trim), nicer
   object (kettle/mustard), `debug_vis=false` for VLA frames, tier-R gate
   automation (wrist-to-object distance from the trajectory).

## Milestones

- **M0 — single-clip PoC (thin thread)**: `small_light_one_hand_pick_up_front_low_R_002__A504`
  → pass A extractor → apple on auto-height stand → pass B recorded episode →
  plays back in `view_x2_recorded_dataset.sh`, palm-to-object gate passes.
  Proves determinism assumption + the whole spine.
- **M1 — teleop-sourced counter corpus (REVISED 2026-08-03, user
  direction)**: boneseed clips are counter-poor (~2 viable front-facing
  arcs — they were mocapped in open space), so counter data comes from
  teleop; the deterministic-replay machinery makes each demo a multiplier.
  - **Record** (MuJoCo, quest3 stack + `x2_dataset_recorder`): counter-
    height fixture (0.876/0.936 m) + Lightwheel MJCF objects — the set at
    `g1-kitchen-sim/assets/nvidia_kitchen/kitchen_assets/objects_extracted/
    lightwheel/` is ALREADY MuJoCo-format (69 categories × variants, pot/
    kettle/saucepan/glass_cup/mustard verified loading) with USD twins in
    `kitchen_assets_usd/` for IsaacLab replay. Scene graft via the
    `build_x2_robocasa_scene_xml.py` pattern.
  - **Spawn diversity without repositioning teleop**: robot RSI'd at
    scripted (x,y,yaw) per demo — operator only demonstrates manipulation.
    Per task record ~5-8 demos at varied body-to-object offsets within the
    ~30 cm spawn disc; replay-scale each laterally along the counter
    (object slides with it — open-loop replay shifts hand endpoint 1:1).
    Valid spawn region per demo computed from its recorded hand arc:
    counter footprint eroded by reach offset ∩ standable esdf band.
  - **Grasping from day one (tier G)**: teleop records the 10-DOF finger
    channel natively; replay open-loop; gate on object z-rise + grasp
    contacts; light objects first (3 Nm fingertips), log yield per
    object; reach-tier fallback for failures.
  - **Replay/render**: each demo = a clip through the proven pass-A/B
    pipeline → splat kitchen, USD-twin object baked at matched pose,
    deterministic execution, render + trajectory + gate → episodes.
  - Volume: ~50 demos × ~12 lateral placements × ~8 objects ≈ 5k episodes
    before jitter/DR/paraphrases. Then fine-tune GR00T
    (`train_groot_vla.sh`) and closed-loop eval via
    `run_x2_vla_runtime.sh` as before.
  - Boneseed clips remain the source for non-counter tiers: floor stands
    (*_low), pantry shelf (*_high references reach 1.3-1.55 m), and
    generic reach variety.
- **M2 — grasp tier**: finger overlay + contact-based success gating; yield
  report per bucket×object; keep survivors.
- **M3 — base-model scale-up**: full feasible pick_up set + put_down/place,
  object library ~10, DR + paraphrases; train the "X2 VLA base" checkpoint;
  compare against the n17 teleop soda-can baseline on its own task.
- **M4 — splat worlds (stretch)**: port pass-B scenes to IsaacLab — NuRec
  kitchen splat + `RigidObjectCfg` objects at the same extracted world poses,
  robot driven via `x2_isaaclab_bridge.py`, cameras per the proven gate-G1
  composite. MuJoCo cannot render splats, so photoreal variants are
  IsaacLab-only; do this only after M3 shows the synthetic data trains.

## Risks & mitigations

1. **Pass A/B determinism** — same seed/tuning should reproduce the rollout;
   M0 verifies (record palm trace twice, diff). Fallback: iterate placement
   once (measure in pass B, nudge object, re-run).
2. **Contact perturbs tracking (tier G)** — object contact changes dynamics
   vs pass A. Light objects + reach-tier fallback; success gate catches the rest.
3. **Camera visibility** — gate every episode on the object projecting into
   `ego_view` with margin; `front_cam` witness stays in the dataset.
4. **Open-loop motion ⇒ weak visual grounding** — the model could ignore
   pixels. Mitigate with object swaps, distractors, micro-jitter, and 6
   location buckets so language+vision must select the motion family. Accept
   that this is a *base* model — real teleop data on top restores closed-loop
   behavior.
5. **Token-channel fidelity** — actions must carry `motion_token(64)` like
   teleop episodes; feed the boneseed reference through the same SONIC
   tokenizer path the recorder uses (`--sonic-tokenizer-device cpu` path in
   the quest3 stack / `live_vla_publish_motion_token.py` precedent).

## Open questions (decide at M0/M1)

- Grasp-event detector: pure kinematic heuristic vs name-tag-guided — check
  extraction quality on ~20 hand-labeled clips first.
- Stand vs floating-object-only ablation (does the fixture matter to the VLA?).
- Whether put_down episodes need a "start holding" weld or a scripted
  pre-grasp splice from a matching pick_up clip.
- Two-hands clips: object sizing so both palms land on it (big_* boxes) —
  probably M2+.
