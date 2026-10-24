#!/usr/bin/env bash
# Kitchen rig + Gemma manipulation hooks (headless, recorded).
#   RUN=<dir>  output dir (log, frames)      default runs/gemma/kitchen/<stamp>
# Stop: by PID only (see run.pid).
set -euo pipefail
G=/home/stickbot/projects/x2-groot-vla
RUN="${RUN:-$G/runs/gemma/kitchen/$(date +%Y%m%dT%H%M%S)}"
mkdir -p "$RUN"
cd "${GEMMA_WBC:-/home/stickbot/Projects/GR00T-WBC-gemma}"
SONIC_DIR=~/x2_cloud_checkpoints/g1teleop_overnight/sonic/softland_173528
FS=~/x2_cloud_checkpoints/fixed_scratch
KITCHEN=~/projects/x2-kitchen-sim/assets/kitchen
export KP_PAD=1 KP_HIDE_TERRAIN=1 KP_GEMMA=1 PYTHONUNBUFFERED=1 KP_RING_S="${KP_RING_S:-1500}" KP_PAD_PORT="${KP_PAD_PORT:-7563}" KP_SPAWN="${KP_SPAWN:-}" \
  KP_GEMMA_SCENE="${KP_GEMMA_SCENE:-$G/gemma_ctl/kitchen_scene.json}" \
  KP_GEMMA_REC_DIR="$RUN/frames" KP_OBS_DUMP_DIR="$RUN"
unset KP_HEAD_CAM
nohup ~/projects/g1-kitchen-sim/.venv/bin/python -u gear_sonic/scripts/x2-navigation/run_x2_kplanner_env2.py \
  --run-dir "$SONIC_DIR" \
  --onnx "$SONIC_DIR/exported/softland_4800_g1.onnx" \
  --vqvae-ckpt "$FS/vqvae/model-step=0300000.ckpt" \
  --pose-ckpt "$FS/pose_500k/model-step=0500000.ckpt" \
  --root-ckpt "$FS/root/model-step=0300000.ckpt" \
  --planner-mode slow_walk \
  +num_envs=1 +headless=True \
  ++manager_env.config.enable_cameras=true \
  ++manager_env.config.render_results=true \
  "++manager_env.config.eval_camera_data_types=[rgb,distance_to_image_plane]" \
  ++manager_env.config.render_width=${RENDER_W:-1920} ++manager_env.config.render_height=${RENDER_H:-1080} \
  ++manager_env.commands.motion.motion_lib_cfg.motion_file=${SPAWN_CLIP:-gear_sonic/data/motions/x2_sonic_feasible_stand_single.pkl} \
  ++manager_env.config.world_usd="$KITCHEN/kitchen_splat.usdz" \
  ++manager_env.config.world_collision_usd="$KITCHEN/kitchen_collision.usd" \
  '++manager_env.config.world_pos=[-19.99,-75.96,0.0]' \
  ++manager_env.terminations.ee_body_pos=null \
  ++manager_env.terminations.foot_pos_xyz=null \
  ++manager_env.terminations.anchor_ori_full=null \
  ++manager_env.commands.motion.debug_vis=false \
  > "$RUN/rig.log" 2>&1 &
echo $! > "$RUN/run.pid"
echo "rig pid $(cat "$RUN/run.pid")  log $RUN/rig.log"
