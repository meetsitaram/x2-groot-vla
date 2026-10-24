#!/usr/bin/env bash
# After a take: manifest -> stop rig -> head-camera kinematic replay -> overlay
# (intro + HUD + head inset) -> paced video.   ./gemma_ctl/finish_take.sh <bundle>
set -uo pipefail
B=$1; G=/home/stickbot/projects/x2-groot-vla; cd "$G"
PY=~/miniconda3/envs/env_isaaclab/bin/python
log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$B/finish.log"; }
until grep -q AGENT_EXIT "$B/agent.log" 2>/dev/null; do sleep 3; done
log "take finished"; python3 gemma_ctl/bundle.py "$B" --note "door start, fixed task mug_to_table" >> "$B/finish.log" 2>&1
P=$(cat "$B/run.pid"); kill "$P" 2>/dev/null; for i in $(seq 1 30); do kill -0 "$P" 2>/dev/null || break; sleep 1; done
kill -0 "$P" 2>/dev/null && kill -9 "$P"; sleep 3; log "rig stopped"
RENDER_W=640 RENDER_H=360 KP_GEMMA_REC_CAM=head_wide KP_GEMMA_CAPTION=0 KP_GEMMA_REPLAY=$B/trajectory.jsonl \
  KP_GEMMA_SCENE=$G/gemma_ctl/kitchen_scene_v1.json RUN=$B/replay_head ./gemma_ctl/launch_kitchen_gemma.sh >> "$B/finish.log"
until grep -qE "REPLAY finished|Traceback" "$B/replay_head/rig.log" 2>/dev/null; do sleep 10; done
log "head replay: $(grep -cE 'REPLAY finished' "$B/replay_head/rig.log") finished, $(ls "$B/replay_head/frames" | wc -l) frames"
P=$(cat "$B/replay_head/run.pid"); kill "$P" 2>/dev/null; sleep 6; kill -0 "$P" 2>/dev/null && kill -9 "$P"
$PY gemma_ctl/overlay_video.py "$B" "$B/live" --intro --head "$B/replay_head" >> "$B/finish.log" 2>&1 && log "overlay done"
$PY gemma_ctl/make_paced.py "$B" frames_ov 4 >> "$B/finish.log" 2>&1 && log "paced done"
ffmpeg -y -loglevel error -i "$B/frames_ov/intro_1.jpg" -vf scale=1280:-2 -q:v 4 "$B/review/paced.jpg"
log "DONE"
