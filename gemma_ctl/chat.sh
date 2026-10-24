#!/usr/bin/env bash
# Interactive Gemma chat with the X2 in the kitchen sim.
#   ./gemma_ctl/chat.sh            start a fresh rig + bundle, then the chat prompt
#   ./gemma_ctl/chat.sh --attach   chat with the rig that is already running (latest bundle)
# Everything is recorded into runs/gemma/bundles/chat_<stamp>/ (render later with
# overlay_video.py + make_paced.py). Type 'stop' any time; Ctrl-D to quit.
set -euo pipefail
G=/home/stickbot/projects/x2-groot-vla
cd "$G"
if [[ "${1:-}" == "--attach" ]]; then
  B=$(ls -dt runs/gemma/bundles/*/ | head -1); B=${B%/}
  echo "attaching to running rig, bundle $B"
else
  if pgrep -f "run_x2_kplanner_en[v]2" >/dev/null; then
    echo "A kitchen rig is already running. Use --attach, or stop it first." >&2; exit 1
  fi
  B=$G/runs/gemma/bundles/chat_$(date +%Y%m%dT%H%M)
  KP_GEMMA_CAPTION=0 KP_GEMMA_SCENE=$G/gemma_ctl/kitchen_scene_v1.json RUN=$B ./gemma_ctl/launch_kitchen_gemma.sh
  echo -n "starting the sim (about a minute) "
  until [ "$(grep -c wall_s "$B/rig.log" 2>/dev/null || echo 0)" -ge 3 ]; do echo -n "."; sleep 3; done; echo " ready"
fi
echo
echo "Talk to Gemma. Try: what do you see? / pick up the black mug next to the stove and put it on the white table"
GEMMA_RUN_DIR=$B/live ~/miniconda3/envs/env_isaaclab/bin/python -u -m gemma_ctl.agent \
  --perception --live --model gemma4:26b --transcript "$B/transcript.json"
echo "session saved in $B"
