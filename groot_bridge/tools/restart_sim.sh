#!/usr/bin/env bash
# Restart the gwbc-x2 MuJoCo simstack (sim only; never contacts PC2) with OmniHands and a scene XML,
# then wait for the deploy verdict. Usage: groot_bridge/tools/restart_sim.sh <scene.xml> [log]
set -euo pipefail
SCENE="$(realpath "$1")"; LOG="$(realpath -m "${2:-$(dirname "$0")/../../runs/groot_bridge/simstack.log}")"
G="$HOME/Projects/gwbc-x2"
"$G/gear_sonic/scripts/simstack_local.sh" --stop > /dev/null 2>&1 || true
cd "$G"
X2_WHOLE_BODY_TELEOP=0 SIMSTACK_OMNIHAND=1 SIMSTACK_EXTRA_DEPLOY_ARGS="--sim-mjcf $SCENE" \
  SONIC_PYTHON="$HOME/Projects/GR00T-WholeBodyControl/.venv/bin/python" SIMSTACK_NO_VIEWER=1 SIM_VIEWER=0 \
  setsid nohup ./gear_sonic/scripts/simstack_local.sh > "$LOG" 2>&1 < /dev/null &
for _ in $(seq 1 100); do
  if grep -q "VERDICT" "$LOG" 2>/dev/null; then grep "VERDICT" "$LOG"; exit 0; fi
  if grep -qE "^FATAL|Traceback" "$LOG" 2>/dev/null; then tail -5 "$LOG"; exit 1; fi
  sleep 3
done
echo "timeout waiting for the simstack verdict"; exit 1
