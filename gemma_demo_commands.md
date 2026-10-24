# X2 kitchen sim + Gemma — run commands

## Terminal 1 — start the sim rig + live viewer

```bash
docker stop x2-gemma-rig 2>/dev/null; docker rm x2-gemma-rig 2>/dev/null
cd ~/projects/agibot_x2/x2-groot-vla && \
FAST=1 RUN=$PWD/runs/gemma/bundles/kitchen_fast_$(date +%Y%m%dT%H%M%S) ~/gemma_setup/launch_kitchen_gemma_thor.sh && \
B=$(cat ~/gemma_setup/current_bundle) && \
until grep -q wall_s $B/rig.log 2>/dev/null; do sleep 5; echo "waiting for rig..."; done && \
echo "RIG READY" && ~/gemma_setup/.venv/bin/python ~/gemma_setup/view_rig.py
```

Takes ~2 min to boot; the viewer window opens once the rig is ready.
Drop `FAST=1` for full-quality (1920x1080, ~3x slower) rendering.

## Terminal 2 — talk to Gemma (start after "RIG READY")

```bash
cd ~/projects/agibot_x2/x2-groot-vla && B=$(cat ~/gemma_setup/current_bundle) && mkdir -p $B/live_chat && \
GEMMA_SURVEY_TIMEOUT_S=1800 GEMMA_VERIFY_RELABEL=0 GEMMA_WBC=/home/thor/projects/GR00T-WBC-gemma GEMMA_RUN_DIR=$B/live_chat \
~/gemma_setup/.venv/bin/python -u -m gemma_ctl.agent --perception --live --model gemma4:26b --transcript $B/transcript_chat.json
```

Type at `user>`, e.g. `pick up the black mug and put it on the white table`.
First request triggers a ~2 min room survey. Ctrl+D / Ctrl+C to quit.

## Stop

```bash
docker stop x2-gemma-rig
```

Run outputs (rig.log, frames, transcripts): the folder printed by `cat ~/gemma_setup/current_bundle`.
