"""Run the MuJoCo sim stack for the Gemma chat and RESET it automatically after a collapse.

Launches simstack_local.sh (GR00T-WBC-X2-supermimic) without its own viewer, plus the follow-camera
viewer. Watches the deploy log; on a [FATAL] (tilt watchdog / SAFE_HOLD = the robot fell):

  FAST RESET (~5 s, the repo's own fall-recovery rehearsal, docs/x2/F10 "Fall recovery"):
    the bridge's ElasticBand (scene_reset hook, :5560) lifts the robot where it lies (knees bent),
    the deploy's SAFE_HOLD -> RECOVER stand-up runs with sim-shortened timings, the band eases to
    stand height and lets go -> CONTROL.
  FALLBACK: if a step stalls, stop + relaunch the stack (~40 s).

State for the chat page goes to runs/gemma/real/sim_state.json.

  python -m gemma_ctl.sim_guard                      # SuperMimic 320k (default)
  python -m gemma_ctl.sim_guard --model sonic        # SONIC incumbent ft8_w18 45000
Ctrl-C stops the guard AND the stack.
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import time
from pathlib import Path

G = Path(__file__).resolve().parents[1]
REPO = Path("/home/stickbot/Projects/GR00T-WBC-X2-supermimic")
CLIPS = G / "runs/gemma/real/clips_merged"
STATE = G / "runs/gemma/real/sim_state.json"
# sim-only RECOVER timings (deploy defaults 2.0 / 1.5 / 4.0 s, 0.4 rad/s, 3 attempts)
RECOVER_FLAGS = ["--recover-dwell-s", "0.4", "--recover-stiffen-s", "0.4", "--recover-standup-s", "1.2",
                 "--recover-max-rate", "3.0", "--recover-max-attempts", "0"]
# this worktree's own colcon volumes (as sim_posture_bench.sh): built from ITS source, which has the
# SAFE_HOLD -> RECOVER stand-up; the default docker_x2_* cache is an older build without it
COMPOSE_PROJECT = "x2sim_supermimic"
DEPLOY_LOG = Path.home() / ".cache/sonic/simstack_pc/logs/deploy.log"
MODELS = {
    "supermimic": dict(
        MODEL=str(REPO / "gear_sonic_deploy/models/x2_supermimic_bones_full_v1_320000.onnx"),
        X2_RITUAL_TUNING=str(REPO / "gear_sonic_deploy/configs/real_deploy_tuning/supermimic.yaml")),
    "sonic": dict(
        MODEL=str(Path.home() / "x2_eval_local/hf_ft8_45000/incumbent_pico_v16_w20_ft8_w18_it45000/"
                  "x2_sonic_v16ft8_45000_g1.onnx")),    # stack default tuning = waistmc preset
}


def state(**kw):
    s = json.loads(STATE.read_text()) if STATE.exists() else {}
    s.update(kw, t=time.time())
    STATE.write_text(json.dumps(s))


def log(msg):
    print(f"[guard {time.strftime('%H:%M:%S')}] {msg}", flush=True)


class Guard:
    def __init__(self, model, reset_after):
        self.model, self.reset_after = model, reset_after
        self.viewer = None
        self.resets = 0
        self._pub = None

    def stop_stack(self, viewer_too=False):
        # the follow viewer is a stateless mirror that reconnects to a restarted stack: keep it
        if viewer_too and self.viewer and self.viewer.poll() is None:
            self.viewer.kill()
        subprocess.run(["./gear_sonic/scripts/simstack_local.sh", "--stop"], cwd=REPO,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=90)

    def start_stack(self):
        env = dict(os.environ, SIMSTACK_NO_VIEWER="1", X2_WHOLE_BODY_TELEOP="0",
                   KPLANNER_DANCES_DIR=str(CLIPS), **MODELS[self.model],
                   # band present but released at once (needed for the fast reset)
                   SIMSTACK_SIM_FLAGS="--sim-init-pose stand-settled --sim-band-release-after-s 0",
                   COMPOSE_PROJECT_NAME=COMPOSE_PROJECT,
                   SIMSTACK_EXTRA_DEPLOY_ARGS=" ".join(f"--deploy-extra-arg {a}" for a in RECOVER_FLAGS))
        out = G / f"runs/gemma/real/stack_{time.strftime('%Y%m%dT%H%M%S')}.log"
        with open(out, "w") as f:
            p = subprocess.Popen(["stdbuf", "-oL", "-eL", "./gear_sonic/scripts/simstack_local.sh"],
                                 cwd=REPO, env=env, stdout=f, stderr=subprocess.STDOUT,
                                 stdin=subprocess.DEVNULL, start_new_session=True)
        for _ in range(90):
            txt = out.read_text(errors="ignore")
            if "VERDICT" in txt or p.poll() is not None:
                break
            time.sleep(1)
        ok = "deploy in CONTROL" in out.read_text(errors="ignore")
        if self.viewer is not None and self.viewer.poll() is None:
            return ok, out                       # one viewer window, reused across resets
        self.viewer = subprocess.Popen(
            [str(REPO / ".venv/bin/python"), "-u", "-m", "gemma_ctl.follow_viewer", "--debug-port", "5659"],
            cwd=G, stdout=open(G / "runs/gemma/real/follow_viewer.log", "w"), stderr=subprocess.STDOUT,
            start_new_session=True)
        return ok, out

    def bring_up(self, why):
        state(status="resetting", why=why, model=self.model, resets=self.resets)
        log(f"{why}: (re)starting the stack with {self.model} ...")
        self.stop_stack()
        ok, out = self.start_stack()
        state(status="up" if ok else "failed", why="", model=self.model, resets=self.resets)
        log(f"stack {'UP (CONTROL)' if ok else 'FAILED, see ' + str(out)}")
        return ok

    def band(self, **payload):
        """scene_reset hook on the bridge (simstack_hold_up.py's wire: PUB bind :5560, CONFLATE SUB)."""
        if self._pub is None:
            import sys as _s
            import zmq
            _s.path.insert(0, str(REPO))
            from gear_sonic.utils.teleop.zmq.scene_state_zmq import SCENE_RESET_TOPIC, pack_json
            self._pack, self._topic = pack_json, SCENE_RESET_TOPIC
            self._pub = zmq.Context.instance().socket(zmq.PUB)
            self._pub.bind("tcp://*:5560")
            time.sleep(0.8)
        for _ in range(3):
            self._pub.send(self._pack(self._topic, payload))
            time.sleep(0.05)

    def wait_log(self, pos, needle, timeout):
        t_end = time.monotonic() + timeout
        while time.monotonic() < t_end:
            with open(DEPLOY_LOG, errors="ignore") as f:
                f.seek(pos); txt = f.read()
            if needle in txt:
                return True
            time.sleep(0.1)
        return False

    def fast_reset(self, pos):
        """Stand the robot back up on the floor where it fell (stand_settled pose), steady it with the
        band at stand height while the deploy's SAFE_HOLD -> RECOVER hands back to CONTROL, then let go.
        Returns True when back in CONTROL."""
        t0 = time.monotonic()
        state(status="resetting", why="fast reset", model=self.model, resets=self.resets)
        self.band(robot_stand_here=True, band_point_here=True, band_length=-0.319, band_enable=True)
        if not self.wait_log(pos, "SAFE_HOLD -> RECOVER", 4.0):
            log("fast reset: RECOVER never started"); return False
        if not self.wait_log(pos, "RECOVER complete", 5.0):
            log("fast reset: RECOVER did not complete"); return False
        time.sleep(1.0)                       # CONTROL soft-start ramp with the band still steadying it
        self.band(band_enable=False)
        log(f"fast reset done in {time.monotonic() - t0:.1f} s (+{self.reset_after:.1f} s down)")
        state(status="up", why="", model=self.model, resets=self.resets)
        return True

    def run(self):
        self.bring_up("start")
        pos = DEPLOY_LOG.stat().st_size if DEPLOY_LOG.exists() else 0
        while True:
            time.sleep(0.5)
            if not DEPLOY_LOG.exists():
                continue
            size = DEPLOY_LOG.stat().st_size
            if size < pos:                       # log restarted with the stack
                pos = 0
            with open(DEPLOY_LOG, errors="ignore") as f:
                f.seek(pos); new = f.read(); pos = f.tell()
            fatal = [l for l in new.splitlines() if "[FATAL]" in l]
            if fatal:
                log("COLLAPSE: " + fatal[0].split("]: ", 1)[-1][:120])
                state(status="fallen", why="robot fell", model=self.model, resets=self.resets)
                time.sleep(self.reset_after)
                self.resets += 1
                mark = DEPLOY_LOG.stat().st_size
                if not self.fast_reset(mark):
                    self.band(band_enable=False)
                    self.bring_up(f"reset #{self.resets} (fallback restart)")
                time.sleep(1.0)                  # skip the recovery's own log lines
                pos = DEPLOY_LOG.stat().st_size if DEPLOY_LOG.exists() else 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=sorted(MODELS), default="supermimic")
    ap.add_argument("--reset-after", type=float, default=1.0, help="seconds to leave it down first")
    a = ap.parse_args()
    g = Guard(a.model, a.reset_after)

    def bye(*_):
        log("stopping guard + stack"); state(status="stopped")
        g.stop_stack(viewer_too=True); raise SystemExit(0)
    signal.signal(signal.SIGTERM, bye)
    try:
        g.run()
    except KeyboardInterrupt:
        bye()


if __name__ == "__main__":
    main()
