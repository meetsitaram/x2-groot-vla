#!/usr/bin/env python
"""kplanner-only check of the `llm` command source
(patches/pc2_kplanner_llm_source.patch). Starts pc2_kplanner_onnx on isolated
58xx ports with a fake deploy x2_debug stream, then checks the reference
root speed on the pose wire:

  A  llm walk + heartbeat         -> moves
  B  llm silent (wifi drop)       -> stops within ~1.5 s
  C  llm walks, pad sends idle    -> pad preempts: stops, llm dropped
  D  >3 s after pad, llm walks    -> llm re-acquires: moves
  E  VR engaged, llm walks        -> dropped

  python -m gemma_ctl.test_llm_source
"""
from __future__ import annotations

import json
import os
import struct
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import zmq

REPO = Path(os.environ.get("GEMMA_WBC", "/home/stickbot/Projects/GR00T-WBC-gemma"))
sys.path.insert(0, str(REPO))
from gear_sonic.utils.pose_pipeline.wire import X2_DEBUG_HEADER_SIZE  # noqa: E402
from gear_sonic.utils.teleop.zmq.zmq_packed_message_decoder import unpack_message  # noqa: E402

PUB, CMD, CLIP, DBG = 5856, 5863, 5868, 5859
OUT = Path(__file__).resolve().parents[1] / "runs/gemma/test_llm_source"


def _debug_frame(body_q, tick):
    fields = [{"name": "tick", "dtype": "i64", "shape": [1]},
              {"name": "base_quat", "dtype": "f64", "shape": [4]},
              {"name": "body_q", "dtype": "f64", "shape": [31]}]
    hdr = json.dumps({"v": 1, "endian": "le", "count": 3, "fields": fields}).encode().ljust(X2_DEBUG_HEADER_SIZE, b"\0")
    return (b"x2_debug" + hdr + struct.pack("<q", tick) + np.array([1.0, 0, 0, 0]).tobytes()
            + np.asarray(body_q, np.float64).tobytes())


# (t_start, t_end, source, payload-or-None, label); payload re-sent every 0.3 s
SCRIPT = [
    (3.0, 6.0, "llm", {"intent": "walk", "magnitude": "forward"}, "A llm walk"),
    (6.0, 9.0, None, None, "B llm silent"),
    (9.0, 14.0, "llm", {"intent": "walk", "magnitude": "forward"}, "C llm walk"),
    # pad: single zero-stick message at 11.0 (operator touch), like the bridge
    (14.0, 15.0, None, None, "C' pad lockout tail"),
    (15.0, 18.0, "llm", {"intent": "walk", "magnitude": "forward"}, "D llm re-acquire"),
    (18.0, 21.0, "llm", {"intent": "walk", "magnitude": "forward"}, "E vr engaged"),
]
PAD_AT = 11.0
VR_AT = 18.0
WINDOWS = {  # label: (t0, t1, expect_moving)
    "A moving": (4.5, 6.0, True),
    "B stopped after silence": (8.6, 9.0, False),
    "C moving before pad": (10.0, 11.0, True),
    "C stopped after pad": (12.5, 14.0, False),
    "D re-acquired": (16.5, 18.0, True),
    "E VR blocks llm": (19.8, 21.0, False),
}
MOVING_MPS = 0.08


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    py = os.path.expanduser("~/miniconda3/envs/env_isaaclab/bin/python")
    from gear_sonic.scripts.pc2_kplanner_onnx import _load_warmup_qpos
    warm = REPO / "gear_sonic/data/motions/kplanner_idle_anchor_g1teleop_v3.pkl"
    anchor = np.asarray(_load_warmup_qpos(warm), np.float64)[7:38]
    log_f = open(OUT / "kplanner.log", "w")
    kp = subprocess.Popen([py, str(REPO / "gear_sonic/scripts/pc2_kplanner_onnx.py"),
        "--onnx", os.path.expanduser("~/x2_cloud_checkpoints/planner_onnx_g1core_s1d_a05_ws/x2_planner_template.onnx"),
        "--planner-mode", "slow_walk", "--cmd-bind", "--replan-threshold-frames", "48",
        "--warmup-qpos", str(warm), "--dances-dir", os.path.expanduser("~/x2_cloud_checkpoints/dances_x2m2"),
        "--ort-gpu", "--pub-host", "127.0.0.1", "--pub-port", str(PUB), "--cmd-port", str(CMD),
        "--clip-cmd-port", str(CLIP), "--arm-port", "0",
        "--x2-debug-host", "127.0.0.1", "--x2-debug-port", str(DBG), "--pid-file", str(OUT / "kplanner.pid")],
        stdout=log_f, stderr=subprocess.STDOUT, env=dict(os.environ, PYTHONUNBUFFERED="1"), cwd=str(REPO))
    ctx = zmq.Context.instance()
    dbg = ctx.socket(zmq.PUB); dbg.bind(f"tcp://127.0.0.1:{DBG}")
    cmd = ctx.socket(zmq.PUB); cmd.connect(f"tcp://127.0.0.1:{CMD}")
    pose = ctx.socket(zmq.SUB); pose.connect(f"tcp://127.0.0.1:{PUB}")
    pose.setsockopt_string(zmq.SUBSCRIBE, "pose")
    send = lambda d: cmd.send_multipart([b"planner_cmd", json.dumps(d).encode()])  # noqa: E731
    tick = 0
    t0 = time.monotonic()
    try:
        while time.monotonic() - t0 < 120:
            dbg.send(_debug_frame(anchor, tick)); tick += 1
            if kp.poll() is not None:
                print("kplanner died:\n" + open(OUT / "kplanner.log").read()[-2000:]); return 2
            if "planner active" in open(OUT / "kplanner.log").read() or "rebase ARMED" in open(OUT / "kplanner.log").read():
                break
            time.sleep(0.02)
        else:
            print("kplanner never armed"); return 2
        ts, xs = [], []
        t_start = time.monotonic(); last_tx = 0.0; pad_done = vr_done = False
        while (tau := time.monotonic() - t_start) < 21.5:
            dbg.send(_debug_frame(anchor, tick)); tick += 1
            if tau - last_tx >= 0.3:
                for (a, b, src, pl, _) in SCRIPT:
                    if a <= tau < b and src:
                        send({**pl, "source": src}); send({"intent": "llm_keepalive", "source": "llm"})
                last_tx = tau
            if not pad_done and tau >= PAD_AT:
                send({"intent": "locomotion", "magnitude": "continuous", "stick_fwd": 0.0,
                      "stick_side": 0.0, "stick_yaw": 0.0, "source": "pad"}); pad_done = True
            if not vr_done and tau >= VR_AT:
                vr_done = True
            if vr_done and int(tau * 2) != int((tau - 0.02) * 2):   # VR keepalive 0.5 s
                send({"intent": "idle", "magnitude": "default", "source": "vr"})
            while True:
                try:
                    raw = pose.recv(zmq.NOBLOCK)
                except zmq.Again:
                    break
                f = unpack_message(raw, expected_topic="pose").fields
                ts.append(tau); xs.append(np.asarray(f["root_xy_world"], np.float64).ravel()[:2])
            time.sleep(0.02)
    finally:
        send({"intent": "shutdown"}); time.sleep(1); kp.terminate(); log_f.close()
    t = np.asarray(ts); X = np.stack(xs)
    ok_all = True
    for name, (a, b, want) in WINDOWS.items():
        m = (t >= a) & (t <= b)
        if m.sum() < 5:
            print(f"  ??  {name}: no samples"); ok_all = False; continue
        sp = np.linalg.norm(X[m][-1] - X[m][0]) / (t[m][-1] - t[m][0])
        ok = (sp > MOVING_MPS) == want
        ok_all &= ok
        print(f"  {'PASS' if ok else 'FAIL'}  {name:28s} speed {sp:.3f} m/s (want {'moving' if want else 'stopped'})")
    owner_lines = [l.strip()[-90:] for l in open(OUT / "kplanner.log") if "cmd owner" in l]
    print("  kplanner owner log:"); [print("    " + l) for l in owner_lines]
    np.savez_compressed(OUT / "trace.npz", t=t, xy=X)
    print("RESULT:", "PASS" if ok_all else "FAIL")
    return 0 if ok_all else 1


if __name__ == "__main__":
    raise SystemExit(main())
