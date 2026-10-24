"""Executor-side palm IK on the X2 MJCF, seeded with the rig's MEASURED qpos.

Natural-posture DLS: the primary task is the palm position; the nullspace
pulls the arm toward a forward-reach posture. (Unregularised IK produced
contorted solutions, e.g. shoulder roll -2.3 rad, which SONIC refuses to
track: it reaches with its own natural arm and lands 10-20 cm off.)
"""
from __future__ import annotations

import os

import numpy as np

WBC = os.environ.get("GEMMA_WBC", "/home/stickbot/Projects/GR00T-WBC-gemma")
MJCF = WBC + "/gear_sonic/data/assets/robot_description/mjcf/x2_ultra.xml"
PALM_OFF = np.array([0.0, 0.0, -0.10])
ARM_DOF = {"left": list(range(21, 28)), "right": list(range(28, 35))}
WAIST_PITCH_QADR = 7 + 13
WRIST = {"left": "left_wrist_roll_link", "right": "right_wrist_roll_link"}
# forward reach at chest height (MJ order: sh pitch, roll, yaw, elbow, wr yaw, pitch, roll)
NOMINAL = {"right": np.array([-0.9, -0.15, 0.0, -1.0, 0.0, 0.0, 0.0]),
           "left": np.array([-0.9, 0.15, 0.0, -1.0, 0.0, 0.0, 0.0])}


class PalmIK:
    def __init__(self):
        import mujoco
        self.mj = mujoco
        self.m = mujoco.MjModel.from_xml_path(MJCF)
        self.d = mujoco.MjData(self.m)

    def palm(self, qpos, side):
        self.d.qpos[:] = qpos
        self.mj.mj_kinematics(self.m, self.d)
        b = self.m.body(WRIST[side]).id
        return self.d.xpos[b] + self.d.xmat[b].reshape(3, 3) @ PALM_OFF

    def solve(self, qpos, target, side="right", waist_pitch=None, iters=300,
              null_gain=0.15):
        m, d, mj = self.m, self.d, self.mj
        d.qpos[:] = np.asarray(qpos, float)
        if waist_pitch is not None:
            d.qpos[WAIST_PITCH_QADR] = waist_pitch
        adr = ARM_DOF[side]
        qadr = [a + 1 for a in adr]
        lo = m.jnt_range[[a - 5 for a in adr], 0] + 0.02
        hi = m.jnt_range[[a - 5 for a in adr], 1] - 0.02
        d.qpos[qadr] = NOMINAL[side]
        b = m.body(WRIST[side]).id
        jacp = np.zeros((3, m.nv))
        target = np.asarray(target, float)
        for _ in range(iters):
            mj.mj_kinematics(m, d)
            mj.mj_comPos(m, d)
            palm = d.xpos[b] + d.xmat[b].reshape(3, 3) @ PALM_OFF
            err = target - palm
            mj.mj_jac(m, d, jacp, None, palm, b)
            J = jacp[:, adr]
            JJ = J @ J.T + 0.03 ** 2 * np.eye(3)
            Jp = J.T @ np.linalg.inv(JJ)
            q = d.qpos[qadr]
            dq = Jp @ err + (np.eye(7) - Jp @ J) @ (null_gain * (NOMINAL[side] - q))
            d.qpos[qadr] = np.clip(q + np.clip(dq, -0.15, 0.15), lo, hi)
            if np.linalg.norm(err) < 0.003 and np.linalg.norm(NOMINAL[side] - q) < 1e-3:
                break
        palm = self.palm(d.qpos.copy(), side)
        return d.qpos[qadr].copy(), float(np.linalg.norm(target - palm))


if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0, WBC)
    from gear_sonic.scripts.pc2_kplanner_onnx import _load_warmup_qpos
    q0 = np.asarray(_load_warmup_qpos(Path(WBC + "/"
                    "gear_sonic/data/motions/kplanner_idle_anchor_g1teleop_v3.pkl")), float)
    q0[0:2] = 0; q0[3:7] = [1, 0, 0, 0]
    ik = PalmIK()
    for wp in (None, 0.25):
        for fx, fy, z in [(0.25, -0.18, 0.95), (0.32, -0.18, 0.95), (0.38, -0.15, 0.92), (0.30, -0.2, 0.80)]:
            q, e = ik.solve(q0, [fx, fy, z], waist_pitch=wp)
            print(f"waist {wp} tgt {fx},{fy},{z}: err {e:.3f} q {np.round(q, 2).tolist()}")
