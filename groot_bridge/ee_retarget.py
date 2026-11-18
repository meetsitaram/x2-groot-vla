"""Hand-space (end-effector) arm retargeting between two humanoids, e.g. G1 <-> X2.

Why: copying joint angles keeps the arm *shape* but not where the hands go. The X2's shoulders are
wider and higher than the G1's, so the G1 BoxLift grasp pose copied joint-for-joint puts the X2 wrists
0.448 m apart instead of 0.234 m (groot_bridge/g1_x2.py docstring). Two-handed tasks need the hands
where the source robot put them.

Map:  source arm joints --FK(source)--> wrist pose in the pelvis frame
      --(fixed per-side frame alignment)--> target wrist pose --IK(target, damped least squares)--> target joints

The per-side alignment is measured once at a reference pose (arms hanging, elbows bent 90 deg on
both robots), so wrist orientation conventions of the two models never need to match.
"""
from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

ARM = ["shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow", "wrist_roll", "wrist_pitch", "wrist_yaw"]


@dataclass
class ArmModel:
    """One robot, one side: joint names in a fixed semantic order (ARM), the wrist body, the pelvis body."""
    model: mujoco.MjModel
    joints: list[str]
    wrist_body: str
    pelvis_body: str = "pelvis"
    tcp_local: np.ndarray | None = None      # control point in the wrist frame (e.g. palm centre); None = wrist origin

    def __post_init__(self):
        m = self.model
        self.data = mujoco.MjData(m)
        self.qadr = np.array([m.jnt_qposadr[m.joint(j).id] for j in self.joints])
        self.dadr = np.array([m.jnt_dofadr[m.joint(j).id] for j in self.joints])
        self.lo = np.array([m.jnt_range[m.joint(j).id][0] for j in self.joints])
        self.hi = np.array([m.jnt_range[m.joint(j).id][1] for j in self.joints])
        self.wid = m.body(self.wrist_body).id
        self.pid = m.body(self.pelvis_body).id
        self.tcp = np.zeros(3) if self.tcp_local is None else np.asarray(self.tcp_local, float)
        self.data.qpos[:] = m.qpos0
        if m.jnt_type[0] == mujoco.mjtJoint.mjJNT_FREE:
            self.data.qpos[:7] = [0, 0, 1.0, 1, 0, 0, 0]

    def fk(self, q):
        """Control-point position (3) and wrist rotation (3x3) in the pelvis frame."""
        d = self.data
        d.qpos[self.qadr] = q
        mujoco.mj_kinematics(self.model, d)
        Rp = d.xmat[self.pid].reshape(3, 3)
        p = Rp.T @ (d.xpos[self.wid] + d.xmat[self.wid].reshape(3, 3) @ self.tcp - d.xpos[self.pid])
        R = Rp.T @ d.xmat[self.wid].reshape(3, 3)
        return p, R

    def ik(self, p_tgt, R_tgt, q0, iters=60, w_rot=0.25, damping=0.03, tol=2e-3):
        """Damped-least-squares IK for this arm's 7 joints; returns (q, position error m, rotation error rad)."""
        m, d = self.model, self.data
        q = np.clip(np.asarray(q0, float).copy(), self.lo, self.hi)
        jacp = np.zeros((3, m.nv)); jacr = np.zeros((3, m.nv))
        for _ in range(iters):
            d.qpos[self.qadr] = q
            mujoco.mj_kinematics(m, d); mujoco.mj_comPos(m, d)
            Rp = d.xmat[self.pid].reshape(3, 3)
            Rw = d.xmat[self.wid].reshape(3, 3)
            pw = d.xpos[self.wid] + Rw @ self.tcp
            p = Rp.T @ (pw - d.xpos[self.pid])
            R = Rp.T @ Rw
            ep = p_tgt - p
            er_m = R_tgt @ R.T
            er = 0.5 * np.array([er_m[2, 1] - er_m[1, 2], er_m[0, 2] - er_m[2, 0], er_m[1, 0] - er_m[0, 1]])
            if np.linalg.norm(ep) < tol and np.linalg.norm(er) < 0.02:
                break
            mujoco.mj_jac(m, d, jacp, jacr, pw, self.wid)
            J = np.vstack([Rp.T @ jacp[:, self.dadr], w_rot * (Rp.T @ jacr[:, self.dadr])])
            e = np.concatenate([ep, w_rot * er])
            dq = J.T @ np.linalg.solve(J @ J.T + damping ** 2 * np.eye(6), e)
            q = np.clip(q + np.clip(dq, -0.3, 0.3), self.lo, self.hi)
        p, R = self.fk(q)
        er_m = R_tgt @ R.T
        return q, float(np.linalg.norm(p_tgt - p)), float(np.arccos(np.clip((np.trace(er_m) - 1) / 2, -1, 1)))


def hand_geometry(model, wrist_body, middle_body, thumb_body, q_joints=None, joints=None, palm_frac=0.5):
    """Hand frame (3x3, columns: finger dir, n x f, palm normal n) and palm-centre point, both in the
    wrist frame, from body positions (wrist, middle-finger base, thumb base) at one pose."""
    d = mujoco.MjData(model)
    d.qpos[:] = model.qpos0
    if model.jnt_type[0] == mujoco.mjtJoint.mjJNT_FREE:
        d.qpos[:7] = [0, 0, 1.0, 1, 0, 0, 0]
    if joints is not None:
        for j, v in zip(joints, q_joints):
            d.qpos[model.jnt_qposadr[model.joint(j).id]] = v
    mujoco.mj_kinematics(model, d)
    Rw = d.xmat[model.body(wrist_body).id].reshape(3, 3); pw = d.xpos[model.body(wrist_body).id]
    f = d.xpos[model.body(middle_body).id] - pw
    t = d.xpos[model.body(thumb_body).id] - pw
    palm = Rw.T @ (palm_frac * f)
    f = f / np.linalg.norm(f); n = np.cross(f, t); n /= np.linalg.norm(n)
    H = np.stack([f, np.cross(n, f), n], 1)
    return Rw.T @ H, palm


class HandSpaceRetarget:
    """src -> dst arm map in hand space, per side. q vectors use the ARM semantic order."""

    def __init__(self, src: ArmModel, dst: ArmModel, src_ref_q, dst_ref_q, pos_offset=None,
                 src_hand_local=None, dst_hand_local=None):
        """src/dst_hand_local: optional 3x3 hand frames (finger dir, palm normal x finger, palm normal) in
        each wrist frame; then the alignment matches hand geometry instead of assuming the wrist frames
        line up at the reference pose."""
        self.src, self.dst = src, dst
        ps, Rs = src.fk(src_ref_q)
        pd, Rd = dst.fk(dst_ref_q)
        if src_hand_local is not None and dst_hand_local is not None:
            self.R_align = src_hand_local @ dst_hand_local.T   # dst wrist = src wrist * Hs * Hd^T
        else:
            self.R_align = Rs.T @ Rd                 # dst wrist frame expressed in src wrist frame, at the reference
        self.pos_offset = np.zeros(3) if pos_offset is None else np.asarray(pos_offset, float)
        self.q_prev = np.asarray(dst_ref_q, float).copy()
        self.last_err = (0.0, 0.0)

    def __call__(self, q_src, extra_offset=None):
        p, R = self.src.fk(q_src)
        off = self.pos_offset if extra_offset is None else self.pos_offset + extra_offset
        q, ep, er = self.dst.ik(p + off, R @ self.R_align, self.q_prev)
        self.q_prev = q
        self.last_err = (ep, er)
        return q
