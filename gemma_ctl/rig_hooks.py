"""Gemma manipulation layer for the IsaacLab kitchen rig
(GR00T-WholeBodyControl/gear_sonic/scripts/x2-navigation/run_x2_kplanner_env2.py,
enabled with KP_GEMMA=1; the rig imports this file by path).

What it adds to the rig (the rig itself only accepts stick commands):
  * scene objects   KP_GEMMA_SCENE=<json>: visual-only USD props on kitchen
                    surfaces (physics/collision disabled: no hand contact).
  * arm overlay     the 7 arm DOFs of every reference row SONIC tracks are
                    replaced by a slewed target (so SONIC executes the reach
                    with its own balance); release slews back to the planner.
  * reach IK        position IK for the palm point, solved on the rig's
                    MuJoCo X2 model seeded with the robot's MEASURED state.
  * grasp           kinematic attach: an attached object follows the wrist
                    link rigidly (sim shortcut — no finger contact physics).
  * state           PUB gemma_state 10 Hz (sim): robot, palms, objects,
                    overlay status. All poses in the KITCHEN frame
                    (world - world_pos; == robot_pose :6570 frame).

Wire (JSON, multipart [topic, json]):
  SUB bind :6581 topic gemma_rig
    {"op":"reach","side":"right","xyz":[x,y,z]}          IK + slew; reply in state
    {"op":"arm_release","side":"right"}
    {"op":"attach","obj":"mug","side":"right"}
    {"op":"detach","obj":"mug","xyz":[x,y,z_surface]}    rests the object's bottom on z
  PUB bind :6582 topic gemma_state
"""
from __future__ import annotations

import json
import math
import os
import time

import numpy as np

RIG_PORT = 6581
STATE_PORT = 6582
ARM_SL = {"left": slice(15, 22), "right": slice(22, 29)}     # MJ dof order
WAIST_SL = slice(12, 15)                                     # yaw, pitch, roll (MJ)
ARM_DOF_ADR = {"left": list(range(21, 28)), "right": list(range(28, 35))}
WRIST_BODY = {"left": "left_wrist_roll_link", "right": "right_wrist_roll_link"}
PALM_OFF = np.array([0.0, 0.0, -0.10])        # palm centre in wrist-roll frame
SLEW_RAD_PER_TICK = 0.025                     # 1.25 rad/s of sim time


def _quat_wxyz_to_R(q):
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def _R_to_quat_wxyz(R):
    t = np.trace(R)
    if t > 0:
        s = math.sqrt(t + 1.0) * 2
        return np.array([0.25 * s, (R[2, 1] - R[1, 2]) / s,
                         (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s])
    i = int(np.argmax(np.diag(R)))
    if i == 0:
        s = math.sqrt(1 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        return np.array([(R[2, 1] - R[1, 2]) / s, 0.25 * s,
                         (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s])
    if i == 1:
        s = math.sqrt(1 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        return np.array([(R[0, 2] - R[2, 0]) / s, (R[0, 1] + R[1, 0]) / s,
                         0.25 * s, (R[1, 2] + R[2, 1]) / s])
    s = math.sqrt(1 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
    return np.array([(R[1, 0] - R[0, 1]) / s, (R[0, 2] + R[2, 0]) / s,
                     (R[1, 2] + R[2, 1]) / s, 0.25 * s])


class GemmaHooks:
    def __init__(self, driver):
        import mujoco
        import zmq
        self.d = driver
        self.mj = mujoco
        self.m = driver.mj_model
        self.ik_data = mujoco.MjData(self.m)
        self.world_pos = np.array([float(v) for v in os.environ.get(
            "KP_WORLD_POS", "-19.99,-75.96,0").split(",")])
        self.target = {"left": None, "right": None}   # q7 goal
        self.cur = {"left": None, "right": None}      # q7 streamed now
        self.releasing = {"left": False, "right": False}
        self.planner_arm = {"left": None, "right": None}
        self.waist_target = None      # (yaw, pitch, roll) rad, None = planner
        self.waist_cur = None
        self.planner_waist = None
        self.last_reach = {}
        self.attached = {}          # obj -> (side, T_wrist_obj 4x4)
        self.objects = {}           # obj -> dict(prim, pos, quat, zmin)
        self._ticks = 0
        self._il_ids = {}
        # recording: eval_camera (needs ++manager_env.config.render_results=true)
        # at a fixed eye in the open middle of the kitchen, tracking the robot;
        # frames -> JPEG files (survive a hard kill), ffmpeg'd afterwards.
        self.rec_dir = os.environ.get("KP_GEMMA_REC_DIR")
        self.rec_every = int(os.environ.get("KP_GEMMA_REC_EVERY", "4"))
        self.cam_eye = np.array([float(v) for v in os.environ.get(
            "KP_GEMMA_CAM_EYE", "0.3,0.0,2.1").split(",")])
        self.rec_n = 0
        self.overlay = ""
        ctx = zmq.Context.instance()
        self.sub = ctx.socket(zmq.SUB)
        self.sub.setsockopt(zmq.SUBSCRIBE, b"gemma_rig")
        self.sub.bind(f"tcp://127.0.0.1:{RIG_PORT}")
        self.pub = ctx.socket(zmq.PUB)
        self.pub.bind(f"tcp://127.0.0.1:{STATE_PORT}")
        # KP_GEMMA_REPLAY=<bundle>/trajectory.jsonl: kinematic replay of a saved
        # run (robot root + joints + props written each tick, physics ignored)
        # so the recording camera can re-shoot it from new viewpoints.
        self.replay = None
        rp = os.environ.get("KP_GEMMA_REPLAY")
        if rp:
            self.replay = [json.loads(l) for l in open(rp)]
            # KP_GEMMA_REPLAY_T0/T1: replay only this sim-time window
            t0 = float(os.environ.get("KP_GEMMA_REPLAY_T0", "0")); t1 = float(os.environ.get("KP_GEMMA_REPLAY_T1", "1e9"))
            self.replay = [r for r in self.replay if t0 <= r["t"] <= t1]
            self._rp_t0 = self.replay[0]["t"] if self.replay else 0.0
            self._rp_i = 0
            print(f"[gemma] REPLAY {len(self.replay)} states from {rp}", flush=True)
        scene = os.environ.get("KP_GEMMA_SCENE")
        if scene:
            self._spawn_scene(json.load(open(scene)))
        print(f"[gemma] hooks up: rig :{RIG_PORT}, state :{STATE_PORT}, "
              f"objects {sorted(self.objects)}", flush=True)

    # ---- scene -----------------------------------------------------------
    def _spawn_scene(self, scene):
        import omni.usd
        from pxr import Gf, Usd, UsdGeom, UsdPhysics
        stage = omni.usd.get_context().get_stage()
        UsdGeom.Xform.Define(stage, "/World/GemmaObj")
        for o in scene["objects"]:
            path = f"/World/GemmaObj/{o['name']}"
            prim = UsdGeom.Xform.Define(stage, path).GetPrim()
            inner = stage.DefinePrim(path + "/asset", "Xform")
            inner.GetReferences().AddReference(o["usd"])
            s = float(o.get("scale", 1.0))
            UsdGeom.XformCommonAPI(inner).SetScale(Gf.Vec3f(s, s, s))
            for p in Usd.PrimRange(prim):
                if p.HasAPI(UsdPhysics.RigidBodyAPI):
                    UsdPhysics.RigidBodyAPI(p).CreateRigidBodyEnabledAttr(False)
                if p.HasAPI(UsdPhysics.CollisionAPI):
                    UsdPhysics.CollisionAPI(p).CreateCollisionEnabledAttr(False)
            bb = UsdGeom.BBoxCache(Usd.TimeCode.Default(),
                                   [UsdGeom.Tokens.default_, UsdGeom.Tokens.render])
            box = bb.ComputeLocalBound(inner).ComputeAlignedRange()
            cxy = ([float((box.GetMin()[i] + box.GetMax()[i]) / 2) for i in (0, 1)]
                   if not box.IsEmpty() else [0.0, 0.0])
            zmin = float(box.GetMin()[2]) if not box.IsEmpty() else 0.0
            zmax = float(box.GetMax()[2]) if not box.IsEmpty() else 0.1
            xf = UsdGeom.Xformable(prim)
            xf.ClearXformOpOrder()
            t_op = xf.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble)
            o_op = xf.AddOrientOp(UsdGeom.XformOp.PrecisionDouble)
            x, y, zs = o["xyz"]
            yaw = float(o.get("yaw", 0.0))
            pos = np.array([x, y, zs - zmin])
            quat = np.array([math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)])
            self.objects[o["name"]] = dict(t_op=t_op, o_op=o_op, pos=pos,
                                           quat=quat, zmin=zmin,
                                           height=zmax - zmin, center_off=cxy,
                                           desc=o.get("desc", o["name"]))
            self._set_obj(o["name"], pos, quat)
            print(f"[gemma] object {o['name']} <- {os.path.basename(o['usd'])} "
                  f"at {np.round(pos, 3).tolist()} (h {zmax - zmin:.3f} m)",
                  flush=True)

    def _set_obj(self, name, pos, quat):
        from pxr import Gf
        ob = self.objects[name]
        ob["pos"], ob["quat"] = np.asarray(pos, float), np.asarray(quat, float)
        w = ob["pos"] + self.world_pos
        ob["t_op"].Set(Gf.Vec3d(*w.tolist()))
        q = ob["quat"]
        ob["o_op"].Set(Gf.Quatd(float(q[0]), Gf.Vec3d(*q[1:].tolist())))

    # ---- robot state -----------------------------------------------------
    def _robot(self):
        env = self.d.cmd._env
        robot = env.scene["robot"]
        org = env.scene.env_origins[0].cpu().numpy()
        return robot, org

    def _wrist_pose(self, side):
        robot, org = self._robot()
        if side not in self._il_ids:
            self._il_ids[side] = robot.body_names.index(WRIST_BODY[side])
        i = self._il_ids[side]
        p = robot.data.body_pos_w[0, i].cpu().numpy() - org
        q = robot.data.body_quat_w[0, i].cpu().numpy()       # wxyz
        return p, _quat_wxyz_to_R(q)

    def _measured_qpos(self):
        robot, org = self._robot()
        rs = robot.data.root_state_w[0].cpu().numpy()
        jp_il = robot.data.joint_pos[0].cpu().numpy()
        jp_mj = np.zeros(31)
        jp_mj[np.asarray(self.d.mj2il)] = jp_il
        q = np.zeros(38)
        q[0:3] = rs[0:3] - org
        q[3:7] = rs[3:7]
        q[7:] = jp_mj
        return q

    # ---- IK --------------------------------------------------------------
    def solve_reach(self, side, target, iters=200):
        m, d = self.m, self.ik_data
        d.qpos[:] = self._measured_qpos()
        if self.cur[side] is not None:
            d.qpos[7 + ARM_SL[side].start:7 + ARM_SL[side].stop] = self.cur[side]
        adr = ARM_DOF_ADR[side]
        qadr = [a + 1 for a in adr]          # free joint: qpos = dof + 1
        lo = m.jnt_range[[a - 5 for a in adr], 0]
        hi = m.jnt_range[[a - 5 for a in adr], 1]
        bid = m.body(WRIST_BODY[side]).id
        jacp = np.zeros((3, m.nv))
        target = np.asarray(target, float)
        lam = 0.05
        for _ in range(iters):
            self.mj.mj_kinematics(m, d)
            self.mj.mj_comPos(m, d)
            R = d.xmat[bid].reshape(3, 3)
            palm = d.xpos[bid] + R @ PALM_OFF
            err = target - palm
            if np.linalg.norm(err) < 0.005:
                break
            self.mj.mj_jac(m, d, jacp, None, palm, bid)
            J = jacp[:, adr]
            dq = J.T @ np.linalg.solve(J @ J.T + lam ** 2 * np.eye(3), err)
            dq = np.clip(dq, -0.2, 0.2)
            q = d.qpos[qadr] + dq
            d.qpos[qadr] = np.clip(q, lo + 0.02, hi - 0.02)
        self.mj.mj_kinematics(m, d)
        R = d.xmat[bid].reshape(3, 3)
        palm = d.xpos[bid] + R @ PALM_OFF
        return d.qpos[qadr].copy(), float(np.linalg.norm(target - palm))

    # ---- rig integration -------------------------------------------------
    def override_row(self, dof_mj):
        """Called by the rig's _write_row on EVERY reference row."""
        dof = np.array(dof_mj, dtype=np.float32, copy=True)
        for side, sl in ARM_SL.items():
            self.planner_arm[side] = dof[sl].copy()
            if self.cur[side] is not None:
                dof[sl] = self.cur[side]
        self.planner_waist = dof[WAIST_SL].copy()
        if self.waist_cur is not None:
            dof[WAIST_SL] = self.waist_cur
        return dof

    def _handle(self, msg):
        op, side = msg.get("op"), msg.get("side", "right")
        if op == "reach":
            if msg.get("q") is not None:
                q, err = np.asarray(msg["q"], np.float32), float(msg.get("ik_err", -1.0))
            else:
                q, err = self.solve_reach(side, msg["xyz"])
            if self.cur[side] is None:
                self.cur[side] = (self.planner_arm[side].copy()
                                  if self.planner_arm[side] is not None else q.copy())
            self.target[side] = q
            self.releasing[side] = False
            self.last_reach[side] = {"id": msg.get("id"), "xyz": msg.get("xyz"),
                                     "ik_err_m": round(err, 3)}
            print(f"[gemma] reach {side} -> {np.round(msg.get('xyz', [0, 0, 0]), 3).tolist()} "
                  f"ik_err {err:.3f} m", flush=True)
        elif op == "arm_release":
            self.releasing[side] = True
            self.target[side] = None
        elif op == "attach":
            name = msg["obj"]
            p, R = self._wrist_pose(side)
            T_w = np.eye(4); T_w[:3, :3] = R; T_w[:3, 3] = p
            ob = self.objects[name]
            T_o = np.eye(4); T_o[:3, :3] = _quat_wxyz_to_R(ob["quat"])
            T_o[:3, 3] = ob["pos"]
            self.attached[name] = (side, np.linalg.inv(T_w) @ T_o)
            print(f"[gemma] attach {name} -> {side} hand", flush=True)
        elif op == "waist":
            w = msg.get("ypr")
            self.waist_target = None if w is None else np.asarray(w, np.float32)
            if self.waist_cur is None and w is not None:
                self.waist_cur = (self.planner_waist.copy() if self.planner_waist is not None
                                  else np.zeros(3, np.float32))
        elif op == "survey":
            # room-camera snapshot for Gemma perception: done on the next tick
            self._survey_req = msg
        elif op == "move":
            # test harness: place an object (bottom on xyz) — never used by Gemma
            ob = self.objects[msg["obj"]]
            x, y, zs = msg["xyz"]
            yaw = float(msg.get("yaw", 0.0))
            self._set_obj(msg["obj"], [x, y, zs - ob["zmin"]],
                          [math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)])
        elif op == "caption":
            self.overlay = str(msg.get("text", ""))[:120]
        elif op == "detach":
            name = msg["obj"]
            self.attached.pop(name, None)
            ob = self.objects[name]
            if msg.get("xyz") is not None:
                x, y, zs = msg["xyz"]
                yaw = math.atan2(2 * (ob["quat"][0] * ob["quat"][3]),
                                 1 - 2 * ob["quat"][3] ** 2)
                self._set_obj(name, [x, y, zs - ob["zmin"]],
                              [math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)])
            print(f"[gemma] detach {name} at {np.round(ob['pos'], 3).tolist()}",
                  flush=True)

    def _apply_replay(self):
        import torch
        t = self._rp_t0 + max(0.0, self._ticks * 0.02 - float(os.environ.get("KP_GEMMA_REPLAY_LEAD", "4")))
        while self._rp_i + 1 < len(self.replay) and self.replay[self._rp_i + 1]["t"] <= t:
            self._rp_i += 1
        r = self.replay[self._rp_i]
        robot, org = self._robot()
        dev = robot.data.root_state_w.device
        root = torch.zeros((1, 13), device=dev)
        root[0, 0:3] = torch.tensor(np.array(r["root"][:3]) + org, device=dev)
        root[0, 3:7] = torch.tensor(r["root"][3:7], device=dev)
        robot.write_root_state_to_sim(root)
        jp = torch.zeros((1, robot.data.joint_pos.shape[1]), device=dev)
        jp[0] = torch.tensor(np.asarray(r["dof"])[np.asarray(self.d.mj2il)], device=dev)
        robot.write_joint_state_to_sim(jp, torch.zeros_like(jp))
        for n, v in r["obj"].items():
            if n in self.objects:
                self._set_obj(n, v[:3], v[3:7])
        if self._rp_i >= len(self.replay) - 1 and not getattr(self, "_rp_done", False):
            self._rp_done = True
            print("[gemma] REPLAY finished", flush=True)

    def tick(self):
        import zmq
        self._ticks += 1
        if self.replay is not None:
            self._apply_replay()
            if self.rec_dir and self._ticks % self.rec_every == 0 and not getattr(self, "_rp_done", False):
                self._capture()
            return
        while True:
            try:
                parts = self.sub.recv_multipart(flags=zmq.NOBLOCK)
            except zmq.Again:
                break
            try:
                self._handle(json.loads(parts[-1].decode()))
            except Exception as e:  # noqa: BLE001 - never kill the sim
                print(f"[gemma] bad rig msg: {e!r}", flush=True)
        # slew arm overlays
        for side in ARM_SL:
            if self.cur[side] is None:
                continue
            goal = (self.planner_arm[side] if self.releasing[side]
                    else self.target[side])
            if goal is None:
                continue
            delta = np.clip(goal - self.cur[side], -SLEW_RAD_PER_TICK,
                            SLEW_RAD_PER_TICK)
            self.cur[side] = self.cur[side] + delta
            if self.releasing[side] and np.abs(goal - self.cur[side]).max() < 0.01:
                self.cur[side] = None
                self.releasing[side] = False
        if self.waist_cur is not None:
            goal = self.waist_target if self.waist_target is not None else self.planner_waist
            if goal is not None:
                self.waist_cur = self.waist_cur + np.clip(goal - self.waist_cur, -0.01, 0.01)
                if self.waist_target is None and np.abs(goal - self.waist_cur).max() < 0.01:
                    self.waist_cur = None
        # attached objects follow the measured wrist
        for name, (side, T_rel) in self.attached.items():
            p, R = self._wrist_pose(side)
            T_w = np.eye(4); T_w[:3, :3] = R; T_w[:3, 3] = p
            T = T_w @ T_rel
            self._set_obj(name, T[:3, 3], _R_to_quat_wxyz(T[:3, :3]))
        if self._ticks % 5 == 0:
            self._publish()
        if self.rec_dir and self._ticks % 2 == 0:
            self._log_traj()
        if getattr(self, "_survey_req", None) is not None:
            req, self._survey_req = self._survey_req, None
            self._survey(req)
        if self.rec_dir and self._ticks % self.rec_every == 0:
            self._capture()

    def _log_traj(self):
        """Full kinematic state at 25 Hz (kitchen frame): robot root + 31 joints
        (MuJoCo order) + every prop pose. Enough to re-render the run from any
        camera with a kinematic replay, without re-running Gemma or physics."""
        try:
            if not hasattr(self, "_trajlog"):
                self._trajlog = open(os.path.join(os.path.dirname(self.rec_dir), "trajectory.jsonl"), "a", buffering=1)
            q = self._measured_qpos()
            rec = {"t": round(self._ticks * 0.02, 2), "root": np.round(q[:7], 4).tolist(),
                   "dof": np.round(q[7:], 4).tolist(),
                   "obj": {n: [*np.round(ob["pos"], 4).tolist(), *np.round(ob["quat"], 4).tolist()]
                           for n, ob in self.objects.items()},
                   "held": {n: s for n, (s, _) in self.attached.items()}}
            self._trajlog.write(json.dumps(rec) + "\n")
        except Exception as e:  # noqa: BLE001
            print(f"[gemma] traj log failed: {e!r}", flush=True)
            self.rec_dir = self.rec_dir

    def _robot_cam_pose(self, kind):
        """(eye, target, hfov) of a robot camera in the kitchen frame."""
        robot, org = self._robot()
        hi = robot.body_names.index("head_pitch_link")
        hp = robot.data.body_pos_w[0, hi].cpu().numpy() - org
        hR = _quat_wxyz_to_R(robot.data.body_quat_w[0, hi].cpu().numpy())
        if kind == "orbbec":
            e = hp + hR @ np.array([0.05761, -0.011183, -0.04837]); f = hR @ np.array([0.766, 0.0, -0.643]); fov = 86.0
        else:
            e = hp + hR @ np.array([0.067995, 0.029784, 0.05]); f = hR @ np.array([1.0, 0.0, 0.0]); fov = 120.0
        return e, e + f, fov

    def _cam_prim(self):
        import omni.usd
        from pxr import UsdGeom
        st = omni.usd.get_context().get_stage()
        return UsdGeom.Camera(st.GetPrimAtPath("/World/envs/env_0/eval_camera"))

    def _base_hfov(self):
        c = self._cam_prim()
        if not hasattr(self, "_focal0"):
            self._focal0 = float(c.GetFocalLengthAttr().Get())
        ap = float(c.GetHorizontalApertureAttr().Get())
        return math.degrees(2 * math.atan(ap / (2 * self._focal0)))

    def _set_hfov(self, cam, hfov_deg):
        """Swap the render camera's lens for one capture (None = restore)."""
        c = self._cam_prim()
        if not hasattr(self, "_focal0"):
            self._focal0 = float(c.GetFocalLengthAttr().Get())
        if hfov_deg is None:
            c.GetFocalLengthAttr().Set(self._focal0)
            return
        ap = float(c.GetHorizontalApertureAttr().Get())
        c.GetFocalLengthAttr().Set(ap / (2 * math.tan(math.radians(hfov_deg) / 2)))

    def _survey(self, req):
        """Render each requested view (eye/target, kitchen frame) and save
        rgb.jpg + depth.npy (distance to image plane, m) + camera.json
        (K, ROS-convention pose in the KITCHEN frame). Ground truth goes to
        truth.json for grading only."""
        try:
            import torch
            from PIL import Image
            env = self.d.cmd._env
            cam = env.scene["eval_camera"]
            _, org = self._robot()
            out = req["dir"]
            os.makedirs(out, exist_ok=True)
            dev = cam.data.pos_w.device
            for k, v in enumerate(req["views"]):
                hfov = None
                if v.get("camera") == "orbbec" or v.get("head"):
                    # X2 Orbbec Gemini 335 RGB-D (rgbd_head_front, the
                    # manipulation camera): URDF mount on head_pitch_link
                    # xyz (0.0576, -0.0112, -0.0484), optical axis 40 deg
                    # below the head's forward axis; ~86 deg HFOV, 1280x720.
                    robot, _ = self._robot()
                    hi = robot.body_names.index("head_pitch_link")
                    hp = robot.data.body_pos_w[0, hi].cpu().numpy() - org
                    hR = _quat_wxyz_to_R(robot.data.body_quat_w[0, hi].cpu().numpy())
                    e = hp + hR @ np.array([0.05761, -0.011183, -0.04837])
                    f = hR @ np.array([0.766, 0.0, -0.643])
                    v = {**v, "eye": e.round(4).tolist(), "target": (e + f).round(4).tolist(),
                         "camera": "orbbec"}
                    req["views"][k] = v
                    hfov = float(v.get("hfov_deg", 86.0))
                if v.get("camera") == "head_wide":
                    # X2 wide head camera (stereo_head_front_left): URDF mount
                    # on head_pitch_link xyz (0.068, 0.0298, 0.05), optical axis
                    # level along the head's forward axis. The real lens is an
                    # equidistant fisheye (~170 deg); a pinhole render can only
                    # approximate its centre, so 120 deg.
                    robot, _ = self._robot()
                    hi = robot.body_names.index("head_pitch_link")
                    hp = robot.data.body_pos_w[0, hi].cpu().numpy() - org
                    hR = _quat_wxyz_to_R(robot.data.body_quat_w[0, hi].cpu().numpy())
                    e = hp + hR @ np.array([0.067995, 0.029784, 0.05])
                    f = hR @ np.array([1.0, 0.0, 0.0])
                    v = {**v, "eye": e.round(4).tolist(), "target": (e + f).round(4).tolist()}
                    req["views"][k] = v
                    hfov = float(v.get("hfov_deg", 120.0))
                self._set_hfov(cam, hfov)
                eye = torch.tensor([np.asarray(v["eye"], float) + org], device=dev, dtype=torch.float32)
                tgt = torch.tensor([np.asarray(v["target"], float) + org], device=dev, dtype=torch.float32)
                cam._view._sync_usd_on_fabric_write = True  # noqa: SLF001
                cam.set_world_poses_from_view(eye, tgt)
                for _ in range(3):
                    env.sim.render()
                cam._is_outdated[:] = True  # noqa: SLF001
                cam.update(dt=0.0, force_recompute=True)
                rgb = cam.data.output["rgb"][0].cpu().numpy()[..., :3]
                out_size = None
                if v.get("camera") in ("orbbec", "head_wide"):
                    out_size = (1280, 720)        # stream size (Orbbec colour; wide cam kept 16:9)
                img = Image.fromarray(rgb)
                if out_size:
                    img = img.resize(out_size, Image.LANCZOS)
                img.save(f"{out}/view{k}.jpg", quality=95)
                dk = "distance_to_image_plane"
                if dk in cam.data.output:
                    d = cam.data.output[dk][0].cpu().numpy().squeeze().astype(np.float32)
                    if out_size:
                        d = np.asarray(Image.fromarray(d).resize(out_size, Image.NEAREST))
                    np.save(f"{out}/view{k}_depth.npy", d.astype(np.float16))
                meta = {"name": v.get("name", f"view{k}"), "eye": v["eye"], "target": v["target"],
                        "K": cam.data.intrinsic_matrices[0].cpu().numpy().tolist(),
                        "pos": (cam.data.pos_w[0].cpu().numpy() - org).tolist(),
                        "quat_ros_wxyz": cam.data.quat_w_ros[0].cpu().numpy().tolist(),
                        "size": list(out_size) if out_size else [int(rgb.shape[1]), int(rgb.shape[0])],
                        "hfov_deg": hfov if hfov else self._base_hfov(),
                        "camera": v.get("camera", "room"), "has_depth": dk in cam.data.output}
                json.dump(meta, open(f"{out}/view{k}.json", "w"), indent=1)
            self._set_hfov(cam, None)
            truth = {}
            for n, ob in self.objects.items():
                base = ob["pos"] + np.array([0, 0, ob["zmin"]])
                # bounding-box centre (asset origins are not always centred)
                Rz = _quat_wxyz_to_R(ob["quat"])
                cb = base + Rz @ np.array([ob["center_off"][0], ob["center_off"][1], 0.0])
                truth[n] = {"base": base.round(4).tolist(), "center_base": cb.round(4).tolist(),
                            "height": round(ob["height"], 4)}
            json.dump(truth, open(f"{out}/truth.json", "w"), indent=1)
            self.survey_done = {"id": req.get("id"), "dir": out, "n": len(req["views"])}
            self._eye = None          # recording camera re-picks its tripod
            print(f"[gemma] survey {req.get('id')} -> {out} ({len(req['views'])} views)", flush=True)
        except Exception as e:  # noqa: BLE001
            self.survey_done = {"id": req.get("id"), "error": repr(e)}
            print(f"[gemma] survey failed: {e!r}", flush=True)

    def _stations(self):
        """Fixed tripod positions: open floor cells (esdf >= 0.45 m) on a
        0.6 m lattice, at 2.1 m, under the ceiling, inside the room."""
        if hasattr(self, "_st_list"):
            return self._st_list
        import trimesh
        self._mesh = getattr(self, "_mesh", None) or trimesh.load(
            "/home/stickbot/projects/x2-kitchen-sim/assets/kitchen/kitchen_collision.obj", force="mesh")
        g = np.load("/home/stickbot/projects/x2-kitchen-sim/assets/nav_grid.npz")
        E, Wk, o, res = g["esdf"], g["walkable"], g["origin"], float(g["res"])
        out = []
        for x in np.arange(-3.0, 3.2, 0.6):
            for y in np.arange(-3.0, 3.2, 0.6):
                i, j = int(round((x - o[0]) / res)), int(round((y - o[1]) / res))
                # inside the room = on walkable floor (the collision mesh has
                # no floor under open space, so a down-ray test rejects all)
                if not (0 <= i < E.shape[0] and 0 <= j < E.shape[1]) or not Wk[i, j] or E[i, j] < 0.45:
                    continue
                eye = np.array([x, y, self.cam_eye[2]])
                up, _, _ = self._mesh.ray.intersects_location([eye], [[0, 0, 1.0]])
                if len(up) == 0 or min(np.linalg.norm(h - eye) for h in up) < 0.15:
                    continue
                out.append(eye)
        self._st_list = out
        print(f"[gemma] camera: {len(out)} tripod stations", flush=True)
        return out

    def _visible(self, eye, r):
        chest = np.array([r[0], r[1], 1.0])
        d = chest - eye
        L = float(np.linalg.norm(d))
        hits, _, _ = self._mesh.ray.intersects_location([eye], [d / L])
        return not any(np.linalg.norm(h - eye) < L - 0.25 for h in hits)

    def _station_eye(self, r, yaw):
        """Hold the current tripod; re-check every 25 frames and CUT (no
        glide) only when the robot is hidden or too near/far."""
        cur = getattr(self, "_eye", None)
        if cur is not None and self.rec_n % 25:
            return cur
        if cur is not None:
            d = float(np.linalg.norm(cur[:2] - r))
            if 1.2 <= d <= 4.2 and self._visible(cur, r):
                return cur
        best, best_s = None, -1e9
        for eye in self._stations():
            v = r - eye[:2]
            d = float(np.linalg.norm(v))
            if not 1.6 <= d <= 3.6:
                continue
            # prefer ~2.4 m and a view of the robot's right (working) side
            side = math.cos(math.atan2(-v[1], -v[0]) - (yaw - math.pi / 2))
            s = -abs(d - 2.4) + 0.6 * side
            if s > best_s and self._visible(eye, r):
                best, best_s = eye, s
        if best is None:
            best = cur if cur is not None else np.r_[r + np.array([1.4, -1.1]), self.cam_eye[2]]
        if cur is None or np.linalg.norm(best - cur) > 1e-6:
            self._cut = True
        self._eye = best
        return best

    def _pick_eye(self, r, yaw):
        """Side-on eye 1.8 m from the robot at 2.1 m: prefer the robot's right
        side (the working hand), require clear line of sight to the chest on
        the kitchen collision mesh and an eye that is not inside geometry."""
        try:
            if not hasattr(self, "_mesh"):
                import trimesh
                self._mesh = trimesh.load(os.environ.get(
                    "KP_GEMMA_COLLISION_OBJ",
                    "/home/stickbot/projects/x2-kitchen-sim/assets/kitchen/kitchen_collision.obj"),
                    force="mesh")
            chest = np.array([r[0], r[1], 1.1])
            for off in (-70, -110, -45, -135, 70, 110, 45, 135, 180, 0, -160, 160):
                a = yaw + math.radians(off)
                for dist in (1.8, 1.5, 1.2):
                    eye = np.array([r[0] + dist * math.cos(a), r[1] + dist * math.sin(a), self.cam_eye[2]])
                    d = chest - eye
                    L = float(np.linalg.norm(d))
                    hits, _, _ = self._mesh.ray.intersects_location([eye], [d / L])
                    if any(np.linalg.norm(h - eye) < L - 0.25 for h in hits):
                        continue
                    up, _, _ = self._mesh.ray.intersects_location([eye], [[0, 0, 1.0]])
                    dn, _, _ = self._mesh.ray.intersects_location([eye], [[0, 0, -1.0]])
                    if len(up) == 0 or len(dn) == 0:     # outside the room
                        continue
                    if min(np.linalg.norm(h - eye) for h in up) < 0.15:
                        continue
                    return eye
        except Exception as ex:  # noqa: BLE001
            print(f"[gemma] eye pick failed: {ex!r}", flush=True)
        return None

    def _finish_capture(self, cam, env, eye, tgt, e):
        try:
            import torch  # noqa: F401
            cam._view._sync_usd_on_fabric_write = True  # noqa: SLF001
            cam.set_world_poses_from_view(eye, tgt)
            env.sim.render(); env.sim.render()
            cam._is_outdated[:] = True  # noqa: SLF001
            cam.update(dt=0.0, force_recompute=True)
            rgb = cam.data.output["rgb"][0].cpu().numpy()[..., :3]
            from PIL import Image, ImageDraw
            im = Image.fromarray(rgb)
            if self.overlay and os.environ.get("KP_GEMMA_CAPTION", "1") != "0":
                from PIL import ImageFont
                if not hasattr(self, "_font"):
                    try:
                        self._font = ImageFont.truetype(
                            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
                            max(22, im.height // 32))
                    except Exception:
                        self._font = ImageFont.load_default()
                dr = ImageDraw.Draw(im)
                dr.rectangle([0, 0, im.width, im.height // 18], fill=(0, 0, 0))
                dr.text((im.height // 60, im.height // 90), self.overlay, fill=(255, 255, 255), font=self._font)
            os.makedirs(self.rec_dir, exist_ok=True)
            im.save(f"{self.rec_dir}/f{self.rec_n:06d}.jpg", quality=93)
            # per-frame camera pose (kitchen frame) so overlays can be projected later
            if not hasattr(self, "_camlog"):
                self._camlog = open(os.path.join(os.path.dirname(self.rec_dir), "cam_poses.jsonl"), "a", buffering=1)
            t_log = (self.replay[self._rp_i]["t"] if self.replay else self._ticks * 0.02)   # replay: original run time
            self._camlog.write(json.dumps({"f": self.rec_n, "t": round(t_log, 2),
                                           "eye": np.round(e, 4).tolist(), "aim": np.round(self._aim, 4).tolist(),
                                           "hfov": round(self._rec_hfov or self._base_hfov(), 3), "w": im.width, "h": im.height}) + "\n")
            self.rec_n += 1
            if self._rec_hfov:
                self._set_hfov(cam, None)
        except Exception as ex:  # noqa: BLE001
            print(f"[gemma] capture failed, recording off: {ex!r}", flush=True)
            self.rec_dir = None

    def _capture(self):
        try:
            import torch
            env = self.d.cmd._env
            cam = env.scene["eval_camera"]
            robot, org = self._robot()
            root = robot.data.root_pos_w[0:1].clone()
            org_t = torch.tensor(org, device=root.device, dtype=root.dtype)
            r = root[0, :2].cpu().numpy() - org[:2]
            rs = robot.data.root_state_w[0].cpu().numpy()
            yaw = math.atan2(2 * (rs[3] * rs[6] + rs[4] * rs[5]), 1 - 2 * (rs[5] ** 2 + rs[6] ** 2))
            rec_cam = os.environ.get("KP_GEMMA_REC_CAM", "tripod")
            if rec_cam in ("head_wide", "orbbec"):
                e, tgt_np, fov = self._robot_cam_pose(rec_cam)
                self._set_hfov(cam, fov)
                self._aim = np.asarray(tgt_np, float)
                eye = (torch.tensor(e, device=root.device, dtype=root.dtype) + org_t).unsqueeze(0)
                tgt = (torch.tensor(tgt_np, device=root.device, dtype=root.dtype) + org_t).unsqueeze(0)
                self._rec_hfov = fov
                return self._finish_capture(cam, env, eye, tgt, e)
            self._rec_hfov = None
            e = self._station_eye(r, yaw)
            eye = (torch.tensor(e, device=root.device, dtype=root.dtype)
                   + org_t).unsqueeze(0)
            # tripod operator: aim follows the robot through a ~2 s low-pass
            # (snapped on a cut), so body sway never shakes the shot
            want = np.r_[r, float(root[0, 2].cpu()) + 0.15]
            if getattr(self, "_aim", None) is None or getattr(self, "_cut", False):
                self._aim = want
                self._cut = False
            self._aim = self._aim + 0.04 * (want - self._aim)
            tgt = (torch.tensor(self._aim, device=root.device, dtype=root.dtype) + org_t).unsqueeze(0)
            return self._finish_capture(cam, env, eye, tgt, e)
        except Exception as e:  # noqa: BLE001
            print(f"[gemma] capture failed, recording off: {e!r}", flush=True)
            self.rec_dir = None

    def _publish(self):
        robot, org = self._robot()
        rs = robot.data.root_state_w[0].cpu().numpy()
        w, x, y, z = rs[3:7]
        st = {"t_sim": round(self._ticks * 0.02, 2),
              "robot": {"xy": [round(float(rs[0] - org[0]), 3),
                               round(float(rs[1] - org[1]), 3)],
                        "z": round(float(rs[2] - org[2]), 3),
                        "yaw": round(math.atan2(2 * (w * z + x * y),
                                                1 - 2 * (y * y + z * z)), 3),
                        "up_cos": round(float(1 - 2 * (x * x + y * y)), 3)},
              "palm": {}, "arm": {}, "objects": {}, "attached": {}}
        for side in ARM_SL:
            p, R = self._wrist_pose(side)
            st["palm"][side] = np.round(p + R @ PALM_OFF, 3).tolist()
            busy = (self.cur[side] is not None and self.target[side] is not None
                    and float(np.abs(self.target[side] - self.cur[side]).max()) > 0.01)
            qm = self._measured_qpos()[7:]
            st["arm"][side + "_q"] = {
                "cmd": None if self.cur[side] is None else np.round(self.cur[side], 3).tolist(),
                "meas": np.round(qm[ARM_SL[side]], 3).tolist()}
            st["arm"][side] = {"overlay": self.cur[side] is not None,
                               "moving": busy or self.releasing[side],
                               "last_reach": self.last_reach.get(side)}
        for name, ob in self.objects.items():
            base = ob["pos"] + np.array([0.0, 0.0, ob["zmin"]])
            st["objects"][name] = {"xyz": np.round(base, 3).tolist(),
                                   "height": round(ob["height"], 3),
                                   "desc": ob["desc"]}
        st["attached"] = {n: s for n, (s, _) in self.attached.items()}
        st["survey"] = getattr(self, "survey_done", None)
        qm_full = self._measured_qpos()
        qm = qm_full[7:]
        st["qpos"] = np.round(qm_full, 4).tolist()
        st["waist"] = {"cmd": None if self.waist_cur is None else np.round(self.waist_cur, 3).tolist(),
                       "meas": np.round(qm[WAIST_SL], 3).tolist()}
        self.pub.send_multipart([b"gemma_state", json.dumps(st).encode()])
