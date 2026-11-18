"""X2 side of the bridge for the gwbc-x2 MuJoCo simstack (sim_onnx_planner.sh / simstack_local.sh).

Reads:   x2_debug   (deploy telemetry, :5659 in the simstack)  body_q[31], hand q[10+10], base_quat
         robot_pose (bridge ground-truth pelvis, :5570)        x, y, z, qw, qx, qy, qz
Renders: the X2 head camera (ego_view, 640x480, 60 deg) with gwbc-x2's camera mount, plus render-only
         props (a console with a red button) and a third-person chase view for the video.
Sends:   planner_cmd    target_velocity [yaw, left, fwd, hip] to the kplanner (:5663), source "vr"
         arm_targets    left_q_rad / right_q_rad [7] (msgpack) to the kplanner arm ingest (:5572)
         hand_finger_cmd left_hand_q / right_hand_q [10]
gwbc-x2 code is imported at run time from ~/Projects/gwbc-x2 (not copied).
"""
from __future__ import annotations

import json
import os
import sys
import threading
import time
from pathlib import Path

import numpy as np

from .embodiment import RobotCommand, RobotState

GWBC_X2 = Path.home() / "Projects/gwbc-x2"
if str(GWBC_X2) not in sys.path:
    sys.path.insert(0, str(GWBC_X2))

PORTS = dict(x2_debug=5659, robot_pose=5570, planner_cmd=5663, arm=5572, scene_state=5559)


def add_props(spec, props):
    """Render-only boxes: (name, pos xyz, half-size xyz, rgba). No collision with the sim robot."""
    import mujoco
    wb = spec.worldbody
    for name, pos, size, rgba in props:
        g = wb.add_geom()
        g.name, g.type = name, mujoco.mjtGeom.mjGEOM_BOX
        g.pos, g.size, g.rgba = pos, size, rgba
        g.contype = g.conaffinity = 0


# A G1-Factory-PushButton-like console 1.6 m ahead: grey desk, red button on top.
DEFAULT_PROPS = [
    ("console", (1.6, 0.0, 0.45), (0.25, 0.45, 0.45), (0.55, 0.57, 0.6, 1)),
    ("button", (1.45, 0.0, 0.915), (0.06, 0.06, 0.02), (0.85, 0.08, 0.08, 1)),
]


class X2Renderer:
    """Head-camera + chase renderer. scene_xml: a physical scene (e.g. tools/build_x2_console_scene.py,
    X2 + OmniHand + console) rendered as simulated; without it, the bare X2 plus render-only props."""

    def __init__(self, props=DEFAULT_PROPS, w=640, h=480, chase_wh=(640, 480), scene_xml=None, ego_fovy=None, ego_pitch_deg=None,
                 g1_look=False):
        os.environ.setdefault("MUJOCO_GL", "egl")
        import mujoco
        from gear_sonic.scripts.x2_pico_video_sender import add_camera_to_spec, resolve_camera_spec
        from gear_sonic.data.robot_model.supplemental_info.x2_ultra.x2_ultra_supplemental_info import (
            X2_BODY_JOINT_NAMES)
        self.scene_xml = scene_xml
        xml = scene_xml or GWBC_X2 / "gear_sonic/data/assets/robot_description/mjcf/x2_ultra.xml"
        spec = mujoco.MjSpec.from_file(str(xml))
        add_camera_to_spec(spec, resolve_camera_spec("ego_view"))
        self.props = [] if scene_xml else props
        add_props(spec, self.props)
        spec.visual.global_.offwidth = max(w, chase_wh[0]); spec.visual.global_.offheight = max(h, chase_wh[1])
        light = spec.worldbody.add_light(); light.pos = (0, 0, 4); light.dir = (0, 0, -1)
        self.m = spec.compile()
        self.d = mujoco.MjData(self.m)
        self.mj = mujoco
        self.qadr = np.array([self.m.jnt_qposadr[self.m.joint(n if n.endswith("_joint") else n + "_joint").id]
                              for n in X2_BODY_JOINT_NAMES])
        self.ego_cam = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_CAMERA, resolve_camera_spec("ego_view").name)
        # analytic camera head: a G1-like virtual camera on the X2 head (wider fovy; optional extra pitch)
        if ego_fovy:
            self.m.cam_fovy[self.ego_cam] = float(ego_fovy)
            # closer to the G1 Factory look: plain grey concrete floor instead of the blue grid (render only)
            fid = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_GEOM, "floor")
            if fid >= 0:
                self.m.geom_matid[fid] = -1; self.m.geom_rgba[fid] = (0.42, 0.42, 0.43, 1)
        if ego_pitch_deg:
            q = self.m.cam_quat[self.ego_cam].copy(); r = np.zeros(4)
            mujoco.mju_axisAngle2Quat(r, np.array([1.0, 0, 0]), np.radians(float(ego_pitch_deg)))
            out = np.zeros(4); mujoco.mju_mulQuat(out, q, r); self.m.cam_quat[self.ego_cam] = out
        self.r_ego = mujoco.Renderer(self.m, height=h, width=w)
        self.r_chase = mujoco.Renderer(self.m, height=chase_wh[1], width=chase_wh[0])
        self.chase = mujoco.MjvCamera(); self.chase.type = mujoco.mjtCamera.mjCAMERA_FREE
        self.chase.distance, self.chase.elevation, self.chase.azimuth = 2.6, -20, 145
        if g1_look:
            # camera head, appearance: paint the X2 body/hands dark metal grey like the G1 in G1 Factory
            # (render model only; the sim's physics model is untouched)
            pel = self.m.body("pelvis").id
            for g in range(self.m.ngeom):
                b = self.m.geom_bodyid[g]
                while b > 0 and b != pel:
                    b = self.m.body_parentid[b]
                if b == pel and self.m.geom_group[g] in (0, 1, 2):
                    self.m.geom_matid[g] = -1
                    self.m.geom_rgba[g] = (0.30, 0.31, 0.33, self.m.geom_rgba[g][3])
        self.hand_layout = None
        if scene_xml:
            from gear_sonic.scripts.compose_x2_with_omnihand import (
                _build_layout, _default_side_configs, apply_active_hand_qpos)
            self.hand_layout = _build_layout(self.m, _default_side_configs())
            self._apply_hand = apply_active_hand_qpos
        bnames = [self.m.body(i).name for i in range(self.m.nbody)]
        jnames = [self.m.joint(i).name for i in range(self.m.njnt)]
        self.has_button = "button" in bnames
        # free scene objects (e.g. the BoxLift box): their qpos follows the sim via scene_state
        self.obj_qadr = {n: int(self.m.jnt_qposadr[self.m.joint(n).id]) for n in jnames
                         if n != jnames[0] and self.m.jnt_type[self.m.joint(n).id] == mujoco.mjtJoint.mjJNT_FREE}
        if self.has_button:
            self.button_bid = self.m.body("button").id
            self.button_gid = self.m.geom("button_cap").id
            self.tip_bids = [self.m.body(n).id for n in (
                "left_L_index_dip", "left_L_middle_dip", "right_R_index_dip", "right_R_middle_dip",
                "left_L_thumb_dip", "right_R_thumb_dip") if n in [self.m.body(i).name for i in range(self.m.nbody)]]

    def set_state(self, root7, body_q, lh=None, rh=None, objects=None):
        self.d.qpos[:7] = root7
        for n, q7 in (objects or {}).items():
            if n in self.obj_qadr:
                self.d.qpos[self.obj_qadr[n]:self.obj_qadr[n] + 7] = q7
        self.d.qpos[self.qadr] = body_q
        if self.hand_layout is not None and lh is not None:
            self._apply_hand(self.d, self.hand_layout, left_active=lh, right_active=rh)
        self.mj.mj_forward(self.m, self.d)

    def button_check(self):
        """Fingertip vs the button top (render model kinematics from the sim's joint state).
        Returns (min distance m from any fingertip to the button top centre, pressed?) where pressed =
        a fingertip within the cap footprint (+1 cm) and within 2 cm above the cap top."""
        if self.has_button and self.hand_layout is not None:
            c = self.d.geom_xpos[self.button_gid]; hx, hy, hz = self.m.geom_size[self.button_gid]
            top = c + np.array([0, 0, hz])
            best, pressed = 9.9, False
            for b in self.tip_bids:
                p = self.d.xpos[b]
                best = min(best, float(np.linalg.norm(p - top)))
                if abs(p[0] - c[0]) < hx + 0.01 and abs(p[1] - c[1]) < hy + 0.01 and -0.01 < p[2] - top[2] < 0.02:
                    pressed = True
            return best, pressed
        return None, False

    def move_body(self, name, pos):
        self.m.body_pos[self.m.body(name).id] = pos

    def place_props(self, root7, props=DEFAULT_PROPS):
        """Re-anchor the render-only props in the robot's frame at root7 (x, y, z, qw, qx, qy, qz)."""
        w, x, y, z = root7[3:7]
        yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
        c, s = np.cos(yaw), np.sin(yaw)
        for name, pos, _, _ in props:
            gid = self.m.geom(name).id
            self.m.geom_pos[gid] = [root7[0] + c * pos[0] - s * pos[1], root7[1] + s * pos[0] + c * pos[1], pos[2]]
            self.m.geom_quat[gid] = [np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]

    def ego(self):
        self.r_ego.update_scene(self.d, camera=self.ego_cam)
        return self.r_ego.render()

    def chase_view(self):
        self.chase.lookat[:] = self.d.qpos[:3] + np.array([0.4, 0, -0.2])
        self.r_chase.update_scene(self.d, camera=self.chase)
        return self.r_chase.render()


class X2SimIO:
    def __init__(self, host="127.0.0.1", ports=PORTS):
        import zmq
        from gear_sonic.utils.teleop.zmq.zmq_packed_message_decoder import unpack_message
        from gear_sonic.utils.teleop.zmq.robot_pose_zmq import unpack_robot_pose
        self._unpack, self._unpack_pose = unpack_message, unpack_robot_pose
        self.ctx = zmq.Context.instance()
        self.sub_dbg = self.ctx.socket(zmq.SUB); self.sub_dbg.setsockopt(zmq.SUBSCRIBE, b"x2_debug")
        self.sub_dbg.setsockopt(zmq.CONFLATE, 1); self.sub_dbg.connect(f"tcp://{host}:{ports['x2_debug']}")
        self.sub_pose = self.ctx.socket(zmq.SUB); self.sub_pose.setsockopt(zmq.SUBSCRIBE, b"robot_pose")
        self.sub_pose.setsockopt(zmq.CONFLATE, 1); self.sub_pose.connect(f"tcp://{host}:{ports['robot_pose']}")
        from gear_sonic.utils.teleop.zmq.scene_state_zmq import unpack_json
        self._unpack_scene = unpack_json
        self.sub_scene = self.ctx.socket(zmq.SUB); self.sub_scene.setsockopt(zmq.SUBSCRIBE, b"scene_state")
        self.sub_scene.setsockopt(zmq.CONFLATE, 1); self.sub_scene.connect(f"tcp://{host}:{ports['scene_state']}")
        self.pub_cmd = self.ctx.socket(zmq.PUB); self.pub_cmd.connect(f"tcp://{host}:{ports['planner_cmd']}")
        self.pub_arm = self.ctx.socket(zmq.PUB); self.pub_arm.connect(f"tcp://{host}:{ports['arm']}")
        self.state = {"body_q": None, "lh": np.zeros(10, np.float32), "rh": np.zeros(10, np.float32),
                      "root7": np.array([0, 0, 0.793, 1, 0, 0, 0], np.float64), "t_dbg": 0.0, "t_pose": 0.0,
                      "objects": {}, "t_scene": 0.0}
        self._stop = False
        self._thr = threading.Thread(target=self._rx, daemon=True); self._thr.start()

    def _rx(self):
        import zmq
        poller = zmq.Poller(); poller.register(self.sub_dbg, zmq.POLLIN); poller.register(self.sub_pose, zmq.POLLIN)
        poller.register(self.sub_scene, zmq.POLLIN)
        while not self._stop:
            for s, _ in poller.poll(100):
                try:
                    if s is self.sub_dbg:
                        f = self._unpack(s.recv(), expected_topic="x2_debug").fields
                        self.state["body_q"] = np.asarray(f["body_q"], np.float32).reshape(-1)[:31]
                        if "left_hand_q" in f:
                            self.state["lh"] = np.asarray(f["left_hand_q"], np.float32).reshape(-1)[:10]
                            self.state["rh"] = np.asarray(f["right_hand_q"], np.float32).reshape(-1)[:10]
                        self.state["t_dbg"] = time.time()
                    elif s is self.sub_scene:
                        _, p = self._unpack_scene(s.recv(), expected_topic="scene_state")
                        self.state["objects"] = {k: np.asarray(v) for k, v in p.get("object_freejoint_qpos", {}).items()}
                        self.state["t_scene"] = time.time()
                    else:
                        p = self._unpack_pose(s.recv())
                        self.state["root7"] = np.asarray(p["pelvis_qpos_wxyz"], np.float64)
                        self.state["t_pose"] = time.time()
                except Exception as e:                                  # never kill the loop on one frame
                    print(f"[x2io] bad frame: {e!r}", flush=True)

    def wait_ready(self, timeout=60):
        t0 = time.time()
        while time.time() - t0 < timeout:
            if self.state["body_q"] is not None and time.time() - self.state["t_pose"] < 1.0:
                return True
            time.sleep(0.2)
        return False

    def robot_state(self, images) -> RobotState:
        s = self.state
        return RobotState(t=time.time(), joint_q=s["body_q"].copy(), left_hand_q=s["lh"].copy(),
                          right_hand_q=s["rh"].copy(), images=images, base_quat_wxyz=s["root7"][3:7].copy())

    def send(self, c: RobotCommand):
        import msgpack
        fwd, left, yaw = (float(v) for v in c.base_velocity)
        self.pub_cmd.send_multipart([b"planner_cmd", json.dumps({
            "intent": "locomotion", "magnitude": "continuous", "stick_fwd": 0.0, "stick_side": 0.0,
            "stick_yaw": 0.0, "target_velocity": [yaw, left, fwd, float(c.hip_height_m)],
            "source": "vr"}).encode()])
        self.pub_arm.send_multipart([b"arm_targets", msgpack.packb({
            "left_q_rad": c.left_arm_q.astype(np.float32).tolist(),
            "right_q_rad": c.right_arm_q.astype(np.float32).tolist()})])
        self.pub_arm.send_multipart([b"hand_finger_cmd", msgpack.packb({
            "left_hand_q": c.left_hand_q.astype(np.float32).tolist(),
            "right_hand_q": c.right_hand_q.astype(np.float32).tolist(), "ts": time.time()})])

    def reset_objects(self, freejoints=None, bodies=None, port=5560):
        """scene_reset to the bridge (sim only): free-joint qpos {joint: [x,y,z,qw,qx,qy,qz]}, welded body
        positions {body: [x,y,z]}. The bridge SUB-connects to localhost:5560, so this side binds."""
        import zmq
        from gear_sonic.utils.teleop.zmq.scene_state_zmq import pack_json
        if getattr(self, "pub_reset", None) is None:
            self.pub_reset = self.ctx.socket(zmq.PUB); self.pub_reset.bind(f"tcp://*:{port}"); time.sleep(0.5)
        self.pub_reset.send(pack_json("scene_reset", {"object_freejoint_qpos": freejoints or {},
                                                      "mutable_body_pos": bodies or {}}))

    def release(self):
        """Stand still, hand the arms back to the planner, release VR ownership."""
        import msgpack
        for _ in range(5):
            self.pub_cmd.send_multipart([b"planner_cmd", json.dumps({
                "intent": "idle", "magnitude": "default", "source": "vr", "vr_release": True}).encode()])
            self.pub_arm.send_multipart([b"arm_targets", msgpack.packb({"passthrough_arm_targets": True})])
            time.sleep(0.05)

    def close(self):
        self._stop = True
