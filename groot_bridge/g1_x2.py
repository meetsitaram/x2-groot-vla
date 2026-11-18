"""Analytic G1 <-> X2 translator for GR00T checkpoints with the G1 full-body + nav-command interface
(unitree_g1_full_body_with_waist_height_nav_cmd layout; e.g. nvidia/G1-Factory-GR00T-1.7).

Model state  (31): left_arm 7, right_arm 7, left_hand 7, right_hand 7, waist 3
Model action (35/step x horizon): the same 31 + base_height_command 1 + navigate_command 3 (vx, vy, wz)

Joint facts (groot_bridge/tools/compare_joint_axes.py on the two MJCFs, zero pose):
  - all 14 arm + 3 waist axes agree in direction (axis dot >= 0.98), so no sign flips;
  - order differs: G1 wrist = roll, pitch, yaw; X2 wrist = yaw, pitch, roll; G1 waist = yaw, roll, pitch;
    X2 waist = yaw, pitch, roll;
  - the elbow zero differs: G1 elbow 0 = 90 deg bent (forearm forward), X2 elbow 0 = straight, so
    X2_elbow = G1_elbow - pi/2 (checked by FK: both give forearm-forward at G1 0 / X2 -pi/2).
Hands: G1 three-finger hand (7) <-> X2 OmniHand (10) through one grasp ratio per hand (the G1 Factory
data drives the hand like a gripper). The X2 open/closed poses come from gwbc-x2 at run time.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from .embodiment import RobotCommand, RobotState, TargetSpec

GWBC_X2 = Path.home() / "Projects/gwbc-x2"
G1_XML = Path.home() / "Projects/GR00T-WholeBodyControl/gear_sonic_deploy/g1/g1_29dof_with_hand.xml"
X2_XML = Path.home() / "Projects/GR00T-WholeBodyControl/gear_sonic/data/assets/robot_description/mjcf/x2_ultra.xml"
# X2 + OmniHand (finger bodies for the hand-geometry alignment); built by tools/build_x2_console_scene.py
X2_HAND_XML = Path(__file__).resolve().parents[1] / "runs/groot_bridge/scenes/x2_boxlift_w21.xml"

# X2 31-joint mujoco order indices
X2_WAIST = {"yaw": 12, "pitch": 13, "roll": 14}
X2_ARM = {"left": 15, "right": 22}            # shoulder_pitch, roll, yaw, elbow, wrist_yaw, wrist_pitch, wrist_roll
# G1 arm order: shoulder_pitch, roll, yaw, elbow, wrist_roll, wrist_pitch, wrist_yaw
G1_ARM_FROM_X2 = [0, 1, 2, 3, 6, 5, 4]         # G1 slot i <- X2 arm slot G1_ARM_FROM_X2[i]
X2_ARM_FROM_G1 = [0, 1, 2, 3, 6, 5, 4]         # (self-inverse permutation)
ELBOW = 3
ELBOW_OFFSET = np.pi / 2                       # G1 = X2 + pi/2
X2_STAND_HIP_M = 0.687                         # kplanner _HIP_HEIGHT_M
X2_MIN_HIP_M = 0.60                            # below this needs crouch mode; v1 stays standing


def _x2_limits():
    import mujoco
    xml = Path.home() / "Projects/GR00T-WholeBodyControl/gear_sonic/data/assets/robot_description/mjcf/x2_ultra.xml"
    m = mujoco.MjModel.from_xml_path(str(xml))
    lo, hi = [], []
    for j in range(m.njnt):
        if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE:
            lo.append(m.jnt_range[j][0]); hi.append(m.jnt_range[j][1])
    return np.array(lo, np.float32), np.array(hi, np.float32)


def _hand_poses():
    if str(GWBC_X2) not in sys.path:
        sys.path.insert(0, str(GWBC_X2))
    from gear_sonic.utils.teleop import x2_hand_retarget as hr   # gwbc-x2, used at run time, not copied
    return {"left": (np.array(hr.HAND_GRASP_OPEN_RAD_LEFT), np.array(hr.HAND_GRASP_CLOSED_RAD_LEFT)),
            "right": (np.array(hr.HAND_GRASP_OPEN_RAD_RIGHT), np.array(hr.HAND_GRASP_CLOSED_RAD_RIGHT))}, \
        hr.grasp_command_from_ratio


class G1NavToX2:
    """ObsEncoder + ActionDecoder pair (analytic). Learned/hybrid heads can wrap either method."""

    def __init__(self, spec: TargetSpec, max_speed=(0.5, 0.3, 0.8), hip_scale=None, arm_mode="hand",
                 squeeze_offset=0.0, squeeze_gap=0.32, hand_align="wrist"):
        """arm_mode: "hand" = hand-space retarget (G1 FK -> X2 IK, keeps where the hands go; needed for
        two-handed tasks), "joint" = copy joint angles (permutation + elbow offset; keeps arm shape)."""
        self.spec = spec
        self.arm_mode = arm_mode
        self.hand_align = hand_align          # "wrist" (wrist frames) or "geom" (hand geometry + palm centre)
        # bimanual squeeze correction (hand space): when the source wrists are closer than squeeze_gap
        # (closing on an object), move each target wrist squeeze_offset further inward. Compensates
        # palm-thickness / compliance differences between the two hands; a learned head can replace it.
        self.squeeze_offset, self.squeeze_gap = float(squeeze_offset), float(squeeze_gap)
        if arm_mode == "hand":
            self._init_hand_space()
        self.lo, self.hi = _x2_limits()
        self.hand_pose, self._grasp = _hand_poses()
        # G1 hand: per-joint "fully closed" extreme from the checkpoint's action statistics
        self.g1_hand_closed = {}
        for side in ("left", "right"):
            lo, hi = spec.action_range[f"{side}_hand"]
            self.g1_hand_closed[side] = np.where(np.abs(lo) > np.abs(hi), lo, hi)
        g1_h_lo, g1_h_hi = spec.action_range["base_height_command"]
        self.g1_stand_h = float(g1_h_hi[0])
        self.hip_scale = hip_scale or X2_STAND_HIP_M / self.g1_stand_h
        self.max_speed = np.array(max_speed, np.float32)            # |vx|, |vy|, |wz| caps on X2
        self.stats = {"clamped_joints": 0, "steps": 0}

    def _init_hand_space(self):
        import mujoco
        from .ee_retarget import ARM, ArmModel, HandSpaceRetarget, hand_geometry
        g1m = mujoco.MjModel.from_xml_path(str(G1_XML))
        x2_xml = X2_HAND_XML if self.hand_align == "geom" else X2_XML
        x2m = mujoco.MjModel.from_xml_path(str(x2_xml))
        g1_ref = np.zeros(7)                                   # G1 elbow 0 = bent 90 deg
        x2_ref = np.array([0, 0, 0, -np.pi / 2, 0, 0, 0])      # same arm shape on the X2
        self.to_x2, self.to_g1 = {}, {}
        for side in ("left", "right"):
            gj = [f"{side}_{j}_joint" for j in ARM]
            kw_g, kw_x, hk = {}, {}, {}
            if self.hand_align == "geom":
                S = "L" if side == "left" else "R"
                Hg, pg = hand_geometry(g1m, f"{side}_wrist_yaw_link", f"{side}_hand_middle_0_link",
                                       f"{side}_hand_thumb_0_link", g1_ref, gj)
                Hx, px = hand_geometry(x2m, f"{side}_wrist_roll_link", f"{side}_{S}_middle_abad",
                                       f"{side}_{S}_thumb_roll", x2_ref, gj)
                kw_g, kw_x = {"tcp_local": pg}, {"tcp_local": px}
                hk = {"g": Hg, "x": Hx}
            g = ArmModel(g1m, gj, f"{side}_wrist_yaw_link", **kw_g)
            x = ArmModel(x2m, gj, f"{side}_wrist_roll_link", **kw_x)
            self.to_x2[side] = HandSpaceRetarget(g, x, g1_ref, x2_ref, src_hand_local=hk.get("g"), dst_hand_local=hk.get("x"))
            g2 = ArmModel(g1m, gj, f"{side}_wrist_yaw_link", **kw_g)
            x2 = ArmModel(x2m, gj, f"{side}_wrist_roll_link", **kw_x)
            self.to_g1[side] = HandSpaceRetarget(x2, g2, x2_ref, g1_ref, src_hand_local=hk.get("x"), dst_hand_local=hk.get("g"))
        self.ik_err = {"x2": [], "g1": []}

    # semantic ARM order (sp, sr, sy, elbow, wrist_roll, wrist_pitch, wrist_yaw) <-> X2 slot order
    @staticmethod
    def _sem_to_x2(q):
        return np.asarray(q, np.float32)[[0, 1, 2, 3, 6, 5, 4]]

    _x2_to_sem = _sem_to_x2                                    # self-inverse

    # ------------------------------------------------------------------ X2 -> G1 (observation)
    def x2_arm_to_g1(self, x2_arm7, side="left"):
        if self.arm_mode == "hand":
            q = self.to_g1[side](self._x2_to_sem(x2_arm7))
            self.ik_err["g1"].append(self.to_g1[side].last_err[0])
            return q.astype(np.float32)
        g = np.asarray(x2_arm7, np.float32)[G1_ARM_FROM_X2].copy()
        g[ELBOW] += ELBOW_OFFSET
        return g

    def x2_hand_to_g1(self, side, x2_hand10):
        o, c = self.hand_pose[side]
        span = np.where(np.abs(c - o) > 1e-3, c - o, 1.0)
        ratio = float(np.clip(np.mean((np.asarray(x2_hand10) - o) / span), 0.0, 1.0))
        return (ratio * self.g1_hand_closed[side]).astype(np.float32)

    def encode(self, state: RobotState, instruction: str) -> dict:
        q = np.asarray(state.joint_q, np.float32)
        st = {
            "left_arm": self.x2_arm_to_g1(q[X2_ARM["left"]:X2_ARM["left"] + 7], "left"),
            "right_arm": self.x2_arm_to_g1(q[X2_ARM["right"]:X2_ARM["right"] + 7], "right"),
            "left_hand": self.x2_hand_to_g1("left", state.left_hand_q),
            "right_hand": self.x2_hand_to_g1("right", state.right_hand_q),
            "waist": np.array([q[X2_WAIST["yaw"]], q[X2_WAIST["roll"]], q[X2_WAIST["pitch"]]], np.float32),
        }
        vk = self.spec.video_keys[0]
        img = state.images[vk]
        H, W = self.spec.image_hw
        if img.shape[:2] != (H, W):
            import cv2
            img = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
        return {"video": {vk: img[None, None]},
                "state": {k: v[None, None] for k, v in st.items()},
                "language": {self.spec.language_key: [[instruction]]}}

    # ------------------------------------------------------------------ G1 -> X2 (action)
    def g1_arm_to_x2(self, side, g1_arm7, extra_offset=None):
        if self.arm_mode == "hand":
            x = self._sem_to_x2(self.to_x2[side](np.asarray(g1_arm7, float), extra_offset))
            self.ik_err["x2"].append(self.to_x2[side].last_err[0])
        else:
            x = np.asarray(g1_arm7, np.float32)[X2_ARM_FROM_G1].copy()
            x[ELBOW] -= ELBOW_OFFSET
        base = X2_ARM[side]
        xc = np.clip(x, self.lo[base:base + 7], self.hi[base:base + 7])
        self.stats["clamped_joints"] += int(np.sum(np.abs(xc - x) > 1e-4))
        return xc

    def g1_hand_to_x2(self, side, g1_hand7):
        closed = self.g1_hand_closed[side]
        ok = np.abs(closed) > 0.05
        ratio = float(np.clip(np.mean(np.asarray(g1_hand7)[ok] / closed[ok]), 0.0, 1.0))
        return self._grasp(side, ratio).astype(np.float32), ratio

    def decode(self, action: dict, state: RobotState | None = None) -> list[RobotCommand]:
        """action: key -> (T, dim) (batch already removed). Returns T RobotCommands at spec.control_hz."""
        T = next(iter(action.values())).shape[0]
        out = []
        for t in range(T):
            a = {k: np.asarray(v[t], np.float32) for k, v in action.items()}
            vx, vy, wz = a["navigate_command"]
            vel = np.clip(np.array([vx, vy, wz], np.float32), -self.max_speed, self.max_speed)
            hip = float(np.clip(a["base_height_command"][0] * self.hip_scale, X2_MIN_HIP_M, X2_STAND_HIP_M))
            w = a["waist"]                                        # G1 yaw, roll, pitch
            waist = np.array([w[0], w[2], w[1]], np.float32)      # X2 yaw, pitch, roll
            waist = np.clip(waist, self.lo[12:15], self.hi[12:15])
            lh, _ = self.g1_hand_to_x2("left", a["left_hand"])
            rh, _ = self.g1_hand_to_x2("right", a["right_hand"])
            offs = {"left": None, "right": None}
            if self.arm_mode == "hand" and self.squeeze_offset:
                pl, _ = self.to_x2["left"].src.fk(a["left_arm"]); pr, _ = self.to_x2["right"].src.fk(a["right_arm"])
                if np.linalg.norm(pl - pr) < self.squeeze_gap:
                    offs = {"left": np.array([0, -self.squeeze_offset, 0]), "right": np.array([0, self.squeeze_offset, 0])}
                    self.stats["squeeze_steps"] = self.stats.get("squeeze_steps", 0) + 1
            out.append(RobotCommand(left_arm_q=self.g1_arm_to_x2("left", a["left_arm"], offs["left"]),
                                    right_arm_q=self.g1_arm_to_x2("right", a["right_arm"], offs["right"]),
                                    left_hand_q=lh, right_hand_q=rh, base_velocity=vel,
                                    hip_height_m=hip, waist_q=waist))
            self.stats["steps"] += 1
        return out
