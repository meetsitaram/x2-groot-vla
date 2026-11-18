"""Compare G1 and X2 arm / waist / hand joints at the zero pose: world axis, limits, and a proposed
G1 -> X2 map (index, sign). Read-only; prints a table and writes the map as JSON.

  python groot_bridge/tools/compare_joint_axes.py --out groot_bridge/specs/g1_to_x2_joint_map.json
"""
import argparse
import json
from pathlib import Path

import mujoco
import numpy as np

WBC = Path.home() / "Projects/GR00T-WholeBodyControl"
G1_XML = WBC / "gear_sonic/data/robots/g1/g1_29dof_with_hand_rev_1_0.xml"
X2_XML = WBC / "gear_sonic/data/assets/robot_description/mjcf/x2_ultra.xml"

# GR00T G1 (unitree_g1_full_body_with_waist_height_nav_cmd) state/action groups, Unitree joint order.
G1_GROUPS = {
    "left_arm": ["left_shoulder_pitch", "left_shoulder_roll", "left_shoulder_yaw", "left_elbow",
                 "left_wrist_roll", "left_wrist_pitch", "left_wrist_yaw"],
    "right_arm": ["right_shoulder_pitch", "right_shoulder_roll", "right_shoulder_yaw", "right_elbow",
                  "right_wrist_roll", "right_wrist_pitch", "right_wrist_yaw"],
    "waist": ["waist_yaw", "waist_roll", "waist_pitch"],
    "left_hand": ["left_hand_thumb_0", "left_hand_thumb_1", "left_hand_thumb_2", "left_hand_middle_0",
                  "left_hand_middle_1", "left_hand_index_0", "left_hand_index_1"],
    "right_hand": ["right_hand_thumb_0", "right_hand_thumb_1", "right_hand_thumb_2", "right_hand_middle_0",
                   "right_hand_middle_1", "right_hand_index_0", "right_hand_index_1"],
}


def joints(model):
    d = mujoco.MjData(model)
    mujoco.mj_forward(model, d)
    out = {}
    for j in range(model.njnt):
        if model.jnt_type[j] != mujoco.mjtJoint.mjJNT_HINGE:
            continue
        name = model.joint(j).name.removesuffix("_joint")
        axis = d.xaxis[j].copy()
        lo, hi = model.jnt_range[j] if model.jnt_limited[j] else (-np.inf, np.inf)
        out[name] = {"axis": axis, "range": (float(lo), float(hi)), "qadr": int(model.jnt_qposadr[j])}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()
    g1 = joints(mujoco.MjModel.from_xml_path(str(G1_XML)))
    x2m = mujoco.MjModel.from_xml_path(str(X2_XML))
    x2 = joints(x2m)
    x2_order = [n for n in x2]                       # hinge order == the 31-joint mujoco order
    print(f"G1 hinges: {len(g1)}   X2 hinges: {len(x2)}")
    print("X2 order:", x2_order)
    jmap = {}
    for group, names in G1_GROUPS.items():
        print(f"\n== {group}")
        jmap[group] = []
        for n in names:
            g = g1.get(n)
            if g is None:
                print(f"  {n:24s} MISSING in G1 model"); jmap[group].append(None); continue
            x = x2.get(n)
            if x is None:
                print(f"  {n:24s} G1 axis {np.round(g['axis'], 2)} range {np.round(g['range'], 2)}   -> no X2 joint")
                jmap[group].append({"g1": n, "x2": None}); continue
            dot = float(np.dot(g["axis"], x["axis"]))
            sign = 1 if dot > 0.5 else (-1 if dot < -0.5 else 0)
            print(f"  {n:24s} G1 {np.round(g['axis'], 2)} {np.round(g['range'], 2)}  X2 {np.round(x['axis'], 2)} "
                  f"{np.round(x['range'], 2)}  dot {dot:+.2f} sign {sign:+d}  x2_idx {x2_order.index(n)}")
            jmap[group].append({"g1": n, "x2": n, "x2_idx": x2_order.index(n), "sign": sign, "axis_dot": round(dot, 3),
                                "g1_range": g["range"], "x2_range": x["range"]})
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps({"g1_xml": str(G1_XML), "x2_xml": str(X2_XML), "x2_order": x2_order,
                                     "map": jmap}, indent=1))
        print("\nwrote", a.out)


if __name__ == "__main__":
    main()
