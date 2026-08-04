#!/usr/bin/env python3
"""Interactive MuJoCo viewer: X2 + counter fixture (0.876 m) + Lightwheel objects.

Teleop-scene prototype for the M1 counter corpus (PLAN.md). Composes the
scene with MjSpec.attach — Lightwheel MJCF objects merge natively.

Run with the GR00T-WholeBodyControl venv:
    cd ~/Projects/GR00T-WholeBodyControl
    .venv/bin/python ~/projects/x2-groot-vla/scripts/view_counter_scene.py \
        [--objects kettle/Kettle001 pot/Pot052] [--counter-top 0.876]
"""
import argparse

import mujoco
import mujoco.viewer
import numpy as np

REPO = "/home/stickbot/Projects/GR00T-WholeBodyControl"
LW = ("/home/stickbot/projects/g1-kitchen-sim/assets/nvidia_kitchen/"
      "kitchen_assets/objects_extracted/lightwheel")

p = argparse.ArgumentParser()
p.add_argument("--objects", nargs="*", default=["kettle/Kettle001", "pot/Pot052"],
               help="category/Variant paths under the lightwheel dir")
p.add_argument("--counter-top", type=float, default=0.876)
p.add_argument("--counter-face-x", type=float, default=0.35)
args = p.parse_args()

spec = mujoco.MjSpec.from_file(
    f"{REPO}/gear_sonic/data/assets/robot_description/mjcf/x2_ultra.xml")
half = (0.325, 0.9, args.counter_top / 2)
counter = spec.worldbody.add_body(
    name="counter", pos=[args.counter_face_x + half[0], 0.0, args.counter_top / 2])
counter.add_geom(name="counter_geom", type=mujoco.mjtGeom.mjGEOM_BOX,
                 size=list(half), rgba=[0.82, 0.80, 0.75, 1.0])

for i, obj in enumerate(args.objects):
    sub = mujoco.MjSpec.from_file(f"{LW}/{obj}/model.xml")
    y = -0.25 + 0.55 * i
    frame = spec.worldbody.add_frame(pos=[0.55, y, args.counter_top],
                                     quat=[1, 0, 0, 0])
    frame.attach_body(sub.worldbody, obj.split("/")[1] + "_", "")

model = spec.compile()
data = mujoco.MjData(model)
# NOTE: no controller is attached — keep the viewer PAUSED for inspection;
# unpausing lets the uncontrolled robot slump (objects are fine either way).
mujoco.mj_forward(model, data)
mujoco.viewer.launch(model, data)
