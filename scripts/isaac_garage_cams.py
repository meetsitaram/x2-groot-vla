#!/usr/bin/env python3
"""Render the 5 fitted camera views of the garage splat in Isaac (NuRec —
no point-splat speckle). Poses/intrinsics from isaac_cams.json."""
import json
import os

from isaaclab.app import AppLauncher

app_launcher = AppLauncher(headless=True, enable_cameras=True)
simulation_app = app_launcher.app

import numpy as np
import torch
from isaaclab.sensors import Camera, CameraCfg
import isaaclab.sim as sim_utils
from isaaclab.sim import SimulationContext
from isaaclab.sim.spawners.from_files import UsdFileCfg

C = "/home/stickbot/projects/x2-groot-vla/assets/garage"
P = f"{C}/calib_2026-08-04/pose_match"
cams = json.load(open(f"{P}/isaac_cams.json"))

sim = SimulationContext(sim_utils.SimulationCfg(dt=0.01))
UsdFileCfg(usd_path=f"{C}/garage_splat.usdz").func(
    "/World/Splat", UsdFileCfg(usd_path=f"{C}/garage_splat.usdz"))
dome = sim_utils.DomeLightCfg(intensity=1000.0)
dome.func("/World/DomeLight", dome)

sensors = {}
SCALE = 0.5
for i, (name, c) in enumerate(cams.items()):
    w, h = int(c["W"] * SCALE), int(c["H"] * SCALE)
    fx = c["fx"] * SCALE
    ha = 20.955
    fl = fx * ha / w
    sensors[name] = Camera(CameraCfg(
        prim_path=f"/World/Cam{i}",
        width=w, height=h, data_types=["rgb"], update_period=0.0,
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=fl, horizontal_aperture=ha,
            clipping_range=(0.05, 60.0)),
    ))

sim.reset()
for _ in range(5):
    sim.step()

from PIL import Image
for name, c in cams.items():
    cam = sensors[name]
    pos = torch.tensor([c["pos"]], dtype=torch.float32)
    quat = torch.tensor([c["quat_wxyz"]], dtype=torch.float32)
    cam.set_world_poses(pos, quat, convention="ros")   # ros = z-forward optical
    for _ in range(40):
        sim.render()
    cam.update(0.01)
    rgb = cam.data.output["rgb"][0].cpu().numpy()
    Image.fromarray(rgb[..., :3].astype(np.uint8)).save(f"{P}/isaac_{name}.png")
    print("saved", name, rgb.shape, flush=True)
simulation_app.close()
