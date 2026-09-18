"""Drone-style orbit capture of a splat world — scene only, no robot, no physics.

Reuses the proven x2_isaaclab_bridge patterns (GR00T-WholeBodyControl
gear_sonic_deploy/scripts/x2_isaaclab_bridge.py): NuRec USDZ load via
UsdFileCfg, the user-approved garagelab lighting (dome 300 + key 300 40 3000),
and the headless-safe Camera-sensor capture. The camera flies an ellipse
around --center at fixed --eye-z, always looking at the center. Emits
numbered PNGs into --out-dir; assemble with ffmpeg afterwards.

Run (env_isaaclab python):
    /home/stickbot/miniconda3/envs/env_isaaclab/bin/python -u \
        scripts/garagelab_orbit_shot.py \
        --scene-usdz assets/garagelab/garagelab_splat.usdz \
        --out-dir /tmp/orbit --frames 288 --headless --enable_cameras
"""

import argparse
import functools
import math
import os
import sys
import threading

print = functools.partial(print, flush=True)  # live logs must survive redirects

from isaaclab.app import AppLauncher  # noqa: E402

parser = argparse.ArgumentParser(description="splat world orbit capture")
parser.add_argument("--scene-usdz", type=str, required=True,
                    help="NuRec splat USDZ (or any USD) world file")
parser.add_argument("--out-dir", type=str, required=True,
                    help="frames land here as frame_0000.png ...")
parser.add_argument("--center", type=float, nargs=2, default=[-0.25, 0.5],
                    metavar=("X", "Y"),
                    help="orbit center on the floor (garagelab open-floor middle)")
parser.add_argument("--target-z", type=float, default=0.7,
                    help="look-at height above floor")
parser.add_argument("--eye-z", type=float, default=1.25,
                    help="camera height; keep <= ~1.3 m (scan fuzz above head height)")
parser.add_argument("--radius", type=float, nargs=2, default=[1.35, 1.95],
                    metavar=("RX", "RY"),
                    help="ellipse semi-axes; default stays ~0.3 m off the "
                         "garagelab open-floor bounds x[-2,1.5] y[-2,3]")
parser.add_argument("--frames", type=int, default=288,
                    help="288 @ 24 fps = 12 s full orbit")
parser.add_argument("--start-deg", type=float, default=-90.0,
                    help="orbit start angle (deg, 0 = +x from center)")
parser.add_argument("--sweep-deg", type=float, default=360.0)
parser.add_argument("--width", type=int, default=1280)
parser.add_argument("--height", type=int, default=720)
parser.add_argument("--accum", type=int, default=12,
                    help="sim.render() calls per frame (RTX sample accumulation)")
parser.add_argument("--dome-intensity", type=float, default=300.0)
parser.add_argument("--key-light", type=float, nargs=3, default=[300.0, 40.0, 3000.0],
                    metavar=("AZ_DEG", "EL_DEG", "INTENSITY"))
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import numpy as np  # noqa: E402
import torch  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.sim import SimulationContext  # noqa: E402
from isaaclab.sim.spawners.from_files import UsdFileCfg  # noqa: E402
from isaaclab.sensors import Camera, CameraCfg  # noqa: E402


def light_quat_from_azel(az_deg: float, el_deg: float):
    """Quaternion (w,x,y,z) that points a DistantLight (shines along local -Z)
    so its rays come FROM azimuth/elevation (deg) toward the scene."""
    az, el = math.radians(az_deg), math.radians(el_deg)
    z = np.array([math.cos(el) * math.cos(az), math.cos(el) * math.sin(az), math.sin(el)])
    x = np.cross([0.0, 0.0, 1.0], z)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    m = np.stack([x, y, z], axis=1)
    w = math.sqrt(max(0.0, 1.0 + m[0, 0] + m[1, 1] + m[2, 2])) / 2.0
    if w > 1e-6:
        qx = (m[2, 1] - m[1, 2]) / (4 * w)
        qy = (m[0, 2] - m[2, 0]) / (4 * w)
        qz = (m[1, 0] - m[0, 1]) / (4 * w)
    else:  # 180-degree case: not reachable for el in (-90, 90)
        qx, qy, qz = 1.0, 0.0, 0.0
    return (float(w), float(qx), float(qy), float(qz))


def main():
    sim = SimulationContext(sim_utils.SimulationCfg(dt=0.005, render_interval=4))

    scene_path = os.path.expanduser(args.scene_usdz)
    if not os.path.exists(scene_path):
        sys.exit(f"[orbit] world file not found: {scene_path}")
    UsdFileCfg(usd_path=scene_path).func("/World/Splat", UsdFileCfg(usd_path=scene_path))
    print(f"[orbit] world: {scene_path}")

    dome = sim_utils.DomeLightCfg(intensity=args.dome_intensity, color=(0.92, 0.92, 0.88))
    dome.func("/World/DomeLight", dome)
    az, el, inten = args.key_light
    key = sim_utils.DistantLightCfg(intensity=inten, color=(1.0, 0.98, 0.94), angle=1.0)
    key.func("/World/KeyLight", key, orientation=light_quat_from_azel(az, el))
    cx, cy = args.center
    fill = sim_utils.SphereLightCfg(intensity=inten * 0.12, radius=0.3,
                                    color=(0.95, 0.97, 1.0))
    fill.func("/World/FillLight", fill, translation=(cx, cy, args.eye_z + 0.4))
    print(f"[orbit] key light az={az:.0f} el={el:.0f} I={inten:.0f}, "
          f"dome {args.dome_intensity:.0f}")

    cam = Camera(CameraCfg(
        prim_path="/World/ShotCam",
        width=args.width, height=args.height, data_types=["rgb"], update_period=0.0,
        spawn=sim_utils.PinholeCameraCfg(clipping_range=(0.05, 100.0)),
    ))

    sim.reset()
    for _ in range(60):  # first-frame RTX warmup
        sim.render()

    os.makedirs(args.out_dir, exist_ok=True)
    from PIL import Image
    rx, ry = args.radius
    tgt = torch.tensor([[cx, cy, args.target_z]], device=sim.device)
    for i in range(args.frames):
        ang = math.radians(args.start_deg + args.sweep_deg * i / args.frames)
        eye = torch.tensor([[cx + rx * math.cos(ang), cy + ry * math.sin(ang),
                             args.eye_z]], device=sim.device)
        cam.set_world_poses_from_view(eye, tgt)
        for _ in range(args.accum):
            sim.render()
        cam.update(0.0)
        rgb = cam.data.output["rgb"][0].cpu().numpy()
        Image.fromarray(rgb[..., :3].astype(np.uint8)).save(
            os.path.join(args.out_dir, f"frame_{i:04d}.png"))
        if i % 10 == 0 or i == args.frames - 1:
            print(f"[orbit] frame {i + 1}/{args.frames} "
                  f"eye=({eye[0,0]:.2f},{eye[0,1]:.2f},{args.eye_z:.2f})")
    print(f"[orbit] DONE: {args.frames} frames in {os.path.abspath(args.out_dir)}")

    # close() is known to hang with cameras enabled — watchdog it, then hard-exit.
    closer = threading.Thread(target=simulation_app.close, daemon=True)
    closer.start()
    closer.join(timeout=20.0)
    if closer.is_alive():
        print("[orbit] close() hung — hard exit")
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
