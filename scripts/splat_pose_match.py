#!/usr/bin/env python3
"""First-light camera pose search: real fisheye frame vs garage splat.

1. Undistorts the real stereo-left fisheye capture to a pinhole view
   (OpenCV equidistant model, K/D from calibration.json).
2. Renders the garage splat point cloud from candidate camera poses with the
   same pinhole intrinsics (fast numpy z-buffer point projector — no Isaac).
3. Writes side-by-side panels per candidate for visual pose refinement.

Usage (g1-kitchen-sim venv has plyfile+cv2):
    ~/projects/g1-kitchen-sim/.venv/bin/python scripts/splat_pose_match.py \
        --pose X Y Z YAW_DEG PITCH_DEG [--tag name]
Multiple --pose flags allowed.
"""
import argparse
import json
import os

import cv2
import numpy as np

A = os.path.expanduser("~/projects/x2-groot-vla/assets/garage")
C = f"{A}/calib_2026-08-04"
OUT = f"{C}/pose_match"
os.makedirs(OUT, exist_ok=True)

calib = json.load(open(f"{C}/calibration.json"))
ci = calib["ci_stereo_head_front_left"]
K = np.array(ci["k"]).reshape(3, 3)
D = np.array(ci["d"])
W, H = ci["w"], ci["h"]

# --- undistort real frame to pinhole (once) -------------------------------
und_path = f"{OUT}/real_undistorted.png"
new_K = None
SCALE = 0.5  # work at half res for speed
if True:
    img = cv2.imread(f"{C}/stereo_head_front_left.jpg")
    new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
        K, D, (W, H), np.eye(3), balance=0.0)
    m1, m2 = cv2.fisheye.initUndistortRectifyMap(
        K, D, np.eye(3), new_K, (W, H), cv2.CV_16SC2)
    und = cv2.remap(img, m1, m2, interpolation=cv2.INTER_LINEAR)
    und_small = cv2.resize(und, (int(W * SCALE), int(H * SCALE)))
    cv2.imwrite(und_path, und_small)

Kp = new_K.copy() * SCALE
Kp[2, 2] = 1.0
w, h = int(W * SCALE), int(H * SCALE)

# --- load splat points (cached) -------------------------------------------
cache = f"{OUT}/splat_pts.npz"
if os.path.exists(cache):
    z = np.load(cache)
    pts, cols = z["pts"], z["cols"]
else:
    from plyfile import PlyData
    p = PlyData.read(f"{A}/garage_splat.ply")
    v = p["vertex"]
    xyz = np.c_[v["x"], v["y"], v["z"]]
    op = 1 / (1 + np.exp(-np.asarray(v["opacity"])))
    C0 = 0.28209479177
    rgb = np.clip(np.c_[v["f_dc_0"], v["f_dc_1"], v["f_dc_2"]] * C0 + 0.5, 0, 1)
    wpts = np.c_[xyz[:, 0], -xyz[:, 2], xyz[:, 1] + 1.356201171875]
    sel = op > 0.45
    pts, cols = wpts[sel].astype(np.float32), (rgb[sel] * 255).astype(np.uint8)
    np.savez_compressed(cache, pts=pts, cols=cols)
print(f"splat points: {len(pts)}")


def render(eye, yaw_deg, pitch_deg):
    """Pinhole render of the splat points. Camera: +z optical forward,
    yaw about world z (0 = +x), pitch down positive."""
    cy, sy = np.cos(np.deg2rad(yaw_deg)), np.sin(np.deg2rad(yaw_deg))
    cp, sp = np.cos(np.deg2rad(pitch_deg)), np.sin(np.deg2rad(pitch_deg))
    fwd = np.array([cy * cp, sy * cp, -sp])
    right = np.array([sy, -cy, 0.0])         # fwd x world_up
    down = np.cross(fwd, right)              # completes right-handed x,y,z=r,d,f
    R = np.stack([right, down, fwd])         # rows: cam x (right), y (down), z (fwd)
    pc = (pts - np.asarray(eye, np.float32)) @ R.T
    vis = pc[:, 2] > 0.15
    pcv, colv = pc[vis], cols[vis]
    uv = (pcv[:, :2] / pcv[:, 2:3]) @ np.array([[Kp[0, 0], 0], [0, Kp[1, 1]]]) + Kp[:2, 2]
    ok = (uv[:, 0] >= 0) & (uv[:, 0] < w) & (uv[:, 1] >= 0) & (uv[:, 1] < h)
    uvv, pz, colz = uv[ok].astype(np.int32), pcv[ok, 2], colv[ok]
    order = np.argsort(-pz)  # far first, near overwrites
    im = np.zeros((h, w, 3), np.uint8)
    # splat as 3x3 blocks for coverage
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            xs = np.clip(uvv[order, 0] + dx, 0, w - 1)
            ys = np.clip(uvv[order, 1] + dy, 0, h - 1)
            im[ys, xs] = colz[order][:, ::-1]
    return im


ap = argparse.ArgumentParser()
ap.add_argument("--pose", nargs=5, action="append", type=float, required=True,
                metavar=("X", "Y", "Z", "YAW", "PITCH"))
ap.add_argument("--tag", default="cand")
args = ap.parse_args()

real = cv2.imread(und_path)
for i, (x, y, z_, yaw, pitch) in enumerate(args.pose):
    sim = render((x, y, z_), yaw, pitch)
    panel = np.concatenate([real, sim], axis=1)
    blend = cv2.addWeighted(real, 0.5, sim, 0.5, 0)
    cv2.putText(panel, f"pose=({x:.2f},{y:.2f},{z_:.2f}) yaw={yaw:.0f} pitch={pitch:.0f}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
    cv2.imwrite(f"{OUT}/{args.tag}_{i:02d}.png", panel)
    cv2.imwrite(f"{OUT}/{args.tag}_{i:02d}_blend.png", blend)
    print(f"{args.tag}_{i:02d}: pose=({x},{y},{z_}) yaw={yaw} pitch={pitch}")
