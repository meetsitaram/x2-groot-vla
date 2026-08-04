#!/usr/bin/env python3
"""Sim-vs-real panels for all four head cameras.

Anchor: manually fitted stereo-left pose in the garage/splat frame.
Propagation: factory /tf_static for the right eye; URDF joints (all on
head_pitch_link) for rear + Orbbec RGBD (with its 180-deg optical roll —
physically inverted mount, driver un-flips images).
Real frames are undistorted per their own model (fisheye k1..k4 for the
IMX900s, 8-coeff rational for the Orbbec); splat rendered with matching
pinhole intrinsics.
"""
import argparse
import json
import os

import cv2
import numpy as np
from scipy.spatial.transform import Rotation as Rot

A = os.path.expanduser("~/projects/x2-groot-vla/assets/garage")
C = f"{A}/calib_2026-08-04"
OUT = f"{C}/pose_match"
calib = json.load(open(f"{C}/calibration.json"))

ap = argparse.ArgumentParser()
ap.add_argument("--anchor", nargs=5, type=float, default=[-0.32, 0.30, 1.25, -100.0, 22.0],
                metavar=("X", "Y", "Z", "YAW", "PITCH"))
ap.add_argument("--dir", default="", help="capture subdir (e.g. standing)")
ap.add_argument("--only", default=None, help="render just this camera")
ap.add_argument("--out", default="all_cams_panel.png")
ap.add_argument("--rear-zoom", type=float, default=1.0,
                help="scale rear camera fx/fy in sim (real rear stream appears ISP-cropped vs camera_info)")
ap.add_argument("--true-fit", nargs=2, type=float, default=None,
                metavar=("DYAW", "PITCH"), help="orbbec (row5) aim fit: upright-equivalent yaw=anchor+DYAW, pitch=PITCH; rendered inverted")
ap.add_argument("--flip-true-real", action="store_true", help="show the raw orbbec real frame rotated 180 for readability")
ap.add_argument("--rgbd-fit", nargs=2, type=float, default=None,
                metavar=("DYAW", "PITCH"), help="override rgbd orientation: yaw=anchor_yaw+DYAW, pitch=PITCH (position stays URDF)")
args = ap.parse_args()
EYE_L = np.array(args.anchor[:3])
YAW_L, PITCH_L = args.anchor[3], args.anchor[4]
CAPDIR = f"{C}/{args.dir}" if args.dir else C


def opt_R(yaw_deg, pitch_deg):
    cy, sy = np.cos(np.deg2rad(yaw_deg)), np.sin(np.deg2rad(yaw_deg))
    cp, sp = np.cos(np.deg2rad(pitch_deg)), np.sin(np.deg2rad(pitch_deg))
    fwd = np.array([cy * cp, sy * cp, -sp])
    right = np.array([sy, -cy, 0.0])
    down = np.cross(fwd, right)
    return np.stack([right, down, fwd])  # world->cam rows


def homog(Rm, t):
    M = np.eye(4)
    M[:3, :3] = Rm
    M[:3, 3] = t
    return M


R_wl = opt_R(YAW_L, PITCH_L)          # world->cam
M_wl = homog(R_wl.T, EYE_L)           # cam->world

# ---------- URDF joints on head_pitch_link (optical frames) ----------
def urdf_M(xyz, rpy):
    return homog(Rot.from_euler("xyz", rpy).as_matrix(), np.array(xyz))

M_h_left = urdf_M([0.067995, 0.029784, 0.05], [-1.5708, 0, -1.574])
M_h_rear = urdf_M([-0.0834, 0.00026495, 0.0], [-1.5708, 0, 1.5676])
M_h_rgbd = urdf_M([0.05761, -0.011183, -0.04837], [2.2689, 0, 1.5708])
RZ180 = homog(Rot.from_euler("z", np.pi).as_matrix(), np.zeros(3))

M_wh = M_wl @ np.linalg.inv(M_h_left)              # head_pitch_link -> world

# factory stereo TF: left -> right (optical frames)
q = [0.0006061240771422848, 0.002241421452001928, -0.0003776389199249995, 0.9999972330123338]
t_lr = [0.0602012025776, -0.0007929398386, 0.0006029608815]
M_lr = homog(Rot.from_quat(q).as_matrix(), np.array(t_lr))

M_rgbd = M_wh @ M_h_rgbd @ RZ180
if args.rgbd_fit is not None:
    dyaw, rp = args.rgbd_fit
    M_fit = np.eye(4)
    M_fit[:3, :3] = opt_R(YAW_L + dyaw, rp).T
    M_fit[:3, 3] = M_rgbd[:3, 3]        # keep URDF chin position
    M_rgbd = M_fit
POSES = {
    "stereo_head_front_left": M_wl,
    "stereo_head_front_right": M_wl @ M_lr,
    "rgb_head_rear": M_wh @ M_h_rear,
    "rgbd_head_front_orbbec": M_rgbd,
    "rgbd_head_front_TRUE": M_wh @ M_h_rgbd,
}
if args.true_fit is not None:
    dy, tp = args.true_fit
    Mt = np.eye(4)
    Mt[:3, :3] = opt_R(YAW_L + dy, tp).T  # upright; pair with --flip-true-real
    Mt[:3, 3] = (M_wh @ M_h_rgbd)[:3, 3]
    POSES["rgbd_head_front_TRUE"] = Mt

# ---------- intrinsics / undistortion per camera ----------
CAMS = {
    "stereo_head_front_left": ("ci_stereo_head_front_left", "fisheye"),
    "stereo_head_front_right": ("ci_stereo_head_front_right", "fisheye"),
    "rgb_head_rear": ("ci_rgb_head_rear", "fisheye"),
    "rgbd_head_front_orbbec": ("ci_rgb_head_front_center", "plumb_bob"),
    "rgbd_head_front_TRUE": ("ci_rgbd_head_front_true", "plumb_bob"),
}
SCALE = 0.4

z = np.load(f"{OUT}/splat_pts.npz")
pts, cols = z["pts"], z["cols"]
rad = z["rad"] if "rad" in z.files else None


def render(M_cw, Kp, w, h, model="fisheye", D=None):
    """Project splat points through the REAL camera model (native FOV)."""
    Rm = M_cw[:3, :3].T
    eye = M_cw[:3, 3]
    pc = (pts - eye.astype(np.float32)) @ Rm.T
    dist = np.linalg.norm(pc, axis=1) + 1e-9
    if model == "fisheye":
        vis = (dist > 0.25) & (pc[:, 2] / dist > np.cos(np.deg2rad(100)))
    else:
        vis = pc[:, 2] > 0.12
    pcv, colv = pc[vis], cols[vis]
    x, y, zc = pcv[:, 0], pcv[:, 1], pcv[:, 2]
    if model == "fisheye":
        r = np.sqrt(x * x + y * y) + 1e-9
        th = np.arctan2(r, zc)
        k1, k2, k3, k4 = D[:4]
        thd = th * (1 + k1 * th**2 + k2 * th**4 + k3 * th**6 + k4 * th**8)
        u = Kp[0, 0] * thd * x / r + Kp[0, 2]
        v = Kp[1, 1] * thd * y / r + Kp[1, 2]
    else:  # rational polynomial (plumb_bob 8-coeff)
        a, b = x / zc, y / zc
        r2 = a * a + b * b
        k1, k2, p1, p2, k3, k4, k5, k6 = D[:8]
        num = 1 + k1 * r2 + k2 * r2**2 + k3 * r2**3
        den = 1 + k4 * r2 + k5 * r2**2 + k6 * r2**3
        f = num / den
        ad = a * f + 2 * p1 * a * b + p2 * (r2 + 2 * a * a)
        bd = b * f + p1 * (r2 + 2 * b * b) + 2 * p2 * a * b
        u = Kp[0, 0] * ad + Kp[0, 2]
        v = Kp[1, 1] * bd + Kp[1, 2]
    depth_key = np.linalg.norm(pcv, axis=1) if model == "fisheye" else zc
    ok = (u >= 0) & (u < w) & (v >= 0) & (v < h)
    uvv = np.stack([u[ok], v[ok]], 1).astype(np.int32)
    pz, colz = depth_key[ok], colv[ok]
    radv = rad[vis][ok] if rad is not None else None
    order = np.argsort(-pz)
    uvv, pz, colz = uvv[order], pz[order], colz[order]
    im = np.zeros((h, w, 3), np.uint8)
    if radv is not None:
        # gaussian-extent-aware: pixel radius = f * sigma / depth (capped)
        rpx = np.clip(1.6 * Kp[0, 0] * radv[order] / np.maximum(pz, 0.2), 1, 13).astype(np.int32)
        for r in range(1, 14):
            m = rpx == r
            if not m.any():
                continue
            uu, cc = uvv[m], colz[m]
            for dy in range(-r, r + 1):
                for dx in range(-r, r + 1):
                    xs = np.clip(uu[:, 0] + dx, 0, w - 1)
                    ys = np.clip(uu[:, 1] + dy, 0, h - 1)
                    im[ys, xs] = cc[:, ::-1]
    else:
        for rr, lo, hi in ((1, 2.5, 1e9), (2, 1.2, 2.5), (3, 0.0, 1.2)):
            m = (pz >= lo) & (pz < hi)
            if not m.any():
                continue
            uu, cc = uvv[m], colz[m]
            for dy in range(-rr, rr + 1):
                for dx in range(-rr, rr + 1):
                    xs = np.clip(uu[:, 0] + dx, 0, w - 1)
                    ys = np.clip(uu[:, 1] + dy, 0, h - 1)
                    im[ys, xs] = cc[:, ::-1]
    return im


rows = []
for name, (ci_key, model) in CAMS.items():
    ci = calib[ci_key]
    K = np.array(ci["k"]).reshape(3, 3)
    D = np.array(ci["d"])
    W, H = ci["w"], ci["h"]
    if args.only and name != args.only:
        continue
    fname = "rgbd_head_front_TRUE" if name == "rgbd_head_front_TRUE" else name
    img = cv2.imread(f"{CAPDIR}/{fname}.jpg")
    if name == "rgbd_head_front_TRUE" and args.flip_true_real:
        img = cv2.rotate(img, cv2.ROTATE_180)
    w, h = int(W * SCALE), int(H * SCALE)
    und = cv2.resize(img, (w, h))          # RAW frame — native fisheye FOV
    Kp = K * SCALE
    Kp[2, 2] = 1.0
    if name == "rgb_head_rear" and args.rear_zoom != 1.0:
        Kp = Kp.copy()
        Kp[0, 0] *= args.rear_zoom
        Kp[1, 1] *= args.rear_zoom
    sim = render(POSES[name], Kp, w, h, model=model, D=D)
    # normalize row width for stacking
    panel = np.concatenate([und, sim], axis=1)
    panel = cv2.resize(panel, (1650, int(panel.shape[0] * 1650 / panel.shape[1])))
    LABELS = {"rgbd_head_front_orbbec": "rgb_head_front_center_5MP",
              "rgbd_head_front_TRUE": "rgbd_head_front_orbbec (raw inverted)"}
    disp = LABELS.get(name, name)
    if name == "rgbd_head_front_TRUE" and args.flip_true_real:
        disp = "rgbd_head_front_orbbec (both flipped upright)"
    cv2.putText(panel, f"{disp}  (real | sim)", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
    rows.append(panel)
    print("done", name)

out = np.concatenate(rows, axis=0)
cv2.imwrite(f"{OUT}/{args.out}", out)
print("wrote", f"{OUT}/{args.out}", out.shape)
