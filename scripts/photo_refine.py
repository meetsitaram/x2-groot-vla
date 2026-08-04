#!/usr/bin/env python3
"""Photometric pose refinement: optimize camera pose so the splat render's
edges + yellow-floor-line mask align with the real (undistorted) frame.
Feature-free — robust to the point-splat appearance gap."""
import json
import os
import sys

import cv2
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation as Rot

A = os.path.expanduser("~/projects/x2-groot-vla/assets/garage")
C = f"{A}/calib_2026-08-04"
OUT = f"{C}/pose_match"
calib = json.load(open(f"{C}/calibration.json"))
z = np.load(f"{OUT}/splat_pts.npz")
pts, cols = z["pts"], z["cols"]

NAME = sys.argv[1] if len(sys.argv) > 1 else "stereo_head_front_left"
CI = {"stereo_head_front_left": ("ci_stereo_head_front_left", "fisheye"),
      "stereo_head_front_right": ("ci_stereo_head_front_right", "fisheye"),
      "rgb_head_rear": ("ci_rgb_head_rear", "fisheye"),
      "rgbd_head_front_orbbec": ("ci_rgb_head_front_center", "plumb_bob")}[NAME]
SEED = [float(v) for v in sys.argv[2:8]] if len(sys.argv) >= 8 else [-0.25, 0.55, 1.42, -75, 15, 0]

ci = calib[CI[0]]
K = np.array(ci["k"]).reshape(3, 3)
D = np.array(ci["d"])
W, H = ci["w"], ci["h"]
img = cv2.imread(f"{C}/standing/{NAME}.jpg")
if CI[1] == "fisheye":
    newK = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(K, D, (W, H), np.eye(3), balance=0.35)
    m1, m2 = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), newK, (W, H), cv2.CV_16SC2)
else:
    newK, _ = cv2.getOptimalNewCameraMatrix(K, D, (W, H), 0.35)
    m1, m2 = cv2.initUndistortRectifyMap(K, D, None, newK, (W, H), cv2.CV_16SC2)
und_full = cv2.remap(img, m1, m2, interpolation=cv2.INTER_LINEAR)

S = 640.0 / W
w, h = int(W * S), int(H * S)
und = cv2.resize(und_full, (w, h))
Kp = newK * S
Kp[2, 2] = 1.0


def feat(im):
    g = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    e = cv2.Sobel(g, cv2.CV_32F, 1, 0) ** 2 + cv2.Sobel(g, cv2.CV_32F, 0, 1) ** 2
    e = np.sqrt(e)
    e = cv2.GaussianBlur(e, (0, 0), 4)
    e /= e.max() + 1e-6
    hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
    yel = cv2.inRange(hsv, (15, 60, 60), (40, 255, 255)).astype(np.float32) / 255
    yel = cv2.GaussianBlur(yel, (0, 0), 4)
    return e, yel


REAL_E, REAL_Y = feat(und)


def pose_M(p):
    x, y, zc, yaw, pitch, roll = p
    cy, sy = np.cos(np.deg2rad(yaw)), np.sin(np.deg2rad(yaw))
    cp, sp = np.cos(np.deg2rad(pitch)), np.sin(np.deg2rad(pitch))
    fwd = np.array([cy * cp, sy * cp, -sp])
    right = np.array([sy, -cy, 0.0])
    down = np.cross(fwd, right)
    R = np.stack([right, down, fwd])
    R = Rot.from_euler("z", np.deg2rad(roll)).as_matrix() @ R  # roll about optical axis
    return R, np.array([x, y, zc])


def render(R, eye):
    pc = (pts - eye.astype(np.float32)) @ R.T
    vis = pc[:, 2] > 0.12
    pcv, colv = pc[vis], cols[vis]
    uv = (pcv[:, :2] / pcv[:, 2:3]) @ np.array([[Kp[0, 0], 0], [0, Kp[1, 1]]]) + Kp[:2, 2]
    ok = (uv[:, 0] >= 0) & (uv[:, 0] < w) & (uv[:, 1] >= 0) & (uv[:, 1] < h)
    uvv, pz, colz = uv[ok].astype(np.int32), pcv[ok, 2], colv[ok]
    order = np.argsort(-pz)
    uvv, colz, pz = uvv[order], colz[order], pz[order]
    im = np.zeros((h, w, 3), np.uint8)
    for rad, lo, hi in ((1, 2.0, 1e9), (2, 0.0, 2.0)):
        m = (pz >= lo) & (pz < hi)
        if not m.any():
            continue
        uu, cc = uvv[m], colz[m]
        for dy in range(-rad, rad + 1):
            for dx in range(-rad, rad + 1):
                xs = np.clip(uu[:, 0] + dx, 0, w - 1)
                ys = np.clip(uu[:, 1] + dy, 0, h - 1)
                im[ys, xs] = cc[:, ::-1]
    return im


def score(p):
    R, eye = pose_M(p)
    sim = render(R, eye)
    se, sy_ = feat(sim)
    def ncc(a, b):
        a = a - a.mean(); b = b - b.mean()
        return float((a * b).sum() / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))
    return -(ncc(REAL_E, se) + 1.5 * ncc(REAL_Y, sy_))


p0 = np.array(SEED, float)
print(f"{NAME} seed score: {score(p0):.4f}")
res = minimize(score, p0, method="Nelder-Mead",
               options=dict(maxfev=220, xatol=5e-3, fatol=1e-4,
                            initial_simplex=p0 + np.vstack([np.zeros(6),
                                np.diag([0.15, 0.15, 0.08, 6, 4, 3])])))
p = res.x
print(f"{NAME} solved: x={p[0]:.3f} y={p[1]:.3f} z={p[2]:.3f} yaw={p[3]:.1f} "
      f"pitch={p[4]:.1f} roll={p[5]:.1f}  score={-res.fun:.4f} evals={res.nfev}")
R, eye = pose_M(p)
sim = render(R, eye)
panel = np.concatenate([und, sim], axis=1)
blend = cv2.addWeighted(und, 0.5, sim, 0.5, 0)
cv2.imwrite(f"{OUT}/photo_{NAME}.png", panel)
cv2.imwrite(f"{OUT}/photo_{NAME}_blend.png", blend)
M = np.eye(4); M[:3, :3] = R.T; M[:3, 3] = eye
sol_path = f"{OUT}/photo_solution.json"
sol = json.load(open(sol_path)) if os.path.exists(sol_path) else {}
sol[NAME] = dict(pose=list(map(float, p)), M=M.tolist(), score=float(-res.fun))
json.dump(sol, open(sol_path, "w"), indent=1)
print("saved", sol_path)
