#!/usr/bin/env python3
"""PnP pose refinement: solve each real camera's pose against the splat.

Per camera: render sim (pinhole, with depth) from a seed pose -> SIFT match
against the undistorted real frame -> back-project sim keypoints via depth ->
solvePnPRansac -> solved world pose. Reports pose + residual, renders a
verification panel, and compares solved extrinsics against URDF/TF chains.
"""
import json
import os

import cv2
import numpy as np
from scipy.spatial.transform import Rotation as Rot

A = os.path.expanduser("~/projects/x2-groot-vla/assets/garage")
C = f"{A}/calib_2026-08-04"
OUT = f"{C}/pose_match"
calib = json.load(open(f"{C}/calibration.json"))
z = np.load(f"{OUT}/splat_pts.npz")
pts, cols = z["pts"], z["cols"]

SEED = {  # from manual standing fit
    "stereo_head_front_left": (np.array([-0.25, 0.55, 1.42]), -75.0, 15.0),
}


def opt_R(yaw, pitch):
    cy, sy = np.cos(np.deg2rad(yaw)), np.sin(np.deg2rad(yaw))
    cp, sp = np.cos(np.deg2rad(pitch)), np.sin(np.deg2rad(pitch))
    fwd = np.array([cy * cp, sy * cp, -sp])
    right = np.array([sy, -cy, 0.0])
    down = np.cross(fwd, right)
    return np.stack([right, down, fwd])


def render_depth(Rm, eye, Kp, w, h):
    pc = (pts - eye.astype(np.float32)) @ Rm.T
    vis = pc[:, 2] > 0.12
    pcv, colv = pc[vis], cols[vis]
    uv = (pcv[:, :2] / pcv[:, 2:3]) @ np.array([[Kp[0, 0], 0], [0, Kp[1, 1]]]) + Kp[:2, 2]
    ok = (uv[:, 0] >= 0) & (uv[:, 0] < w) & (uv[:, 1] >= 0) & (uv[:, 1] < h)
    uvv, pz, colz = uv[ok].astype(np.int32), pcv[ok, 2], colv[ok]
    order = np.argsort(-pz)
    uvv, pz, colz = uvv[order], pz[order], colz[order]
    im = np.zeros((h, w, 3), np.uint8)
    dep = np.full((h, w), np.inf, np.float32)
    for rad, lo, hi in ((1, 2.5, 1e9), (2, 1.2, 2.5), (3, 0.0, 1.2)):
        m = (pz >= lo) & (pz < hi)
        if not m.any():
            continue
        uu, cc, dd = uvv[m], colz[m], pz[m]
        for dy in range(-rad, rad + 1):
            for dx in range(-rad, rad + 1):
                xs = np.clip(uu[:, 0] + dx, 0, w - 1)
                ys = np.clip(uu[:, 1] + dy, 0, h - 1)
                im[ys, xs] = cc[:, ::-1]
                dep[ys, xs] = dd
    return im, dep


def solve_cam(name, ci_key, model, seed_M, tag):
    ci = calib[ci_key]
    K = np.array(ci["k"]).reshape(3, 3)
    D = np.array(ci["d"])
    W, H = ci["w"], ci["h"]
    img = cv2.imread(f"{C}/standing/{name}.jpg")
    # undistort real to pinhole newK
    if model == "fisheye":
        newK = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
            K, D, (W, H), np.eye(3), balance=0.35)
        m1, m2 = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), newK, (W, H), cv2.CV_16SC2)
    else:
        newK, _ = cv2.getOptimalNewCameraMatrix(K, D, (W, H), 0.35)
        m1, m2 = cv2.initUndistortRectifyMap(K, D, None, newK, (W, H), cv2.CV_16SC2)
    und = cv2.remap(img, m1, m2, interpolation=cv2.INTER_LINEAR)
    S = 0.5
    w, h = int(W * S), int(H * S)
    und = cv2.resize(und, (w, h))
    Kp = newK * S
    Kp[2, 2] = 1.0

    Rm = seed_M[:3, :3].T
    eye = seed_M[:3, 3]
    sim, dep = render_depth(Rm, eye, Kp, w, h)

    sift = cv2.SIFT_create(4000)
    g1 = cv2.cvtColor(und, cv2.COLOR_BGR2GRAY)
    g2 = cv2.cvtColor(sim, cv2.COLOR_BGR2GRAY)
    g1 = cv2.equalizeHist(g1)
    g2 = cv2.equalizeHist(g2)
    k1, d1 = sift.detectAndCompute(g1, None)
    k2, d2 = sift.detectAndCompute(g2, None)
    if d1 is None or d2 is None:
        print(f"{name}: no features")
        return None
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(d1, d2, k=2)
    good = [m for m, n in matches if m.distance < 0.75 * n.distance]
    obj, imgp = [], []
    for m in good:
        u2, v2 = k2[m.trainIdx].pt
        d_ = dep[int(v2), int(u2)]
        if not np.isfinite(d_):
            continue
        x = (u2 - Kp[0, 2]) / Kp[0, 0] * d_
        y = (v2 - Kp[1, 2]) / Kp[1, 1] * d_
        p_cam = np.array([x, y, d_])
        p_world = Rm.T @ p_cam + eye
        obj.append(p_world)
        imgp.append(k1[m.queryIdx].pt)
    print(f"{name}: {len(good)} ratio-matches, {len(obj)} with depth")
    if len(obj) < 12:
        return None
    okflag, rvec, tvec, inl = cv2.solvePnPRansac(
        np.array(obj, np.float32), np.array(imgp, np.float32), Kp, None,
        reprojectionError=6.0, iterationsCount=500)
    if not okflag or inl is None or len(inl) < 10:
        print(f"{name}: PnP failed ({0 if inl is None else len(inl)} inliers)")
        return None
    Rw = cv2.Rodrigues(rvec)[0]          # world->cam
    eye_s = (-Rw.T @ tvec).ravel()
    M = np.eye(4)
    M[:3, :3] = Rw.T
    M[:3, 3] = eye_s
    # reprojection residual on inliers
    proj, _ = cv2.projectPoints(np.array(obj, np.float32)[inl.ravel()], rvec, tvec, Kp, None)
    err = np.linalg.norm(proj.reshape(-1, 2) - np.array(imgp, np.float32)[inl.ravel()], axis=1)
    print(f"{name}: PnP OK — {len(inl)} inliers, reproj {err.mean():.2f}px  eye={np.round(eye_s,3)}")
    sim2, _ = render_depth(Rw, eye_s, Kp, w, h)
    panel = np.concatenate([und, sim2], axis=1)
    blend = cv2.addWeighted(und, 0.5, sim2, 0.5, 0)
    cv2.imwrite(f"{OUT}/pnp_{tag}.png", panel)
    cv2.imwrite(f"{OUT}/pnp_{tag}_blend.png", blend)
    return M


# --- solve stereo-left from manual seed ---
eye0, yaw0, pitch0 = SEED["stereo_head_front_left"]
M_seed = np.eye(4)
M_seed[:3, :3] = opt_R(yaw0, pitch0).T
M_seed[:3, 3] = eye0
M_left = solve_cam("stereo_head_front_left", "ci_stereo_head_front_left", "fisheye", M_seed, "left")
if M_left is None:
    raise SystemExit("left solve failed")

# --- URDF/TF chains from solved left ---
def urdf_M(xyz, rpy):
    M = np.eye(4)
    M[:3, :3] = Rot.from_euler("xyz", rpy).as_matrix()
    M[:3, 3] = xyz
    return M

M_h_left = urdf_M([0.067995, 0.029784, 0.05], [-1.5708, 0, -1.574])
M_h_rgbd = urdf_M([0.05761, -0.011183, -0.04837], [2.2689, 0, 1.5708])
RZ180 = urdf_M([0, 0, 0], [0, 0, np.pi])
M_wh = M_left @ np.linalg.inv(M_h_left)
M_rgbd_urdf = M_wh @ M_h_rgbd @ RZ180

# --- solve rgbd independently, seeded from LEFT's pose (similar view) ---
M_rgbd = solve_cam("rgbd_head_front_orbbec", "ci_rgb_head_front_center", "plumb_bob", M_left, "rgbd")

results = {"left_solved": M_left.tolist()}
if M_rgbd is not None:
    delta = np.linalg.inv(M_rgbd_urdf) @ M_rgbd
    rot_err = Rot.from_matrix(delta[:3, :3]).as_euler("xyz", degrees=True)
    results["rgbd_solved"] = M_rgbd.tolist()
    results["rgbd_urdf_predicted"] = M_rgbd_urdf.tolist()
    results["rgbd_delta_rpy_deg"] = rot_err.tolist()
    results["rgbd_delta_t_m"] = delta[:3, 3].tolist()
    print("\nRGBD URDF-vs-solved delta: rpy(deg) =", np.round(rot_err, 1),
          " t(m) =", np.round(delta[:3, 3], 3))
json.dump(results, open(f"{OUT}/pnp_solution.json", "w"), indent=1)
print("saved", f"{OUT}/pnp_solution.json")
