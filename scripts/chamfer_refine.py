#!/usr/bin/env python3
"""Chamfer pose refinement for the stereo-left anchor (native fisheye space).

The garage floor mat's yellow circle + tape is a rigid, high-contrast planar
landmark that locks camera x/y/yaw (and helps z). We render ONLY the yellow
subset of the splat through the real fisheye model, threshold yellow in the
raw real frame, and minimise a symmetric distance-transform (chamfer) cost.

Pitch and roll are seeded from the stereo-head IMU gravity; z from standing
kinematics. Coarse grid over (x, y, yaw) then Nelder-Mead over all 6 DOF.

Runs in the g1-kitchen-sim venv (cv2 + scipy + numpy):
    ~/projects/g1-kitchen-sim/.venv/bin/python scripts/chamfer_refine.py --probe
    ~/projects/g1-kitchen-sim/.venv/bin/python scripts/chamfer_refine.py --solve
"""
import argparse
import json
import os

import cv2
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation as Rot

A = os.path.expanduser("~/projects/x2-groot-vla/assets/garage")
C = f"{A}/calib_2026-08-04"
OUT = f"{C}/pose_match"
calib = json.load(open(f"{C}/calibration.json"))

ci = calib["ci_stereo_head_front_left"]
K = np.array(ci["k"]).reshape(3, 3)
D = np.array(ci["d"])
W, H = ci["w"], ci["h"]

SCALE = 0.4
w, h = int(W * SCALE), int(H * SCALE)
Kp = K * SCALE
Kp[2, 2] = 1.0

z = np.load(f"{OUT}/splat_pts.npz")
PTS, COLS = z["pts"], z["cols"]           # COLS is RGB uint8


def yellow_mask_bgr(img_bgr, clean=False, min_area=250):
    """Yellow floor tape/circle mask from a BGR image. With clean=True, keep
    only large connected components (drops window-reflection speckle so the
    remaining mask is the floor circle/tape)."""
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    m = cv2.inRange(hsv, (15, 60, 60), (45, 255, 255))
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    if clean:
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
        n, lab, stats, _ = cv2.connectedComponentsWithStats(m, 8)
        keep = np.zeros_like(m)
        for i in range(1, n):
            if stats[i, cv2.CC_STAT_AREA] >= min_area:
                keep[lab == i] = 255
        m = keep
    return m


def yellow_splat_subset(pts_xyz, cols_rgb, floor_z=0.25, smin=120):
    """Yellow floor-tape/circle splat points: saturated yellow AND near the
    floor plane (the circle is painted on the mat, so gating on world height
    kills the desaturated wall/ceiling false positives)."""
    hsv = cv2.cvtColor(cols_rgb.reshape(-1, 1, 3), cv2.COLOR_RGB2HSV).reshape(-1, 3)
    hh, ss, vv = hsv[:, 0], hsv[:, 1], hsv[:, 2]
    floor = pts_xyz[:, 2] < floor_z
    return floor & (hh >= 15) & (hh <= 45) & (ss >= smin) & (vv >= 70)


def opt_R(yaw_deg, pitch_deg, roll_deg=0.0):
    cy, sy = np.cos(np.deg2rad(yaw_deg)), np.sin(np.deg2rad(yaw_deg))
    cp, sp = np.cos(np.deg2rad(pitch_deg)), np.sin(np.deg2rad(pitch_deg))
    fwd = np.array([cy * cp, sy * cp, -sp])
    right = np.array([sy, -cy, 0.0])
    down = np.cross(fwd, right)
    R = np.stack([right, down, fwd])                     # world->cam rows
    if roll_deg:
        R = Rot.from_euler("z", np.deg2rad(roll_deg)).as_matrix() @ R
    return R


def render_fisheye_mask(sub_pts, eye, Rwc, splat=2):
    """Binary mask of a point subset projected through the fisheye model."""
    pc = (sub_pts - eye.astype(np.float32)) @ Rwc.T       # world->cam
    dist = np.linalg.norm(pc, axis=1) + 1e-9
    vis = (dist > 0.25) & (pc[:, 2] / dist > np.cos(np.deg2rad(100)))
    pcv = pc[vis]
    if len(pcv) == 0:
        return np.zeros((h, w), np.uint8)
    x, y, zc = pcv[:, 0], pcv[:, 1], pcv[:, 2]
    r = np.sqrt(x * x + y * y) + 1e-9
    th = np.arctan2(r, zc)
    k1, k2, k3, k4 = D[:4]
    thd = th * (1 + k1 * th**2 + k2 * th**4 + k3 * th**6 + k4 * th**8)
    u = Kp[0, 0] * thd * x / r + Kp[0, 2]
    v = Kp[1, 1] * thd * y / r + Kp[1, 2]
    ok = (u >= 0) & (u < w) & (v >= 0) & (v < h)
    uu = u[ok].astype(np.int32)
    vv = v[ok].astype(np.int32)
    im = np.zeros((h, w), np.uint8)
    for dy in range(-splat, splat + 1):
        for dx in range(-splat, splat + 1):
            xs = np.clip(uu + dx, 0, w - 1)
            ys = np.clip(vv + dy, 0, h - 1)
            im[ys, xs] = 255
    return im


# ---- real yellow mask + its distance transform (fixed target) ----
REAL = cv2.resize(cv2.imread(f"{C}/standing/stereo_head_front_left.jpg"), (w, h))
REAL_Y = yellow_mask_bgr(REAL, clean=True)
# The floor circle lives in the lower frame; zero the upper rows so the
# warm workbench shelf + ceiling rafters don't pollute the chamfer target.
ROI_TOP = int(0.58 * h)
REAL_Y[:ROI_TOP] = 0
DT_REAL = cv2.distanceTransform(255 - REAL_Y, cv2.DIST_L2, 3)

YIDX = yellow_splat_subset(PTS, COLS, floor_z=0.12)
YPTS = PTS[YIDX]


def chamfer(p, splat=2):
    x, y, zc, yaw, pitch, roll = p
    Rwc = opt_R(yaw, pitch, roll)
    sim_y = render_fisheye_mask(YPTS, np.array([x, y, zc]), Rwc, splat=splat)
    sim_y[:ROI_TOP] = 0
    n_sim = int((sim_y > 0).sum())
    if n_sim < 200:                       # basically nothing rendered
        return 1e6, sim_y
    c_s2r = float(DT_REAL[sim_y > 0].mean())
    dt_sim = cv2.distanceTransform(255 - sim_y, cv2.DIST_L2, 3)
    c_r2s = float(dt_sim[REAL_Y > 0].mean()) if (REAL_Y > 0).any() else 0.0
    return c_s2r + c_r2s, sim_y


def render_color(eye, Rwc):
    """Full-scene BGR fisheye render (depth-splat), like urdf_cam_panels."""
    pc = (PTS - eye.astype(np.float32)) @ Rwc.T
    dist = np.linalg.norm(pc, axis=1) + 1e-9
    vis = (dist > 0.25) & (pc[:, 2] / dist > np.cos(np.deg2rad(100)))
    pcv, colv = pc[vis], COLS[vis]
    x, y, zc = pcv[:, 0], pcv[:, 1], pcv[:, 2]
    r = np.sqrt(x * x + y * y) + 1e-9
    th = np.arctan2(r, zc)
    k1, k2, k3, k4 = D[:4]
    thd = th * (1 + k1 * th**2 + k2 * th**4 + k3 * th**6 + k4 * th**8)
    u = Kp[0, 0] * thd * x / r + Kp[0, 2]
    v = Kp[1, 1] * thd * y / r + Kp[1, 2]
    ok = (u >= 0) & (u < w) & (v >= 0) & (v < h)
    uvv = np.stack([u[ok], v[ok]], 1).astype(np.int32)
    pz = np.linalg.norm(pcv[ok], axis=1)
    colz = colv[ok]
    order = np.argsort(-pz)
    uvv, pz, colz = uvv[order], pz[order], colz[order]
    im = np.zeros((h, w, 3), np.uint8)
    for rad, lo, hi in ((1, 2.5, 1e9), (2, 1.2, 2.5), (3, 0.0, 1.2)):
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


REAL_GRAY = cv2.GaussianBlur(cv2.cvtColor(REAL, cv2.COLOR_BGR2GRAY), (0, 0), 9)
REAL_BRIGHT = (REAL_GRAY > np.percentile(REAL_GRAY, 55)).astype(np.uint8)


def structural_cost(sim_bgr):
    """1 - IoU of the bright/dark layout (walls+door+ceiling vs floor),
    evaluated only where the sim render has coverage. Robust to splat gaps
    via heavy blur; breaks the circle's rotational degeneracy."""
    cov = (sim_bgr.sum(2) > 0).astype(np.uint8)
    cov = cv2.dilate(cov, np.ones((7, 7), np.uint8))
    g = cv2.GaussianBlur(cv2.cvtColor(sim_bgr, cv2.COLOR_BGR2GRAY), (0, 0), 9)
    sb = (g > np.percentile(g[cov > 0], 55)).astype(np.uint8) if (cov > 0).any() else cov
    m = cov > 0
    a, b = REAL_BRIGHT[m].astype(bool), sb[m].astype(bool)
    inter = (a & b).sum()
    union = (a | b).sum() + 1e-6
    frac = m.mean()                    # penalise tiny coverage
    return (1 - inter / union) / max(frac, 0.15)


def fit_xy(yaw, z0, pitch0, roll0, seedxy, rng=0.35):
    """Circle-chamfer fit of (x, y) only, at a fixed yaw."""
    best = (1e9, seedxy)
    xs = np.round(np.arange(seedxy[0] - rng, seedxy[0] + rng + 1e-6, 0.08), 3)
    ys = np.round(np.arange(seedxy[1] - rng, seedxy[1] + rng + 1e-6, 0.08), 3)
    for x in xs:
        for y in ys:
            cost, _ = chamfer([x, y, z0, yaw, pitch0, roll0], splat=2)
            if cost < best[0]:
                best = (cost, (x, y))
    return best


def yaw_sweep(seed):
    z0, pitch0, roll0 = seed[2], seed[4], seed[5]
    tiles = []
    for yaw in range(-85, -29, 8):
        cost, (x, y) = fit_xy(yaw, z0, pitch0, roll0, (seed[0], seed[1]))
        Rwc = opt_R(yaw, pitch0, roll0)
        sim = render_color(np.array([x, y, z0]), Rwc)
        _, sim_y = chamfer([x, y, z0, yaw, pitch0, roll0], splat=2)
        sc = structural_cost(sim)
        sim[sim_y > 0] = (0, 255, 0)
        sim[REAL_Y > 0] = (0, 0, 255)
        tile = np.concatenate([REAL, sim], axis=1)
        cv2.putText(tile, f"yaw={yaw} x={x:.2f} y={y:.2f} circ={cost:.1f} struct={sc:.3f}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        tiles.append(tile)
        print(f"yaw {yaw}: x={x:.2f} y={y:.2f} circle_cost={cost:.1f} struct={sc:.3f}")
    grid = np.concatenate(tiles, axis=0)
    s = 1500 / grid.shape[1]
    grid = cv2.resize(grid, None, fx=s, fy=s)
    cv2.imwrite(f"{OUT}/chamfer_yaw_sweep.jpg", grid,
                [cv2.IMWRITE_JPEG_QUALITY, 82])
    print("wrote chamfer_yaw_sweep.jpg")


def probe():
    print(f"real yellow px: {(REAL_Y>0).sum()}  |  yellow splat pts: {len(YPTS)} / {len(PTS)}")
    cv2.imwrite(f"{OUT}/chamfer_real_yellow.png", REAL_Y)
    seed = [-0.20, 0.20, 1.42, -75, 9, 0]
    cost, sim_y = chamfer(seed)
    print(f"seed cost {cost:.2f}  sim yellow px {(sim_y>0).sum()}")
    overlay = REAL.copy()
    overlay[REAL_Y > 0] = (0, 0, 255)         # real yellow -> red
    overlay[sim_y > 0] = (0, 255, 0)          # sim yellow  -> green
    cv2.imwrite(f"{OUT}/chamfer_probe_overlay.png", overlay)
    # also dump the yellow splat points rendered in full color context
    print("wrote chamfer_real_yellow.png + chamfer_probe_overlay.png")


def solve(seed, xr=0.35, yr=0.35, yawr=30):
    """3-DOF fit over (x, y, yaw); z / pitch / roll held at the IMU +
    kinematics priors in `seed` (freeing them creates degenerate minima
    because the floor circle is near rotationally symmetric)."""
    z0, pitch0, roll0 = seed[2], seed[4], seed[5]
    best = (1e9, None)
    xs = np.round(np.arange(seed[0] - xr, seed[0] + xr + 1e-6, 0.10), 3)
    ys = np.round(np.arange(seed[1] - yr, seed[1] + yr + 1e-6, 0.10), 3)
    yaws = np.arange(seed[3] - yawr, seed[3] + yawr + 1e-6, 5)
    print(f"coarse grid: {len(xs)}x{len(ys)}x{len(yaws)} = {len(xs)*len(ys)*len(yaws)}")
    for x in xs:
        for y in ys:
            for yaw in yaws:
                cost, _ = chamfer([x, y, z0, yaw, pitch0, roll0], splat=2)
                if cost < best[0]:
                    best = (cost, [x, y, z0, yaw, pitch0, roll0])
    print(f"coarse best cost {best[0]:.2f} at {np.round(best[1],3)}")

    def f(q):                                  # q = (x, y, yaw)
        return chamfer([q[0], q[1], z0, q[2], pitch0, roll0], splat=2)[0]
    q0 = np.array([best[1][0], best[1][1], best[1][3]], float)
    res = minimize(f, q0, method="Nelder-Mead",
                   options=dict(maxfev=250, xatol=1e-3, fatol=1e-3,
                                initial_simplex=q0 + np.vstack(
                                    [np.zeros(3), np.diag([0.06, 0.06, 3.0])])))
    p = [res.x[0], res.x[1], z0, res.x[2], pitch0, roll0]
    cost, sim_y = chamfer(p, splat=2)
    print(f"refined cost {cost:.2f}  evals {res.nfev}")
    print(f"SOLVED anchor: x={p[0]:.3f} y={p[1]:.3f} z={p[2]:.3f} "
          f"yaw={p[3]:.2f} pitch={p[4]:.2f} roll={p[5]:.2f}")
    overlay = REAL.copy()
    overlay[REAL_Y > 0] = (0, 0, 255)
    overlay[sim_y > 0] = (0, 255, 0)
    cv2.imwrite(f"{OUT}/chamfer_solved_overlay.png", overlay)
    sol = {"stereo_head_front_left_anchor": list(map(float, p)),
           "chamfer_cost": float(cost), "coarse_cost": float(best[0]),
           "dof": "x,y,yaw fit; z,pitch,roll fixed from IMU+kinematics"}
    json.dump(sol, open(f"{OUT}/chamfer_solution.json", "w"), indent=1)
    print("wrote chamfer_solution.json + chamfer_solved_overlay.png")
    return p


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", action="store_true")
    ap.add_argument("--solve", action="store_true")
    ap.add_argument("--yaw-sweep", action="store_true")
    ap.add_argument("--seed", nargs=6, type=float,
                    default=[-0.20, 0.20, 1.42, -75, 9, 0],
                    metavar=("X", "Y", "Z", "YAW", "PITCH", "ROLL"))
    args = ap.parse_args()
    if args.probe:
        probe()
    if args.yaw_sweep:
        yaw_sweep(args.seed)
    if args.solve:
        solve(args.seed)
    if not (args.probe or args.solve or args.yaw_sweep):
        probe()
