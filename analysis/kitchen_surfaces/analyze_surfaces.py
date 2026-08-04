#!/usr/bin/env python3
"""Quantify horizontal support surfaces in the x2-kitchen-sim collision mesh.

Frame: kitchen world frame (floor z=0) — kitchen_collision.obj is exported in
this frame by scripts/01c_mesh_align_to_usd.py (same frame as the collision
USD, nav_grid.npz and configs/waypoints.json).

Method:
  1. Load kitchen_collision.obj with trimesh.
  2. Select up-facing faces (face normal z > 0.95) with centroid z in
     [0.15, 2.0] m (mesh floor faces < 0.08 m were dropped at export).
  3. Voxelize up-face area at 4 cm XY x 3 cm Z and take 3-D connected
     components (26-connectivity) -> individual surfaces. A surface must be
     locally flat: component z-spread is reported; components are split no
     further (scan roughness ~2-3 cm).
  4. For each surface >= MIN_SURF_AREA, rasterize its footprint into the nav
     grid, distance-transform, and check for walkable floor cells 0.4-0.8 m
     from the footprint (robot standing band); pick a sample standing spot
     and the nearest named waypoint.
"""
import json
import os

import numpy as np
import trimesh
from scipy import ndimage

ROOT = "/home/stickbot/projects/x2-kitchen-sim"
OUT_DIR = os.path.dirname(os.path.abspath(__file__))

NORMAL_Z_MIN = 0.95
Z_MIN, Z_MAX = 0.15, 2.0
XY_RES = 0.04            # m, voxel XY
Z_RES = 0.03             # m, voxel Z
MIN_SURF_AREA = 0.05     # m^2, report surfaces at least this big
MAX_Z_SPREAD = 0.12      # m, warn if a component spans more than this in z
STAND_NEAR, STAND_FAR = 0.40, 0.80  # m, robot standing band from surface edge

# ---------------------------------------------------------------- mesh ------
mesh = trimesh.load(os.path.join(ROOT, "assets/kitchen/kitchen_collision.obj"),
                    process=False)
if isinstance(mesh, trimesh.Scene):
    mesh = trimesh.util.concatenate(list(mesh.geometry.values()))
print(f"mesh: {len(mesh.vertices):,} verts, {len(mesh.faces):,} faces, "
      f"bounds {mesh.bounds.round(2).tolist()}")

normals = mesh.face_normals
cent = mesh.triangles_center
areas = mesh.area_faces

up = (normals[:, 2] > NORMAL_Z_MIN) & (cent[:, 2] > Z_MIN) & (cent[:, 2] < Z_MAX)
print(f"up-facing faces (nz>{NORMAL_Z_MIN}, {Z_MIN}<z<{Z_MAX}): {up.sum():,}, "
      f"total area {areas[up].sum():.2f} m^2")

fz = cent[up, 2]
fxy = cent[up, :2]
fa = areas[up]

# ------------------------------------------------- height histogram ---------
hist_edges = np.arange(Z_MIN, Z_MAX + 0.025, 0.025)
hist, _ = np.histogram(fz, bins=hist_edges, weights=fa)
histogram = [
    {"z_lo": round(float(hist_edges[i]), 3),
     "z_hi": round(float(hist_edges[i + 1]), 3),
     "area_m2": round(float(hist[i]), 3)}
    for i in range(len(hist)) if hist[i] > 0.01
]
print("\nheight histogram (2.5 cm bins, area-weighted, bins > 0.01 m^2):")
for h in histogram:
    bar = "#" * int(h["area_m2"] / 0.05)
    print(f"  {h['z_lo']:.3f}-{h['z_hi']:.3f} m  {h['area_m2']:6.3f} m^2 {bar}")

# ------------------------------------------------- nav grid -----------------
nav = np.load(os.path.join(ROOT, "assets/nav_grid.npz"))
walkable = nav["walkable"]
esdf = nav["esdf"]
origin = nav["origin"]
res = float(nav["res"])
waypoints = json.load(open(os.path.join(ROOT, "configs/waypoints.json")))


def world_to_grid(xy, axis_order):
    """axis_order 'yx': grid[i,j] -> i=y, j=x. 'xy': grid[i,j] -> i=x, j=y."""
    xy = np.asarray(xy)
    if axis_order == "yx":
        j = np.round((xy[..., 0] - origin[0]) / res).astype(int)
        i = np.round((xy[..., 1] - origin[1]) / res).astype(int)
    else:
        i = np.round((xy[..., 0] - origin[0]) / res).astype(int)
        j = np.round((xy[..., 1] - origin[1]) / res).astype(int)
    return i, j


# Axis convention: waypoints are robot standing spots (radius 0.4), so the
# correct convention gives positive ESDF clearance at every waypoint.
conv_report = {}
for conv in ("yx", "xy"):
    vals = []
    for name, wp in waypoints.items():
        i, j = world_to_grid(np.array(wp["xy"]), conv)
        if 0 <= i < esdf.shape[0] and 0 <= j < esdf.shape[1]:
            vals.append(float(esdf[i, j]))
        else:
            vals.append(-1.0)
    conv_report[conv] = {
        "min_esdf": round(min(vals), 3),
        "mean_esdf": round(float(np.mean(vals)), 3),
        "walkable_hits": int(sum(v > 0 for v in vals)),
    }
axis_order = max(conv_report, key=lambda c: conv_report[c]["mean_esdf"])
print(f"\nnav grid: shape {walkable.shape}, res {res} m, origin {origin.round(3).tolist()}")
print(f"axis convention check (ESDF at 8 waypoints): {conv_report} -> '{axis_order}'")


def grid_to_world(i, j):
    if axis_order == "yx":
        return origin[0] + j * res, origin[1] + i * res
    return origin[0] + i * res, origin[1] + j * res


# --------------------------------------- 3-D voxel connected components -----
# Sample points on the up-facing submesh proportional to area (face centroids
# alone collapse large flat triangles to a single point and break clustering).
DENSITY = 5000.0  # points per m^2 -> each point represents 2 cm^2
sub = mesh.submesh([np.nonzero(up)[0]], append=True)
n_samples = max(int(sub.area * DENSITY), 1000)
pts, _ = trimesh.sample.sample_surface(sub, n_samples, seed=0)
pts = np.asarray(pts)
print(f"sampled {len(pts):,} points on up-facing submesh ({sub.area:.2f} m^2)")
fxy = pts[:, :2]
fz = pts[:, 2]
fa = np.full(len(pts), sub.area / len(pts))  # area per sample point

lo = np.array([fxy[:, 0].min(), fxy[:, 1].min(), fz.min()]) - 1e-6
gi = ((fxy[:, 0] - lo[0]) / XY_RES).astype(int)
gj = ((fxy[:, 1] - lo[1]) / XY_RES).astype(int)
gk = ((fz - lo[2]) / Z_RES).astype(int)
vox = np.zeros((gi.max() + 1, gj.max() + 1, gk.max() + 1), dtype=bool)
vox[gi, gj, gk] = True
lab, nlab = ndimage.label(vox, structure=np.ones((3, 3, 3)))
face_lab = lab[gi, gj, gk]
print(f"3-D voxel CC ({XY_RES} m XY x {Z_RES} m Z, 26-conn): {nlab} components")

surfaces = []
for c in range(1, nlab + 1):
    m = face_lab == c
    ca = float(fa[m].sum())
    if ca < MIN_SURF_AREA:
        continue
    cxy = fxy[m]
    w = fa[m]
    zvals = fz[m]
    zmed = float(np.average(zvals, weights=w))

    # placement spots: peaks of the distance transform inside the occupied
    # footprint = points with maximal clearance to the surface edge
    slo = cxy.min(axis=0) - XY_RES
    si = ((cxy[:, 0] - slo[0]) / XY_RES).astype(int)
    sj = ((cxy[:, 1] - slo[1]) / XY_RES).astype(int)
    occ = np.zeros((si.max() + 2, sj.max() + 2), dtype=bool)
    occ[si, sj] = True
    spots = []
    dt = ndimage.distance_transform_edt(occ) * XY_RES
    for _ in range(3):
        k = np.unravel_index(np.argmax(dt), dt.shape)
        r = float(dt[k])
        if r < 0.06:
            break
        sx = slo[0] + k[0] * XY_RES
        sy = slo[1] + k[1] * XY_RES
        near = m & (np.abs(fxy[:, 0] - sx) < 0.1) & (np.abs(fxy[:, 1] - sy) < 0.1)
        sz = float(np.median(fz[near])) if near.any() else zmed
        spots.append({"xyz": [round(sx, 3), round(sy, 3), round(sz, 3)],
                      "clear_radius_m": round(r, 3)})
        ki, kj = np.mgrid[:dt.shape[0], :dt.shape[1]]
        dt[(ki - k[0]) ** 2 + (kj - k[1]) ** 2 < (2 * r / XY_RES) ** 2] = 0

    surfaces.append({
        "z": round(zmed, 3),
        "z_spread": round(float(np.percentile(zvals, 95) - np.percentile(zvals, 5)), 3),
        "area_m2": round(ca, 3),
        "xy_min": [round(float(v), 3) for v in cxy.min(axis=0)],
        "xy_max": [round(float(v), 3) for v in cxy.max(axis=0)],
        "size_xy": [round(float(v), 2) for v in (cxy.max(axis=0) - cxy.min(axis=0))],
        "centroid_xy": [round(float(v), 3) for v in np.average(cxy, axis=0, weights=w)],
        "placement_spots": spots,
        "_cells_xy": cxy,
    })
surfaces.sort(key=lambda s: -s["area_m2"])
print(f"{len(surfaces)} surfaces with area >= {MIN_SURF_AREA} m^2")

# --------------------------------------- standing-spot analysis -------------
wp_names = list(waypoints)
wp_xy = np.array([waypoints[n]["xy"] for n in wp_names])

for s in surfaces:
    foot = np.zeros_like(walkable)
    i, j = world_to_grid(s.pop("_cells_xy"), axis_order)
    ok = (i >= 0) & (i < foot.shape[0]) & (j >= 0) & (j < foot.shape[1])
    foot[i[ok], j[ok]] = True
    foot = ndimage.binary_dilation(foot, iterations=int(XY_RES / res) + 1)
    dist = ndimage.distance_transform_edt(~foot) * res
    ring = walkable & (dist >= STAND_NEAR) & (dist <= STAND_FAR)
    n_ring = int(ring.sum())
    s["walkable_cells_0.4-0.8m"] = n_ring
    s["stand_area_m2"] = round(n_ring * res * res, 3)
    s["can_stand"] = bool(n_ring * res * res > 0.04)  # >= ~0.2x0.2 m of floor
    if n_ring:
        ii, jj = np.nonzero(ring)
        wx, wy = grid_to_world(ii, jj)
        cx, cy = s["centroid_xy"]
        k = int(np.argmin((wx - cx) ** 2 + (wy - cy) ** 2))
        spot = [round(float(wx[k]), 3), round(float(wy[k]), 3)]
        s["sample_stand_xy"] = spot
        dw = np.linalg.norm(wp_xy - np.array(spot), axis=1)
        s["nearest_waypoint"] = wp_names[int(np.argmin(dw))]
        s["nearest_waypoint_dist_m"] = round(float(dw.min()), 2)
    else:
        s["sample_stand_xy"] = None
        s["nearest_waypoint"] = None
        s["nearest_waypoint_dist_m"] = None

# ---------------------------------------------------------------- report ----
print("\n=== support surfaces (largest first) ===")
for s in surfaces:
    print(f"  z={s['z']:.3f}+-{s['z_spread']:.2f} m  area={s['area_m2']:6.2f} m^2  "
          f"xy [{s['xy_min'][0]:6.2f},{s['xy_min'][1]:6.2f}]..[{s['xy_max'][0]:6.2f},{s['xy_max'][1]:6.2f}]  "
          f"({s['size_xy'][0]:.2f}x{s['size_xy'][1]:.2f})  "
          f"stand={s['stand_area_m2']:5.2f} m^2 {'YES' if s['can_stand'] else 'no '}  "
          f"spot={s['sample_stand_xy']}  wp={s['nearest_waypoint']} "
          f"({s['nearest_waypoint_dist_m']} m)")

# ---------------------------------------------------------------- debug png -
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(11, 10))
ii, jj = np.nonzero(walkable)
wx, wy = grid_to_world(ii, jj)
ax.scatter(wx, wy, s=1, c="0.85", label="walkable")
sc = ax.scatter(fxy[:, 0], fxy[:, 1], s=2, c=fz, cmap="viridis", vmin=0.15, vmax=1.6)
plt.colorbar(sc, ax=ax, label="surface z [m]")
for s in surfaces:
    x0, y0 = s["xy_min"]; x1, y1 = s["xy_max"]
    ax.add_patch(plt.Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, ec="r", lw=1))
    ax.annotate(f"{s['z']:.2f}m/{s['area_m2']:.2f}m2", (x0, y1), fontsize=6, color="r")
    if s["sample_stand_xy"]:
        ax.plot(*s["sample_stand_xy"], "r*", ms=8)
for name, wp in waypoints.items():
    ax.plot(*wp["xy"], "b^", ms=7)
    ax.annotate(name, wp["xy"], fontsize=7, color="b")
ax.set_aspect("equal"); ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]")
ax.set_title("kitchen support surfaces (red boxes) + walkable (grey) + waypoints (blue)")
png = os.path.join(OUT_DIR, "kitchen_surfaces_debug.png")
fig.savefig(png, dpi=110, bbox_inches="tight")
print(f"debug plot: {png}")

summary = {
    "frame": "kitchen world frame (floor z=0; same as kitchen_collision.usd, nav_grid.npz, waypoints.json)",
    "mesh": os.path.join(ROOT, "assets/kitchen/kitchen_collision.obj"),
    "params": {
        "normal_z_min": NORMAL_Z_MIN, "z_range_m": [Z_MIN, Z_MAX],
        "voxel_xy_m": XY_RES, "voxel_z_m": Z_RES,
        "min_surface_area_m2": MIN_SURF_AREA,
        "stand_band_m": [STAND_NEAR, STAND_FAR],
    },
    "nav_grid": {
        "shape": list(walkable.shape), "res_m": res,
        "origin_xy": origin.tolist(),
        "axis_order": axis_order,
        "axis_check_esdf_at_waypoints": conv_report,
    },
    "height_histogram_area_weighted": histogram,
    "surfaces": surfaces,
}
out_json = os.path.join(OUT_DIR, "kitchen_surfaces.json")
json.dump(summary, open(out_json, "w"), indent=2)
print(f"wrote {out_json}")
