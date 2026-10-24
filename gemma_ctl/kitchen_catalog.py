"""Build the Gemma kitchen catalog: named object/placement LOCATIONS, each
with a surface point (ray-cast on kitchen_collision.obj) and a STAND pose the
arm can reach it from (offline IK limit: palm <=0.45 m ahead of the pelvis at
z 0.85-1.05; we use 0.30-0.40 m), plus the nearest nav waypoint to route via.

  python -m gemma_ctl.kitchen_catalog    -> gemma_ctl/kitchen_catalog.json + map png
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

KS = Path("/home/stickbot/projects/x2-kitchen-sim")
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent

# name: (rough xy seed on the surface, z band, waypoint to approach via, description)
LOCATIONS = {
    "stove_counter": ((0.30, 1.75), (0.80, 0.95), "cooking_range", "counter next to the stove"),
    "sink_counter": ((-0.95, 2.45), (0.80, 0.95), "sink", "counter beside the sink"),
    "dishwasher_counter": ((-0.45, 2.60), (0.80, 0.95), "dishwasher", "counter above the dishwasher"),
    "pantry_shelf": ((-1.20, -1.55), (0.85, 1.20), "pantry", "pantry shelf"),
    "side_table": ((2.50, -1.05), (0.65, 0.95), "dining_table", "side table by the dining area"),
}
REACH_MIN, REACH_MAX = 0.24, 0.34
STAND_ESDF_MIN = 0.20


def main():
    import trimesh
    mesh = trimesh.load(KS / "assets/kitchen/kitchen_collision.obj", force="mesh")
    g = np.load(KS / "assets/nav_grid.npz")
    E, W, o, r = g["esdf"], g["walkable"], g["origin"], float(g["res"])
    wps = json.loads((KS / "configs/waypoints.json").read_text())

    def idx(x, y):
        return int(round((x - o[0]) / r)), int(round((y - o[1]) / r))

    def esdf(x, y):
        i, j = idx(x, y)
        if 0 <= i < E.shape[0] and 0 <= j < E.shape[1]:
            return float(E[i, j])
        return -1.0

    def top_z(pts):
        pts = np.asarray(pts, float)
        orig = np.c_[pts, np.full(len(pts), 2.2)]
        loc, ray, _ = mesh.ray.intersects_location(orig, np.tile([0, 0, -1.0], (len(pts), 1)))
        z = np.full(len(pts), np.nan)
        for p, k in zip(loc, ray):
            z[k] = p[2] if np.isnan(z[k]) else max(z[k], p[2])
        return z

    out = {}
    for name, ((sx, sy), (z0, z1), via, desc) in LOCATIONS.items():
        # candidate object points on a 0.6 m patch around the seed
        gx, gy = np.meshgrid(np.arange(-0.4, 0.41, 0.04) + sx, np.arange(-0.4, 0.41, 0.04) + sy)
        P = np.c_[gx.ravel(), gy.ravel()]
        Z = top_z(P)
        best = None
        for (px, py), pz in zip(P, Z):
            if not (z0 <= pz <= z1):
                continue
            # surface must be locally flat (object footprint 6 cm)
            ring = top_z([[px + dx, py + dy] for dx, dy in ((.04, 0), (-.04, 0), (0, .04), (0, -.04))])
            if np.nanmax(np.abs(ring - pz)) > 0.02 or np.isnan(ring).any():
                continue
            for a in np.radians(np.arange(0, 360, 10)):
                for d in np.arange(REACH_MIN, REACH_MAX + 1e-6, 0.02):
                    qx, qy = px - d * math.cos(a), py - d * math.sin(a)
                    e = esdf(qx, qy)
                    if e < STAND_ESDF_MIN:
                        continue
                    # score: clearance, then shorter reach, then closer to via waypoint
                    wx, wy = wps[via]["xy"]
                    score = min(e, 0.35) * 4 - d - 0.1 * math.hypot(qx - wx, qy - wy)
                    if best is None or score > best[0]:
                        best = (score, dict(
                            obj_xyz=[round(float(px), 3), round(float(py), 3), round(float(pz), 3)],
                            stand={"xy": [round(qx, 3), round(qy, 3)], "yaw": round(a, 3)},
                            reach_m=round(float(d), 2), stand_esdf=round(e, 2)))
                    break   # nearest feasible stand along this bearing
        if best is None:
            print(f"{name:20s} NO feasible stand")
            continue
        rec = best[1] | {"via": via, "desc": desc}
        # second spot on the same surface (for placing next to an object)
        out[name] = rec
        print(f"{name:20s} obj {rec['obj_xyz']}  stand {rec['stand']}  reach {rec['reach_m']} "
              f"esdf {rec['stand_esdf']}  via {via} ({math.hypot(rec['stand']['xy'][0]-wps[via]['xy'][0], rec['stand']['xy'][1]-wps[via]['xy'][1]):.2f} m)")
    (HERE / "kitchen_catalog.json").write_text(json.dumps(out, indent=2))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(W.T, origin="lower", cmap="gray",
              extent=[o[0], o[0] + W.shape[0] * r, o[1], o[1] + W.shape[1] * r])
    for k, v in wps.items():
        ax.plot(*v["xy"], "b^"); ax.text(v["xy"][0], v["xy"][1] + 0.08, k, color="b", fontsize=8)
    for k, v in out.items():
        sx, sy = v["stand"]["xy"]; a = v["stand"]["yaw"]
        ax.plot(sx, sy, "go"); ax.arrow(sx, sy, 0.25 * math.cos(a), 0.25 * math.sin(a), color="g", head_width=0.05)
        ax.plot(*v["obj_xyz"][:2], "r*", ms=12); ax.text(v["obj_xyz"][0], v["obj_xyz"][1] - 0.15, k, color="r", fontsize=9)
    ax.set_xlim(-4, 3.5); ax.set_ylim(-4.5, 3.5); ax.set_aspect("equal"); ax.grid(alpha=.3)
    (ROOT / "runs/gemma/kitchen").mkdir(parents=True, exist_ok=True)
    plt.savefig(ROOT / "runs/gemma/kitchen/catalog_map.png", dpi=80, bbox_inches="tight")


if __name__ == "__main__":
    main()
