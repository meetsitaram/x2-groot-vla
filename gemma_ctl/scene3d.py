"""3D world-model scene export for the browser viewer.

  kitchen mesh  kitchen_collision.obj, cropped to the kitchen, ceiling removed,
                decimated, vertices coloured from the nearest splat points
  objects       every world-model object as a box (fused centre, footprint,
                height, sigma, label votes, sources); sim truth as wireframes
  places        Gemma-labelled surfaces as flat rectangles at their height
  cameras       room cams + robot cams as frusta; robot position

  ~/miniconda3/envs/env_isaaclab/bin/python -m gemma_ctl.scene3d <run_dir> <out_dir>
"""
from __future__ import annotations

import base64
import json
import sys
from pathlib import Path

import numpy as np

KS = Path("/home/stickbot/projects/x2-kitchen-sim/assets/kitchen")
CROP = dict(x=(-3.3, 3.7), y=(-3.4, 3.35), z=(-0.1, 2.2))


def _b64(a):
    return base64.b64encode(np.ascontiguousarray(a).tobytes()).decode()


def export_mesh(out: Path, target_tris=40000):
    import open3d as o3d
    from scipy.spatial import cKDTree
    m = o3d.io.read_triangle_mesh(str(KS / "kitchen_collision.obj"))
    V = np.asarray(m.vertices); F = np.asarray(m.triangles)
    c = V[F].mean(1)
    keep = ((c[:, 0] > CROP["x"][0]) & (c[:, 0] < CROP["x"][1]) & (c[:, 1] > CROP["y"][0]) &
            (c[:, 1] < CROP["y"][1]) & (c[:, 2] < CROP["z"][1]))
    m.triangles = o3d.utility.Vector3iVector(F[keep]); m.remove_unreferenced_vertices()
    if len(m.triangles) > target_tris:
        m = m.simplify_quadric_decimation(target_tris)
    m.remove_degenerate_triangles(); m.remove_unreferenced_vertices()
    V = np.asarray(m.vertices).astype(np.float32); F = np.asarray(m.triangles).astype(np.uint32)
    # colour from the splat (median of nearby gaussians)
    s = np.load("/tmp/claude-1000/splat_world_rgb.npz")
    sx, sc = s["xyz"], s["rgb"].astype(np.float32)
    box = ((sx[:, 0] > CROP["x"][0] - .2) & (sx[:, 0] < CROP["x"][1] + .2) &
           (sx[:, 1] > CROP["y"][0] - .2) & (sx[:, 1] < CROP["y"][1] + .2))
    tree = cKDTree(sx[box]); scb = sc[box]
    d, idx = tree.query(V, k=12, distance_upper_bound=0.08)
    col = np.full((len(V), 3), 150, np.float32)
    for i in range(len(V)):
        ok = np.isfinite(d[i])
        if ok.any():
            col[i] = np.median(scb[idx[i][ok]], 0)
    mesh = {"n_v": int(len(V)), "n_f": int(len(F)), "pos": _b64(V), "idx": _b64(F),
            "col": _b64(col.clip(0, 255).astype(np.uint8))}
    (out / "mesh.json").write_text(json.dumps(mesh))
    print(f"mesh: {len(V)} verts, {len(F)} tris, {(out / 'mesh.json').stat().st_size / 1e6:.1f} MB")


def _frustum(meta):
    eye = np.array(meta["eye"], float); tgt = np.array(meta["target"], float)
    return {"eye": eye.round(3).tolist(), "target": tgt.round(3).tolist(),
            "hfov": meta.get("hfov_deg", 53.13), "aspect": meta["size"][0] / meta["size"][1],
            "kind": meta.get("camera", "room"), "name": meta.get("name", "")}


def export_scene(run: Path, out: Path):
    """Rebuild the world model from a run's saved surveys (gemma_raw boxes ->
    lift -> fuse), keeping footprints, and write scene.json."""
    from .perceive import View, lift_object, lift_place
    from .world_model import WorldModel
    wm = WorldModel()
    cams, sizes = [], {}
    dirs = sorted(p for p in run.iterdir() if p.is_dir() and (p / "gemma_raw.json").exists())
    if (run / "room").is_dir():
        dirs = [run / "room"] + [d for d in dirs if d.name != "room"]
    truth = json.loads((dirs[-1] / "truth.json").read_text())   # latest = final object poses
    robot_poses = []
    for d in dirs:
        if not (d / "gemma_raw.json").exists():
            continue
        raw = json.loads((d / "gemma_raw.json").read_text())
        for r in raw:
            v = View(d, r["view"])
            cams.append(_frustum(v.meta))
            src = "room" if v.meta.get("camera", "room") == "room" else "head"
            if src == "head":
                robot_poses.append(v.t.round(3).tolist())
            for b in r["boxes"]:
                if b.get("type") == "place":
                    p = lift_place(v, b["box_2d"])
                    if p:
                        wm.observe_place(b["label"], p["center"], p["extent"])
                    continue
                o = lift_object(v, b["box_2d"])
                if not o:
                    continue
                rng = float(np.linalg.norm(np.array(o["base"]) - v.t))
                oid, _ = wm.observe(b["label"], o["base"], o["height"], rng, src)
                sizes.setdefault(oid, []).append(o.get("size_xy", [0.08, 0.08]))
    objs = []
    for oid, o in wm.objects.items():
        sz = np.median(np.array(sizes.get(oid, [[0.08, 0.08]])), 0)
        votes = wm.label_votes(oid) if hasattr(wm, "label_votes") else {}
        objs.append({"id": oid, "label": wm.label(oid), "votes": votes,
                     "center": [round(float(o["mean"][0]), 3), round(float(o["mean"][1]), 3),
                                round(float(o["mean"][2] + o["height"] / 2), 3)],
                     "size": [round(float(max(0.04, sz[0])), 3), round(float(max(0.04, sz[1])), 3),
                              round(float(max(0.03, o["height"])), 3)],
                     "sigma_xy": round(o["sigma_xy"], 3), "n": o["n"], "sources": dict(o["sources"]),
                     "ready": o["sigma_xy"] <= 0.03})
    places = [{"id": k, "label": max(p["labels"], key=p["labels"].get), "center": p["center"],
               "extent": p["extent"]} for k, p in wm.places.items()]
    tr = [{"name": n, "center": [t.get("center_base", t["base"])[0], t.get("center_base", t["base"])[1],
                                 t["base"][2] + t["height"] / 2], "height": t["height"]}
          for n, t in truth.items()]
    # errors of nearest perceived object to each truth (for the panel)
    for t in tr:
        if objs:
            d = [np.linalg.norm(np.array(o["center"][:2]) - np.array(t["center"][:2])) for o in objs]
            i = int(np.argmin(d)); t["nearest"] = objs[i]["id"]; t["err_xy_cm"] = round(float(d[i]) * 100, 1)
    scene = {"objects": objs, "places": places, "truth": tr, "cameras": cams, "robot": robot_poses,
             "run": str(run.name)}
    (out / "scene.json").write_text(json.dumps(scene, indent=1))
    print(f"scene: {len(objs)} objects, {len(places)} places, {len(cams)} camera views")


if __name__ == "__main__":
    run, out = Path(sys.argv[1]), Path(sys.argv[2])
    out.mkdir(parents=True, exist_ok=True)
    if not (out / "mesh.json").exists():
        export_mesh(out)
    export_scene(run, out)
