"""V1 Gemma perception: room-camera survey -> Gemma boxes -> 3D world model.

Gemma provides the boxes itself (no separate detector, user ruling); depth +
the camera's intrinsics/pose lift them to the kitchen frame. Sim ground
truth (truth.json) is used ONLY by grade().

  python -m gemma_ctl.perceive              # survey the live rig, perceive, grade
  python -m gemma_ctl.perceive --dir D      # re-run perception on a saved survey
"""
from __future__ import annotations

import argparse
import base64
import json
import math
import os
import re
import time
import urllib.request
from pathlib import Path

import numpy as np

OLLAMA = "http://127.0.0.1:11434"
ROOT = Path(__file__).resolve().parents[1]

# room cameras (kitchen frame, eye at 2.1 m) — together they cover the counters
# by the stove/sink/dishwasher, the fridge, the dining side table
ROOM_VIEWS = [
    {"name": "counters", "eye": [0.6, -0.2, 2.1], "target": [-0.35, 1.9, 0.85]},
    {"name": "dining", "eye": [0.0, 0.2, 2.1], "target": [2.2, -1.4, 0.7]},
    {"name": "fridge_side", "eye": [1.2, 0.6, 2.1], "target": [-1.1, 1.6, 0.9]},
]

PROMPT = """You are the eyes of a kitchen robot. Look at this camera image.
Detect:
1. every small object a robot hand could pick up (cups, mugs, cans, bottles, fruit, ...), type "object";
2. every place where an object could be put down (countertop areas, sink, stove top, tables, shelves), type "place".
"label" says WHAT it is in 1-3 words: for objects e.g. "white mug", "soda can", "green bottle", "bowl";
for places a short name that says where it is, e.g. "counter next to the stove", "counter by the sink", "sink".
Never use the word "object" or "place" as a label.
Do not include the robot itself, walls, floor or cabinet doors.
Return ONLY JSON: a list of {"label": str, "type": "object"|"place", "box_2d": [ymin, xmin, ymax, xmax]}
with coordinates normalized to 0-1000."""


def gemma_boxes(img_path, model="gemma4:26b", prompt=PROMPT):
    b = base64.b64encode(Path(img_path).read_bytes()).decode()
    body = json.dumps({"model": model, "stream": False, "think": False,
                       "options": {"temperature": 0.0, "num_ctx": 16384,
                                   "num_predict": 3000},
                       "messages": [{"role": "user", "content": prompt, "images": [b]}]}).encode()
    t0 = time.monotonic()
    r = json.loads(urllib.request.urlopen(urllib.request.Request(
        f"{OLLAMA}/api/chat", data=body, headers={"Content-Type": "application/json"}),
        timeout=300).read())
    txt = r["message"]["content"]
    m = re.search(r"\[.*\]", txt, re.S)
    try:
        boxes = json.loads(m.group(0)) if m else []
    except ValueError:
        boxes = []
        for obj in re.findall(r"\{[^{}]*\}", txt):
            try:
                boxes.append(json.loads(obj))
            except ValueError:
                pass
        boxes = list({json.dumps(x, sort_keys=True): x for x in boxes}.values())
    return [b for b in boxes if isinstance(b, dict) and len(b.get("box_2d", [])) == 4], \
        time.monotonic() - t0, txt


_SCENE = {}


def _mesh_depth(K, R, t, W, H):
    import open3d as o3d
    if "scene" not in _SCENE:
        m = o3d.io.read_triangle_mesh(
            "/home/stickbot/projects/x2-kitchen-sim/assets/kitchen/kitchen_collision.obj")
        sc = o3d.t.geometry.RaycastingScene()
        sc.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(m))
        _SCENE["scene"] = sc
    u, v = np.meshgrid(np.arange(W) + 0.5, np.arange(H) + 0.5)
    dc = np.stack([(u - K[0, 2]) / K[0, 0], (v - K[1, 2]) / K[1, 1], np.ones_like(u)], -1)
    dw = dc.reshape(-1, 3) @ R.T                     # camera z = 1 -> t_hit = image-plane depth
    rays = np.concatenate([np.tile(t, (len(dw), 1)), dw], 1).astype(np.float32)
    th = _SCENE["scene"].cast_rays(o3d.core.Tensor(rays))["t_hit"].numpy().reshape(H, W)
    th[~np.isfinite(th)] = np.inf
    return th.astype(np.float32)


def _R_from_quat_wxyz(q):
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


class View:
    def __init__(self, d: Path, k: int):
        self.meta = json.loads((d / f"view{k}.json").read_text())
        self.img = d / f"view{k}.jpg"
        self.depth = np.load(d / f"view{k}_depth.npy").astype(np.float32)
        self.W, self.H = self.meta["size"]
        self.K = np.array(self.meta["K"])
        if self.meta.get("hfov_deg"):
            # K from the lens we set (the tiled camera's reported K can be
            # stale after a focal-length swap)
            fx = self.W / (2 * math.tan(math.radians(self.meta["hfov_deg"]) / 2))
            self.K = np.array([[fx, 0, self.W / 2], [0, fx, self.H / 2], [0, 0, 1.0]])
        # The tiled camera's reported pose is stale after set_world_poses_from_view,
        # so rebuild it from the eye/target we commanded (same look-at, up=+z).
        # ROS camera frame: x right, y down, z forward.
        self.t = np.array(self.meta["eye"], float)
        fwd = np.array(self.meta["target"], float) - self.t
        fwd /= np.linalg.norm(fwd)
        right = np.cross(fwd, [0.0, 0.0, 1.0]); right /= np.linalg.norm(right)
        down = np.cross(fwd, right)
        self.R = np.stack([right, down, fwd], 1)
        # The splat kitchen writes no depth (only meshes do: props, robot, the
        # hidden ground plane). Fill the scene from the aligned collision mesh
        # of the same scan (the sim stand-in for a depth camera) and keep the
        # nearer of the two per pixel.
        self.depth_render = self.depth
        self.depth_mesh = _mesh_depth(self.K, self.R, self.t, self.W, self.H)
        self.depth = np.fmin(self.depth_render, self.depth_mesh)

    def px(self, box):
        y0, x0, y1, x1 = [float(v) for v in box]
        return (x0 * self.W / 1000, y0 * self.H / 1000, x1 * self.W / 1000, y1 * self.H / 1000)

    def lift(self, us, vs):
        us = np.asarray(us, float); vs = np.asarray(vs, float)
        ui = np.clip(us.astype(int), 0, self.W - 1); vi = np.clip(vs.astype(int), 0, self.H - 1)
        d = self.depth[vi, ui]
        ok = np.isfinite(d) & (d > 0.1) & (d < 15)
        fx, fy, cx, cy = self.K[0, 0], self.K[1, 1], self.K[0, 2], self.K[1, 2]
        pc = np.stack([(us - cx) / fx * d, (vs - cy) / fy * d, d], 1)
        return (pc @ self.R.T + self.t)[ok]

    def box_points(self, box, shrink=0.15, step=2):
        x0, y0, x1, y1 = self.px(box)
        dx, dy = (x1 - x0) * shrink, (y1 - y0) * shrink
        us, vs = np.meshgrid(np.arange(x0 + dx, x1 - dx, step), np.arange(y0 + dy, y1 - dy, step))
        return self.lift(us.ravel(), vs.ravel())


def lift_object(v: View, box):
    """Object base point: xy = centroid of the nearest depth layer in the box;
    z = the support surface sampled in a band just below the box."""
    # pad the box (Gemma's boxes on small objects sit a few px off); only the
    # nearest depth layer is kept, so the margin cannot pull in background
    y0b, x0b, y1b, x1b = [float(b) for b in box]
    py, px_ = 0.25 * (y1b - y0b) + 4, 0.25 * (x1b - x0b) + 4
    pbox = [y0b - py, x0b - px_, y1b + py, x1b + px_]
    pts = v.box_points(pbox, shrink=0.0, step=1)
    if len(pts) < 5:
        return None
    # support surface = dominant flat height in the padded box; the object is
    # what stands above it (works the same with a real depth camera)
    zz = pts[:, 2]
    band = zz[(zz > 0.3) & (zz < 1.6)]
    if len(band) < 5:
        return None
    hist, edges = np.histogram(band, bins=np.arange(0.3, 1.61, 0.01))
    zs = float(edges[np.argmax(hist)] + 0.005)
    above = pts[(zz > zs + 0.012) & (zz < zs + 0.45)]
    if len(above) < 5:
        return None
    med = np.median(above[:, :2], 0)
    near = above[np.linalg.norm(above[:, :2] - med, axis=1) < 0.10]
    if len(near) < 5:
        near = above
    c = near.mean(0)
    top = float(np.percentile(near[:, 2], 95))
    # visible footprint (robust extent of the object's points)
    sx = float(np.percentile(near[:, 0], 95) - np.percentile(near[:, 0], 5))
    sy = float(np.percentile(near[:, 1], 95) - np.percentile(near[:, 1], 5))
    return {"base": [round(float(c[0]), 3), round(float(c[1]), 3), round(zs, 3)],
            "height": round(max(0.02, top - zs), 3), "size_xy": [round(sx, 3), round(sy, 3)],
            "n_px": int(len(near))}


def lift_place(v: View, box):
    """Place: dominant flat height inside the box (0.4-1.6 m), its centroid."""
    pts = v.box_points(box, shrink=0.1, step=3)
    pts = pts[(pts[:, 2] > 0.4) & (pts[:, 2] < 1.6)]
    if len(pts) < 20:
        return None
    hist, edges = np.histogram(pts[:, 2], bins=np.arange(0.4, 1.61, 0.02))
    zc = edges[np.argmax(hist)] + 0.01
    surf = pts[np.abs(pts[:, 2] - zc) < 0.025]
    c = surf.mean(0)
    return {"center": [round(float(c[0]), 3), round(float(c[1]), 3), round(float(zc), 3)],
            "extent": [round(float(np.ptp(surf[:, 0])), 2), round(float(np.ptp(surf[:, 1])), 2)],
            "n_px": int(len(surf))}


VERIFY = """This is a zoomed crop around one item in a kitchen camera image.
What is the item in the CENTRE of the crop? Answer with JSON only:
{"label": "<1-3 words, e.g. white mug, soda can, glass jar>", "graspable": true|false}"""


def verify_label(v: View, box, model):
    """Zoom check: crop around the box (3x its size) and ask Gemma again."""
    from PIL import Image
    import io
    x0, y0, x1, y1 = v.px(box)
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    s_ = max(x1 - x0, y1 - y0, 40) * 1.5
    im = Image.open(v.img).crop((int(cx - s_), int(cy - s_), int(cx + s_), int(cy + s_))).resize((448, 448))
    buf = io.BytesIO(); im.save(buf, "JPEG", quality=92)
    tmp = Path("/tmp/claude-1000/gemma_verify_crop.jpg"); tmp.parent.mkdir(parents=True, exist_ok=True)
    tmp.write_bytes(buf.getvalue())
    _, dt, txt = gemma_boxes(tmp, model, VERIFY)
    m = re.search(r"\{.*\}", txt, re.S)
    try:
        return json.loads(m.group(0)) if m else None
    except ValueError:
        return None


def perceive(d: Path, model="gemma4:26b", log=print):
    d = Path(d)
    views = [View(d, k) for k in range(len(list(d.glob("view*.json"))))]
    objects, places, raw = [], [], []
    for k, v in enumerate(views):
        boxes, dt, txt = gemma_boxes(v.img, model)
        raw.append({"view": k, "name": v.meta["name"], "llm_s": round(dt, 2), "boxes": boxes, "text": txt})
        log(f"  view {k} ({v.meta['name']}): {len(boxes)} boxes in {dt:.1f}s")
        for b in boxes:
            if b.get("type") == "place":
                p = lift_place(v, b["box_2d"])
                if p:
                    places.append({"label": b["label"], "view": k, **p, "box_2d": b["box_2d"]})
            else:
                o = lift_object(v, b["box_2d"])
                if o:
                    ver = verify_label(v, b["box_2d"], model)
                    lab = (ver or {}).get("label") or b["label"]
                    if os.environ.get("GEMMA_VERIFY_RELABEL", "1") == "0":
                        lab = b["label"]
                    objects.append({"label": lab, "first_label": b["label"],
                                    "verify_label": (ver or {}).get("label"), "view": k, **o,
                                    "graspable": (ver or {}).get("graspable", True), "box_2d": b["box_2d"]})
    world = {"objects": _merge(objects, "base", 0.15), "places": _merge(places, "center", 0.35)}
    (d / "gemma_raw.json").write_text(json.dumps(raw, indent=1))
    (d / "world.json").write_text(json.dumps(world, indent=1))
    return world, views, raw


def _merge(items, key, radius):
    """Same thing seen from several views -> one entry (label of the first,
    position = mean, all labels kept)."""
    out = []
    for it in items:
        p = np.array(it[key][:2])
        for o in out:
            if np.linalg.norm(np.array(o[key][:2]) - p) < radius:
                o["_pts"].append(it[key]); o["labels"].append(it["label"]); o["views"].append(it["view"])
                o[key] = np.round(np.mean(o["_pts"], 0), 3).tolist()
                break
        else:
            out.append({**it, "labels": [it["label"]], "views": [it["view"]], "_pts": [it[key]]})
    for o in out:
        o.pop("_pts")
    return out


def grade(d: Path, world, log=print):
    truth = json.loads((Path(d) / "truth.json").read_text())
    rows = []
    for name, t in truth.items():
        tb = np.array(t["base"])
        cands = world["objects"]
        if not cands:
            rows.append({"truth": name, "found": False}); continue
        dist = [float(np.linalg.norm(np.array(o["base"]) - tb)) for o in cands]
        i = int(np.argmin(dist))
        o = cands[i]
        e = np.array(o["base"]) - tb
        rows.append({"truth": name, "found": dist[i] < 0.25, "label": o["label"], "err_m": round(dist[i], 3),
                     "err_xy_m": round(float(np.linalg.norm(e[:2])), 3), "err_z_m": round(float(e[2]), 3)})
        log(f"  {name:4s} -> '{o['label']}'  3D err {dist[i] * 100:.1f} cm (xy {np.linalg.norm(e[:2]) * 100:.1f}, z {e[2] * 100:+.1f})"
            + ("" if dist[i] < 0.25 else "  NOT FOUND"))
    extra = len(world["objects"]) - sum(r.get("found", False) for r in rows)
    return rows, extra


def overlay(d: Path, views, raw):
    from PIL import Image, ImageDraw, ImageFont
    f = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 22)
    for v, r in zip(views, raw):
        im = Image.open(v.img).convert("RGB"); dr = ImageDraw.Draw(im)
        for b in r["boxes"]:
            x0, y0, x1, y1 = v.px(b["box_2d"])
            c = (255, 60, 60) if b.get("type") != "place" else (60, 200, 255)
            dr.rectangle([x0, y0, x1, y1], outline=c, width=4)
            dr.text((x0 + 4, max(0, y0 - 26)), b["label"], fill=c, font=f)
        im.save(Path(d) / f"view{raw.index(r)}_boxes.jpg", quality=85)


def request_survey(dir_, views=ROOM_VIEWS, timeout_s=None, ex=None):
    if timeout_s is None:
        timeout_s = float(os.environ.get("GEMMA_SURVEY_TIMEOUT_S", "120"))
    from .executor_kitchen import KitchenExecutor
    if ex is None:
        if not hasattr(request_survey, "_obs"):
            request_survey._obs = KitchenExecutor(log=lambda *a: None, drive=False)
        ex = request_survey._obs
    for _ in range(200):
        if ex._st():
            break
        time.sleep(0.1)
    sid = int(time.time())
    ex._rig(op="survey", id=sid, dir=str(dir_), views=views)
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        s = (ex._st() or {}).get("survey")
        if s and s.get("id") == sid:
            if s.get("error"):
                raise RuntimeError(s["error"])
            return ex
        time.sleep(0.2)
    raise RuntimeError("survey timed out")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", type=Path, default=None)
    ap.add_argument("--model", default="gemma4:26b")
    a = ap.parse_args()
    d = a.dir or ROOT / "runs/gemma/perception" / time.strftime("survey_%Y%m%dT%H%M%S")
    if a.dir is None:
        print(f"survey -> {d}")
        request_survey(d)
    world, views, raw = perceive(d, a.model)
    overlay(d, views, raw)
    print("objects:", [(o["label"], o["base"]) for o in world["objects"]])
    print("places:", [(p["label"], p["center"]) for p in world["places"]])
    grade(d, world)


if __name__ == "__main__":
    main()


# ---- targeted grounding: "find the <thing>" (used for actions) --------------
FIND = """Find every {q} in this kitchen image (part of a larger camera view).
Return ONLY JSON: a list of {{"label": "<what it is, 1-3 words>", "box_2d": [ymin, xmin, ymax, xmax]}}
normalized to 0-1000 for THIS image. Return [] if there is none."""


def find(views, query, model="gemma4:26b", tiles=(2, 2), overlap=0.2, log=print):
    """Ask Gemma for one named thing in every view, on overlapping tiles (small
    objects are ~25 px at 1080p; Gemma downsizes whole frames), then lift."""
    from PIL import Image
    hits = []
    for k, v in enumerate(views):
        im = Image.open(v.img)
        nx, ny = tiles
        tw, th = v.W / (nx - overlap * (nx - 1)), v.H / (ny - overlap * (ny - 1))
        for iy in range(ny):
            for ix in range(nx):
                x0 = int(ix * tw * (1 - overlap)); y0 = int(iy * th * (1 - overlap))
                x1, y1 = int(min(v.W, x0 + tw)), int(min(v.H, y0 + th))
                tile = Path(f"/tmp/claude-1000/gemma_tile_{k}_{ix}{iy}.jpg")
                im.crop((x0, y0, x1, y1)).save(tile, quality=93)
                boxes, dt, _ = gemma_boxes(tile, model, FIND.format(q=query))
                for b in boxes:
                    ty0, tx0, ty1, tx1 = [float(c) for c in b["box_2d"]]
                    # tile-normalized -> full-view-normalized
                    fb = [(y0 + ty0 / 1000 * (y1 - y0)) / v.H * 1000, (x0 + tx0 / 1000 * (x1 - x0)) / v.W * 1000,
                          (y0 + ty1 / 1000 * (y1 - y0)) / v.H * 1000, (x0 + tx1 / 1000 * (x1 - x0)) / v.W * 1000]
                    o = lift_object(v, fb)
                    if o:
                        hits.append({"label": b.get("label", query), "view": k, "tile": f"{ix}{iy}",
                                     "box_2d": fb, **o})
    return _merge(hits, "base", 0.12)


# ---- coarse-to-fine grounding (point, then zoomed box) ----------------------
POINT = """Is there a {q} anywhere in this kitchen image? If yes, point to each one.
Return ONLY JSON: a list of {{"label": "<1-3 words>", "point": [y, x]}} normalized to 0-1000, or [] if none."""
ZOOM_BOX = """Find the {q} in this zoomed image. Return ONLY JSON: a list of
{{"label": "<1-3 words>", "box_2d": [ymin, xmin, ymax, xmax]}} normalized to 0-1000, or [] if none."""


def _ask_json(img, prompt, model):
    _, dt, txt = gemma_boxes(img, model, prompt)
    m = re.search(r"\[.*\]", txt, re.S)
    try:
        out = json.loads(m.group(0)) if m else []
    except ValueError:
        out = []
    return [o for o in out if isinstance(o, dict)], dt


def find2(views, query, model="gemma4:26b", crop_px=220, log=print):
    from PIL import Image
    hits, t_llm = [], 0.0
    for k, v in enumerate(views):
        pts, dt = _ask_json(v.img, POINT.format(q=query), model)
        t_llm += dt
        im = Image.open(v.img)
        for p in pts:
            if len(p.get("point", [])) != 2:
                continue
            py, px_ = float(p["point"][0]) * v.H / 1000, float(p["point"][1]) * v.W / 1000
            x0, y0 = int(max(0, px_ - crop_px)), int(max(0, py - crop_px))
            x1, y1 = int(min(v.W, px_ + crop_px)), int(min(v.H, py + crop_px))
            tile = Path(f"/tmp/claude-1000/gemma_zoom_{k}.jpg")
            im.crop((x0, y0, x1, y1)).resize((448, int(448 * (y1 - y0) / max(1, x1 - x0)))).save(tile, quality=93)
            boxes, dt = _ask_json(tile, ZOOM_BOX.format(q=query), model)
            t_llm += dt
            for b in boxes:
                if len(b.get("box_2d", [])) != 4:
                    continue
                ty0, tx0, ty1, tx1 = [float(c) for c in b["box_2d"]]
                fb = [(y0 + ty0 / 1000 * (y1 - y0)) / v.H * 1000, (x0 + tx0 / 1000 * (x1 - x0)) / v.W * 1000,
                      (y0 + ty1 / 1000 * (y1 - y0)) / v.H * 1000, (x0 + tx1 / 1000 * (x1 - x0)) / v.W * 1000]
                o = lift_object(v, fb)
                if o:
                    hits.append({"label": b.get("label", query), "view": k, "box_2d": fb, **o})
    return _merge(hits, "base", 0.12), t_llm
