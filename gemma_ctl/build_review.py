"""Build phone-friendly review assets for a kitchen run: per-step GIFs,
grasp/release stills and a route map, into <run>/review/.

  python gemma_ctl/build_review.py runs/gemma/kitchen/demo1
"""
import json
import math
import os
import re
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

RUN = Path(sys.argv[1])
OUT = RUN / "review"
OUT.mkdir(exist_ok=True)
FR = sorted((RUN / "frames").iterdir())
DT = 0.08                     # sim s per captured frame (KP_GEMMA_REC_EVERY=4)
FONT = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 15)

STEPS = [  # key, title, t0, t1 (sim s), speed
    ("walk_stove", "Walk to the stove counter", 0, 50, 6),
    ("pick_mug", "Pick the mug", 50, 92, 3),
    ("carry_mug", "Carry it to the sink", 92, 124, 6),
    ("place_mug", "Place it on the sink counter", 124, 158, 3),
    ("walk_table", "Walk to the side table", 158, 240, 6),
    ("pick_can", "Pick the can (miss, then retry)", 240, 316, 4),
    ("carry_can", "Carry it to the stove", 316, 385, 6),
    ("place_can", "Place it by the stove", 385, 423, 3),
    ("walk_fridge", "Walk to the fridge", 423, 451, 6),
]
STILLS = [("grasp_mug", "Mug grasped", 77.5), ("release_mug", "Mug released at the sink", 147.5),
          ("grasp_can", "Can grasped (retry)", 302.5), ("release_can", "Can released by the stove", 412.5)]


def frame(t):
    return Image.open(FR[min(len(FR) - 1, max(0, int(t / DT)))]).convert("RGB")


def gif(key, t0, t1, speed, width=420, fps=8):
    step = speed / fps                        # sim s between gif frames
    ims = []
    t = t0
    while t < t1:
        im = frame(t)
        h = int(im.height * width / im.width)
        im = im.resize((width, h), Image.LANCZOS)
        d = ImageDraw.Draw(im)
        d.rectangle([width - 70, h - 24, width, h], fill=(0, 0, 0))
        d.text((width - 64, h - 21), f"{speed}x", fill=(255, 255, 255), font=FONT)
        ims.append(im.quantize(colors=96, method=Image.MEDIANCUT, dither=Image.NONE))
        t += step
    p = OUT / f"{key}.gif"
    ims[0].save(p, save_all=True, append_images=ims[1:], duration=int(1000 / fps), loop=0, optimize=True)
    return p


for key, title, t0, t1, sp in STEPS:
    p = gif(key, t0, t1, sp)
    print(f"{p.name}: {p.stat().st_size / 1e3:.0f} kB")
for key, title, t in STILLS:
    im = frame(t).resize((800, 450), Image.LANCZOS)
    im.save(OUT / f"{key}.jpg", quality=80)

# route map from rig heartbeats (world -> kitchen frame)
wp = np.array([-19.99, -75.96])
pts = []
for line in open(RUN / "rig.log"):
    m = re.search(r'\{"wall_s".*\}', line)
    if m:
        try:
            d = json.loads(m.group(0))
        except ValueError:
            continue
        pts.append((d["sim_s"], d["robot"][0] - wp[0], d["robot"][1] - wp[1]))
P = np.array(pts)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
g = np.load("/home/stickbot/projects/x2-kitchen-sim/assets/nav_grid.npz")
W, o, r = g["walkable"], g["origin"], float(g["res"])
cat = json.load(open(Path(__file__).with_name("kitchen_catalog.json")))
fig, ax = plt.subplots(figsize=(7, 7), dpi=110)
ax.imshow(W.T, origin="lower", cmap="Greys_r", alpha=0.35,
          extent=[o[0], o[0] + W.shape[0] * r, o[1], o[1] + W.shape[1] * r])
cols = plt.cm.viridis(np.linspace(0, 1, len(STEPS)))
for (key, title, t0, t1, _), c in zip(STEPS, cols):
    m = (P[:, 0] >= t0) & (P[:, 0] <= t1)
    ax.plot(P[m, 1], P[m, 2], "-", color=c, lw=2.5, label=title)
for k, v in cat.items():
    if k == "dishwasher_counter":
        continue
    ax.plot(*v["obj_xyz"][:2], "s", color="#d1495b", ms=8)
    ax.annotate(k.replace("_", " "), v["obj_xyz"][:2], textcoords="offset points", xytext=(6, 6), fontsize=8, color="#d1495b")
ax.plot(P[0, 1], P[0, 2], "o", color="k"); ax.annotate("start", P[0, 1:], textcoords="offset points", xytext=(6, -12), fontsize=8)
ax.set_xlim(-2.2, 3.2); ax.set_ylim(-2.4, 3.0); ax.set_aspect("equal")
ax.set_xlabel("x (m, kitchen frame)"); ax.set_ylabel("y (m)")
ax.legend(fontsize=7, loc="lower left", framealpha=0.85)
ax.grid(alpha=0.25)
plt.savefig(OUT / "route_map.png", bbox_inches="tight")
print("total", sum(p.stat().st_size for p in OUT.iterdir()) / 1e6, "MB")
