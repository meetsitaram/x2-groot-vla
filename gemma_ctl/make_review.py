"""Review media for a recorded kitchen run -> <run>/review/:
  full.mp4         whole run at FULL_SPEED x, 1280x720, speed badge bottom-right
  <step>.mp4/.jpg  one clip + poster per step (walks faster, reaches slower)
  route_map.png    robot path coloured by step
  steps.json       step list for the page

Steps are found from the rig log: [gemma] reach events within 10 s of each
other form one manipulation phase (labelled by its attach/detach), gaps
between phases are walks.

  python gemma_ctl/make_review.py runs/gemma/kitchen/demo2
"""
import json
import re
import subprocess
import sys
from pathlib import Path

import numpy as np

RUN = Path(sys.argv[1])
OUT = RUN / "review"
OUT.mkdir(exist_ok=True)
DT = 0.08                    # sim s per captured frame
FULL_SPEED = 3
WALK_SPEED, MANIP_SPEED = 4, 2
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
import os
FRAMES = os.environ.get("REVIEW_FRAMES", "frames")      # e.g. frames_ov (with Gemma overlay)
N_FRAMES = len(list((RUN / FRAMES).glob("f*.jpg")))
T_END = N_FRAMES * DT

# ---- events from the rig log ------------------------------------------------
sim, ev, path = 0.0, [], []
for line in open(RUN / "rig.log"):
    m = re.search(r'\{"wall_s".*\}', line)
    if m:
        try:
            d = json.loads(m.group(0))
            sim = d["sim_s"]
            path.append((sim, d["robot"][0] + 19.99, d["robot"][1] + 75.96))
        except ValueError:
            pass
        continue
    m = re.match(r"\[gemma\] (reach|attach|detach) ?(\w+)?", line)
    if m:
        ev.append((sim, m.group(1), m.group(2)))

phases = []
for t, kind, obj in ev:
    if phases and t - phases[-1]["t1"] <= 10:
        ph = phases[-1]
    else:
        ph = {"t0": t, "t1": t, "kind": None, "obj": None, "reaches": 0}
        phases.append(ph)
    ph["t1"] = t
    if kind == "reach":
        ph["reaches"] += 1
    else:
        ph["kind"], ph["obj"] = kind, obj
# pick failures per object, from Gemma's transcript (order of pick calls)
fails = {}
try:
    for r in json.load(open(RUN / "transcript.json")):
        if r.get("tool") == "pick" and not r["result"].get("ok"):
            o_ = r["args"].get("object"); fails[o_] = fails.get(o_, 0) + 1
except (OSError, ValueError, KeyError):
    pass
dests = {"mug": ["sink", "sink counter"], "can": ["stove", "stove counter"]}
steps, t = [], 0.0
held = None
for ph in phases:
    a, b = ph["t0"] - 2, ph["t1"] + 6
    walk = "Walk to the next target" if held is None else f"Carry the {held}"
    if a - t > 3:
        steps.append({"key": f"s{len(steps):02d}_walk", "title": walk, "t0": t, "t1": a, "speed": WALK_SPEED})
    if ph["kind"] == "attach":
        title = f"Pick the {ph['obj']}" + (" (miss, then retry)" if fails.get(ph["obj"]) else "")
        held = ph["obj"]
    elif ph["kind"] == "detach":
        title = f"Place the {ph['obj']}"
        held = None
    else:
        title = "Reach (no grasp)"
    steps.append({"key": f"s{len(steps):02d}_" + (ph["kind"] or "reach"), "title": title,
                  "t0": a, "t1": b, "speed": MANIP_SPEED})
    t = b
if T_END - t > 3:
    steps.append({"key": f"s{len(steps):02d}_walk", "title": "Walk / look again", "t0": t, "t1": T_END,
                  "speed": WALK_SPEED})
# name the walk between the first place and the second pick
for i, s in enumerate(steps):
    if s["title"] == "Walk to the next target" and i + 1 < len(steps) and "Pick" in steps[i + 1]["title"]:
        obj = steps[i + 1]["title"].split()[2]
        s["title"] = f"Walk to the {obj}"


def encode(src_start, n, speed, out, width=1280, crf=24, maxrate="2600k", two_pass_kbps=None):
    badge = (f"drawbox=x=iw-118:y=ih-62:w=104:h=48:color=black@0.62:t=fill,"
             f"drawtext=fontfile={FONT}:text='{speed}×':fontcolor=white:fontsize=30:x=w-104:y=h-54")
    base = ["ffmpeg", "-y", "-loglevel", "error", "-start_number", str(src_start),
            "-framerate", str(12.5 * speed), "-i", str(RUN / FRAMES / "f%06d.jpg"),
            "-frames:v", str(n), "-vf", f"scale={width}:-2:flags=lanczos,fps=25,{badge}",
            "-c:v", "libx264", "-profile:v", "high", "-level", "4.0", "-preset", "slow",
            "-tune", "film", "-pix_fmt", "yuv420p", "-an"]
    if two_pass_kbps:     # single pass, capped (2-pass segfaulted on this ffmpeg build)
        k = two_pass_kbps
        subprocess.run(base + ["-crf", "23", "-maxrate", f"{k}k", "-bufsize", f"{2 * k}k",
                               "-movflags", "+faststart", str(out)], check=True)
    else:
        subprocess.run(base + ["-crf", str(crf), "-maxrate", maxrate, "-bufsize", "5200k",
                               "-movflags", "+faststart", str(out)], check=True)


full_s = N_FRAMES * DT / FULL_SPEED
encode(0, N_FRAMES, FULL_SPEED, OUT / "full.mp4", two_pass_kbps=int(13.5e6 * 8 / 1000 / full_s))
for s in steps:
    f0, n = int(s["t0"] / DT), max(1, int((s["t1"] - s["t0"]) / DT))
    encode(f0, n, s["speed"], OUT / f"{s['key']}.mp4")
    mid = f0 + n // 2
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(RUN / FRAMES / f"f{mid:06d}.jpg"),
                    "-vf", "scale=1280:-2", "-q:v", "4", str(OUT / f"{s['key']}.jpg")], check=True)
    s["mp4_kb"] = (OUT / f"{s['key']}.mp4").stat().st_size // 1000
    print(f"{s['key']:14s} {s['title']:36s} {s['t0']:6.1f}-{s['t1']:6.1f}s @{s['speed']}x  {s['mp4_kb']} kB")
subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(RUN / FRAMES / f"f{N_FRAMES // 2:06d}.jpg"),
                "-vf", "scale=1280:-2", "-q:v", "4", str(OUT / "full.jpg")], check=True)

# ---- route map ---------------------------------------------------------------
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
P = np.array(path)
g = np.load("/home/stickbot/projects/x2-kitchen-sim/assets/nav_grid.npz")
W, o, r = g["walkable"], g["origin"], float(g["res"])
cat = json.load(open(Path(__file__).with_name("kitchen_catalog.json")))
fig, ax = plt.subplots(figsize=(7, 7), dpi=120)
ax.imshow(W.T, origin="lower", cmap="Greys_r", alpha=0.35,
          extent=[o[0], o[0] + W.shape[0] * r, o[1], o[1] + W.shape[1] * r])
cols = plt.cm.viridis(np.linspace(0, 1, len(steps)))
for s, c in zip(steps, cols):
    m = (P[:, 0] >= s["t0"]) & (P[:, 0] <= s["t1"])
    ax.plot(P[m, 1], P[m, 2], "-", color=c, lw=2.5, label=s["title"])
for k, v in cat.items():
    if k == "dishwasher_counter":
        continue
    ax.plot(*v["obj_xyz"][:2], "s", color="#d1495b", ms=8)
    ax.annotate(k.replace("_", " "), v["obj_xyz"][:2], textcoords="offset points", xytext=(6, 6),
                fontsize=8, color="#d1495b")
ax.plot(P[0, 1], P[0, 2], "o", color="k")
ax.annotate("start", P[0, 1:], textcoords="offset points", xytext=(6, -12), fontsize=8)
ax.set_xlim(-2.2, 3.2); ax.set_ylim(-2.4, 3.0); ax.set_aspect("equal")
ax.set_xlabel("x (m, kitchen frame)"); ax.set_ylabel("y (m)")
ax.legend(fontsize=7, loc="lower left", framealpha=0.85); ax.grid(alpha=0.25)
plt.savefig(OUT / "route_map.png", bbox_inches="tight")
json.dump({"steps": steps, "full_speed": FULL_SPEED, "sim_s": round(T_END, 1)},
          open(OUT / "steps.json", "w"), indent=1)
print("full.mp4", (OUT / "full.mp4").stat().st_size // 1000, "kB")
