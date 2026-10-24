"""Draw Gemma's world model into the recorded video frames.

Per frame: the world-model snapshot in force at that sim time (world_timeline.jsonl)
is projected through that frame's camera pose (cam_poses.jsonl): every object
as a 3D box, coloured by confidence, label + ±sigma; boxes hidden behind the
kitchen (scan mesh) are dimmed. After each perception call, an inset shows the
camera image(s) Gemma just boxed.

  python gemma_ctl/overlay_video.py <rig_run_dir> <live_run_dir>
  -> <rig_run_dir>/frames_ov/ (same names as frames/)
"""
import json
import os
import math
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

RIG, LIVE = Path(sys.argv[1]), Path(sys.argv[2])
HEAD = Path(sys.argv[sys.argv.index("--head") + 1]) if "--head" in sys.argv else None   # replay_head bundle dir
INTRO = "--intro" in sys.argv
OUT = RIG / "frames_ov"
OUT.mkdir(exist_ok=True)
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
INSET_S = 6.0

poses = {}
for l in open(RIG / "cam_poses.jsonl"):
    d = json.loads(l); poses[d["f"]] = d
snaps = [json.loads(l) for l in open(LIVE / "world_timeline.jsonl")]
convo = [json.loads(l) for l in open(LIVE / "conversation.jsonl")] if (LIVE / "conversation.jsonl").exists() else []
convo.sort(key=lambda e: e["t"])
import re


def _name_at(oid, t):
    """World-model label of an object id at sim time t (ids alone mean nothing to a viewer)."""
    lab = None
    for sn in snaps:
        if sn["t"] > t + 1.0:
            break
        if oid in sn["objects"]:
            lab = sn["objects"][oid]["label"]
    return lab


for e in convo:      # show names next to ids in tool calls
    if e["kind"] == "call":
        e["text"] = re.sub(r"\b(obj\d+)\b(?!\s*\()",
                           lambda m: f"{m.group(1)} · {_name_at(m.group(1), e['t'])}" if _name_at(m.group(1), e["t"]) else m.group(1),
                           e["text"])
for _i, _e in enumerate(convo):     # global log order (events can share a sim instant)
    _e["i"] = _i
stack_ev = [e for e in convo if e["kind"] in ("exec", "kplanner", "ik", "percept")]
convo = [e for e in convo if e["kind"] not in ("task", "exec", "kplanner", "ik", "percept")]
for e in convo:      # strip Gemma's leaked thinking marker ("thought ... <channel|>")
    if e["kind"] == "gemma":
        e["text"] = re.sub(r"^\s*thought\s*", "", e["text"], flags=re.I).split("<channel|>")[-1].strip()
        # ids are for tool calls; on screen Gemma speaks in names
        e["text"] = re.sub(r"\s*\(obj\d+\)", "", e["text"])
        e["text"] = re.sub(r"\bobj\d+\b", "it", e["text"])


def _clean_result(txt):
    """Readable one-liners: drop look() boilerplate, shorten skill results."""
    if txt.startswith("FAILED"):
        if len(txt) >= 116:            # logged text was cut mid-word: end cleanly
            txt = txt[:txt.rfind(" ")] + " …"
        return txt
    out = []
    for p in (q.strip() for q in txt.split("·")):
        if p.startswith("you_are_near") or p == "holding: None":
            continue
        p = re.sub(r"^picked: .*", "picked", p)
        p = re.sub(r"^placed: .*", "placed", p)
        p = re.sub(r"^on: (.*)", r"on \1", p)
        p = re.sub(r"^palm_error_m ([0-9.]+) cm", r"palm \1 cm", p)
        p = re.sub(r"^position_sigma_cm ([0-9.]+)", r"±\1 cm", p)
        p = re.sub(r"^orbbec_confirms: True", "camera confirms", p)
        p = re.sub(r"^orbbec_confirms: False", "camera did not confirm", p)
        p = re.sub(r"^refined position ±([0-9.]+) -> ±([0-9.]+) cm with the head cameras", r"head-cam look ±\1→±\2 cm", p)
        out.append(p)
    return " · ".join(out)


# one HUD line per tool call: "Gemma → call → result"
merged = []
for e in convo:
    if e["kind"] == "result" and merged and merged[-1]["kind"] == "call" and "res" not in merged[-1]:
        merged[-1]["res"] = _clean_result(e["text"]); merged[-1]["t_res"] = e["t"]
        continue
    merged.append(dict(e))
convo = merged
for i, e in enumerate(convo):        # a call with no result before the next request was interrupted
    if e["kind"] == "call" and "res" not in e and any(x["kind"] == "user" for x in convo[i + 1:]):
        nxt = next(x["t"] for x in convo[i + 1:] if x["kind"] == "user")
        e["res"], e["t_res"] = "(interrupted)", nxt
ROLE = {"user": (os.environ.get("HUD_USER_LABEL", "Operator (scripted):"), (255, 255, 255)), "call": ("Gemma →", (110, 205, 255)),
        "result": ("←", (240, 190, 90)), "gemma": ("Gemma", (120, 220, 160))}


def wrap(dr, text, font, width):
    out, line = [], ""
    for w in text.split():
        t = (line + " " + w).strip()
        if dr.textlength(t, font=font) > width and line:
            out.append(line); line = w
        else:
            line = t
    if line:
        out.append(line)
    return out


STACK_PREFIX = {"exec": "Executor:", "kplanner": "kplanner:", "ik": "IK + SONIC:", "percept": "Perception:"}


def draw_hud(im, t, status, n_lines=3, upto=None):
    """Caption strip with three fixed rows and clear ownership:
    1 conversation · 2 Gemma's latest request to the robot stack (+ result) ·
    3 what the stack is doing now (executor / kplanner / IK + SONIC / perception)."""
    W, H = im.size
    h = int(H * 0.125)
    y0 = H - h - int(H * 0.11)          # lifted clear of video-player controls
    dr = ImageDraw.Draw(im, "RGBA")
    for i in range(h + 10):
        a = int(215 * min(1.0, i / (h * 0.22), (h + 10 - i) / (h * 0.22)))
        dr.line([(0, y0 + i), (W, y0 + i)], fill=(6, 10, 8, a))
    fb = ImageFont.truetype(FONT, max(18, H // 46)); fs = ImageFont.truetype(FONT, max(14, H // 70))
    ev = [e for i, e in enumerate(convo) if e["t"] <= t and (upto is None or i <= upto)]
    talk = next((e for e in reversed(ev) if e["kind"] in ("user", "gemma")), None)
    req = next((e for e in reversed(ev) if e["kind"] == "call"), None)
    cut_i = (convo[upto].get("i_cut", convo[upto]["i"])) if upto is not None else None
    stack = next((e for e in reversed(stack_ev) if e["t"] <= t and (cut_i is None or e["i"] <= cut_i)), None)
    rows = []
    if talk:
        if talk["kind"] == "user":
            who = "You:" if talk.get("source") == "typed" else "Operator (scripted):"
            rows.append((f"{who} {talk['text']}", (255, 255, 255)))
        else:
            rows.append((f"Gemma: {talk['text']}", (120, 220, 160)))
    else:
        rows.append(("", (255, 255, 255)))
    if req:
        txt = re.sub(r"\s*\[[0-9.]+ s\]$", "", req["text"])
        c = (110, 205, 255)
        if "res" in req and req["t_res"] <= t:
            txt = f"Gemma requests: {txt} → {req['res']}"
            if req["res"].startswith("FAILED"):
                c = (255, 110, 100)
        else:
            txt = f"Gemma requests: {txt} → executing on the robot stack…"
        rows.append((txt, c))
    else:
        rows.append(("", (110, 205, 255)))
    if stack and t - stack["t"] < 20:
        rows.append((f"{STACK_PREFIX.get(stack['kind'], '')} {stack['text']}", (235, 200, 140)))
    else:
        rows.append(("", (235, 200, 140)))
    maxw = W - 60
    if not rows[2][0] and rows[0][0] and dr.textlength(rows[0][0], font=fb) > maxw:
        words, l1 = rows[0][0].split(), ""
        while words and dr.textlength((l1 + " " + words[0]).strip(), font=fb) <= maxw:
            l1 = (l1 + " " + words.pop(0)).strip()
        rows = [(l1, rows[0][1]), ("   " + " ".join(words), rows[0][1]), rows[1]]
    st_w = dr.textlength(status, font=fs)
    dr.text((W - st_w - 20, y0 - 22), status, fill=(200, 212, 205, 235), font=fs)
    lh = (h - 22) / 3
    for i, (line, c) in enumerate(rows):
        if not line:
            continue
        if dr.textlength(line, font=fb) > maxw:
            while line and dr.textlength(line + " …", font=fb) > maxw:
                line = line[:-1]
            line = line.rstrip() + " …"
        dr.text((28, y0 + 16 + i * lh), line, fill=(*c, 255), font=fb)


def colour(s):
    return (34, 160, 107) if s <= 0.03 else (217, 138, 28) if s <= 0.06 else (214, 69, 61)


def cam(p):
    eye = np.array(p["eye"], float); f = np.array(p["aim"], float) - eye; f /= np.linalg.norm(f)
    r = np.cross(f, [0, 0, 1.0]); r /= np.linalg.norm(r); dn = np.cross(f, r)
    R = np.stack([r, dn, f], 1)
    fx = p["w"] / (2 * math.tan(math.radians(p["hfov"]) / 2))
    return eye, R, fx, p["w"] / 2, p["h"] / 2


def project(P, eye, R, fx, cx, cy):
    pc = (np.asarray(P) - eye) @ R
    ok = pc[:, 2] > 0.1
    uv = np.stack([fx * pc[:, 0] / np.maximum(pc[:, 2], 1e-6) + cx, fx * pc[:, 1] / np.maximum(pc[:, 2], 1e-6) + cy], 1)
    return uv, ok


import open3d as o3d
_m = o3d.io.read_triangle_mesh("/home/stickbot/projects/x2-kitchen-sim/assets/kitchen/kitchen_collision.obj")
SCENE = o3d.t.geometry.RaycastingScene()
SCENE.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(_m))


def occluded(eye, pt):
    """True when the kitchen mesh blocks the line of sight from eye to pt (box drawn faint, no label)."""
    d = np.asarray(pt) - eye; L = float(np.linalg.norm(d))
    r = o3d.core.Tensor(np.array([[*eye, *(d / L)]], np.float32))
    return float(SCENE.cast_rays(r)["t_hit"].numpy()[0]) < L - 0.12


EDGES = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7)]

head_idx = []
if HEAD is not None and (HEAD / "cam_poses.jsonl").exists():
    for l in open(HEAD / "cam_poses.jsonl"):
        d = json.loads(l); head_idx.append((d["t"], HEAD / "frames" / f"f{d['f']:06d}.jpg"))
    head_idx.sort()


def head_frame(t):
    if not head_idx:
        return None
    import bisect
    i = bisect.bisect_left([x[0] for x in head_idx], t)
    i = min(max(i, 0), len(head_idx) - 1)
    return head_idx[i][1] if abs(head_idx[i][0] - t) < 1.0 else None


# intro: after the opening look returns, until the first robot action
ACTIONS = ("pick", "place", "go_to", "walk", "turn")
t_look = next((sn["t"] for sn in snaps if sn["event"] == "look"), None)
t_act = next((e["t"] for e in convo if e["kind"] == "call" and e["text"].split("(")[0] in ACTIONS), None)
intro_views = []
if INTRO and t_look is not None:
    first = next(sn for sn in snaps if sn["event"] == "look")
    if first.get("survey"):
        intro_views = sorted(Path(first["survey"]).glob("view*_boxes.jpg"))[:1]   # one wide room view
insets = {}


def current_target(t):
    """Object id of the latest pick/place request at time t (its label gets coordinates)."""
    tid = None
    for e in convo:
        if e["t"] > t:
            break
        if e["kind"] == "call" and e["text"].startswith(("pick", "place")):
            m = re.search(r"\b(obj\d+)\b", e["text"])
            tid = m.group(1) if m else tid
    return tid


def inset_for(s):
    if not s.get("survey"):
        return None
    if s["survey"] in insets:
        return insets[s["survey"]]
    d = Path(s["survey"])
    ims = [Image.open(p).convert("RGB") for p in sorted(d.glob("view*_boxes.jpg"))]
    if not ims:
        return None
    n = len(ims); cols = 2 if n > 1 else 1; rows = math.ceil(n / cols)
    tw, th = 320, 180
    g = Image.new("RGB", (cols * tw, rows * th + 30), (0, 0, 0))
    for i, im in enumerate(ims):
        g.paste(im.resize((tw, th)), ((i % cols) * tw, (i // cols) * th + 30))
    dr = ImageDraw.Draw(g); dr.text((8, 5), f"What Gemma sees ({s['event']})", fill=(255, 255, 255),
                                    font=ImageFont.truetype(FONT, 18))
    insets[s["survey"]] = g
    return g


import os
frames = sorted((RIG / "frames").glob("f*.jpg"))
if os.environ.get("OV_ONLY"):        # preview: render just these frame numbers
    only = {int(x) for x in os.environ["OV_ONLY"].split(",")}
    frames = [f for f in frames if int(f.stem[1:]) in only]
f_lab = ImageFont.truetype(FONT, 22); f_hud = ImageFont.truetype(FONT, 24)
for fp in frames:
    k = int(fp.stem[1:])
    im = Image.open(fp).convert("RGB")
    t_here = poses.get(k, {}).get("t", 0.0)
    in_intro = False          # the intro is rendered as separate stills (below), timed by make_paced
    if in_intro:
        span = max(0.5, (t_act - t_look) / len(intro_views))
        vi = min(len(intro_views) - 1, int((t_here - t_look) / span))
        bg = Image.open(intro_views[vi]).convert("RGB").resize(im.size, Image.LANCZOS)
        d0 = ImageDraw.Draw(bg, "RGBA"); f0 = ImageFont.truetype(FONT, 26)
        names = ["Room camera 1 · counters", "Room camera 2 · dining", "Room camera 3 · fridge side", "Robot head camera"]
        lab = f"What Gemma sees: {names[vi] if vi < len(names) else intro_views[vi].stem}"
        d0.rectangle([0, 64, d0.textlength(lab, font=f0) + 30, 104], fill=(0, 0, 0, 170))
        d0.text((14, 68), lab, fill=(255, 255, 255, 255), font=f0)
        im = bg
    p = poses.get(k)
    if p is None:
        im.save(OUT / fp.name, quality=90); continue
    t = p["t"]
    s = None
    for sn in snaps:
        if sn["t"] <= t:
            s = sn
        else:
            break
    status = f"world model: {len(s['objects'])} objects" if s is not None else "world model: empty"
    if s is not None and not in_intro:
        eye, R, fx, cx, cy = cam(p)
        dr = ImageDraw.Draw(im, "RGBA")
        sx, sy = im.width / p["w"], im.height / p["h"]
        placed = []      # label rectangles already drawn (de-overlap)
        for oid, o in s["objects"].items():
            if oid == s.get("held"):
                continue
            mx, my, mz = o["mean"]; hx, hy = [max(0.04, v) / 2 for v in o["size_xy"]]; h = max(0.03, o["height"])
            if np.linalg.norm(np.array([mx, my, mz]) - eye) > 5.0:
                continue
            P = [[mx - hx, my - hy, mz], [mx + hx, my - hy, mz], [mx + hx, my + hy, mz], [mx - hx, my + hy, mz],
                 [mx - hx, my - hy, mz + h], [mx + hx, my - hy, mz + h], [mx + hx, my + hy, mz + h], [mx - hx, my + hy, mz + h]]
            uv, ok = project(P, eye, R, fx, cx, cy)
            if not ok.all():
                continue
            uv = uv * [sx, sy]
            if (uv[:, 0].max() < 0) or (uv[:, 0].min() > im.width) or (uv[:, 1].max() < 0) or (uv[:, 1].min() > im.height):
                continue
            c = colour(o["sigma_xy"])
            hidden = occluded(eye, [mx, my, mz + h / 2])
            a = 90 if hidden else 255
            for i, j in EDGES:
                dr.line([tuple(uv[i]), tuple(uv[j])], fill=(*c, a), width=2 if hidden else 3)
            if not hidden:
                tx, ty = uv[:, 0].min(), uv[:, 1].min() - 28
                txt = f"{o['label']} ±{o['sigma_xy'] * 100:.0f}cm"
                if oid == current_target(t):
                    txt += f"  ({mx:+.2f}, {my:+.2f}, {mz:.2f}) m"
                w = dr.textlength(txt, font=f_lab)
                for _ in range(8):     # nudge down until it clears earlier labels
                    if not any(tx < r[2] and tx + w > r[0] and ty < r[3] and ty + 27 > r[1] for r in placed):
                        break
                    ty += 30
                placed.append((tx - 3, ty - 2, tx + w + 3, ty + 25))
                dr.rectangle([tx - 3, ty - 2, tx + w + 3, ty + 25], fill=(0, 0, 0, 170))
                dr.text((tx, ty), txt, fill=(*c, 255), font=f_lab)
        held = s["objects"].get(s.get("held") or "", {}).get("label")
        status = f"world model: {len(s['objects'])} objects" + (f" · holding: {held}" if held else "")
        if t - s["t"] < INSET_S and not (INTRO and s["event"] == "look"):
            ins = inset_for(s)
            if ins is not None:
                scale = 0.36 * im.width / ins.width
                ins2 = ins.resize((int(ins.width * scale), int(ins.height * scale)))
                im.paste(ins2, (im.width - ins2.width - 12, 52))
    if p is not None and not in_intro:
        hf = head_frame(t)
        if hf is not None and hf.exists():
            hw = int(im.width * 0.22); hh = int(hw * 9 / 16)
            hi = Image.open(hf).convert("RGB").resize((hw, hh), Image.LANCZOS)
            x = im.width - hw - 16; y = int(im.height * (1 - 0.125 - 0.11)) - hh - 34
            dh = ImageDraw.Draw(im, "RGBA")
            dh.rectangle([x - 3, y - 26, x + hw + 3, y + hh + 3], fill=(0, 0, 0, 190))
            dh.text((x + 4, y - 23), "Robot head camera", fill=(255, 255, 255, 255), font=ImageFont.truetype(FONT, 17))
            im.paste(hi, (x, y))
    if p is not None:
        draw_hud(im, t, status)
    im.save(OUT / fp.name, quality=90)
# intro stills (one room view, full frame), timed by make_paced:
#   0: after the opening look  -> HUD shows Gemma's description, boxes only
#   1: after the first request -> HUD shows the request + Gemma's tool call; the
#      TARGET's box gets a tag: Gemma box (0-1000 image coords) -> global position
if intro_views and t_act is not None:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from gemma_ctl.perceive import View, lift_object
    vp = intro_views[0]
    sd = vp.parent
    W0, H0 = Image.open(frames[0]).size
    i_reply = next((i for i, e in enumerate(convo) if e["kind"] == "gemma" and e["t"] >= t_look), None)
    i_look = next((i for i, e in enumerate(convo) if e["kind"] == "call" and e["text"].startswith("look")), 0)
    i_act = next(i for i, e in enumerate(convo) if e["kind"] == "call" and e["t"] >= t_act - 1e-6
                 and e["text"].split("(")[0] in ACTIONS)
    act_call = convo[i_act]
    m = re.search(r"\b(obj\d+)\b", act_call["text"])
    tgt_id = m.group(1) if m else None
    s_at = None
    for sn in snaps:
        if sn["t"] <= t_act + 1e-3:
            s_at = sn
    tgt_xyz = np.array(s_at["objects"][tgt_id]["mean"]) if (s_at and tgt_id in s_at["objects"]) else None
    st = f"world model: {len(s_at['objects'])} objects" if s_at else "world model: empty"
    raw = json.loads((sd / "gemma_raw.json").read_text())
    rv = next((r for r in raw if r["view"] == 0), None)
    v = View(sd, 0)
    best = None
    if rv is not None and tgt_xyz is not None:
        for bx in rv["boxes"]:
            if bx.get("type") == "place":
                continue
            o = lift_object(v, bx["box_2d"])
            if o and (best is None or np.linalg.norm(np.array(o["base"][:2]) - tgt_xyz[:2]) < best[0]):
                best = (float(np.linalg.norm(np.array(o["base"][:2]) - tgt_xyz[:2])), bx, o)
    beats = ((i_look, False, False),                                           # question, no boxes yet
             (i_reply if i_reply is not None else i_act - 1, True, False),    # boxes + Gemma's answer
             (i_act, True, True))                                             # request + target tag
    first_exec = next((e for e in stack_ev if e["kind"] == "exec" and e["i"] > convo[i_act]["i"]), None)
    for k, (upto, boxed, tag) in enumerate(beats):
        hud_t = convo[upto]["t"] + 1e-3
        if tag and first_exec is not None:
            convo[upto]["i_cut"] = first_exec["i"]
        src_img = vp if boxed else vp.with_name(vp.name.replace("_boxes", ""))
        im = Image.open(src_img).convert("RGB").resize((W0, H0), Image.LANCZOS)
        d0 = ImageDraw.Draw(im, "RGBA"); f0 = ImageFont.truetype(FONT, 28)
        lab = "What Gemma sees · Room camera 1 · counters"
        d0.rectangle([0, 64, d0.textlength(lab, font=f0) + 30, 106], fill=(0, 0, 0, 175))
        d0.text((14, 69), lab, fill=(255, 255, 255, 255), font=f0)
        if not boxed:          # the question, asked over the plain camera view
            draw_hud(im, hud_t, "world model: empty", upto=upto)
            im.save(OUT / f"intro_{k}.jpg", quality=90)
            continue
        if tag and best is not None and best[0] < 0.25:
            _, bx, o = best
            sx0, sy0 = W0 / v.W, H0 / v.H
            x0, y0, x1, y1 = v.px(bx["box_2d"])
            d0.rectangle([x0 * sx0 - 4, y0 * sy0 - 4, x1 * sx0 + 4, y1 * sy0 + 4], outline=(255, 215, 0, 255), width=5)
            ft = ImageFont.truetype(FONT, 22)
            l1 = f"target {tgt_id}: {bx['label']}"
            l2 = "Gemma box [" + ", ".join(str(int(round(float(c)))) for c in bx["box_2d"]) + "]  (0-1000 image coords)"
            l3 = f"→ depth + camera pose → ({o['base'][0]:+.2f}, {o['base'][1]:+.2f}, {o['base'][2]:.2f}) m kitchen frame"
            w = max(d0.textlength(l, font=ft) for l in (l1, l2, l3))
            tx = min(x0 * sx0, W0 - w - 20); ty = max(120, y0 * sy0 - 98)
            d0.rectangle([tx - 8, ty - 6, tx + w + 8, ty + 84], fill=(0, 0, 0, 200), outline=(255, 215, 0, 255), width=2)
            d0.text((tx, ty), l1, fill=(255, 215, 0, 255), font=ft)
            d0.text((tx, ty + 27), l2, fill=(255, 230, 170, 255), font=ft)
            d0.text((tx, ty + 54), l3, fill=(255, 255, 255, 255), font=ft)
        draw_hud(im, hud_t, st, upto=upto)
        im.save(OUT / f"intro_{k}.jpg", quality=90)
    json.dump({"t_look": t_look, "t_act": t_act, "n": 3, "hold_s": [3.5, 7.0, 5.5]}, open(OUT / "intro.json", "w"))
    print("intro stills: 3, target", tgt_id, "box match", None if best is None else round(best[0], 3))
print("overlaid", len(frames), "->", OUT)
