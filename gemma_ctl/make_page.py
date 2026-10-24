"""Write the review page for a run (references review/* files, published
alongside the page as a multi-file artifact).

  python gemma_ctl/make_page.py runs/gemma/kitchen/demo2
"""
import html
import json
import sys
from pathlib import Path

RUN = Path(sys.argv[1])
R = json.load(open(RUN / "review/steps.json"))
T = json.load(open(RUN / "transcript.json"))
esc = html.escape

# transcript -> turns (a turn ends at each reply)
utter = ["Hey robot, what's in the kitchen right now?", "Please bring the mug over to the sink.",
         "Now take the soda can and put it on the counter next to the stove.", "Thanks! Go wait by the fridge."]
turns, cur = [], []
for r in T:
    cur.append(r)
    if "reply" in r:
        turns.append(cur); cur = []


def res_text(r):
    x = r["result"]
    if not x.get("ok"):
        return x.get("error", "failed"), "retry"
    if r["tool"] == "look":
        return " · ".join(o.replace(" - ", " on ").replace("(white mug) ", "").replace("(soda can) ", "")
                          for o in x.get("objects", [])), "ok"
    if "palm_error_m" in x:
        return f"palm {x['palm_error_m'] * 100:.1f} cm from target", "ok"
    return x.get("info", "done"), "ok"


# executor detail per tool call, from agent.log: route legs + palm targets
import re
blocks, curb = [], None
for line in open(RUN / "agent.log"):
    if line.startswith("→ "):
        curb = {"route": None, "reach": []}
    elif line.startswith("  ← ") and curb is not None:
        blocks.append(curb); curb = None
    elif curb is not None:
        m = re.search(r"\[route\] \d+ legs: (.*)", line)
        if m:
            curb["route"] = m.group(1).split(" -> ")[-1].strip()
        m = re.search(r"\[reach\] target \[([^\]]+)\] palm err ([0-9.]+)", line)
        if m:
            curb["reach"].append(([float(v) for v in m.group(1).split(",")], float(m.group(2))))


def detail(b):
    if not b:
        return ""
    parts = []
    if b["route"]:
        parts.append(f"stand {b['route']}")
    if len(b["reach"]) >= 2:
        names = ["hover", "grasp" , "lift", "carry"]
        for (xyz, e), n in zip(b["reach"][:2], names):
            parts.append(f"{n} ({xyz[0]:+.2f}, {xyz[1]:+.2f}, {xyz[2]:.2f})")
    return " · ".join(parts)


bi = iter(blocks)
trace = []
for u, turn in zip(utter, turns):
    rows = [f'<p class="say">“{esc(u)}”</p>']
    for r in turn:
        if "tool" in r:
            args = ", ".join(f"{k}={v}" for k, v in r["args"].items())
            txt, cls = res_text(r)
            det = detail(next(bi, None))
            rows.append(f'<div class="call"><span class="fn">{esc(r["tool"])}({esc(args)})</span>'
                        f'<span class="res">{esc(txt)}</span><span class="pill {cls}">'
                        f'{"ok" if cls == "ok" else "failed"}</span>'
                        f'<span class="lat">{r["llm_s"]:.2f} s</span>'
                        + (f'<span class="xyz">{esc(det)}</span>' if det else "") + '</div>')
        else:
            rows.append(f'<p class="reply">{esc(r["reply"])}</p>')
    trace.append('<div class="turn">' + "".join(rows) + "</div>")

cards = []
for s in R["steps"]:
    dur = (s["t1"] - s["t0"]) / s["speed"]
    cards.append(f'''<figure class="clip">
  <video controls playsinline muted preload="none" poster="review/{s['key']}.jpg" src="review/{s['key']}.mp4"></video>
  <figcaption><b>{esc(s['title'])}</b><span>{s['speed']}× · {dur:.0f} s clip · sim {s['t0']:.0f}–{s['t1']:.0f} s</span></figcaption>
</figure>''')

n_ok = sum(1 for r in T if "tool" in r and r["result"].get("ok"))
n_calls = sum(1 for r in T if "tool" in r)
page = open(Path(__file__).with_name("review_template.html")).read()
page = (page.replace("__TRACE__", "\n".join(trace)).replace("__CARDS__", "\n".join(cards))
        .replace("__SIM__", f"{R['sim_s']:.0f}").replace("__FULLSPEED__", str(R["full_speed"]))
        .replace("__CALLS__", f"{n_ok} / {n_calls}"))
(RUN / "gemma_kitchen_review.html").write_text(page)
print("wrote", RUN / "gemma_kitchen_review.html")
