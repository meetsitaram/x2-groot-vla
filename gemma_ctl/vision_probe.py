"""Standalone Gemma vision probe: image in, boxes out, no sim needed.

  python -m gemma_ctl.vision_probe IMG [IMG ...]               # perception prompt
  python -m gemma_ctl.vision_probe IMG --find "black mug"      # targeted search
  python -m gemma_ctl.vision_probe IMG --ask "what is on the table?"

Uses the same Ollama call and prompt as perceive.py. Writes <img>_boxes.jpg
next to --out (default runs/gemma/probe/) with the boxes drawn.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from .perceive import PROMPT, gemma_boxes

ROOT = Path(__file__).resolve().parents[1]

FIND = """Look at this camera image and find every "{noun}".
Return ONLY JSON: a list of {{"label": str, "box_2d": [ymin, xmin, ymax, xmax]}}
with coordinates normalized to 0-1000. Return [] if there is none."""


def draw(img_path, boxes, out_path):
    from PIL import Image, ImageDraw
    im = Image.open(img_path).convert("RGB")
    W, H = im.size
    d = ImageDraw.Draw(im)
    lw = max(2, W // 400)
    for b in boxes:
        y0, x0, y1, x1 = [float(v) for v in b["box_2d"]]
        xy = [x0 * W / 1000, y0 * H / 1000, x1 * W / 1000, y1 * H / 1000]
        col = (255, 160, 0) if b.get("type") == "place" else (0, 255, 80)
        d.rectangle(xy, outline=col, width=lw)
        d.text((xy[0] + 3, max(0, xy[1] - 12)), str(b.get("label", "?")), fill=col)
    im.save(out_path, quality=90)


def ask(img_path, question, model):
    import base64
    import urllib.request
    from .perceive import OLLAMA
    b = base64.b64encode(Path(img_path).read_bytes()).decode()
    body = json.dumps({"model": model, "stream": False, "think": False,
                       "options": {"temperature": 0.0, "num_ctx": 16384},
                       "messages": [{"role": "user", "content": question, "images": [b]}]}).encode()
    t0 = time.monotonic()
    r = json.loads(urllib.request.urlopen(urllib.request.Request(
        f"{OLLAMA}/api/chat", data=body, headers={"Content-Type": "application/json"}),
        timeout=300).read())
    return r["message"]["content"], time.monotonic() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("images", nargs="+")
    ap.add_argument("--model", default="gemma4:26b")
    ap.add_argument("--find", help="noun for a targeted search instead of the full survey prompt")
    ap.add_argument("--ask", help="free-form question; prints the answer, no boxes")
    ap.add_argument("--out", default=str(ROOT / "runs/gemma/probe"))
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for img in a.images:
        if a.ask:
            txt, dt = ask(img, a.ask, a.model)
            print(f"== {img}  ({dt:.2f} s)\n{txt}")
            continue
        prompt = FIND.format(noun=a.find) if a.find else PROMPT
        boxes, dt, raw = gemma_boxes(img, model=a.model, prompt=prompt)
        dst = out / f"{Path(img).stem}_boxes.jpg"
        draw(img, boxes, dst)
        print(f"== {img}  {len(boxes)} boxes  {dt:.2f} s  -> {dst}")
        print(json.dumps(boxes, indent=1) if boxes else raw)


if __name__ == "__main__":
    main()
