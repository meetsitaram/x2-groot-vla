"""Variable-speed cut of a run bundle: real time (1x) while the conversation
with Gemma updates, so the HUD can be read; fast (FAST x) in between.

Reading window per conversation event: 1.5 s + 1 s per 18 characters of new
text, clamped to [2.5, 9] s (sim time at 1x). Overlapping windows merge.
The speed badge (bottom right) shows the current playback speed.

  python gemma_ctl/make_paced.py <bundle> [frames_ov] [FAST]
  -> <bundle>/review/paced.mp4
"""
import json
import re
import subprocess
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

B = Path(sys.argv[1])
SRC = B / (sys.argv[2] if len(sys.argv) > 2 else "frames_ov")
FAST = float(sys.argv[3]) if len(sys.argv) > 3 else 4.0
DT = 0.08                      # sim s per recorded frame
OUT_FPS = 25.0
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"

frames = sorted(SRC.glob("f*.jpg"))
INTRO_EACH = 5.5                       # video seconds per intro still (description, then the request)
intro = json.loads((SRC / "intro.json").read_text()) if (SRC / "intro.json").exists() else None
poses = {json.loads(l)["f"]: json.loads(l)["t"] for l in open(B / "cam_poses.jsonl")}
conv = [json.loads(l) for l in open(B / "live/conversation.jsonl")]
for e in conv:
    if e["kind"] == "gemma":
        e["text"] = re.sub(r"^\s*thought\s*", "", e["text"], flags=re.I).split("<channel|>")[-1].strip()

# per-message pacing: each conversation line gets its own reading time in VIDEO
# seconds before the next one appears. If messages arrive faster than that in
# sim time, the video slows (down to a hold); otherwise the reading time plays
# at 1x and the rest of the gap at FAST.
def read_s(e):
    return min(9.0, max(2.5, 1.5 + len(e["text"]) / 18.0))


# kplanner / IK lines are live telemetry: drawn on the HUD, but they get no reading pause
TELEMETRY = {"kplanner", "ik"}
evs = sorted((e for e in conv if e["kind"] not in TELEMETRY), key=lambda e: e["t"])
t_end = len(frames) * DT
segs = []                                 # (sim_t0, sim_t1, speed)
cur = 0.0
for i, e in enumerate(evs):
    t0 = max(cur, e["t"])
    if t0 > cur:
        segs.append((cur, t0, FAST))
    t1 = evs[i + 1]["t"] if i + 1 < len(evs) else min(t_end, t0 + read_s(e) + 2)
    gap = max(0.0, t1 - t0)
    R = read_s(e)
    if gap >= R:
        segs.append((t0, t0 + R, 1.0))
        if t1 > t0 + R:
            segs.append((t0 + R, t1, FAST))
    else:
        segs.append((t0, t1, max(gap / R, 0.02)))   # slow / hold to give R seconds of reading
    cur = max(cur, t1)
if cur < t_end:
    segs.append((cur, t_end, FAST))
if intro:   # the video OPENS on the intro stills; the robot idling before the first action is skipped
    segs = [(a, b, 1e6 if a < intro["t_act"] else sp) for a, b, sp in segs]
wins = [(a, b) for a, b, sp in segs if sp <= 1.0]


def speed_at(t):
    for a, b, sp in segs:
        if a <= t < b:
            return sp
    return FAST


out = B / "frames_paced"
out.mkdir(exist_ok=True)
for p in out.glob("*.jpg"):
    p.unlink()
font = ImageFont.truetype(FONT, 30)
# walk sim time at the local speed: each output frame advances speed/OUT_FPS sim s
t, n, last_img, last_key = 0.0, 0, None, None
intro_done = intro is None
while t < t_end:
    if not intro_done:
        holds = intro.get("hold_s", [INTRO_EACH] * intro["n"])
        for vi in range(intro["n"]):
            im = Image.open(SRC / f"intro_{vi}.jpg").convert("RGB")
            dr = ImageDraw.Draw(im, "RGBA"); txt = "what Gemma sees"
            w = dr.textlength(txt, font=font)
            dr.rounded_rectangle([14, 118, 14 + w + 28, 168], radius=10, fill=(20, 120, 80, 210))
            dr.text((28, 124), txt, fill=(255, 255, 255, 255), font=font)
            for _ in range(int(holds[vi] * OUT_FPS)):
                im.save(out / f"p{n:06d}.jpg", quality=88); n += 1
        intro_done = True
        t = intro["t_act"]          # cut to the robot as it starts moving
        continue
        last_key = None
    k = min(len(frames) - 1, int(t / DT))
    sp = speed_at(t)
    key = (k, sp)
    if key != last_key:
        im = Image.open(frames[k]).convert("RGB")
        dr = ImageDraw.Draw(im, "RGBA")
        txt = ("1×  real time" if abs(sp - 1.0) < 1e-6 else
               "paused to read" if sp < 0.15 else f"{sp:.2g}×  slowed to read" if sp < 1.0 else f"{sp:g}×")
        w = dr.textlength(txt, font=font)
        # top-left, under the caption bar (the bottom strip is the conversation)
        x0, y0 = 14, 74
        dr.rounded_rectangle([x0, y0, x0 + w + 28, y0 + 50], radius=10,
                             fill=(20, 120, 80, 210) if sp <= 1.0 else (0, 0, 0, 170))
        dr.text((x0 + 14, y0 + 6), txt, fill=(255, 255, 255, 255), font=font)
        last_img, last_key = im, key
    last_img.save(out / f"p{n:06d}.jpg", quality=88)
    n += 1
    t += sp / OUT_FPS

res = B / "review" / "paced.mp4"
subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(OUT_FPS), "-i", str(out / "p%06d.jpg"),
                "-vf", "scale=1280:-2:flags=lanczos", "-c:v", "libx264", "-profile:v", "high", "-crf", "24", "-tune", "stillimage",
                "-maxrate", "760k", "-bufsize", "1520k", "-preset", "slow", "-pix_fmt", "yuv420p",
                "-movflags", "+faststart", "-an", str(res)], check=True)
real = sum(min(b, t_end) - max(a, 0) for a, b in wins)   # sim seconds at <=1x
print(f"paced: {n} frames = {n / OUT_FPS:.0f} s video; {len(wins)} reading windows, {real:.0f} s at 1x; "
      f"{res.stat().st_size / 1e6:.1f} MB -> {res}")
