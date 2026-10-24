"""Live robot camera view with Gemma's objects, boxes and distances (READ-ONLY, nothing moves).

Reads the PC2 streamer (gemma_ctl/pc2_live_streamer.py, plain HTTP :8793: Orbbec colour + depth),
rotates it upright (the Orbbec is mounted inverted), re-detects objects with Gemma in a loop on the
newest frame, and gives each box a distance from the aligned depth and a direction from the
calibrated URDF mount. Serves http://127.0.0.1:8791 (MJPEG video + object list).

  python -m gemma_ctl.live_view --pc2 192.168.0.185 [--model gemma4:e4b]
"""
from __future__ import annotations

import argparse
import json
import math
import threading
import time
import urllib.request
import zlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import cv2
import numpy as np

from .multicam_look import urdf_frames
from .perceive import gemma_boxes
from .real_robot import LOOK_PROMPT

# Orbbec colour intrinsics at 1280x720 (camera_info, 2026-10-24 grab); scaled to the stream size
K_FULL = np.array([[692.0, 0, 640.0], [0, 692.0, 360.0], [0, 0, 1]])
TMP = Path("/tmp/claude-1000/gemma_live"); TMP.mkdir(parents=True, exist_ok=True)


class Live:
    def __init__(self, pc2, model):
        self.base = f"http://{pc2}:8793"
        self.model = model
        self.lock = threading.Lock()
        self.frame = None; self.frame_t = 0.0; self.n_frames = 0; self.fps = 0.0
        self.depth = None                 # upright, metres, 1/4 resolution
        self.dets = []; self.det_t = 0.0; self.det_s = 0.0; self.det_age_frame = None
        self.err = ""
        R, t = urdf_frames()["orbbec"]
        self.R_upright = R
        self.K = None
        try:
            info = json.loads(Path(__file__).resolve().parents[1].joinpath(
                "runs/gemma/real/pc2_all_122332/orbbec_rgb_info.json").read_text())
            self.K_full = np.array(info["k"], float).reshape(3, 3)
        except (OSError, ValueError):
            self.K_full = K_FULL

    def _get(self, path, timeout=5):
        return urllib.request.urlopen(self.base + path, timeout=timeout).read()

    def loop_rgb(self):
        t_last, n_last = time.monotonic(), 0
        while True:
            try:
                img = cv2.imdecode(np.frombuffer(self._get("/rgb.jpg"), np.uint8), cv2.IMREAD_COLOR)
                img = cv2.rotate(img, cv2.ROTATE_180)
                with self.lock:
                    self.frame, self.frame_t = img, time.time(); self.n_frames += 1; self.err = ""
                if time.monotonic() - t_last > 2:
                    self.fps = (self.n_frames - n_last) / (time.monotonic() - t_last)
                    t_last, n_last = time.monotonic(), self.n_frames
            except Exception as e:
                self.err = f"camera: {type(e).__name__}"; time.sleep(0.5)

    def loop_depth(self):
        while True:
            try:
                raw = self._get("/depth.bin")
                hdr, blob = raw.split(b"\n", 1); h = json.loads(hdr)
                d = np.frombuffer(zlib.decompress(blob), np.uint16).reshape(h["h"], h["w"])
                with self.lock:
                    self.depth = cv2.rotate(d, cv2.ROTATE_180).astype(np.float32) / 1000.0
            except Exception:
                pass
            time.sleep(0.4)

    def measure(self, b, W, H):
        """Distance (m, along the ray) and direction (deg) of a box centre."""
        y0, x0, y1, x1 = b["box_2d"]
        u, v = (x0 + x1) / 2000 * W, (y0 + y1) / 2000 * H
        s = W / 1280.0
        K = self.K_full.copy(); K[:2] *= s
        K[0, 2], K[1, 2] = W - 1 - K[0, 2], H - 1 - K[1, 2]   # upright image: principal point rotates
        xn, yn = (u - K[0, 2]) / K[0, 0], (v - K[1, 2]) / K[1, 1]
        ray = self.R_upright @ np.array([xn, yn, 1.0]); ray /= np.linalg.norm(ray)
        az = math.degrees(math.atan2(ray[1], ray[0]))
        dist = None
        with self.lock:
            dep = self.depth
        if dep is not None:
            dh, dw = dep.shape
            sx, sy = dw / 1000, dh / 1000
            cx0, cx1 = x0 + (x1 - x0) * 0.3, x1 - (x1 - x0) * 0.3
            cy0, cy1 = y0 + (y1 - y0) * 0.3, y1 - (y1 - y0) * 0.3
            patch = dep[int(cy0 * sy):int(cy1 * sy) + 1, int(cx0 * sx):int(cx1 * sx) + 1]
            vals = patch[(patch > 0.2) & (patch < 12)]
            if vals.size >= 4:
                z = float(np.median(vals)); dist = z * math.sqrt(1 + xn * xn + yn * yn)
        return dist, az

    def loop_detect(self):
        while True:
            with self.lock:
                img = None if self.frame is None else self.frame.copy()
            if img is None:
                time.sleep(0.3); continue
            p = TMP / "det.jpg"; cv2.imwrite(str(p), img, [cv2.IMWRITE_JPEG_QUALITY, 90])
            t0 = time.monotonic()
            try:
                boxes, _, _ = gemma_boxes(p, self.model, LOOK_PROMPT)
            except Exception as e:
                self.err = f"gemma: {type(e).__name__}"; time.sleep(1); continue
            H, W = img.shape[:2]
            dets = []
            for b in boxes:
                dist, az = self.measure(b, W, H)
                dets.append({"label": b.get("label", "?"), "box": b["box_2d"], "dist": dist, "az": az})
            dets.sort(key=lambda d: (d["dist"] is None, d["dist"] or 0))
            with self.lock:
                self.dets, self.det_t, self.det_s = dets, time.time(), time.monotonic() - t0

    def render(self):
        with self.lock:
            img = None if self.frame is None else self.frame.copy()
            dets = list(self.dets)
        if img is None:
            img = np.zeros((360, 640, 3), np.uint8)
            cv2.putText(img, "waiting for the robot camera...", (150, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
            return img
        H, W = img.shape[:2]
        for d in dets:
            y0, x0, y1, x1 = d["box"]
            p0, p1 = (int(x0 * W / 1000), int(y0 * H / 1000)), (int(x1 * W / 1000), int(y1 * H / 1000))
            near = d["dist"] is not None and d["dist"] < 1.5
            c = (60, 90, 255) if near else (60, 220, 140)
            cv2.rectangle(img, p0, p1, c, 2)
            txt = d["label"] + (f"  {d['dist']:.1f} m" if d["dist"] is not None else "")
            (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            ty = max(th + 4, p0[1] - 4)
            cv2.rectangle(img, (p0[0], ty - th - 4), (p0[0] + tw + 6, ty + 3), (20, 20, 20), -1)
            cv2.putText(img, txt, (p0[0] + 3, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.5, c, 1, cv2.LINE_AA)
        return img

    def state(self):
        with self.lock:
            return {"fps": round(self.fps, 1), "frames": self.n_frames,
                    "det_age_s": round(time.time() - self.det_t, 1) if self.det_t else None,
                    "det_s": round(self.det_s, 1), "model": self.model, "err": self.err,
                    "objects": [{"label": d["label"], "dist": None if d["dist"] is None else round(d["dist"], 2),
                                 "az": round(d["az"])} for d in self.dets]}


PAGE = """<!doctype html><html lang="en" data-theme="dark"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>X2 Live View</title>
<style>
:root{--bg:#101418;--panel:#171d22;--line:#283039;--fg:#e8edf1;--muted:#8d99a3;--ok:#4fc39a;--near:#ff7a5c}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font:15px/1.4 system-ui,sans-serif;display:flex;flex-direction:column;height:100vh}
header{display:flex;gap:12px;align-items:center;padding:8px 14px;border-bottom:1px solid var(--line);flex-wrap:wrap}
header b{font-size:16px}.st{color:var(--muted);font:13px ui-monospace,monospace}
main{flex:1;display:flex;gap:12px;padding:12px;min-height:0;flex-wrap:wrap}
.vid{flex:3 1 480px;min-width:0;display:flex;align-items:center;justify-content:center;background:#000;border-radius:8px;overflow:hidden}
.vid img{max-width:100%;max-height:100%;display:block}
aside{flex:1 1 240px;min-width:0;background:var(--panel);border:1px solid var(--line);border-radius:8px;padding:10px;overflow:auto}
aside h2{font-size:13px;text-transform:uppercase;letter-spacing:.08em;color:var(--muted);margin:0 0 8px}
.row{display:flex;justify-content:space-between;gap:8px;padding:5px 0;border-bottom:1px solid var(--line);font-variant-numeric:tabular-nums}
.row .d{color:var(--ok);white-space:nowrap}.row.near .d{color:var(--near)}.row .w{color:var(--muted);font-size:12px}
.badge{font:12px ui-monospace,monospace;padding:2px 8px;border-radius:99px;background:#20372e;color:var(--ok)}
</style></head><body>
<header><b>X2 live · Orbbec</b><span class="badge">read-only · no motion</span><span class="st" id="st">connecting…</span></header>
<main><div class="vid"><img id="v" alt="Live robot camera with Gemma's object boxes"></div>
<aside><h2>Objects (nearest first)</h2><div id="list"></div></aside></main>
<script>
// one frame per request, swapped in only when fully loaded (no long-lived MJPEG to stall)
const v=document.getElementById('v');
function nextFrame(){const im=new Image();let done=false;
  const go=()=>{if(done)return;done=true;setTimeout(nextFrame,120)};
  im.onload=()=>{v.src=im.src;go()}; im.onerror=go; setTimeout(go,4000);
  im.src='/frame.jpg?t='+Date.now();}
nextFrame();
async function poll(){try{const s=await (await fetch('/api/state')).json();
document.getElementById('st').textContent=`video ${s.fps} fps · Gemma ${s.model} ${s.det_s}s/frame · boxes ${s.det_age_s??'–'}s old`+(s.err?` · ${s.err}`:'');
document.getElementById('list').innerHTML=s.objects.map(o=>{const side=o.az>15?'left':o.az<-15?'right':'ahead';
return `<div class="row${o.dist!==null&&o.dist<1.5?' near':''}"><span>${o.label.replace(/</g,'&lt;')}<br><span class="w">${side} ${Math.abs(o.az)}°</span></span><span class="d">${o.dist===null?'–':o.dist.toFixed(2)+' m'}</span></div>`}).join('')||'<span class="w">no objects yet</span>';
}catch(e){document.getElementById('st').textContent='viewer offline'} setTimeout(poll,700)} poll();
</script></body></html>"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pc2", default="192.168.0.185")
    ap.add_argument("--model", default="gemma4:26b")
    ap.add_argument("--port", type=int, default=8791)
    a = ap.parse_args()
    L = Live(a.pc2, a.model)
    for f in (L.loop_rgb, L.loop_depth, L.loop_detect):
        threading.Thread(target=f, daemon=True).start()

    class H(BaseHTTPRequestHandler):
        def log_message(self, *x):
            pass

        def do_GET(self):
            if self.path == "/":
                b = PAGE.encode(); self.send_response(200); self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(b))); self.end_headers(); self.wfile.write(b)
            elif self.path == "/api/state":
                b = json.dumps(L.state()).encode(); self.send_response(200); self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(b))); self.end_headers(); self.wfile.write(b)
            elif self.path.startswith("/frame.jpg"):
                b = cv2.imencode(".jpg", L.render(), [cv2.IMWRITE_JPEG_QUALITY, 80])[1].tobytes()
                self.send_response(200); self.send_header("Content-Type", "image/jpeg")
                self.send_header("Cache-Control", "no-store"); self.send_header("Content-Length", str(len(b)))
                self.end_headers(); self.wfile.write(b)
            elif self.path == "/stream":
                self.send_response(200); self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=f"); self.end_headers()
                try:
                    while True:
                        jpg = cv2.imencode(".jpg", L.render(), [cv2.IMWRITE_JPEG_QUALITY, 80])[1].tobytes()
                        self.wfile.write(b"--f\r\nContent-Type: image/jpeg\r\nContent-Length: " + str(len(jpg)).encode() + b"\r\n\r\n" + jpg + b"\r\n")
                        time.sleep(0.15)
                except (BrokenPipeError, ConnectionResetError):
                    pass
            else:
                self.send_error(404)

    print(f"[live_view] http://127.0.0.1:{a.port}/  (PC2 {a.pc2}, model {a.model}, read-only)", flush=True)
    ThreadingHTTPServer(("127.0.0.1", a.port), H).serve_forever()


if __name__ == "__main__":
    main()
