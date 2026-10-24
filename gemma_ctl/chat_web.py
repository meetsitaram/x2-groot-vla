"""Small web chat with Gemma driving the X2 sim stack (gestures, dances, the motion library).

  GEMMA_CLIP_PORT=5668 python -m gemma_ctl.chat_web            # supermimic simstack_local.sh
  python -m gemma_ctl.chat_web --open                          # + compact browser window

Serves http://127.0.0.1:8790 (local only). The page is meant to sit small at the bottom of the
screen under the MuJoCo viewer. Same agent and tools as `agent --real --live`.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from .agent import Agent, OLLAMA
from .real_robot import RealExecutor, real_system, real_tool_schemas

PAGE = Path(__file__).with_name("chat_web.html")


class Session:
    def __init__(self, model, live, pc2_host=None):
        self.ex = RealExecutor("robot" if pc2_host else "sim", pc2_host, live=live, model=model)
        self.model, self.live = model, live
        self.chat_lock = threading.Lock()
        self.reset()

    def reset(self):
        self.agent = Agent(self.model, self.ex, {}, OLLAMA, log=lambda *a: None,
                           tools=real_tool_schemas(self.ex.target), system=real_system(self.ex.target))

    def chat(self, text):
        with self.chat_lock:
            n0 = len(self.agent.trace)
            t0 = time.monotonic()
            reply = self.agent.run(text, source="web")
            acts = []
            for t in self.agent.trace[n0:]:
                if "tool" not in t:
                    continue
                r = t["result"] or {}
                if t["tool"] == "find_motions":
                    continue          # internal search step, not worth a chip
                label = (r.get("gesture") or r.get("dance") or r.get("motion") or t["tool"])
                acts.append({"tool": t["tool"], "label": str(label).replace("_", " "),
                             "seconds": r.get("seconds"), "ok": bool(r.get("ok")),
                             "error": r.get("error")})
            return {"reply": reply, "actions": acts, "secs": round(time.monotonic() - t0, 1)}

    def status(self):
        left = self.ex.playing_until - time.monotonic()
        name = self.ex.last_motion if left > 0 else None
        if name:
            from .motion_library import nice_name
            name = nice_name(name)
        sim = {}
        st = Path(__file__).resolve().parents[1] / "runs/gemma/real/sim_state.json"
        if st.exists():
            try:
                sim = json.loads(st.read_text())
            except ValueError:
                pass
        return {"playing": name, "left_s": round(max(0.0, left - 1.0), 1),
                "target": self.ex.target, "live": self.live, "page": PAGE.stat().st_mtime,
                "sim": sim.get("status"), "model": sim.get("model"), "resets": sim.get("resets", 0)}


def make_handler(sess):
    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _json(self, obj, code=200):
            b = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(b)))
            self.end_headers()
            self.wfile.write(b)

        def do_GET(self):
            if self.path in ("/", "/index.html"):
                b = PAGE.read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(b)))
                self.end_headers()
                self.wfile.write(b)
            elif self.path == "/api/status":
                self._json(sess.status())
            else:
                self._json({"error": "not found"}, 404)

        def do_POST(self):
            n = int(self.headers.get("Content-Length") or 0)
            body = json.loads(self.rfile.read(n) or b"{}")
            if self.path == "/api/chat":
                text = str(body.get("text", "")).strip()[:500]
                if not text:
                    return self._json({"error": "empty"}, 400)
                try:
                    self._json(sess.chat(text))
                except Exception as e:  # keep the page alive
                    self._json({"reply": f"(error: {type(e).__name__}: {e})", "actions": []})
            elif self.path == "/api/stop":
                self._json(sess.ex.call("stop", {}))
            elif self.path == "/api/reset":
                sess.reset(); self._json({"ok": True})
            else:
                self._json({"error": "not found"}, 404)
    return H


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8790)
    ap.add_argument("--model", default="gemma4:26b")
    ap.add_argument("--dry-run", action="store_true", help="print clip commands instead of sending")
    ap.add_argument("--pc2-host", default=None, help="real robot (gestures only); omit = sim stack")
    ap.add_argument("--open", action="store_true", help="open a compact browser window bottom-right")
    a = ap.parse_args()
    sess = Session(a.model, live=not a.dry_run, pc2_host=a.pc2_host)
    srv = ThreadingHTTPServer(("127.0.0.1", a.port), make_handler(sess))
    url = f"http://127.0.0.1:{a.port}/"
    print(f"[chat_web] {url}  target={sess.ex.target} live={sess.live} model={a.model}", flush=True)
    if a.open:
        subprocess.Popen(["google-chrome", f"--app={url}", "--window-size=520,340",
                          "--window-position=3300,1780"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        sess.ex.call("stop", {}); sess.ex.close()


if __name__ == "__main__":
    main()
