"""Gemma -> real X2, limited first milestone: LOOK + GESTURES, no locomotion.

Tools Gemma gets:
  look()           one frame from the robot's wide head camera (stereo_head_front_left,
                   ~170 deg fisheye) -> undistorted -> Gemma lists every object it sees,
                   with a rough direction (left / ahead / right).
  gesture(name)    one stationary gesture clip (wave, high five, blow kiss, handshake,
                   heart) on the kplanner's motion_clip_cmd -- the robot stays planted.
  stop()           stop the clip: kplanner blends back to its idle stand. gesture()
                   returns at once, so "stop" works mid-gesture; exit/Ctrl-C also stops.

Targets (same rule as run_motion_replay.sh: no host = sim, an ip = the robot):
  sim    local sim stack (sim_onnx_planner.sh), clip port 5668. MuJoCo has no head
         camera, so look() uses --image (default: a real garage capture).
  robot  --pc2-host <ip>, clip port 5568; look() reads the PC2 camera bridge
         (x2_pc2_cameras.sh serve, ZMQ :5555, key stereo_left).
Default is DRY RUN (commands printed, nothing sent); --live publishes.

  python -m gemma_ctl.agent --real --say "what do you see?" --say "can you wave?"
"""
from __future__ import annotations

import json
import math
import os
import struct
import sys
import time
from pathlib import Path

import numpy as np

from .perceive import gemma_boxes

ROOT = Path(__file__).resolve().parents[1]
WBC = Path(os.environ.get("GEMMA_WBC", "/home/stickbot/Projects/GR00T-WBC-gemma"))
CALIB = ROOT / "assets/garage/calib_2026-08-04/calibration.json"
DEFAULT_IMAGE = ROOT / "assets/garage/calib_2026-08-04/captures_circle/stereo_head_front_left.jpg"
from .motion_library import CLIPS_DIR as DANCES_DIR      # the stack's KPLANNER_DANCES_DIR
CLIP_PORT = {"sim": int(os.environ.get("GEMMA_CLIP_PORT", 5568)), "robot": 5568}   # sim_onnx_planner stack = 5568
OUT_HFOV = 110.0           # undistorted pinhole view of the fisheye (deg)

# stationary gestures only (no travel, no turning) -> x2m2 motion keys in the dances bank
GESTURES = {
    "wave": "right_wave_001",
    "high_five": "right_five_001",
    "blow_kiss": "right_kiss_001",
    "handshake": "right_shake_001",
    "heart": "both_heart_002",
}

# SIM TARGET ONLY: dances from the same bank (some travel or turn, so never offered to the
# robot target). One friendly name per dance; Gemma picks from this enum.
DANCES = {
    "macarena": "victory_dance_macarena_R_001__A306",
    "gangnam_style": "victory_dance_gangam_style_rodeo_R_001__A324",
    "beyonce_moves": "victory_dance_beyonce_moves_front_R_003__A322",
    "brick_slide": "victory_dance_brick_brick_slide_slide_R_001__A323",
    "victory_dance": "victory_dance_asarahe_180_R_003__A324",
    "disco_fever": "dance_disco_fever_001__A465",
    "raise_the_roof": "dance_raise_the_roof_001__A464",
    "party_hips": "dance_party_hips_003__A465",
    "hip_hop": "dance_hiphop_stick_n_roll_dancehall_R_loop_003__A324",
    "freedom_wheels": "dance_freedom_wheels_001__A464",
    "distraction_dance": "dance_distraction_dance_001__A466",
    "vogue": "dance_vouge_butterfly_step_180_R_fast_002__A319",
    "egyptian": "egipt_dance_R_001__A438",
    "shadow_boxing": "shadow_boxing_R_001__A359",
}

LOOK_PROMPT = """You are the eyes of a humanoid robot. This is its head camera view.
List every distinct object you can see (furniture, boxes, containers, tools, devices, bags,
bottles, doors, windows, shelves, ...). "label" says WHAT it is in 1-4 words, with colour
when useful, e.g. "white cardboard box", "grey storage case", "white door".
Do not include the walls, floor or ceiling, and do not include the robot itself.
Return ONLY JSON: a list of {"label": str, "box_2d": [ymin, xmin, ymax, xmax]}
with coordinates normalized to 0-1000."""

REAL_SYSTEM = """You control a real humanoid robot (AgiBot X2) by calling tools.
It stands in place: it cannot walk in this mode. It can look through its head camera and
perform gestures.
Rules:
- When asked what the robot sees, call look and describe the main things briefly, saying
  roughly where they are (left, ahead, right). Do not invent objects look did not return.
- When asked for a gesture, call gesture with the closest match. If the request is not one
  of the gestures (for example walking or picking something up), say you can't do that yet.
- "Can you <specific move>?" ("can you wave?", "can you do a high jump?") is a polite REQUEST:
  do the move (call the tool). Only GENERAL questions about abilities ("what can you do?",
  "what gestures do you know?", "which dances do you support?", "can you dance?") are
  answered WITHOUT calling a tool. When asked which moves, name all of them (from the tool lists) as normal words
  without underscores (say "Beyonce moves", "blow a kiss", "high five", not beyonce_moves),
  then offer to do one. Only move on a clear request ("wave", "do the macarena", "show me a dance").
- Every request to move gets a NEW tool call, even if you did the same move a moment ago.
  Never say you are doing or already doing a move unless you called its tool in this turn;
  earlier motions have finished. If the user repeats or rephrases a request ("I mean the robot
  wave", "again", "do it again", "one more time"), call the tool again: they did not see it.
- If the user says stop, call stop immediately.
- Reply in one or two short sentences (a list of moves may be longer)."""


SIM_DANCE_RULE = """
- In this SIMULATION you can also dance: call dance with the closest match. If asked to
  dance with no style ("dance for me", "show me a dance"), pick one. For several moves in a
  row, call one, then the next.
- For walking, styled walks (injured, crutch, catwalk, march, carrying, walking the dog), jogging,
  running, crouching, squats, boxing and kicks, and more dances, use move with the closest name.
- Beyond those you have a library of about 9000 motions (jumps, jogging, running, squats, kicks,
  martial arts, crouching, catching, one-leg balance, stretches, ...). For any move that is not
  a gesture or dance above, call find_motions with a short query (1-3 words), then
  play_motion with the best match's name. Asked what moves you know, mention the library and
  give a few examples by category; do not read out long lists.
- Motions that travel (travels_m above 1) move the robot across the floor; that is fine here."""


def real_tool_schemas(target="sim"):
    from .tools import _fn
    tools = [
        _fn("look", "Look through the robot's head camera. Returns the objects it can see "
            "and roughly where they are."),
        _fn("gesture", "Perform a gesture in place.",
            {"name": {"type": "string", "enum": sorted(GESTURES)}}, ["name"]),
        _fn("stop", "Stop the current gesture or dance immediately."),
    ]
    if target == "sim":
        tools.insert(2, _fn("dance", "Perform a dance (simulation only).",
                            {"name": {"type": "string", "enum": sorted(DANCES)}}, ["name"]))
        from .motion_catalog import load as _cat
        cat = _cat()
        if cat:
            by = {}
            for n, m in cat.items():
                by.setdefault(m["category"], []).append(n)
            desc = "Perform a named move (simulation only). Moves by category: " + "; ".join(
                f"{c}: {', '.join(ns)}" for c, ns in by.items())
            tools.insert(3, _fn("move", desc, {"name": {"type": "string", "enum": sorted(cat)}}, ["name"]))
        tools.insert(3, _fn("find_motions", "Search the motion library (simulation only). "
                            "Returns up to 8 matching motions with name, description, seconds and "
                            "how far each travels.",
                            {"query": {"type": "string", "description": "1-3 words, e.g. 'high jump', 'squat', 'kick'"}},
                            ["query"]))
        tools.insert(4, _fn("play_motion", "Play a motion from find_motions (simulation only).",
                            {"name": {"type": "string", "description": "the exact name from find_motions"}},
                            ["name"]))
    return tools


def real_system(target="sim"):
    return REAL_SYSTEM + (SIM_DANCE_RULE if target == "sim" else "")


def _clip_seconds(key):
    """Duration of a bake from its x2m2 header (<III magic, n, ndof> then <d fps>, as wire.load_x2m2)."""
    p = DANCES_DIR / f"{key}.x2m2"
    try:
        raw = p.read_bytes()[:20]
        _, n, _ = struct.unpack("<III", raw[:12])
        fps = struct.unpack("<d", raw[12:20])[0]
        if fps > 0:
            return n / fps
    except (OSError, struct.error):
        pass
    return 8.0


class Fisheye:
    """Calibrated stereo_head_front_left fisheye -> pinhole of OUT_HFOV, at any frame size."""

    def __init__(self):
        c = json.loads(CALIB.read_text())["ci_stereo_head_front_left"]
        self.w0, self.h0 = c["w"], c["h"]
        self.K0 = np.array(c["k"], float).reshape(3, 3)
        self.D = np.array(c["d"], float).reshape(4, 1)

    def undistort(self, img):
        import cv2
        h, w = img.shape[:2]
        K = self.K0.copy(); K[0] *= w / self.w0; K[1] *= h / self.h0
        f = (w / 2) / math.tan(math.radians(OUT_HFOV) / 2)
        P = np.array([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1]])
        m1, m2 = cv2.fisheye.initUndistortRectifyMap(K, self.D, np.eye(3), P, (w, h), cv2.CV_16SC2)
        return cv2.remap(img, m1, m2, cv2.INTER_LINEAR)


class RealExecutor:
    def __init__(self, target="sim", pc2_host=None, live=False, image=None,
                 model="gemma4:26b", run_dir=None):
        if target == "robot" and not pc2_host:
            raise ValueError("robot target needs an explicit --pc2-host (no robot default)")
        self.target, self.host, self.live, self.model = target, pc2_host, live, model
        self.image = Path(image) if image else DEFAULT_IMAGE
        self.run_dir = Path(run_dir or ROOT / f"runs/gemma/real/{time.strftime('%Y%m%dT%H%M%S')}")
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.fish = Fisheye()
        self.n_look = 0
        self.playing_until = 0.0
        self.last_motion = None
        self._lock = __import__("threading").Lock()
        self._clip = self._cam = None

    # ---- transport -------------------------------------------------------
    def _send_clip(self, payload):
        with self._lock:          # web UI: stop arrives on another thread than the chat
            self._send_clip_locked(payload)

    def _send_clip_locked(self, payload):
        line = json.dumps(payload)
        if not self.live:
            print(f"  [dry-run] motion_clip_cmd -> {self.target}: {line}", flush=True)
            return
        import zmq
        if self._clip is None:
            self._clip = zmq.Context.instance().socket(zmq.PUB)
            host = "127.0.0.1" if self.target == "sim" else self.host
            self._clip.connect(f"tcp://{host}:{CLIP_PORT[self.target]}")
            time.sleep(0.5)        # PUB slow-joiner: let the kplanner SUB attach
        self._clip.send_multipart([b"motion_clip_cmd", line.encode()])

    def _frame(self):
        import cv2
        if self.target == "robot" and self.live:
            if self._cam is None:
                sys.path.insert(0, str(WBC))
                from gear_sonic.camera.composed_camera import ComposedCameraClientSensor
                self._cam = ComposedCameraClientSensor(server_ip=self.host, port=5555)
            for _ in range(50):
                d = self._cam.read()
                if d and "stereo_left" in (d.get("images") or {}):
                    return cv2.cvtColor(np.asarray(d["images"]["stereo_left"]), cv2.COLOR_RGB2BGR), "stereo_left (live)"
                time.sleep(0.1)
            raise RuntimeError("no stereo_left frame from the PC2 camera bridge in 5 s")
        return cv2.imread(str(self.image)), f"{self.image.name} (saved capture)"

    # ---- tools -----------------------------------------------------------
    def t_look(self):
        import cv2
        raw, src = self._frame()
        img = self.fish.undistort(raw)
        self.n_look += 1
        p = self.run_dir / f"look{self.n_look:02d}.jpg"
        cv2.imwrite(str(p), img, [cv2.IMWRITE_JPEG_QUALITY, 92])
        boxes, dt, _ = gemma_boxes(p, self.model, LOOK_PROMPT)
        seen = []
        for b in boxes:
            y0, x0, y1, x1 = b["box_2d"]
            u = ((x0 + x1) / 2000 - 0.5) * 2 * math.tan(math.radians(OUT_HFOV) / 2)
            ang = math.degrees(math.atan(u))
            where = "left" if ang < -20 else "right" if ang > 20 else "ahead"
            seen.append({"label": b.get("label", "?"), "where": where, "bearing_deg": round(ang)})
        self._draw(img, boxes, p.with_name(p.stem + "_boxes.jpg"))
        (self.run_dir / f"look{self.n_look:02d}.json").write_text(json.dumps(
            {"source": src, "boxes": boxes, "seen": seen, "gemma_s": round(dt, 2)}, indent=1))
        return {"ok": True, "camera": "wide head camera", "objects": seen}

    def t_gesture(self, name):
        if name not in GESTURES:
            return {"ok": False, "error": f"unknown gesture {name!r}; have {sorted(GESTURES)}"}
        key = GESTURES[name]
        dur = _clip_seconds(key)
        self._send_clip({"action": "play", "motion_key": key, "kind": "locomotion"})
        # returns right away (does NOT wait for the clip), so a "stop" typed while it
        # plays reaches Gemma -> t_stop at once; close() also stops a clip still playing
        self.playing_until = time.monotonic() + dur + 1.0
        self.last_motion = key
        return {"ok": True, "gesture": name, "status": "started", "seconds": round(dur, 1),
                "sent": self.live, "note": "" if self.live else "dry run: not sent"}

    def t_dance(self, name):
        if self.target != "sim":
            return {"ok": False, "error": "dances are simulation-only; on the robot use gesture"}
        if name not in DANCES:
            return {"ok": False, "error": f"unknown dance {name!r}; have {sorted(DANCES)}"}
        if time.monotonic() < self.playing_until:     # queue behind the clip still playing
            time.sleep(self.playing_until - time.monotonic())
        key = DANCES[name]
        dur = _clip_seconds(key)
        self._send_clip({"action": "play", "motion_key": key, "kind": "locomotion"})
        self.playing_until = time.monotonic() + dur + 1.0
        self.last_motion = key
        return {"ok": True, "dance": name, "status": "started", "seconds": round(dur, 1),
                "sent": self.live, "note": "" if self.live else "dry run: not sent"}

    def t_move(self, name):
        from .motion_catalog import load as _cat
        m = _cat().get(name)
        if self.target != "sim" or m is None:
            return {"ok": False, "error": f"unknown move {name!r}" if m is None else "moves are simulation-only"}
        r = self.t_play_motion(m["key"])
        if r.get("ok"):
            r["motion"] = name
        return r

    def t_find_motions(self, query):
        if self.target != "sim":
            return {"ok": False, "error": "the motion library is simulation-only"}
        from .motion_library import search
        hits = search(query)
        return {"ok": True, "query": query, "motions": hits} if hits else \
            {"ok": False, "error": f"no motion matches {query!r}; try other words"}

    def t_play_motion(self, name):
        if self.target != "sim":
            return {"ok": False, "error": "the motion library is simulation-only"}
        from .motion_library import ensure_baked, index
        if name not in index():
            return {"ok": False, "error": f"unknown motion {name!r}; use a name from find_motions"}
        ensure_baked(name)
        if time.monotonic() < self.playing_until:
            time.sleep(self.playing_until - time.monotonic())
        dur = _clip_seconds(name)
        self._send_clip({"action": "play", "motion_key": name, "kind": "locomotion"})
        self.playing_until = time.monotonic() + dur + 1.0
        self.last_motion = name
        return {"ok": True, "motion": index()[name]["name"], "status": "started",
                "seconds": round(dur, 1), "sent": self.live}

    def t_stop(self):
        """Clip stop: the kplanner drops the gesture, blends back to its idle stand and
        holds. Legs and base are never commanded in this mode."""
        self._send_clip({"action": "stop"})
        self.playing_until = 0.0
        return {"ok": True, "stopped": True}

    def call(self, name, args):
        fn = getattr(self, f"t_{name}", None)
        if fn is None:
            return {"ok": False, "error": f"no tool {name}"}
        try:
            return fn(**args)
        except Exception as e:      # report, never crash the chat loop
            return {"ok": False, "error": f"{type(e).__name__}: {e}"}

    def wait_idle(self):
        """Block until the gesture/dance that is playing has finished (normal exit)."""
        while self.live and time.monotonic() < self.playing_until:
            time.sleep(0.2)

    def close(self):
        """On exit (also Ctrl-C): stop a gesture that may still be playing."""
        if self.live and time.monotonic() < self.playing_until:
            self._send_clip({"action": "stop"})
        if self._clip is not None:
            self._clip.close(linger=500)

    @staticmethod
    def summarize(result):
        if "objects" in result:
            return f"{len(result['objects'])} objects: " + ", ".join(o["label"] for o in result["objects"][:8])
        return json.dumps(result)[:150]

    @staticmethod
    def _draw(img, boxes, path):
        import cv2
        out = img.copy(); h, w = out.shape[:2]
        for b in boxes:
            y0, x0, y1, x1 = b["box_2d"]
            p0, p1 = (int(x0 * w / 1000), int(y0 * h / 1000)), (int(x1 * w / 1000), int(y1 * h / 1000))
            cv2.rectangle(out, p0, p1, (40, 200, 255), 2)
            cv2.putText(out, b.get("label", "?"), (p0[0] + 3, max(16, p0[1] - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (40, 200, 255), 2, cv2.LINE_AA)
        cv2.imwrite(str(path), out, [cv2.IMWRITE_JPEG_QUALITY, 90])
