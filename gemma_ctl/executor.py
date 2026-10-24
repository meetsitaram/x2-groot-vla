"""Tool executor: Gemma tool calls -> pc2_kplanner_onnx.py ZMQ inputs.

Channels (pc2_kplanner_onnx defaults; simstack_local.sh uses the isolated
56xx set, see PORTS_SIMSTACK):
  planner_cmd     JSON multipart, kplanner SUB (bind with --cmd-bind) -> we PUB-connect
  motion_clip_cmd JSON multipart, kplanner SUB bind                   -> we PUB-connect
  pose            packed v4, kplanner PUB bind (reference root xy + quat) -> feedback
  x2_debug        packed, deploy PUB (measured IMU base_quat)          -> upright check

Velocity intents are LATCHED in the kplanner, so distance/angle tools are
closed here on the planner's own reference pose (root_xy_world /
root_quat_xyzw on the pose wire): same signal in sim and on the robot.

Safety: the LLM can only reach this whitelist; every motion has a timeout
that falls back to idle; dry_run prints the wire messages instead.
"""
from __future__ import annotations

import json
import os
import math
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path

from . import tools as T

WBC_REPO = Path(os.environ.get("GEMMA_WBC", "/home/stickbot/Projects/GR00T-WBC-gemma"))

PORTS_DEFAULT = dict(cmd=5563, clip=5568, pose=5556, debug=5557)
PORTS_SIMSTACK = dict(cmd=5663, clip=5668, pose=5656, debug=5659)

SOURCE = "llm"           # kplanner ownership: lowest priority (patches/pc2_kplanner_llm_source.patch)
HEARTBEAT_S = 0.3        # kplanner idles us after 1.0 s of silence (wifi loss)
CMD_KEEPALIVE_S = 0.5
TURN_LEAD_RAD = math.radians(4.0)    # stop early: planner coasts a few degrees
WALK_LEAD_M = 0.05
UPRIGHT_MIN_COS = math.cos(math.radians(35.0))


def _wrap(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi


@dataclass
class Pose:
    x: float = 0.0
    y: float = 0.0
    yaw: float = 0.0
    t: float = 0.0          # monotonic receive time
    upright_cos: float = 1.0
    debug_t: float = 0.0


class Feedback(threading.Thread):
    """SUBs the planner pose wire + deploy x2_debug; keeps the latest Pose."""

    def __init__(self, host: str, pose_port: int, debug_port: int):
        super().__init__(daemon=True, name="gemma-feedback")
        self.host, self.pose_port, self.debug_port = host, pose_port, debug_port
        self.pose = Pose()
        self.lock = threading.Lock()
        self._stop = threading.Event()

    def run(self):
        import zmq
        sys.path.insert(0, str(WBC_REPO))
        from gear_sonic.utils.teleop.zmq.zmq_packed_message_decoder import unpack_message
        ctx = zmq.Context.instance()
        socks = {}
        for name, port, topic in (("pose", self.pose_port, "pose"),
                                  ("x2_debug", self.debug_port, "x2_debug")):
            s = ctx.socket(zmq.SUB)
            s.setsockopt_string(zmq.SUBSCRIBE, topic)
            s.connect(f"tcp://{self.host}:{port}")
            socks[s] = topic
        poller = zmq.Poller()
        for s in socks:
            poller.register(s, zmq.POLLIN)
        while not self._stop.is_set():
            for s, _ in poller.poll(200):
                topic = socks[s]
                try:
                    msg = b"".join(s.recv_multipart())
                    f = unpack_message(msg, topic).fields
                except Exception:
                    continue
                now = time.monotonic()
                with self.lock:
                    if topic == "pose":
                        xy = f["root_xy_world"].reshape(-1)
                        qx, qy, qz, qw = f["root_quat_xyzw"].reshape(-1)[:4]
                        self.pose.x, self.pose.y = float(xy[0]), float(xy[1])
                        self.pose.yaw = math.atan2(2 * (qw * qz + qx * qy),
                                                   1 - 2 * (qy * qy + qz * qz))
                        self.pose.t = now
                    elif "base_quat" in f:
                        w, x, y, z = f["base_quat"].reshape(-1)[:4]
                        # body z-axis . world z  (wxyz IMU quat)
                        self.pose.upright_cos = float(1 - 2 * (x * x + y * y))
                        self.pose.debug_t = now

    def get(self) -> Pose:
        with self.lock:
            return Pose(**vars(self.pose))

    def stop(self):
        self._stop.set()


class Executor:
    def __init__(self, ports: dict | None = None, host: str = "127.0.0.1",
                 dry_run: bool = True, places: dict | None = None,
                 log=print):
        self.ports = ports or PORTS_DEFAULT
        self.host = host
        self.dry_run = dry_run
        self.places = places or {}
        self.log = log
        self.posture_active = False
        self.abort = threading.Event()
        self.fb = None
        self.sent: list[tuple[str, dict]] = []     # wire log (bench / tests)
        if not dry_run:
            import zmq
            ctx = zmq.Context.instance()
            self.cmd = ctx.socket(zmq.PUB)
            self.cmd.connect(f"tcp://{host}:{self.ports['cmd']}")
            self.clip = ctx.socket(zmq.PUB)
            self.clip.connect(f"tcp://{host}:{self.ports['clip']}")
            self.fb = Feedback(host, self.ports["pose"], self.ports["debug"])
            self.fb.start()
            self._wire_lock = threading.Lock()
            self._hb_stop = threading.Event()
            threading.Thread(target=self._heartbeat, daemon=True,
                             name="gemma-heartbeat").start()
            time.sleep(0.5)      # slow-joiner

    def _heartbeat(self):
        while not self._hb_stop.wait(HEARTBEAT_S):
            with self._wire_lock:
                self.cmd.send_multipart([b"planner_cmd", json.dumps(
                    {"intent": "llm_keepalive", "source": SOURCE}).encode()])

    # ---- wire ------------------------------------------------------------
    def _send(self, topic: str, payload: dict):
        if topic == "planner_cmd":
            payload = {**payload, "source": SOURCE}
        self.sent.append((topic, payload))
        if self.dry_run:
            self.log(f"  [dry] {topic} {json.dumps(payload)}")
            return
        sock = self.cmd if topic == "planner_cmd" else self.clip
        with self._wire_lock:
            sock.send_multipart([topic.encode(), json.dumps(payload).encode()])

    def _idle(self):
        self._send("planner_cmd", {"intent": "idle", "magnitude": "default"})

    def _pose(self) -> Pose | None:
        if self.fb is None:
            return None
        p = self.fb.get()
        return p if time.monotonic() - p.t < 1.0 else None

    def _check_upright(self):
        if self.fb is None:
            return
        p = self.fb.get()
        if p.debug_t and p.upright_cos < UPRIGHT_MIN_COS:
            self._idle()
            raise RuntimeError(f"robot not upright (tilt "
                               f"{math.degrees(math.acos(max(-1, min(1, p.upright_cos)))):.0f} deg)"
                               " - halted")

    def _hold_until(self, payload: dict, done, timeout_s: float) -> bool:
        """Re-send a latched intent until done(pose) or timeout; then idle."""
        t0 = time.monotonic()
        last = 0.0
        ok = False
        try:
            while time.monotonic() - t0 < timeout_s:
                if self.abort.is_set():
                    break
                if time.monotonic() - last > CMD_KEEPALIVE_S:
                    self._send("planner_cmd", payload)
                    last = time.monotonic()
                self._check_upright()
                p = self._pose()
                if p is not None and done(p):
                    ok = True
                    break
                time.sleep(0.02)
        finally:
            self._idle()
        return ok

    def _settle(self, s: float = 1.0):
        if not self.dry_run:
            time.sleep(s)

    # ---- tools -----------------------------------------------------------
    def call(self, name: str, args: dict) -> dict:
        fn = getattr(self, f"t_{name}", None)
        if fn is None:
            return {"ok": False, "error": f"unknown tool {name}"}
        try:
            return fn(**(args or {}))
        except TypeError as e:
            return {"ok": False, "error": f"bad arguments: {e}"}
        except (ValueError, RuntimeError) as e:
            return {"ok": False, "error": str(e)}

    def t_stop(self):
        self.abort.set()
        self._send("motion_clip_cmd", {"action": "stop"})
        if self.posture_active:
            self._send("planner_cmd", {"intent": "primitive", "name": "exit"})
            self.posture_active = False
        self._idle()
        self.abort.clear()
        return {"ok": True}

    def t_walk(self, direction: str, meters: float):
        meters = float(meters)
        if direction not in ("forward", "backward"):
            raise ValueError("direction must be forward|backward")
        if not 0.05 <= meters <= T.MAX_WALK_M:
            raise ValueError(f"meters must be in [0.05, {T.MAX_WALK_M}]; for longer distances call walk several times (e.g. 3 + 2)")
        intent = {"intent": "walk", "magnitude": direction}
        return self._translate(intent, meters, speed=0.3 if direction == "forward" else 0.25)

    def t_sidestep(self, direction: str, meters: float):
        meters = float(meters)
        if direction not in ("left", "right"):
            raise ValueError("direction must be left|right")
        if not 0.05 <= meters <= 1.0:
            raise ValueError("meters must be in [0.05, 1.0]")
        return self._translate({"intent": f"side_{direction}", "magnitude": "default"},
                               meters, speed=0.2)

    def _translate(self, intent: dict, meters: float, speed: float):
        timeout = 4.0 + 3.0 * meters / speed
        if self.dry_run:
            self._send("planner_cmd", intent)
            self._idle()
            return {"ok": True, "moved_m": meters, "dry_run": True}
        p0 = self._pose()
        if p0 is None:
            raise RuntimeError("no pose feedback from kplanner (pose wire silent)")
        target = max(0.0, meters - WALK_LEAD_M)
        ok = self._hold_until(
            intent, lambda p: math.hypot(p.x - p0.x, p.y - p0.y) >= target, timeout)
        self._settle()
        p1 = self._pose() or p0
        moved = math.hypot(p1.x - p0.x, p1.y - p0.y)
        return {"ok": ok, "moved_m": round(moved, 2),
                **({} if ok else {"error": "timeout before reaching distance"})}

    def t_turn(self, direction: str, degrees: float):
        degrees = float(degrees)
        if direction not in ("left", "right"):
            raise ValueError("direction must be left|right")
        if not 5 <= degrees <= T.MAX_TURN_DEG:
            raise ValueError(f"degrees must be in [5, {T.MAX_TURN_DEG}]")
        intent = {"intent": f"turn_{direction}", "magnitude": "deg_45"}
        if self.dry_run:
            self._send("planner_cmd", intent)
            self._idle()
            return {"ok": True, "turned_deg": degrees, "dry_run": True}
        p0 = self._pose()
        if p0 is None:
            raise RuntimeError("no pose feedback from kplanner (pose wire silent)")
        sign = 1.0 if direction == "left" else -1.0
        target = max(0.0, math.radians(degrees) - TURN_LEAD_RAD)
        acc = {"yaw": 0.0, "prev": p0.yaw}

        def done(p):
            acc["yaw"] += _wrap(p.yaw - acc["prev"])
            acc["prev"] = p.yaw
            return sign * acc["yaw"] >= target
        ok = self._hold_until(intent, done, timeout_s=4.0 + degrees / 20.0)
        self._settle()
        p1 = self._pose() or p0
        acc["yaw"] += _wrap(p1.yaw - acc["prev"])
        return {"ok": ok, "turned_deg": round(math.degrees(sign * acc["yaw"]), 1),
                **({} if ok else {"error": "timeout before reaching angle"})}

    def t_gesture(self, name: str):
        key = T.GESTURES.get(name)
        if key is None:
            raise ValueError(f"unknown gesture {name}")
        self._send("motion_clip_cmd", {"action": "play", "pkl": "dances_x2m2",
                                       "kind": "locomotion", "motion_key": key})
        if not self.dry_run:
            # Clip length unknown here; wait while the reference moves, then
            # until it has been still ~1.5 s (post-dance idle anchor).
            t0 = time.monotonic()
            still_since = None
            last = self._pose()
            time.sleep(1.0)
            while time.monotonic() - t0 < 60:
                if self.abort.is_set():
                    break
                self._check_upright()
                time.sleep(0.25)
                p = self._pose()
                if p is None or last is None:
                    last = p
                    continue
                moving = (abs(_wrap(p.yaw - last.yaw)) > 1e-3
                          or math.hypot(p.x - last.x, p.y - last.y) > 1e-3)
                last = p
                if moving:
                    still_since = None
                elif still_since is None:
                    still_since = time.monotonic()
                elif time.monotonic() - still_since > 1.5 and time.monotonic() - t0 > 3:
                    break
        return {"ok": True, "gesture": name}

    def t_posture(self, name: str):
        if name == "stand":
            self._send("planner_cmd", {"intent": "primitive", "name": "exit"})
            self.posture_active = False
            self._settle(2.5)
            return {"ok": True, "posture": "stand"}
        prim = T.POSTURES.get(name)
        if prim is None:
            raise ValueError(f"unknown posture {name}")
        self._send("planner_cmd", {"intent": "primitive", "name": prim})
        self.posture_active = True
        self._settle(3.0)
        return {"ok": True, "posture": name, "note": "holding; call posture stand to release"}

    def t_look(self, direction: str):
        yaw = {"left": 0.6, "right": -0.6, "center": 0.0}.get(direction)
        if yaw is None:
            raise ValueError("direction must be left|right|center")
        if direction == "center":
            self._send("planner_cmd", {"intent": "head_targets", "release": True})
        else:
            self._send("planner_cmd", {"intent": "head_targets", "yaw_rad": yaw})
        self._settle(1.0)
        return {"ok": True, "looking": direction}

    def t_get_state(self):
        if self.dry_run:
            return {"ok": True, "x": 0.0, "y": 0.0, "heading_deg": 0.0,
                    "upright": True, "dry_run": True}
        p = self._pose()
        if p is None:
            return {"ok": False, "error": "no pose feedback"}
        return {"ok": True, "x": round(p.x, 2), "y": round(p.y, 2),
                "heading_deg": round(math.degrees(p.yaw), 1),
                "upright": p.upright_cos >= UPRIGHT_MIN_COS,
                "posture_held": self.posture_active}

    def t_go_to(self, place: str):
        if place not in self.places:
            raise ValueError(f"unknown place {place}")
        gx, gy = self.places[place]["xy"]
        if self.dry_run:
            self._send("planner_cmd", {"intent": "walk", "magnitude": "forward"})
            self._idle()
            return {"ok": True, "place": place, "dry_run": True}
        # Open-ground straight-line go-to in the planner's odometry frame
        # (G1 / MuJoCo). The kitchen version (G2) swaps in nav_policy_bridge.
        p = self._pose()
        if p is None:
            raise RuntimeError("no pose feedback")
        for _ in range(3):
            dist = math.hypot(gx - p.x, gy - p.y)
            if dist < 0.25:
                break
            bearing = _wrap(math.atan2(gy - p.y, gx - p.x) - p.yaw)
            if abs(bearing) > math.radians(10):
                self.t_turn("left" if bearing > 0 else "right",
                            min(T.MAX_TURN_DEG, math.degrees(abs(bearing))))
            self.t_walk("forward", min(T.MAX_WALK_M, dist))
            p = self._pose() or p
        dist = math.hypot(gx - p.x, gy - p.y)
        return {"ok": dist < 0.4, "place": place, "remaining_m": round(dist, 2)}

    def t_pick(self, object: str, height: str):
        return {"ok": False, "error": "pick is not wired yet (G1 pick clips "
                "pending bake); tell the user it is not available"}

    def close(self):
        if self.fb:
            self._hb_stop.set()
            self.fb.stop()
