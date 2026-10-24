"""Kitchen-rig executor: Gemma tools -> IsaacLab kitchen rig
(run_x2_kplanner_env2.py with KP_PAD=1 KP_GEMMA=1).

  locomotion  planner_cmd sticks on :5563 (the rig honours sticks only;
              side>0 = RIGHT, yaw>0 = TURN RIGHT), closed on the MEASURED
              robot pose (robot_pose :6570, kitchen frame).
  go_to       nav_policy_bridge.py --goal <waypoint> (trained nav policy),
              then a short closed-loop approach to the catalog stand pose.
  pick/place  rig_hooks ops on :6581 (reach IK, kinematic attach/detach),
              arm/object/palm state from :6582.

The rig runs at ~0.27x real time, so every timeout here is in sim seconds
(from robot_pose t_sim), never wall seconds.
"""
from __future__ import annotations

import json
import os
import math
import re
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
WBC = Path(os.environ.get("GEMMA_WBC", "/home/stickbot/Projects/GR00T-WBC-gemma"))
NAV_PY = str(Path.home() / "miniconda3/envs/env_isaaclab/bin/python")
NAV_BRIDGE = WBC / "gear_sonic/scripts/x2-navigation/nav_policy_bridge.py"
WAYPOINTS = Path("/home/stickbot/projects/x2-kitchen-sim/configs/waypoints.json")

CMD_PORT = int(os.environ.get("KP_PAD_PORT", "7563"))   # rig pad-in port (launcher default 7563)
POSE_PORT, RIG_PORT, STATE_PORT = 6570, 6581, 6582
SHOULDER_Y = 0.18          # palm target lateral offset for the right arm
HOVER = 0.12               # m above grasp / place point
ARM_TOL = 0.06             # palm-to-target error accepted as "reached"


def _wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


class KitchenExecutor:
    def __init__(self, log=print, dry_run=False, drive=True):
        import zmq
        self.log = log
        self.dry_run = dry_run
        self.catalog = json.loads((HERE / "kitchen_catalog.json").read_text())
        self.waypoints = json.loads(WAYPOINTS.read_text())
        self.pose = None            # (x, y, yaw, t_sim)
        self.state = None           # gemma_state dict
        self.abort = threading.Event()
        self.nav_proc = None
        self.lock = threading.Lock()
        if dry_run:
            return
        sys.path.insert(0, str(WBC))
        from gear_sonic.utils.teleop.zmq.robot_pose_zmq import unpack_robot_pose
        self._unpack = unpack_robot_pose
        ctx = zmq.Context.instance()
        self.cmd = ctx.socket(zmq.PUB); self.cmd.connect(f"tcp://127.0.0.1:{CMD_PORT}")
        self.rig = ctx.socket(zmq.PUB); self.rig.connect(f"tcp://127.0.0.1:{RIG_PORT}")
        self._pose_sub = ctx.socket(zmq.SUB); self._pose_sub.setsockopt(zmq.SUBSCRIBE, b"")
        self._pose_sub.connect(f"tcp://127.0.0.1:{POSE_PORT}")
        self._st_sub = ctx.socket(zmq.SUB); self._st_sub.setsockopt(zmq.SUBSCRIBE, b"gemma_state")
        self._st_sub.connect(f"tcp://127.0.0.1:{STATE_PORT}")
        self.sticks = (0.0, 0.0, 0.0)
        threading.Thread(target=self._rx, daemon=True).start()
        # drive=False: observer only (surveys, probes). Exactly ONE driving
        # executor may exist: a second one's zero-stick keepalives fight the
        # first (seen 2026-10-23: turns/walks stalled, sticks 0.7/0.0 flapping)
        if drive:
            threading.Thread(target=self._stick_tx, daemon=True).start()
        time.sleep(0.5)

    @property
    def held(self):
        st = self.state
        return next(iter(st["attached"]), None) if st else None

    @held.setter
    def held(self, _):
        pass          # source of truth = the rig's attached set

    def _ev(self, kind, text):
        """Timestamped execution event for the video HUD (no-op without a run log)."""
        f = getattr(self, "log_event", None)
        if f:
            f(kind, text)

    # ---- wire ------------------------------------------------------------
    def _rx(self):
        import zmq
        poller = zmq.Poller()
        poller.register(self._pose_sub, zmq.POLLIN)
        poller.register(self._st_sub, zmq.POLLIN)
        while True:
            for s, _ in poller.poll(100):
                try:
                    raw = s.recv_multipart()
                    if s is self._pose_sub:
                        d = self._unpack(b"".join(raw) if len(raw) > 1 else raw[0])
                        x, y, _, w, qx, qy, qz = d["pelvis_qpos_wxyz"]
                        yaw = math.atan2(2 * (w * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz))
                        with self.lock:
                            self.pose = (float(x), float(y), yaw, float(d.get("sim_time", 0.0)))
                    else:
                        with self.lock:
                            self.state = json.loads(raw[-1].decode())
                except Exception:
                    continue

    def _stick_tx(self):
        """Sticks are re-sent every 0.3 s wall (rig zeroes sticks >1 s stale),
        except while the nav bridge owns the pad port."""
        while True:
            if self.nav_proc is None or self.nav_proc.poll() is not None:
                f, s, y = self.sticks
                self.cmd.send_multipart([b"planner_cmd", json.dumps(
                    {"intent": "locomotion", "magnitude": "continuous",
                     "stick_fwd": f, "stick_side": s, "stick_yaw": y,
                     "source": "pad"}).encode()])
            time.sleep(0.3)

    def _rig(self, **msg):
        self.log(f"  [rig] {json.dumps(msg)}")
        if not self.dry_run:
            self.rig.send_multipart([b"gemma_rig", json.dumps(msg).encode()])

    def _p(self):
        with self.lock:
            return self.pose

    def _st(self):
        with self.lock:
            return self.state

    def _sim_t(self):
        st = self._st()
        return st["t_sim"] if st else 0.0

    def _wait_sim(self, seconds):
        t0 = self._sim_t()
        while self._sim_t() - t0 < seconds:
            if self.abort.is_set():
                return
            time.sleep(0.05)

    def _check_upright(self):
        st = self._st()
        if st and st["robot"]["up_cos"] < 0.8:
            self.sticks = (0.0, 0.0, 0.0)
            raise RuntimeError("robot fell or is tilting badly - halted")

    # ---- closed-loop primitives -----------------------------------------
    def _drive(self, sticks, done, timeout_sim):
        t0 = self._sim_t()
        try:
            self.sticks = sticks
            self._ev("kplanner", f"planner_cmd sticks fwd {sticks[0]:+.1f} · side {sticks[1]:+.1f} · yaw {sticks[2]:+.1f}")
            while self._sim_t() - t0 < timeout_sim:
                if self.abort.is_set():
                    return False
                self._check_upright()
                if done(self._p()):
                    return True
                time.sleep(0.03)
            return False
        finally:
            self.sticks = (0.0, 0.0, 0.0)
            p = self._p()
            if p:
                self._ev("kplanner", f"planner_cmd sticks 0 → stand · at ({p[0]:+.2f}, {p[1]:+.2f}) heading {math.degrees(p[2]):.0f}°")

    # The rig's planner keeps executing its committed chunk after the sticks
    # go to zero, so the robot COASTS by ~velocity x latency (measured: +29
    # deg after a 90 deg turn, ~+4 deg after a short one). Stop when
    # progress + v * LAT reaches the goal, then correct (up to `tries`).
    LAT_TURN = 0.40       # s sim
    LAT_WALK = 0.90       # s sim

    def _progress_drive(self, sticks, progress, goal, lat, timeout_sim):
        """progress(p) -> scalar travelled so far (monotone); stop predictively."""
        hist = []

        def done(p):
            v = progress(p)
            t = p[3]
            hist.append((t, v))
            while len(hist) > 2 and t - hist[0][0] > 0.3:
                hist.pop(0)
            rate = ((hist[-1][1] - hist[0][1]) / (hist[-1][0] - hist[0][0])
                    if len(hist) > 1 and hist[-1][0] > hist[0][0] else 0.0)
            return v + max(0.0, rate) * lat >= goal
        return self._drive(sticks, done, timeout_sim)

    def _turn_to(self, yaw_goal, tol=math.radians(10), tries=3):
        for _ in range(tries):
            err = _wrap(yaw_goal - self._p()[2])
            if abs(err) < tol:
                return True
            sign = 1.0 if err > 0 else -1.0     # +err = turn left = stick_yaw < 0
            acc = {"v": 0.0, "prev": self._p()[2]}

            def prog(p):
                acc["v"] += _wrap(p[2] - acc["prev"]); acc["prev"] = p[2]
                return sign * acc["v"]
            self._progress_drive((0.0, 0.0, -sign * 0.7), prog, abs(err),
                                 self.LAT_TURN, timeout_sim=4.0 + abs(err) * 4)
            self._wait_sim(1.5)
            self.log(f"  [turn] goal err {math.degrees(err):+.0f} -> now "
                     f"{math.degrees(_wrap(yaw_goal - self._p()[2])):+.0f} deg")
        return abs(_wrap(yaw_goal - self._p()[2])) < tol

    def _straight(self, fwd, meters):
        p0 = self._p()
        self._progress_drive((fwd, 0.0, 0.0),
                             lambda p: math.hypot(p[0] - p0[0], p[1] - p0[1]),
                             meters, self.LAT_WALK, timeout_sim=4.0 + meters / 0.2)
        self._wait_sim(1.5)
        p1 = self._p()
        return math.hypot(p1[0] - p0[0], p1[1] - p0[1])

    def _walk_to(self, xy, tol=0.10, tries=3):
        for _ in range(tries):
            p = self._p()
            dx, dy = xy[0] - p[0], xy[1] - p[1]
            dist = math.hypot(dx, dy)
            if dist < tol:
                return True
            head = math.atan2(dy, dx)
            fwd = 1.0
            if abs(_wrap(head - p[2])) > math.radians(100) and dist < 0.5:
                fwd, head = -1.0, _wrap(head + math.pi)   # short & behind: back up
            self._turn_to(head, tol=math.radians(15), tries=2)
            moved = self._straight(fwd, dist)
            self.log(f"  [walk] wanted {dist:.2f} moved {moved:.2f}")
        p = self._p()
        return math.hypot(xy[0] - p[0], xy[1] - p[1]) < tol

    def _robot_frame(self, xyz):
        x, y, yaw, _ = self._p()
        dx, dy = xyz[0] - x, xyz[1] - y
        c, s = math.cos(yaw), math.sin(yaw)
        return (c * dx + s * dy, -s * dx + c * dy, xyz[2])

    REF_LATENCY = 1.4     # s sim: ring rows are written ~1 s ahead of playback

    LEAN_BELOW_Z = 1.0     # lean the waist forward for low targets
    LEAN_PITCH = 0.25

    def _reach_once(self, cmd_xyz, side):
        if not hasattr(self, "_ik"):
            from .ik import PalmIK
            self._ik = PalmIK()
            self._lean = None
        lean = self.LEAN_PITCH if cmd_xyz[2] < self.LEAN_BELOW_Z else None
        if lean != self._lean:
            self._rig(op="waist", ypr=None if lean is None else [0.0, lean, 0.0])
            self._lean = lean
        qpos = np.asarray(self._st()["qpos"], float)
        q, ik_err = self._ik.solve(qpos, cmd_xyz, side, waist_pitch=lean)
        self._ev("ik", f"IK palm → ({cmd_xyz[0]:+.2f}, {cmd_xyz[1]:+.2f}, {cmd_xyz[2]:.2f}) · "
                       f"arm q [{', '.join(f'{v:+.2f}' for v in q[:4])}…]" + (f" · waist lean {lean:.2f}" if lean else ""))
        self._rig(op="reach", side=side, xyz=[round(float(v), 3) for v in cmd_xyz],
                  q=[round(float(v), 4) for v in q], ik_err=round(ik_err, 3))
        t0 = self._sim_t()
        time.sleep(0.3)
        while self._sim_t() - t0 < 8.0:
            st = self._st()
            if st and not st["arm"][side]["moving"] and self._sim_t() - t0 > 0.4:
                break
            time.sleep(0.05)
        self._wait_sim(self.REF_LATENCY)
        return (self._st()["arm"][side]["last_reach"] or {}).get("ik_err_m")

    def _palm(self, side, n=6):
        ps = []
        for _ in range(n):
            ps.append(self._st()["palm"][side]); self._wait_sim(0.1)
        return np.mean(np.array(ps), axis=0)

    def _reach(self, xyz, side="right", settle=None, tol=0.03, corrections=3):
        """IK reach, then closed-loop correction of SONIC's arm tracking
        error (measured 4-21 cm, growing with extension): command
        target + (target - palm) and re-solve."""
        xyz = np.asarray(xyz, float)
        cmd = xyz.copy()
        ik = self._reach_once(cmd, side)
        err_v = xyz - self._palm(side)
        best = (float(np.linalg.norm(err_v)), cmd.copy())
        for _ in range(corrections):
            if np.linalg.norm(err_v) < tol:
                break
            step = 0.8 * err_v
            shift = (cmd + step) - xyz
            if np.linalg.norm(shift) > 0.10:            # never chase far past the goal
                shift *= 0.10 / np.linalg.norm(shift)
            cmd = xyz + shift
            ik = self._reach_once(cmd, side)
            err_v = xyz - self._palm(side)
            e = float(np.linalg.norm(err_v))
            if e < best[0]:
                best = (e, cmd.copy())
            elif e > best[0] + 0.02:                    # diverging: go back to best
                ik = self._reach_once(best[1], side)
                err_v = xyz - self._palm(side)
                break
        err = float(np.linalg.norm(err_v))
        self.log(f"  [reach] target {np.round(xyz, 3).tolist()} palm err {err:.3f} (ik {ik})")
        self._ev("ik", f"SONIC tracked palm within {err * 100:.1f} cm of ({xyz[0]:+.2f}, {xyz[1]:+.2f}, {xyz[2]:.2f})")
        return err, ik

    # ---- tools -----------------------------------------------------------
    def caption(self, text):
        self.utterance = text
        self._rig(op="caption", text=f'"{text}"')

    def call(self, name, args):
        if not self.dry_run:
            u = getattr(self, "utterance", "")
            a = ", ".join(f"{k}={v}" for k, v in (args or {}).items())
            self._rig(op="caption", text=f'"{u}"   ->  {name}({a})')
        fn = getattr(self, f"t_{name}", None)
        if fn is None:
            return {"ok": False, "error": f"unknown tool {name}"}
        try:
            if self.dry_run:
                return {"ok": True, "dry_run": True, **(args or {})}
            return fn(**(args or {}))
        except TypeError as e:
            return {"ok": False, "error": f"bad arguments: {e}"}
        except (ValueError, RuntimeError, KeyError) as e:
            return {"ok": False, "error": str(e)}

    def t_stop(self):
        self.abort.set()
        if self.nav_proc and self.nav_proc.poll() is None:
            self.nav_proc.terminate()
        self.sticks = (0.0, 0.0, 0.0)
        time.sleep(0.3)
        self.abort.clear()
        return {"ok": True}

    def t_walk(self, direction, meters):
        meters = float(meters)
        if not 0.05 <= meters <= 2.0:
            raise ValueError("meters must be 0.05..2.0; split longer walks into several calls")
        moved = self._straight(1.0 if direction == "forward" else -1.0, meters)
        ok = abs(moved - meters) < max(0.15, 0.2 * meters)
        return {"ok": ok, "moved_m": round(moved, 2),
                **({} if ok else {"note": "blocked or drifted; call look/get_state"})}

    def t_turn(self, direction, degrees):
        degrees = float(degrees)
        if not 5 <= degrees <= 180:
            raise ValueError("degrees must be 5..180")
        y0 = self._p()[2]
        goal = y0 + math.radians(degrees) * (1 if direction == "left" else -1)
        ok = self._turn_to(goal)
        return {"ok": ok, "turned_deg": round(math.degrees(_wrap(self._p()[2] - y0)), 1)}

    def _route_to(self, xy, final_tol=0.12):
        """A* over the nav grid + closed-loop segment following (sim-time,
        measured pose). Replaces nav_policy_bridge, whose wall-clock stuck
        detector misfires when the rig runs at ~0.23x real time."""
        if not hasattr(self, "_planner"):
            from .nav_plan import NavPlanner
            self._planner = NavPlanner()
        p = self._p()
        path = self._planner.plan((p[0], p[1]), tuple(xy))
        if path is None:
            return False, "no path"
        self.log(f"  [route] {len(path) - 1} legs: "
                 + " -> ".join(f"({x:+.2f},{y:+.2f})" for x, y in path[1:]))
        self._ev("exec", f"route A* {len(path) - 1} legs → stand ({xy[0]:+.2f}, {xy[1]:+.2f})")
        for k, wp in enumerate(path[1:], 1):
            self._ev("exec", f"route leg {k}/{len(path) - 1} → ({wp[0]:+.2f}, {wp[1]:+.2f})")
            last = k == len(path) - 1
            self._walk_to(wp, tol=final_tol if last else 0.25, tries=3 if last else 2)
        p = self._p()
        d = math.hypot(xy[0] - p[0], xy[1] - p[1])
        return d < 0.3, f"arrived within {d:.2f} m"

    def _nav(self, waypoint):
        wx, wy = self.waypoints[waypoint]["xy"]
        ok, info = self._route_to((wx, wy), final_tol=0.2)
        if ok:
            self._turn_to(self.waypoints[waypoint]["yaw"], tol=math.radians(15), tries=2)
        return ok, info

    # right-arm comfortable envelope in the robot frame (pelvis): x fwd, y left
    ENV_X = (0.16, 0.34)
    ENV_Y = (-0.32, -0.05)

    def _in_envelope(self, xyz):
        fx, fy, _ = self._robot_frame(xyz)
        return self.ENV_X[0] <= fx <= self.ENV_X[1] and self.ENV_Y[0] <= fy <= self.ENV_Y[1]

    def _face_for_right_arm(self, xyz):
        p = self._p()
        d = math.hypot(xyz[0] - p[0], xyz[1] - p[1])
        bearing = math.atan2(xyz[1] - p[1], xyz[0] - p[0])
        off = math.asin(min(0.9, SHOULDER_Y / max(d, 0.2)))
        self._turn_to(bearing + off, tol=math.radians(12), tries=2)

    def _go_stand(self, loc, side="right"):
        c = self.catalog[loc]
        target = c["obj_xyz"]
        ok, info = self._route_to(c["stand"]["xy"])
        if not ok:
            return False, f"could not get to {loc}: {info}"
        for _ in range(3):
            self._face_for_right_arm(target)
            if self._in_envelope(target):
                break
            fx, fy, _ = self._robot_frame(target)
            # fore/aft correction along the facing direction
            want = fx - 0.26
            if abs(want) > 0.06:
                self._straight(1.0 if want > 0 else -1.0, abs(want))
        fx, fy, _ = self._robot_frame(target)
        return True, f"at {loc}; target {fx:.2f} m ahead, {-fy:.2f} m right"

    def _loc_of(self, xyz):
        best = min(self.catalog, key=lambda k: math.hypot(
            self.catalog[k]["obj_xyz"][0] - xyz[0], self.catalog[k]["obj_xyz"][1] - xyz[1]))
        dist = math.hypot(self.catalog[best]["obj_xyz"][0] - xyz[0],
                          self.catalog[best]["obj_xyz"][1] - xyz[1])
        return best if dist < 0.5 else None

    def t_go_to(self, place):
        if place in self.catalog:
            ok, info = self._go_stand(place)
            return {"ok": ok, "info": info}
        if place in self.waypoints:
            ok, info = self._nav(place)
            return {"ok": ok, "info": info}
        raise ValueError(f"unknown place {place}")

    def t_look(self):
        st = self._st()
        p = self._p()
        near = min(self.waypoints, key=lambda k: math.hypot(
            self.waypoints[k]["xy"][0] - p[0], self.waypoints[k]["xy"][1] - p[1]))
        objs = []
        for n, o in st["objects"].items():
            where = "in your hand" if n in st["attached"] else (self._loc_of(o["xyz"]) or "somewhere")
            objs.append(f"{n} ({o['desc']}) - {where}")
        return {"ok": True, "you_are_near": near, "holding": self.held,
                "objects": objs}

    def t_pick(self, object):
        st = self._st()
        if object not in st["objects"]:
            raise ValueError(f"no object called {object}; call look first")
        if self.held:
            raise ValueError(f"already holding {self.held}; place it first")
        o = st["objects"][object]
        loc = self._loc_of(o["xyz"])
        if loc is None:
            raise ValueError(f"{object} is not at a known reachable location")
        ok, info = self._go_stand(loc)
        if not ok:
            return {"ok": False, "error": info}
        st = self._st()
        o = st["objects"][object]
        top = o["xyz"][2] + o["height"]
        grasp = [o["xyz"][0], o["xyz"][1], o["xyz"][2] + 0.6 * o["height"]]
        # palm stops 4 cm short of the object centre, on the robot side
        x, y, _, _ = self._p()
        v = np.array(grasp[:2]) - np.array([x, y]); v /= np.linalg.norm(v)
        grasp[0] -= 0.04 * v[0]; grasp[1] -= 0.04 * v[1]
        e1, ik1 = self._reach([grasp[0], grasp[1], top + HOVER])
        e2, ik2 = self._reach(grasp)
        if e2 > 0.10:
            self._reach([grasp[0], grasp[1], top + HOVER])
            self._rig(op="arm_release", side="right")
            return {"ok": False, "error": f"could not reach {object} (palm {e2:.2f} m off, ik {ik2})"}
        self._rig(op="attach", obj=object, side="right")
        self.held = object
        self._reach([grasp[0], grasp[1], top + HOVER + 0.05])
        # carry pose: hand close to the body
        cx = self._p()
        c, s = math.cos(cx[2]), math.sin(cx[2])
        carry = [cx[0] + 0.25 * c + SHOULDER_Y * s, cx[1] + 0.25 * s - SHOULDER_Y * c, 1.0]
        self._reach(carry, settle=0.5, tol=0.15)
        return {"ok": True, "picked": object, "from": loc, "palm_error_m": round(e2, 3)}

    def t_place(self, location):
        if not self.held:
            raise ValueError("not holding anything")
        if location not in self.catalog:
            raise ValueError(f"unknown location {location}")
        st = self._st()
        for n, o in st["objects"].items():
            if n != self.held and self._loc_of(o["xyz"]) == location:
                raise ValueError(f"{location} is occupied by {n}")
        obj = self.held
        ok, info = self._go_stand(location)
        if not ok:
            return {"ok": False, "error": info}
        c = self.catalog[location]
        h = self._st()["objects"][obj]["height"]
        px, py, pz = c["obj_xyz"]
        x, y, _, _ = self._p()
        v = np.array([px - x, py - y]); v /= np.linalg.norm(v)
        gx, gy = px - 0.04 * v[0], py - 0.04 * v[1]
        self._reach([gx, gy, pz + h + HOVER])
        e, ik = self._reach([gx, gy, pz + 0.6 * h + 0.01])
        self._rig(op="detach", obj=obj, xyz=[px, py, pz])
        self.held = None
        self._reach([gx, gy, pz + h + HOVER])
        self._rig(op="arm_release", side="right")
        self._rig(op="waist", ypr=None); self._lean = None
        self._wait_sim(2.0)
        return {"ok": True, "placed": obj, "at": location, "palm_error_m": round(e, 3)}

    def t_get_state(self):
        p = self._p()
        return {"ok": True, "x": round(p[0], 2), "y": round(p[1], 2),
                "heading_deg": round(math.degrees(p[2]), 1), "holding": self.held}

    def close(self):
        self.sticks = (0.0, 0.0, 0.0)
        time.sleep(0.4)


def kitchen_tool_schemas(catalog, waypoints, objects):
    from .tools import _fn
    places = sorted(set(waypoints) | set(catalog))
    return [
        _fn("stop", "Stop all motion immediately."),
        _fn("look", "Look around: lists the objects in the kitchen and where they are, "
            "what you are holding, and where you are."),
        _fn("go_to", "Walk to a named place in the kitchen.",
            {"place": {"type": "string", "enum": places}}, ["place"]),
        _fn("pick", "Walk to an object and pick it up with the right hand "
            "(only one object can be held at a time).",
            {"object": {"type": "string", "enum": sorted(objects)}}, ["object"]),
        _fn("place", "Walk to a location and put down the object you are holding.",
            {"location": {"type": "string", "enum": sorted(catalog)}}, ["location"]),
        _fn("walk", "Walk straight forward or backward (metres, max 2).",
            {"direction": {"type": "string", "enum": ["forward", "backward"]},
             "meters": {"type": "number"}}, ["direction", "meters"]),
        _fn("turn", "Turn in place.",
            {"direction": {"type": "string", "enum": ["left", "right"]},
             "degrees": {"type": "number"}}, ["direction", "degrees"]),
        _fn("get_state", "Report position, heading and what you are holding."),
    ]


KITCHEN_SYSTEM = """You control a humanoid robot (AgiBot X2) in a kitchen by calling tools.
Rules:
- Call look first if you do not know where objects are.
- To move an object: pick it (this walks there automatically), then place it at a
  location (this walks there automatically). You hold at most one object.
- Call tools one at a time and wait for each result. If a tool fails, explain why
  in one sentence instead of retrying forever (at most one retry).
- Only use places, objects and locations that exist in the tool enums.
- If the user says stop, call stop immediately.
- When done, reply with one short sentence."""
