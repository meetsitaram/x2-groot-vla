"""Perception-driven kitchen executor: Gemma's tools run on the world model
Gemma itself builds from the cameras (no sim object list, no hand catalog).

  look()            room-camera survey + the robot's wide head camera ->
                    Gemma boxes -> 3D -> world model; returns Gemma's own list
  pick(object)      object id/label from look(). If its position is not yet
                    reach-grade (sigma > 3 cm): walk to a look spot, refine
                    with the wide head cam + Orbbec, then step in and reach
  place(place)      a Gemma-labelled place (or waypoint): free reachable spot
                    on that surface at its detected height; Orbbec check after
  go_to(place)      Gemma-labelled place or named waypoint

The ONLY sim shortcut left is the grasp itself: once the palm is at the
perceived object, the kinematic attach takes the sim prop nearest the palm,
and only within ATTACH_RADIUS (else the pick fails: nothing was there).
"""
from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np

from .executor_kitchen import HOVER, SHOULDER_Y, KitchenExecutor
from .nav_plan import NavPlanner, stand_for
from .perceive import ROOM_VIEWS, ROOT, View, find, overlay, perceive, request_survey
from .world_model import REACH_TOL, WorldModel, sigma_for

ATTACH_RADIUS = 0.12


class PerceptionExecutor(KitchenExecutor):
    def __init__(self, run_dir: Path | None = None, **kw):
        super().__init__(**kw)
        self.wm = WorldModel()
        self.planner = NavPlanner()
        import os
        self.run_dir = Path(run_dir or os.environ.get("GEMMA_RUN_DIR")
                            or ROOT / "runs/gemma/live" / time.strftime("live_%Y%m%dT%H%M%S"))
        self.run_dir.mkdir(parents=True, exist_ok=True)
        # continue numbering across sessions sharing a bundle (no overwrites)
        self.n_survey = len([p for p in self.run_dir.glob("s[0-9]*") if p.is_dir()])
        self.held_oid = None
        self.held_prop = None

    # ---- conversation log (sim time) for the video HUD -------------------------
    def log_event(self, kind, text, source=None):
        rec = {"t": self._sim_t(), "kind": kind, "text": text}
        if source:
            rec["source"] = source            # "typed" (a person at the prompt) | "script"
        with open(self.run_dir / "conversation.jsonl", "a") as f:
            f.write(json.dumps(rec) + "\n")

    @staticmethod
    def summarize(result):
        if not isinstance(result, dict):
            return str(result)[:120]
        if not result.get("ok", True):
            return "FAILED: " + str(result.get("error", ""))[:200]
        if "objects" in result and "places" in result:          # look()
            out = f"{len(result['objects'])} objects · {len(result['places'])} places"
            return out + (f" · holding {result['holding']}" if result.get("holding") else "")
        parts = []
        for k, v in result.items():
            if k in ("ok", "steps", "dry_run"):
                continue
            if isinstance(v, list):
                parts.append(f"{len(v)} {k}")
            elif isinstance(v, float):
                parts.append(f"{k} {v * 100:.1f} cm" if k.endswith("_m") else f"{k} {v}")
            else:
                parts.append(f"{k}: {v}")
        for st in result.get("steps", []):
            parts.append(st)
        return " · ".join(parts)[:200]

    # ---- perception ----------------------------------------------------------
    def _survey(self, views, tag):
        self.n_survey += 1
        d = self.run_dir / f"s{self.n_survey:02d}_{tag}"
        request_survey(d, views=views, ex=self)
        return d

    def _ingest(self, d, source_of=lambda v: "room", label_only_near=None):
        world, views, raw = perceive(d, log=lambda m: self.log("  [percept]" + m))
        overlay(d, views, raw)
        for o in world["objects"]:
            v = views[o["views"][0]]
            src = source_of(v)
            rng = float(np.linalg.norm(np.array(o["base"]) - v.t))
            if label_only_near is not None:
                oid, gate = label_only_near
                t = self.wm.objects[oid]
                if np.linalg.norm(np.array(o["base"][:2]) - t["mean"][:2]) < gate:
                    t["labels"][o["label"]] += 1.0 / sigma_for(rng, src)[0]
                    continue
            self.wm.observe(o["label"], o["base"], o["height"], rng, src, size_xy=o.get("size_xy"))
        for p in world["places"]:
            self.wm.observe_place(p["label"], p["center"], p["extent"])
        return world

    def t_look(self):
        """Room cameras + the robot's own wide head camera."""
        self._ev("percept", "survey: 3 room cameras + wide head camera → Gemma boxes → depth → kitchen frame")
        d = self._survey(ROOM_VIEWS + [{"name": "head_wide", "camera": "head_wide"}], "look")
        self._ingest(d, source_of=lambda v: "head" if v.meta.get("camera") != "room" else "room")
        self._save_world("look", d)
        desc = self.wm.describe()
        p = self._p()
        here = self.wm.place_of([p[0], p[1], 0.9], radius=1.2)
        return {"ok": True, "you_are_near": (here or "the middle of the kitchen").replace("_", " "),
                "holding": None if not self.held_oid else self.wm.label(self.held_oid),
                "objects": desc["objects"], "places": desc["places"]}

    def _save_world(self, event="update", survey_dir=None):
        (self.run_dir / "world.json").write_text(json.dumps(self.wm.to_json(), indent=1, default=str))
        # timeline snapshot (sim time) for the video overlay
        snap = {"t": self._sim_t(), "event": event, "survey": str(survey_dir) if survey_dir else None,
                "held": self.held_oid,
                "objects": {k: {"label": self.wm.label(k), "mean": np.round(o["mean"], 3).tolist(),
                                "height": round(float(o["height"]), 3),
                                "size_xy": [round(float(v), 3) for v in o.get("size_xy", [0.08, 0.08])],
                                "sigma_xy": round(float(o["sigma_xy"]), 3)} for k, o in self.wm.objects.items()}}
        with open(self.run_dir / "world_timeline.jsonl", "a") as f:
            f.write(json.dumps(snap) + "\n")

    def _resolve_object(self, ref):
        if ref in self.wm.objects:
            return ref
        hits = self.wm.find(ref)
        if not hits:
            raise ValueError(f"no object matching '{ref}' in the world model; call look first. "
                             f"Known: {', '.join(f'{k} ({self.wm.label(k)})' for k in self.wm.objects)}")
        return hits[0]

    def _resolve_place(self, ref):
        if ref in self.wm.places:
            return ref
        key = ref.lower().replace(" ", "_")
        if key in self.wm.places:
            return key
        for pid, p in self.wm.places.items():
            if any(ref.lower() in l.lower() or l.lower() in ref.lower() for l in p["labels"]):
                return pid
        return None

    def _refine(self, oid):
        """Walk to a look spot, view with the wide head cam + Orbbec, fuse."""
        o = self.wm.objects[oid]
        p = self._p()
        st = stand_for(self.planner, o["mean"][:2], (p[0], p[1]), reach=(0.45, 0.90))
        if st:
            self._route_to(st["xy"])
        p = self._p()
        self._turn_to(math.atan2(o["mean"][1] - p[1], o["mean"][0] - p[0]), tol=math.radians(8), tries=2)
        d = self._survey([{"name": "head_wide", "camera": "head_wide"},
                          {"name": "orbbec", "camera": "orbbec"}], f"refine_{oid}")
        views = [View(d, 0), View(d, 1)]
        noun = self.wm.label(oid).lower().split()[-1]
        confirmed, in_view = False, False
        for vi, tiles in ((0, (2, 2)), (1, (1, 1))):
            in_view |= self._in_frame(views[vi], self.wm.objects[oid]["mean"])
            hits = find([views[vi]], noun, tiles=tiles, log=lambda *a: None)
            cur = self.wm.objects[oid]
            gate = max(0.30, 3 * cur["sigma_xy"])
            near = [h for h in hits if np.linalg.norm(np.array(h["base"][:2]) - cur["mean"][:2]) < gate]
            if not near:
                continue
            h = min(near, key=lambda h: np.linalg.norm(np.array(h["base"][:2]) - cur["mean"][:2]))
            rng = float(np.linalg.norm(np.array(h["base"]) - views[vi].t))
            new, _ = self.wm.observe(h["label"], h["base"], h["height"], rng, "head", size_xy=h.get("size_xy"))
            confirmed = True
            bx = [round(float(v)) for v in h["box_2d"]]
            self._ev("percept", f"{'wide head cam' if vi == 0 else 'Orbbec'} box {bx} → "
                                f"({h['base'][0]:+.2f}, {h['base'][1]:+.2f}, {h['base'][2]:.2f}) m · fused ±"
                                f"{self.wm.objects[new]['sigma_xy'] * 100:.1f} cm")
            if new != oid:      # fused elsewhere: keep tracking the target id
                oid = new
        o = self.wm.objects[oid]
        if confirmed:
            o["confirmed"] = True
        elif in_view:
            # NEGATIVE EVIDENCE: the head cameras looked right at it and Gemma
            # did not find a <noun> there -> likely a misdetection from afar
            o["misses"] = o.get("misses", 0) + 1
            o["confirmed"] = False
        self._ingest(d, source_of=lambda v: "head",
                     label_only_near=(oid, max(0.15, 3 * self.wm.objects[oid]["sigma_xy"])))
        self._save_world(f"refine {oid}", d)
        return oid, confirmed, in_view

    @staticmethod
    def _in_frame(v, xyz, margin=0.08):
        pc = v.R.T @ (np.asarray(xyz, float) - v.t)
        if pc[2] < 0.15:
            return False
        u = v.K[0, 0] * pc[0] / pc[2] + v.K[0, 2]; w = v.K[1, 1] * pc[1] / pc[2] + v.K[1, 2]
        return margin * v.W < u < (1 - margin) * v.W and margin * v.H < w < (1 - margin) * v.H

    # ---- approach + reach ------------------------------------------------------
    def _reach_stand(self, xy):
        p = self._p()
        st = stand_for(self.planner, xy, (p[0], p[1]), reach=(0.24, 0.36))
        if st is None:
            return False, "no free spot to stand within reach"
        ok, info = self._route_to(st["xy"])
        for _ in range(3):
            self._face_for_right_arm([xy[0], xy[1], 0.9])
            if self._in_envelope([xy[0], xy[1], 0.9]):
                break
            fx, _, _ = self._robot_frame([xy[0], xy[1], 0.9])
            want = fx - 0.26
            if abs(want) > 0.06:
                self._straight(1.0 if want > 0 else -1.0, abs(want))
        fx, fy, _ = self._robot_frame([xy[0], xy[1], 0.9])
        return True, f"{fx:.2f} m ahead, {-fy:.2f} m right"

    def t_pick(self, object):
        if self.held_oid:
            raise ValueError(f"already holding {self.wm.label(self.held_oid)}; place it first")
        oid = self._resolve_object(object)
        lab = self.wm.label(oid)
        steps = []
        if self.wm.objects[oid].get("misses"):
            return {"ok": False, "error": f"{oid} ({lab}) was already checked up close and is not there "
                                           f"(likely a misdetection); pick another object"}
        if not self.wm.ready_for_reach(oid):
            s0 = self.wm.objects[oid]["sigma_xy"]
            oid, confirmed, in_view = self._refine(oid)
            if not confirmed and in_view:
                noun = lab.lower().split()[-1]
                others = [f"{k} ({self.wm.label(k)})" for k in self.wm.find(noun) if k != oid
                          and not self.wm.objects[k].get("misses")]
                return {"ok": False, "error": f"looked closer with the head cameras and there is no {noun} where "
                                               f"{oid} was seen; it was probably a misdetection from far away. "
                                               f"Other candidates: {', '.join(others) or 'none'}"}
            steps.append(f"refined position ±{s0 * 100:.0f} -> ±{self.wm.objects[oid]['sigma_xy'] * 100:.1f} cm "
                         f"with the head cameras" + ("" if confirmed else " (not in view up close)"))
        o = self.wm.objects[oid]
        self._ev("exec", f"target {oid} {self.wm.label(oid)} at ({o['mean'][0]:+.2f}, {o['mean'][1]:+.2f}, "
                         f"{o['mean'][2]:.2f}) ±{o['sigma_xy'] * 100:.1f} cm → reach stand")
        ok, info = self._reach_stand(o["mean"][:2])
        if not ok:
            return {"ok": False, "error": info}
        base, h = o["mean"], max(0.05, o["height"])
        grasp = [base[0], base[1], base[2] + 0.6 * h]
        x, y, _, _ = self._p()
        v = np.array(grasp[:2]) - np.array([x, y]); v /= np.linalg.norm(v)
        grasp[0] -= 0.04 * v[0]; grasp[1] -= 0.04 * v[1]
        self._reach([grasp[0], grasp[1], base[2] + h + HOVER])
        e, ik = self._reach(grasp)
        # grasp (sim shortcut): the prop nearest the palm, if any is close enough
        st = self._st()
        palm = np.array(st["palm"]["right"])
        props = {n: np.array(p["xyz"]) + [0, 0, p["height"] / 2] for n, p in st["objects"].items()}
        name, dist = min(((n, float(np.linalg.norm(c[:2] - palm[:2]))) for n, c in props.items()),
                         key=lambda t: t[1])
        if dist > ATTACH_RADIUS:
            self._reach([grasp[0], grasp[1], base[2] + h + HOVER])
            self._rig(op="arm_release", side="right")
            return {"ok": False, "error": f"reached the perceived {lab} but nothing was under the hand "
                                           f"(nearest item {dist * 100:.0f} cm away)", "steps": steps}
        self._rig(op="attach", obj=name, side="right")
        self._ev("exec", f"grasp (sim attach, palm {dist * 100:.0f} cm from prop) → lift, carry pose")
        self.held_oid, self.held_prop = oid, name
        self._save_world(f"picked {oid}")
        self._reach([grasp[0], grasp[1], base[2] + h + HOVER + 0.05])
        cx = self._p(); c, s_ = math.cos(cx[2]), math.sin(cx[2])
        self._reach([cx[0] + 0.25 * c + SHOULDER_Y * s_, cx[1] + 0.25 * s_ - SHOULDER_Y * c, 1.0],
                    settle=0.5, tol=0.15)
        return {"ok": True, "picked": f"{oid} ({lab})", "palm_error_m": round(e, 3),
                "position_sigma_cm": round(o["sigma_xy"] * 100, 1), "steps": steps}

    def _place_spot(self, pid):
        """A free point on the place's surface with a reachable stand."""
        p = self.wm.places[pid]
        cx, cy, cz = p["center"]
        ex = max(0.2, p["extent"][0] or 0.3) / 2; ey = max(0.2, p["extent"][1] or 0.3) / 2
        rx, ry, _, _ = self._p()
        cands = []
        for dx in np.linspace(-ex * 0.8, ex * 0.8, 7):
            for dy in np.linspace(-ey * 0.8, ey * 0.8, 7):
                q = (cx + dx, cy + dy)
                if any(np.linalg.norm(o["mean"][:2] - q) < 0.12 for k, o in self.wm.objects.items()
                       if k != self.held_oid):
                    continue
                st = stand_for(self.planner, q, (rx, ry), reach=(0.24, 0.36))
                if st:
                    cands.append((math.hypot(dx, dy), q))
        if not cands:
            return None
        return (*min(cands)[1], cz)

    def t_place(self, place):
        if not self.held_oid:
            raise ValueError("not holding anything")
        pid = self._resolve_place(place)
        if pid is None:
            raise ValueError(f"no place matching '{place}'. Known places: {', '.join(self.wm.places)}")
        spot = self._place_spot(pid)
        if spot is None:
            return {"ok": False, "error": f"no free, reachable spot on {pid}"}
        self._ev("exec", f"place spot on {pid}: ({spot[0]:+.2f}, {spot[1]:+.2f}, {spot[2]:.2f}) → reach stand")
        ok, info = self._reach_stand(spot[:2])
        if not ok:
            return {"ok": False, "error": info}
        h = max(0.05, self.wm.objects[self.held_oid]["height"])
        x, y, _, _ = self._p()
        v = np.array(spot[:2]) - np.array([x, y]); v /= np.linalg.norm(v)
        gx, gy = spot[0] - 0.04 * v[0], spot[1] - 0.04 * v[1]
        self._reach([gx, gy, spot[2] + h + HOVER])
        e, _ = self._reach([gx, gy, spot[2] + 0.6 * h + 0.01])
        self._rig(op="detach", obj=self.held_prop, xyz=list(spot))
        self._ev("exec", f"release at ({spot[0]:+.2f}, {spot[1]:+.2f}, {spot[2]:.2f}) → retract arm")
        oid, lab = self.held_oid, self.wm.label(self.held_oid)
        o = self.wm.objects[oid]
        o["mean"] = np.array(spot, float); o["sigma_xy"] = max(0.03, e); o["sigma_z"] = 0.02
        self.held_oid = self.held_prop = None
        self._save_world(f"placed {oid}")
        self._reach([gx, gy, spot[2] + h + HOVER])
        self._rig(op="arm_release", side="right")
        self._rig(op="waist", ypr=None); self._lean = None
        self._wait_sim(2.0)
        # check with the Orbbec that it is there
        d = self._survey([{"name": "orbbec", "camera": "orbbec"}], f"verify_{oid}")
        hits = find([View(d, 0)], lab.lower().split()[-1], tiles=(1, 1), log=lambda *a: None)
        self._save_world(f"verify {oid}", d)
        seen = [hh for hh in hits if np.linalg.norm(np.array(hh["base"][:2]) - np.array(spot[:2])) < 0.15]
        self._save_world()
        return {"ok": True, "placed": f"{oid} ({lab})", "on": pid.replace("_", " "),
                "palm_error_m": round(e, 3), "orbbec_confirms": bool(seen)}

    def t_go_to(self, place):
        pid = self._resolve_place(place)
        if pid:
            c = self.wm.places[pid]["center"]
            p = self._p()
            st = stand_for(self.planner, c[:2], (p[0], p[1]), reach=(0.45, 0.9))
            ok, info = self._route_to(st["xy"]) if st else (False, "no free spot nearby")
            return {"ok": ok, "info": f"{pid}: {info}"}
        return super().t_go_to(place)

    def t_get_state(self):
        r = super().t_get_state()
        r["holding"] = None if not self.held_oid else self.wm.label(self.held_oid)
        return r


def perception_tool_schemas(waypoints):
    from .tools import _fn
    return [
        _fn("stop", "Stop all motion immediately."),
        _fn("look", "Look around with the room cameras and your own head camera. Returns the objects "
            "you can see (with ids like obj4, labels, where they are, and how precisely each "
            "position is known) and the places where things can be put."),
        _fn("pick", "Walk to an object and pick it up with the right hand. If its position is not "
            "precise enough yet, you first walk closer and look at it with your head cameras.",
            {"object": {"type": "string", "description": "object id from look (e.g. obj4) or its label"}},
            ["object"]),
        _fn("place", "Walk to a place and put down what you are holding.",
            {"place": {"type": "string", "description": "place id or name from look"}}, ["place"]),
        _fn("go_to", "Walk to a place from look, or to a room landmark.",
            {"place": {"type": "string", "description": "place id/name from look, or one of: "
                       + ", ".join(sorted(waypoints))}}, ["place"]),
        _fn("walk", "Walk straight forward or backward (metres, max 2).",
            {"direction": {"type": "string", "enum": ["forward", "backward"]},
             "meters": {"type": "number"}}, ["direction", "meters"]),
        _fn("turn", "Turn in place.",
            {"direction": {"type": "string", "enum": ["left", "right"]},
             "degrees": {"type": "number"}}, ["direction", "degrees"]),
        _fn("get_state", "Report position, heading and what you are holding."),
    ]


PERCEPTION_SYSTEM = """You control a humanoid robot (AgiBot X2) in a kitchen by calling tools.
You perceive the kitchen yourself through cameras: call look to find out what is where.
Rules:
- Always call look first. In TOOL CALLS refer to objects by the ids look returns (e.g. obj4) -
  ids are unambiguous when several things share a name.
- When TALKING TO THE USER never say ids like "obj4". Describe objects by what they are and
  where they are, e.g. "the black mug on the counter by the stove".
- If the user's object is ambiguous (several match), pick the closest match and say which.
- To move an object: pick it, then place it. You hold at most one object.
- Call tools one at a time and wait for each result. If a tool fails, say why in one
  sentence; you may call look again and retry once.
- If the user says stop, call stop immediately.
- When done, reply with one short sentence."""
