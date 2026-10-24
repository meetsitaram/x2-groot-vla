"""World model built from Gemma's detections, with per-object uncertainty.

Room cameras see from 2-3 m and give a coarse fix; the robot's head camera
sees from <1 m and gives a fine one. Each detection carries a sigma from a
range model; detections associate to an existing object when they fall
inside a 3-sigma gate (combined), then fuse by inverse-variance weighting, so
the estimate tightens as the robot gets close. The executor reaches only when
sigma_xy is under the reach tolerance; otherwise it goes and looks closer.
"""
from __future__ import annotations

import json
import math
import time
from collections import Counter

import numpy as np

# sigma model (metres): fitted to V1 (room cams at 2-3 m: 3-6 cm observed
# error) with margin; head cam at 0.6-1 m lands near 2-3 cm
SIGMA_XY0, SIGMA_XY_PER_M = 0.010, 0.020
SIGMA_Z_RATIO = 0.5          # z comes from the support surface: tighter
GATE_SIGMAS = 3.0
REACH_TOL = 0.03


def sigma_for(range_m: float, source: str, pose_sigma_xy: float = 0.0) -> tuple[float, float]:
    """Detection sigma in the GLOBAL frame.

    Pixel+depth error grows with range. A head-camera detection also inherits
    the camera's global-pose error = robot localization error (base pose in
    the kitchen frame) through the kinematic chain; room cameras inherit their
    extrinsic-calibration error. In sim both are ~0 (true poses); on the robot
    pass the localizer's sigma, or the head camera would confidently move
    objects whenever odometry drifts."""
    sxy = SIGMA_XY0 + SIGMA_XY_PER_M * range_m
    if source == "room":
        sxy *= 1.3            # room views: smaller pixels, more clutter
    sxy = math.hypot(sxy, pose_sigma_xy)
    return sxy, math.hypot(sxy * SIGMA_Z_RATIO, 0.5 * pose_sigma_xy)


# label compatibility for association: same noun, a shared non-adjective word,
# or a synonym group. Distance alone merged a 'black pot' into a 'glass jar'
# 30 cm away when room-camera sigma made the gate ~38 cm wide.
SYNONYMS = [{"mug", "cup"}, {"pot", "pan", "saucepan", "cookware"}, {"can", "soda", "drink"},
            {"kettle", "teapot"}, {"pitcher", "jug", "carafe"}, {"jar", "container"},
            {"bowl"}, {"plate", "dish"}, {"bottle"}]
ADJ = {"white", "black", "red", "blue", "green", "yellow", "brown", "orange", "silver", "grey", "gray",
       "glass", "small", "big", "large", "tall", "short", "metal", "plastic", "dark", "light", "clear",
       "and", "of", "the", "a"}


def labels_compatible(a: str, b: str) -> bool:
    wa = set(a.lower().replace("-", " ").split()) - ADJ
    wb = set(b.lower().replace("-", " ").split()) - ADJ
    if not wa or not wb:
        return True               # bare adjectives: no evidence either way
    if wa & wb:
        return True
    return any(wa & g and wb & g for g in SYNONYMS)


class WorldModel:
    def __init__(self):
        self.objects: dict[str, dict] = {}
        self.places: dict[str, dict] = {}
        self._n = 0

    # ---- objects -------------------------------------------------------------
    def observe(self, label, base, height, range_m, source, view=None, pose_sigma_xy=0.0, size_xy=None):
        base = np.asarray(base, float)
        sxy, sz = sigma_for(range_m, source, pose_sigma_xy)
        best, best_d = None, 1e9
        for oid, o in self.objects.items():
            s_comb = math.hypot(o["sigma_xy"], sxy)
            gxy = GATE_SIGMAS * s_comb
            dxy = float(np.linalg.norm(o["mean"][:2] - base[:2]))
            if dxy >= gxy or dxy >= best_d:
                continue
            # semantic gate: beyond 1 sigma, labels must be compatible
            if dxy > s_comb and not any(labels_compatible(label, l) for l in o["labels"]):
                continue
            best, best_d = oid, dxy
        if best is None:
            self._n += 1
            oid = f"obj{self._n}"
            self.objects[oid] = {"labels": Counter({label: 1.0 / sxy}), "mean": base, "sigma_xy": sxy,
                                 "size_xy": list(size_xy) if size_xy else [0.08, 0.08],
                                 "sigma_z": sz, "height": height, "n": 1,
                                 "sources": Counter([source]), "t": time.time(), "history": []}
            self.objects[oid]["history"].append((source, base.round(3).tolist(), round(sxy, 3)))
            return oid, "new"
        o = self.objects[best]
        w_old, w_new = 1 / o["sigma_xy"] ** 2, 1 / sxy ** 2
        mxy = (o["mean"][:2] * w_old + base[:2] * w_new) / (w_old + w_new)
        wz_old, wz_new = 1 / o["sigma_z"] ** 2, 1 / sz ** 2
        mz = (o["mean"][2] * wz_old + base[2] * wz_new) / (wz_old + wz_new)
        o["mean"] = np.r_[mxy, mz]
        o["sigma_xy"] = (w_old + w_new) ** -0.5
        o["sigma_z"] = (wz_old + wz_new) ** -0.5
        o["height"] = (o["height"] * o["n"] + height) / (o["n"] + 1)
        if size_xy:
            o["size_xy"] = [(a * o["n"] + b) / (o["n"] + 1) for a, b in zip(o.get("size_xy", size_xy), size_xy)]
        o["n"] += 1
        o["labels"][label] += 1.0 / sxy      # close (small-sigma) views outvote distant ones
        o["sources"][source] += 1
        o["t"] = time.time()
        o["history"].append((source, base.round(3).tolist(), round(sxy, 3)))
        return best, "fused"

    def label(self, oid):
        return self.objects[oid]["labels"].most_common(1)[0][0]

    def find(self, query: str):
        """Objects whose labels share a word with the query (Gemma's own words)."""
        qw = set(query.lower().replace("-", " ").split())
        out = []
        for oid, o in self.objects.items():
            words = set(" ".join(o["labels"]).lower().split())
            if qw & words:
                out.append(oid)
        return sorted(out, key=lambda i: (self.objects[i].get("misses", 0), self.objects[i]["sigma_xy"]))

    def ready_for_reach(self, oid, tol=REACH_TOL):
        return self.objects[oid]["sigma_xy"] <= tol

    # ---- places ----------------------------------------------------------------
    def observe_place(self, label, center, extent):
        for pid, p in self.places.items():
            if np.linalg.norm(np.asarray(p["center"][:2]) - np.asarray(center[:2])) < 0.35:
                p["labels"][label] += 1
                return pid
        pid = label.lower().replace(" ", "_")
        k = 2
        while pid in self.places:
            pid = f"{label.lower().replace(' ', '_')}_{k}"; k += 1
        self.places[pid] = {"labels": Counter([label]), "center": list(center), "extent": extent}
        return pid

    def place_of(self, xyz, radius=0.6):
        best = min(self.places.items(), default=None,
                   key=lambda kv: np.linalg.norm(np.asarray(kv[1]["center"][:2]) - np.asarray(xyz[:2])))
        if best and np.linalg.norm(np.asarray(best[1]["center"][:2]) - np.asarray(xyz[:2])) < radius:
            return best[0]
        return None

    # ---- for Gemma -------------------------------------------------------------
    def describe(self):
        objs = []
        for oid, o in self.objects.items():
            where = self.place_of(o["mean"]) or "somewhere"
            tag = (" — NOT there when checked up close" if o.get("misses") else
                   " — confirmed up close" if o.get("confirmed") else "")
            objs.append(f"{oid}: {self.label(oid)} on {where.replace('_', ' ')} "
                        f"(position known to ±{o['sigma_xy'] * 100:.0f} cm, seen {o['n']}x){tag}")
        return {"objects": objs, "places": sorted(self.places)}

    def to_json(self):
        return {"objects": {k: {**{kk: vv for kk, vv in o.items() if kk not in ("labels", "sources", "mean")},
                                "label": self.label(k), "labels": dict(o["labels"]),
                                "sources": dict(o["sources"]), "mean": np.round(o["mean"], 3).tolist()}
                            for k, o in self.objects.items()},
                "places": {k: {**p, "labels": dict(p["labels"])} for k, p in self.places.items()}}

    def label_votes(self, oid):
        tot = sum(self.objects[oid]["labels"].values())
        return {k: round(v / tot, 2) for k, v in self.objects[oid]["labels"].most_common()}
