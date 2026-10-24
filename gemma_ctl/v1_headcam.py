"""Coarse (room cams) -> fine (head cam) perception test, through the world model.

For each target: room survey -> Gemma boxes -> world model (room sigma);
robot walks to an online stand spot near the coarse estimate, faces it,
takes a head-camera view, Gemma looks for the same label, observations fuse.
Sim truth is used only to grade.

  ~/miniconda3/envs/env_isaaclab/bin/python -u -m gemma_ctl.v1_headcam
"""
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

from .executor_kitchen import KitchenExecutor
from .nav_plan import NavPlanner, stand_for
from .perceive import ROOM_VIEWS, ROOT, View, find, overlay, perceive, request_survey
from .world_model import WorldModel

TARGETS = sys.argv[1:] or ["mug", "pot", "kettle", "pitcher", "can"]
HEAD_Z = 1.28          # approx head-camera height when standing (m)


def main():
    out = ROOT / "runs/gemma/perception" / time.strftime("headcam_%Y%m%dT%H%M%S")
    out.mkdir(parents=True)
    log = open(out / "log.txt", "a", buffering=1)

    def say(m):
        print(m, flush=True); log.write(m + "\n")

    ex = KitchenExecutor(log=lambda m: log.write(m + "\n"))
    for _ in range(200):
        if ex._st() and ex._p():
            break
        time.sleep(0.1)
    wm = WorldModel()
    planner = NavPlanner()

    # 1. room survey -> world model
    d0 = out / "room"
    request_survey(d0, ex=ex)
    world, views, raw = perceive(d0, log=say)
    overlay(d0, views, raw)
    truth = json.loads((d0 / "truth.json").read_text())
    for o in world["objects"]:
        v = views[o["views"][0]]
        rng = float(np.linalg.norm(np.array(o["base"]) - v.t))
        wm.observe(o["label"], o["base"], o["height"], rng, "room", o["views"][0])
    for p in world["places"]:
        wm.observe_place(p["label"], p["center"], p["extent"])
    say(f"room: {len(wm.objects)} objects, {len(wm.places)} places")

    rows = []
    for tgt in TARGETS:
        tb = np.array(truth[tgt].get("center_base", truth[tgt]["base"]))
        # which world-model entry is the target? (grading only: nearest to truth
        # among entries not already claimed by another target)
        taken = {r.get("oid") for r in rows}
        cands = sorted((i for i in wm.objects if i not in taken),
                       key=lambda i: np.linalg.norm(wm.objects[i]["mean"][:2] - tb[:2]))
        oid = cands[0] if cands and np.linalg.norm(wm.objects[cands[0]]["mean"][:2] - tb[:2]) < 0.30 else None
        if oid is None:
            say(f"{tgt}: NOT SEEN by room cameras"); rows.append({"target": tgt, "room_seen": False}); continue
        lab = wm.label(oid)
        room_err = float(np.linalg.norm(wm.objects[oid]["mean"] - tb))
        room_sig = wm.objects[oid]["sigma_xy"]
        coarse = wm.objects[oid]["mean"].copy()
        say(f"{tgt}: room sees '{lab}' ({oid}) err {room_err * 100:.1f} cm, sigma ±{room_sig * 100:.1f} cm")

        # 2. walk near + face it
        p = ex._p()
        # a LOOK stand (not a reach stand): 0.35-0.9 m back, with clearance
        st = stand_for(planner, coarse[:2], (p[0], p[1]), reach=(0.45, 0.90))
        if st is None:
            say(f"   no free spot to look from near {np.round(coarse[:2], 2).tolist()}")
            rows.append({"target": tgt, "oid": fused or oid, "room_seen": True, "label": lab, "final_label": wm.label(fused or oid),
                     "label_votes": wm.label_votes(fused or oid), "no_stand": True}); continue
        ok, info = ex._route_to(st["xy"])
        p = ex._p()
        ex._turn_to(math.atan2(coarse[1] - p[1], coarse[0] - p[0]), tol=math.radians(8), tries=2)
        p = ex._p()
        horiz = math.hypot(coarse[0] - p[0], coarse[1] - p[1])
        pitch = math.degrees(math.atan2(HEAD_Z - coarse[2], max(0.15, horiz)))
        say(f"   walked ({info}); target {horiz:.2f} m ahead; head pitch {pitch:.0f} deg")

        # 3. robot cameras: wide head cam (search-grade) + Orbbec (manipulation-grade)
        dh = out / f"robot_{tgt}"
        request_survey(dh, views=[{"name": "head_wide", "camera": "head_wide"},
                                  {"name": "orbbec", "camera": "orbbec"}], ex=ex)
        hv = [View(dh, 0), View(dh, 1)]
        fused, head_err = None, {}
        for vi, (cam_name, tiles) in enumerate((("head_wide", (2, 2)), ("orbbec", (1, 1)))):
            # search by the NOUN only: colour words from a far view mislead
            # (room said "white mug" for a black mug -> Orbbec query found nothing)
            noun = lab.lower().split()[-1]
            hits = find([hv[vi]], noun, tiles=tiles, log=say)
            overlay(dh, [hv[vi]], [{"view": vi, "boxes": [{"label": h["label"], "box_2d": h["box_2d"]} for h in hits]}])
            if not hits:
                say(f"   {cam_name}: Gemma did not find it"); continue
            # refine the TARGET only with detections inside its gate: 3 sigma
            # of the current estimate, at least 0.3 m (else a same-label item
            # across the room gets fused, as a 5.8 m 'tea kettle' once was)
            cur = wm.objects[fused or oid]
            gate = max(0.30, 3 * cur["sigma_xy"])
            near = [h for h in hits if np.linalg.norm(np.array(h["base"][:2]) - cur["mean"][:2]) < gate]
            for h in hits:
                if h not in near:
                    say(f"   {cam_name}: '{h['label']}' {np.linalg.norm(np.array(h['base'][:2]) - cur['mean'][:2]):.2f} m from the target - outside its ±{gate:.2f} m gate, not used for it")
            if not near:
                say(f"   {cam_name}: nothing inside the target's gate"); continue
            h = min(near, key=lambda h: np.linalg.norm(np.array(h["base"][:2]) - cur["mean"][:2]))
            rng = float(np.linalg.norm(np.array(h["base"]) - hv[vi].t))
            e = float(np.linalg.norm(np.array(h["base"]) - tb))
            head_err[cam_name] = round(e * 100, 1)
            oid2, how = wm.observe(h["label"], h["base"], h["height"], rng, "head")
            fused = oid2
            say(f"   {cam_name} sees '{h['label']}' at {rng:.2f} m: err {e * 100:.1f} cm -> {how} into {oid2} "
                f"(sigma now ±{wm.objects[oid2]['sigma_xy'] * 100:.1f} cm)")
        # full labelling pass on the robot cameras too: every object/place they
        # see enters the world model (labels weighted by view quality)
        rworld, rviews, rraw = perceive(dh, log=lambda m: say("   robot-cam" + m))
        overlay(dh, rviews, rraw)
        n_new = 0
        tgt_id = fused or oid
        for ro in rworld["objects"]:
            rv = rviews[ro["views"][0]]
            rng = float(np.linalg.norm(np.array(ro["base"]) - rv.t))
            t_o = wm.objects[tgt_id]
            if fused and np.linalg.norm(np.array(ro["base"][:2]) - t_o["mean"][:2]) < max(0.15, 3 * t_o["sigma_xy"]):
                # same pixels already fused by the targeted search: label vote only
                from .world_model import sigma_for
                t_o["labels"][ro["label"]] += 1.0 / sigma_for(rng, "head")[0]
                continue
            _, how = wm.observe(ro["label"], ro["base"], ro["height"], rng, "head")
            n_new += how == "new"
        for rp in rworld["places"]:
            wm.observe_place(rp["label"], rp["center"], rp["extent"])
        say(f"   robot cams labelled {len(rworld['objects'])} objects ({n_new} new), {len(rworld['places'])} places; "
            f"target label votes: {wm.label_votes(fused or oid)}")
        o = wm.objects[fused or oid]
        f_err = float(np.linalg.norm(o["mean"] - tb))
        rows.append({"target": tgt, "oid": fused or oid, "room_seen": True, "label": lab, "final_label": wm.label(fused or oid),
                     "label_votes": wm.label_votes(fused or oid), "room_err_cm": round(room_err * 100, 1),
                     "room_sigma_cm": round(room_sig * 100, 1),
                     "robot_cam_err_cm": head_err,
                     "fused_err_cm": round(f_err * 100, 1), "fused_sigma_cm": round(o["sigma_xy"] * 100, 1),
                     "ready_for_reach": wm.ready_for_reach(fused or oid)})
        say(f"   fused: err {f_err * 100:.1f} cm, sigma ±{o['sigma_xy'] * 100:.1f} cm, "
            f"ready_for_reach={rows[-1]['ready_for_reach']}")
    (out / "summary.json").write_text(json.dumps({"rows": rows, "world": wm.to_json()}, indent=1, default=str))
    say("-> " + str(out))


if __name__ == "__main__":
    main()
