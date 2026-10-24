"""Curated, named moves for Gemma, picked from the motion library by category.

Each move is a name pattern over the clip keys; resolve() picks the best clip (unmirrored,
3-15 s, shortest name = least qualified), bake() writes the x2m2s into the kplanner's dances
dir loading each source pkl once.

  python -m gemma_ctl.motion_catalog            # resolve + bake + print the table
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

from .motion_library import CLIPS_DIR, MOTIONS, BAKER_REPO, index

OUT = Path(__file__).resolve().parents[1] / "runs/gemma/real/motion_catalog.json"

# category -> {move name: (regex over the clip key, optional regex the key must NOT match)}
CATALOG = {
    "walk": {
        "walk forward": (r"^Relaxed_walk_forward|^walk_forward|basic_walk_ff_loop", None),
        "walk slowly": (r"slow.*walk|walk.*slow", r"injur|crouch|hold"),
        "walk backward": (r"walk.*back(ward)?|back(ward)?.*walk", r"injur|crouch|hold"),
        "sideways jog": (r"^jog_sideway_045_loop", None),
        "walk in a circle": (r"^walk_circle", None),
        "walk and turn": (r"^slow_walk_turns", None),
    },
    "styled walk": {
        "injured leg walk": (r"injured_[RL]_leg_walk", None),
        "injured torso walk": (r"injured_torso_walk", None),
        "impatient waiting": (r"^hurry_idle_loop", None),
        "crutch walk": (r"crutch_walk_ff_loop", None),
        "walk the dog": (r"walk_the_dog", None),
        "relaxed walk": (r"^Relaxed_walk_forward", None),
        "catwalk": (r"cat_walk", None),
        "march": (r"march", None),
        "carry something heavy": (r"big_heavy.*walk_ff_loop|heavy.*walk", None),
    },
    "jog and run": {
        "jog": (r"^Loop_Forward_Jog", None),
        "walk in place": (r"Walk_In_Place_clip", r"Punch"),
        "start jogging": (r"^jog_ff_start", None),
        "run": (r"^run_start|run_ff_loop|^titan_run", None),
        # "run like crazy" (run_like_crazy_R_001__A533) dropped: SONIC ft8_w18 45000 falls on it
        # (tilt watchdog -> SAFE_HOLD, sim sweep 2026-10-24)
        "injured jog": (r"injured_[RL]_leg_jog_ff_loop", None),
    },
    "crouch and squat": {
        "crouch": (r"crouch_idle", r"walk"),
        "crouch down and stand up": (r"stand_to_crouch|crouch_to_stand", None),
        "crouch walk": (r"crouch.*walk|crouch_ff_loop|crouch_walk", r"injur"),
        "squat": (r"^squat_00", None),
        "yoga squat": (r"Yogi_Squat", None),
        "jump squat": (r"jump.*squat|squat.*jump", None),
    },
    "boxing": {
        "shadow boxing": (r"^shadow_boxing_R_001", None),
        "boxing combo": (r"punch.*combo|combo.*punch|boxing.*combo", None),
        "jab": (r"jab", None),
        "hook punch": (r"hook", r"hookah"),
        "punches": (r"punches|punch", r"combo|jab|hook|uppercut"),
        "front kick": (r"Front_Kick", None),
        "roundhouse kick": (r"roundhouse|round_kick", None),
        "thai boxing": (r"Boxing_kicking_thai", None),
    },
    "dance": {
        "salsa": (r"salsa", None),
        "mambo": (r"mambo", None),
        "jazz dance": (r"jazz", None),
        "latino dance": (r"latino", None),
        "charleston": (r"charleston", None),
        "country dance": (r"country", None),
        "lasso dance": (r"lasso", None),
        "retro dance": (r"retro", None),
        "street dance": (r"street", None),
        "pop dance": (r"dance_pop|pop_dance|_pop_", None),
        "shuffle": (r"^dance_hiphop_shuffle_square_R_001", None),
        "ballet plie": (r"plie", None),
        "ballet turns": (r"chaines", None),
        "hip hop march": (r"hiphop_march", None),
        "running man": (r"running_man", None),
        "moonwalk": (r"moon_?walk", None),
    },
}


def _pick(pat, neg):
    rx, nx = re.compile(pat, re.I), re.compile(neg, re.I) if neg else None
    cands = []
    for k, c in index().items():
        if not rx.search(k) or (nx and nx.search(k)):
            continue
        s = c["seconds"] or 0
        score = (0 if not c["mirrored"] else 1, 0 if 3 <= s <= 15 else 1,
                 0 if c["src"] != "x2_train5pct.pkl" else 1, len(k))
        cands.append((score, k))
    return sorted(cands)[0][1] if cands else None


def resolve():
    out = {}
    for cat, moves in CATALOG.items():
        for name, (pat, neg) in moves.items():
            k = _pick(pat, neg)
            if k:
                c = index()[k]
                out[name] = {"category": cat, "key": k, "seconds": c["seconds"],
                             "travels_m": c["travel_m"]}
    return out


def bake(cat):
    sys.path.insert(0, str(BAKER_REPO))
    sys.path.insert(0, str(BAKER_REPO / "gear_sonic/scripts"))
    import joblib
    import numpy as np
    from pkl_to_x2m2 import write_x2m2
    by_src = {}
    for m in cat.values():
        src = index()[m["key"]]["src"]
        if src != "x2m2_bank" and not (CLIPS_DIR / f"{m['key']}.x2m2").exists():
            by_src.setdefault(src, []).append(m["key"])
    for src, keys in by_src.items():
        lib = joblib.load(MOTIONS / src)
        for k in keys:
            c = lib[k]
            write_x2m2(CLIPS_DIR / f"{k}.x2m2", np.asarray(c["dof"], np.float64),
                       np.asarray(c["root_rot"], np.float64), float(c.get("fps", 30.0)))
        print(f"  baked {len(keys)} from {src}")


def load():
    return json.loads(OUT.read_text()) if OUT.exists() else {}


if __name__ == "__main__":
    cat = resolve()
    bake(cat)
    OUT.write_text(json.dumps(cat, indent=1))
    missing = [n for moves in CATALOG.values() for n in moves if n not in cat]
    for n, m in cat.items():
        print(f"{m['category']:16s} {n:26s} {m['seconds']:5.1f}s  travel {m['travels_m']}  {m['key']}")
    print(f"{len(cat)} moves; not found: {missing}")
