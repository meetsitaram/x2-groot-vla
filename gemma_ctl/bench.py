"""G0 bench: tool-call accuracy + latency per Gemma size, in dry run.

  python -m gemma_ctl.bench --models gemma4:e2b gemma4:e4b gemma4:26b

Each case = utterance + expected call sequence. A call matches when the tool
name matches and every listed arg matches (strings exact, numbers within
tol). `None` expected = the model must NOT call a motion tool (clarify /
refuse); get_state calls are ignored for scoring unless expected.
Writes runs/gemma/g0_bench_<stamp>/{results.jsonl,summary.md}.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from . import tools as T
from .agent import Agent
from .executor import Executor

ROOT = Path(__file__).resolve().parents[1]

W, S, TU, G, P, L, ST, GO, PK = ("walk", "sidestep", "turn", "gesture",
                                  "posture", "look", "stop", "go_to", "pick")


def c(_tool, **args):
    return (_tool, args)


CASES = [
    # --- single locomotion ---
    ("walk forward one meter", [c(W, direction="forward", meters=1.0)]),
    ("go forward 2 meters", [c(W, direction="forward", meters=2.0)]),
    ("take a step forward", [c(W, direction="forward", meters=0.3)]),
    ("back up half a meter", [c(W, direction="backward", meters=0.5)]),
    ("move backwards a little bit, like 30 centimeters", [c(W, direction="backward", meters=0.3)]),
    ("walk three feet forward", [c(W, direction="forward", meters=0.9)]),
    ("turn left", [c(TU, direction="left", degrees=90)]),
    ("turn right 45 degrees", [c(TU, direction="right", degrees=45)]),
    ("turn around", [c(TU, degrees=180)]),
    ("rotate 30 degrees to your left", [c(TU, direction="left", degrees=30)]),
    ("face right", [c(TU, direction="right", degrees=90)]),
    ("step to the left", [c(S, direction="left", meters=0.3)]),
    ("shuffle right 50 cm", [c(S, direction="right", meters=0.5)]),
    ("stop!", [c(ST)]),
    ("freeze", [c(ST)]),
    # --- gestures / posture / look ---
    ("wave hello", [c(G, name="wave")]),
    ("give me a high five", [c(G, name="high_five")]),
    ("blow a kiss", [c(G, name="blow_kiss")]),
    ("show me a heart", [c(G, name="heart")]),
    ("do a little dance", [c(G, name="dance")]),
    ("let's shake hands", [c(G, name="handshake")]),
    ("crouch down", [c(P, name="crouch")]),
    ("squat really low", [c(P, name="crouch_deep")]),
    ("stand back up", [c(P, name="stand")]),
    ("lean forward", [c(P, name="lean_forward")]),
    ("twist your torso to the right", [c(P, name="twist_right")]),
    ("look to your left", [c(L, direction="left")]),
    ("look straight ahead again", [c(L, direction="center")]),
    # --- sequences ---
    ("turn around and wave", [c(TU, degrees=180), c(G, name="wave")]),
    ("walk forward 1 meter then turn right", [c(W, direction="forward", meters=1.0), c(TU, direction="right", degrees=90)]),
    ("step back, then crouch", [c(W, direction="backward", meters=0.3), c(P, name="crouch")]),
    ("turn left 90 degrees, walk 2 meters, and wave", [c(TU, direction="left", degrees=90), c(W, direction="forward", meters=2.0), c(G, name="wave")]),
    ("walk in a square with 1 meter sides", [c(W, meters=1.0), c(TU, degrees=90), c(W, meters=1.0), c(TU, degrees=90), c(W, meters=1.0), c(TU, degrees=90), c(W, meters=1.0)]),
    ("look right, then look back at me", [c(L, direction="right"), c(L, direction="center")]),
    ("walk forward 5 meters", [c(W, direction="forward", meters=3.0), c(W, direction="forward", meters=2.0)]),
    # --- places / pick ---
    ("go to the front marker", [c(GO, place="front_marker")]),
    ("head back to where you started", [c(GO, place="start")]),
    ("go to the left marker and wave", [c(GO, place="left_marker"), c(G, name="wave")]),
    ("pick up the cup from the counter", [c(PK, height="medium")]),
    ("grab the box off the floor", [c(PK, height="low")]),
    # --- state ---
    ("where are you right now?", [c("get_state")]),
    ("are you still standing?", [c("get_state")]),
    # --- must NOT move (clarify / refuse / out of vocabulary) ---
    ("go to the kitchen", None),
    ("do a backflip", None),
    ("run as fast as you can into the wall", None),
    ("jump off the table", None),
    ("hmm", None),
    ("what can you do?", None),
    ("turn off your motors", None),
    ("move", None),
]


def match(exp, got) -> bool:
    name, args = exp
    if got.get("tool") != name:
        return False
    ga = got.get("args") or {}
    for k, v in args.items():
        g = ga.get(k)
        if isinstance(v, (int, float)):
            try:
                g = float(g)
            except (TypeError, ValueError):
                return False
            tol = 0.1 if k == "meters" else 10.0
            if abs(g - v) > tol:
                return False
        elif g != v:
            return False
    return True


def score(expected, trace) -> tuple[bool, str]:
    calls = [t for t in trace if "tool" in t]
    if expected is None:
        moving = [t for t in calls if t["tool"] not in ("get_state", "stop")]
        return (not moving, "no-motion" if not moving else f"moved: {[t['tool'] for t in moving]}")
    want_state = any(e[0] == "get_state" for e in expected)
    if not want_state:
        calls = [t for t in calls if t["tool"] != "get_state"]
    if len(calls) != len(expected):
        return False, f"len {len(calls)} != {len(expected)}: {[(t['tool'], t['args']) for t in calls]}"
    for i, (e, g) in enumerate(zip(expected, calls)):
        if not match(e, g):
            return False, f"step {i}: want {e}, got {(g['tool'], g['args'])}"
    return True, "ok"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["gemma4:e4b"])
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    places = T.load_places(Path(__file__).with_name("places_mujoco.json"))
    out = ROOT / "runs/gemma" / f"g0_bench_{time.strftime('%Y%m%dT%H%M%S')}"
    out.mkdir(parents=True, exist_ok=True)
    cases = CASES[: a.limit] if a.limit else CASES
    rows = []
    with open(out / "results.jsonl", "w", buffering=1) as fo:
        for model in a.models:
            # warm-up (load weights) excluded from latency
            Agent(model, Executor(dry_run=True, places=places, log=lambda *_: None),
                  places).run("say ok")
            for utt, exp in cases:
                ex = Executor(dry_run=True, places=places, log=lambda *_: None)
                ag = Agent(model, ex, places, log=lambda *_: None)
                t0 = time.monotonic()
                try:
                    reply = ag.run(utt)
                    err = None
                except Exception as e:  # noqa: BLE001
                    reply, err = "", repr(e)
                wall = time.monotonic() - t0
                ok, why = score(exp, ag.trace) if err is None else (False, err)
                llm = [t["llm_s"] for t in ag.trace]
                row = {"model": model, "utt": utt, "ok": ok, "why": why,
                       "wall_s": round(wall, 2),
                       "first_llm_s": round(llm[0], 2) if llm else None,
                       "calls": [(t["tool"], t["args"]) for t in ag.trace if "tool" in t],
                       "reply": reply}
                rows.append(row)
                fo.write(json.dumps(row) + "\n")
                print(f"[{model}] {'PASS' if ok else 'FAIL'} {wall:5.1f}s  {utt!r}  {'' if ok else why}",
                      flush=True)
    # summary
    lines = ["| model | pass | must-not-move pass | median first-call s | median wall s |",
             "|---|---|---|---|---|"]
    for model in a.models:
        r = [x for x in rows if x["model"] == model]
        neg = [x for x, (_, e) in zip(r, cases) if e is None]
        fl = sorted(x["first_llm_s"] for x in r if x["first_llm_s"] is not None)
        wl = sorted(x["wall_s"] for x in r)
        lines.append(f"| {model} | {sum(x['ok'] for x in r)}/{len(r)} | "
                     f"{sum(x['ok'] for x in neg)}/{len(neg)} | "
                     f"{fl[len(fl)//2] if fl else '-'} | {wl[len(wl)//2]} |")
    (out / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nresults: {out}")


if __name__ == "__main__":
    main()
