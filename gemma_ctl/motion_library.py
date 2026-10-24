"""Searchable motion library for Gemma: every clip in ~/x2_eval_motions_supermimic/*.pkl plus
the baked x2m2 banks, behind two tools -- find_motions(query) and play_motion(name).

A clip is baked to x2m2 (pkl_to_x2m2.py, the run_motion_replay.sh way) into the kplanner's
dances dir on first play; the kplanner resolves <dances-dir>/<key>.x2m2 at play time, so new
clips work on a running stack.

  python -m gemma_ctl.motion_library --build            # (re)index, ~10 s
  python -m gemma_ctl.motion_library jump               # search from the shell
"""
from __future__ import annotations

import json
import os
import re
import struct
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MOTIONS = Path(os.environ.get("GEMMA_MOTIONS_DIR", Path.home() / "x2_eval_motions_supermimic"))
CLIPS_DIR = Path(os.environ.get("GEMMA_CLIPS_DIR", ROOT / "runs/gemma/real/clips_merged"))
INDEX = ROOT / "runs/gemma/real/motion_index.json"
BAKER_REPO = Path(os.environ.get("GEMMA_BAKER_REPO", "/home/stickbot/Projects/GR00T-WBC-X2-supermimic"))
SKIP_FILES = {"x2_train5pct.pkl"}          # 1.7 GB; add with --with-train
_STOP = {"r", "m", "l", "loop", "clip", "clip1", "chunk", "animation", "the", "a", "of", "to", "and",
         "ff", "subset", "opt", "baseline", "x2", "novel500", "kungfu"}
# the corpus' own spellings
_ALIAS = {"gangnam": "gangam", "vogue": "vouge", "egyptian": "egipt", "karate": "kungfu"}


def _words(key):
    k = re.sub(r"__A\d+(_M)?$|_\d{3}$", "", key)
    k = re.sub(r"([a-z])([A-Z])", r"\1 \2", k)
    toks = [t for t in re.split(r"[^A-Za-z0-9]+", k.lower()) if t and not t.isdigit()]
    return [t for t in toks if t not in _STOP]


def nice_name(key):
    """'victory_dance_gangam_style_rodeo_R_001__A324' -> 'victory dance gangam style rodeo'."""
    return " ".join(_words(key.split("__", 1)[-1] if key.count("__") > 1 else key)) or key


def _x2m2_seconds(p):
    raw = p.read_bytes()[:20]
    _, n, _ = struct.unpack("<III", raw[:12])
    fps = struct.unpack("<d", raw[12:20])[0]
    return n / fps if fps > 0 else 0.0


def build(with_train=False):
    import joblib
    import numpy as np
    clips = {}
    for f in sorted(MOTIONS.glob("*.pkl")):
        if f.name in SKIP_FILES and not with_train:
            continue
        d = joblib.load(f)
        for k, c in d.items():
            fps = float(c.get("fps", 30.0)); n = len(c["dof"])
            rt = np.asarray(c["root_trans_offset"], float)
            travel = float(np.linalg.norm(rt[-1, :2] - rt[0, :2])) if rt.ndim == 2 and len(rt) else 0.0
            clips[k] = {"src": f.name, "seconds": round(n / fps, 1), "travel_m": round(travel, 2),
                        "mirrored": k.endswith("_M")}
    for p in sorted(CLIPS_DIR.glob("*.x2m2")):        # already-baked banks (gestures, dances)
        k = p.stem
        if k not in clips:
            clips[k] = {"src": "x2m2_bank", "seconds": round(_x2m2_seconds(p), 1), "travel_m": None,
                        "mirrored": k.endswith("_M")}
    for k, c in clips.items():
        c["name"] = nice_name(k); c["words"] = _words(k)
    INDEX.parent.mkdir(parents=True, exist_ok=True)
    INDEX.write_text(json.dumps(clips))
    return clips


_CACHE = None


def index():
    global _CACHE
    if _CACHE is None:
        _CACHE = json.loads(INDEX.read_text()) if INDEX.exists() else build()
    return _CACHE


def search(query, k=8):
    """Best matches for a short query ('jump', 'high jump spin', 'boxing', 'squat')."""
    q = [t for t in re.split(r"[^a-z0-9]+", query.lower()) if t and t not in _STOP]
    q = [t[:-1] if t.endswith("s") and len(t) > 3 else t for t in q]       # jumps -> jump
    q = [_ALIAS.get(t, t) for t in q]
    scored = []
    for key, c in index().items():
        w = c["words"]; text = " ".join(w)
        hit = sum(1 for t in q if t in w) + 0.5 * sum(1 for t in q if t not in w and t in text)
        if hit == 0:
            continue
        score = hit / len(q) - 0.002 * len(w) - (0.05 if c["mirrored"] else 0) \
            + (0.1 if c["src"] in ("supermimic_demo_clips.pkl", "x2m2_bank") else 0)
        scored.append((score, key))
    scored.sort(reverse=True)
    seen, out = set(), []
    for _, key in scored:
        c = index()[key]
        if c["name"] in seen:          # mirrored / numbered twins: keep the best one
            continue
        seen.add(c["name"])
        out.append({"name": key, "description": c["name"], "seconds": c["seconds"],
                    "travels_m": c["travel_m"]})
        if len(out) >= k:
            break
    return out


def ensure_baked(key):
    """x2m2 for a clip in the kplanner's dances dir; bakes it from its pkl on first use."""
    out = CLIPS_DIR / f"{key}.x2m2"
    if out.exists():
        return out
    c = index().get(key)
    if c is None or c["src"] == "x2m2_bank":
        raise KeyError(f"unknown motion {key!r}")
    py = BAKER_REPO / ".venv/bin/python"
    r = subprocess.run([str(py), str(BAKER_REPO / "gear_sonic/scripts/pkl_to_x2m2.py"),
                        "--pkl", str(MOTIONS / c["src"]), "--key", key, "--start-s", "0",
                        "--dur-s", str(c["seconds"]), "--out", str(out), "--no-hand-sidecar"],
                       capture_output=True, text=True, timeout=120)
    if r.returncode != 0 or not out.exists():
        raise RuntimeError(f"bake failed for {key}: {r.stderr.strip()[-300:]}")
    return out


if __name__ == "__main__":
    if "--build" in sys.argv:
        c = build(with_train="--with-train" in sys.argv)
        print(f"indexed {len(c)} motions -> {INDEX}")
    else:
        for m in search(" ".join(a for a in sys.argv[1:] if not a.startswith("--"))):
            print(f"{m['seconds']:5.1f}s  travel {m['travels_m']}  {m['name']}   ({m['description']})")
