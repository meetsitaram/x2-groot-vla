"""V1 gate: Gemma perception accuracy over several object placements.

The test harness moves the props (rig "move" op), then each scene is surveyed
and perceived from scratch; sim truth is used only to grade.

  ~/miniconda3/envs/env_isaaclab/bin/python -m gemma_ctl.v1_batch
"""
import json
import time
from pathlib import Path

import numpy as np

from .perceive import ROOT, grade, overlay, perceive, request_survey

CAT = json.loads((Path(__file__).with_name("kitchen_catalog.json")).read_text())


def surface_z(x, y):
    import open3d as o3d
    from .perceive import _SCENE, _mesh_depth  # noqa: F401 (scene cache)
    if "scene" not in _SCENE:
        _mesh_depth(np.eye(3), np.eye(3), np.zeros(3), 1, 1)
    r = o3d.core.Tensor(np.array([[x, y, 2.2, 0, 0, -1]], np.float32))
    return float(2.2 - _SCENE["scene"].cast_rays(r)["t_hit"].numpy()[0])


def spot(loc, dx=0.0, dy=0.0):
    x, y, _ = CAT[loc]["obj_xyz"]
    x, y = x + dx, y + dy
    return [round(x, 3), round(y, 3), round(surface_z(x, y), 3)]


SCENES = [
    {"mug": spot("stove_counter"), "can": spot("side_table")},
    {"mug": spot("sink_counter"), "can": spot("stove_counter")},
    {"mug": spot("side_table"), "can": spot("sink_counter")},
    {"mug": spot("dishwasher_counter", 0.10, 0.05), "can": spot("stove_counter", -0.10, 0.12)},
    {"mug": spot("stove_counter", -0.12, 0.18), "can": spot("side_table", 0.12, 0.10)},
    {"mug": spot("side_table", -0.10, 0.08), "can": spot("dishwasher_counter", 0.05, -0.05)},
]


def main():
    out = ROOT / "runs/gemma/perception" / time.strftime("v1_batch_%Y%m%dT%H%M%S")
    out.mkdir(parents=True)
    rows = []
    ex = None
    for i, sc in enumerate(SCENES):
        d = out / f"scene{i}"
        if ex is None:
            from .executor_kitchen import KitchenExecutor
            ex = KitchenExecutor(log=lambda *a: None, drive=False)
            time.sleep(1.0)
        for name, xyz in sc.items():
            ex._rig(op="move", obj=name, xyz=xyz)
        time.sleep(1.5)
        print(f"scene {i}: {sc}", flush=True)
        request_survey(d, ex=ex)
        world, views, raw = perceive(d, log=lambda m: print(m, flush=True))
        overlay(d, views, raw)
        g, extra = grade(d, world, log=lambda m: print(m, flush=True))
        for r in g:
            rows.append({"scene": i, **r})
    found = [r for r in rows if r.get("found")]
    errs = np.array([r["err_m"] for r in found]) if found else np.array([np.nan])
    summary = {"scenes": len(SCENES), "objects": len(rows), "found": len(found),
               "median_err_cm": round(float(np.median(errs)) * 100, 1),
               "max_err_cm": round(float(np.max(errs)) * 100, 1),
               "within_5cm": int((errs < 0.05).sum()), "rows": rows}
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps({k: v for k, v in summary.items() if k != "rows"}))
    print("->", out)


if __name__ == "__main__":
    main()
