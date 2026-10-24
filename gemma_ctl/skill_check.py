"""Drive KitchenExecutor tools directly (no LLM) against a live kitchen rig.

  python -m gemma_ctl.skill_check turn:left:90 walk:forward:0.5 pick:mug place:sink_counter
"""
import json
import sys
import time

from .executor_kitchen import KitchenExecutor


def main():
    ex = KitchenExecutor()
    for _ in range(200):
        if ex._p() and ex._st():
            break
        time.sleep(0.2)
    print("start pose", ex._p(), flush=True)
    for spec in sys.argv[1:]:
        name, *a = spec.split(":")
        args = {"turn": lambda: {"direction": a[0], "degrees": float(a[1])},
                "walk": lambda: {"direction": a[0], "meters": float(a[1])},
                "pick": lambda: {"object": a[0]},
                "place": lambda: {"location": a[0]},
                "go_to": lambda: {"place": a[0]},
                "look": lambda: {}}[name]()
        t0 = time.time()
        ex.caption(f"skill check: {spec}")
        r = ex.call(name, args)
        print(f"{spec}: {json.dumps(r)}  ({time.time() - t0:.0f}s wall, pose {ex._p()})", flush=True)
    ex.close()


if __name__ == "__main__":
    main()
