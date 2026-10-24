"""Gemma <-> X2 agent loop (Ollama /api/chat with native tool calling).

  # dry run (prints wire JSON, no robot/sim needed)
  python -m gemma_ctl.agent --say "turn around and wave"
  # live against simstack_local.sh (isolated 56xx ports)
  python -m gemma_ctl.agent --live --ports simstack
  # interactive REPL: omit --say

Stdlib HTTP only (urllib), so it runs in the WBC .venv or on the Jetson.
"""
from __future__ import annotations

import argparse
import json
import time
import urllib.request
from pathlib import Path

from . import tools as T
from .executor import PORTS_DEFAULT, PORTS_SIMSTACK, Executor

OLLAMA = "http://127.0.0.1:11434"
MAX_STEPS = 12


def chat(model: str, messages: list, tools: list, host: str = OLLAMA,
         num_ctx: int = 16384) -> dict:
    body = json.dumps({"model": model, "messages": messages, "tools": tools,
                       "stream": False, "think": False,
                       "options": {"temperature": 0.0, "num_ctx": num_ctx}}).encode()
    req = urllib.request.Request(f"{host}/api/chat", data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as r:
        return json.loads(r.read())


class Agent:
    def __init__(self, model: str, executor, places: dict,
                 host: str = OLLAMA, log=print, tools=None, system=None):
        self.model, self.ex, self.host, self.log = model, executor, host, log
        self.tools = tools or T.tool_schemas(places)
        sys_prompt = system or T.SYSTEM_PROMPT
        if places and system is None:
            sys_prompt += "\nKnown places: " + ", ".join(sorted(places)) + "."
        self.messages = [{"role": "system", "content": sys_prompt}]
        self.trace: list[dict] = []

    def run(self, user_text: str, source: str = "script") -> str:
        self.messages.append({"role": "user", "content": user_text})
        ev = getattr(self.ex, "log_event", None)
        if ev:
            ev("user", user_text, source)
        if hasattr(self.ex, "caption"):
            self.ex.caption(user_text)
        for _ in range(MAX_STEPS):
            t0 = time.monotonic()
            resp = chat(self.model, self.messages, self.tools, self.host)
            dt = time.monotonic() - t0
            msg = resp["message"]
            self.messages.append(msg)
            calls = msg.get("tool_calls") or []
            if not calls:
                # Gemma 4 occasionally leaks its thinking marker ("thought") as the
                # whole reply after a tool result: ask once more for the sentence
                txt = (msg.get("content") or "").strip()
                if txt.lower().startswith("thought") or "<channel|>" in txt:
                    import re as _re
                    rest = _re.sub(r"^\s*thought\s*", "", txt, flags=_re.I)
                    rest = rest.split("<channel|>")[-1].strip()
                    if not rest:
                        self.messages.append({"role": "user", "content": "(Reply to the user in one short sentence.)"})
                        resp = chat(self.model, self.messages, [], self.host)
                        rest = (resp["message"].get("content") or "").strip()
                    msg["content"] = rest
                self.trace.append({"reply": msg.get("content", ""), "llm_s": dt})
                if ev:
                    ev("gemma", msg.get("content", ""))
                return msg.get("content", "")
            for c in calls:
                fn = c["function"]
                name, args = fn["name"], fn.get("arguments") or {}
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except ValueError:
                        args = {}
                self.log(f"→ {name}({json.dumps(args)})   [llm {dt:.2f}s]")
                if ev:
                    ev("call", f"{name}({', '.join(f'{k}={v}' for k, v in args.items())})  [{dt:.1f} s]")
                result = self.ex.call(name, args)
                if ev:
                    ev("result", self.ex.summarize(result) if hasattr(self.ex, "summarize") else str(result)[:150])
                self.log(f"  ← {json.dumps(result)}")
                self.trace.append({"tool": name, "args": args, "result": result,
                                   "llm_s": dt})
                self.messages.append({"role": "tool", "tool_name": name,
                                      "content": json.dumps(result)})
        self.ex.call("stop", {})
        return "(step limit reached; stopped)"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gemma4:e4b")
    ap.add_argument("--host", default=OLLAMA)
    ap.add_argument("--live", action="store_true",
                    help="publish to the kplanner (default: dry run)")
    ap.add_argument("--ports", choices=["default", "simstack"], default="simstack")
    ap.add_argument("--places", type=Path,
                    default=Path(__file__).with_name("places_mujoco.json"))
    ap.add_argument("--perception", action="store_true",
                    help="kitchen rig, tools run on Gemma's own camera perception (world model)")
    ap.add_argument("--kitchen", action="store_true",
                    help="drive the IsaacLab kitchen rig (KP_GEMMA=1) instead of the kplanner")
    ap.add_argument("--real", action="store_true",
                    help="real-robot first milestone: look + gestures only (gemma_ctl/real_robot.py)")
    ap.add_argument("--pc2-host", default=None, help="--real: robot IP (omit = local sim stack)")
    ap.add_argument("--image", default=None, help="--real: image for look() when no live camera")
    ap.add_argument("--transcript", type=Path, default=None)
    ap.add_argument("--task", type=Path, default=None,
                    help="fixed task file (gemma_ctl/tasks/*.json): opening, request, one scripted "
                         "clarification if Gemma asks, then the check question")
    ap.add_argument("--say", action="append",
                    help="one-shot utterance(s); omit for an interactive REPL")
    a = ap.parse_args()

    if a.real:
        from .real_robot import RealExecutor, real_system, real_tool_schemas
        ex = RealExecutor("robot" if a.pc2_host else "sim", a.pc2_host, live=a.live,
                          image=a.image, model=a.model)
        print(f"[real] target={ex.target} live={ex.live} runs -> {ex.run_dir}")
        agent = Agent(a.model, ex, {}, a.host, tools=real_tool_schemas(ex.target),
                      system=real_system(ex.target))
    elif a.perception:
        from .executor_perception import PERCEPTION_SYSTEM, PerceptionExecutor, perception_tool_schemas
        ex = PerceptionExecutor()
        for _ in range(100):
            if ex._st() and ex._p():
                break
            time.sleep(0.2)
        agent = Agent(a.model, ex, {}, a.host, tools=perception_tool_schemas(ex.waypoints),
                      system=PERCEPTION_SYSTEM)
    elif a.kitchen:
        from .executor_kitchen import KITCHEN_SYSTEM, KitchenExecutor, kitchen_tool_schemas
        ex = KitchenExecutor(dry_run=not a.live)
        objs = (ex._st() or {}).get("objects", {}) if a.live else {"mug": 0, "can": 0}
        if a.live:
            for _ in range(100):
                if ex._st() and ex._p():
                    break
                time.sleep(0.2)
            objs = ex._st()["objects"]
        agent = Agent(a.model, ex, {}, a.host,
                      tools=kitchen_tool_schemas(ex.catalog, ex.waypoints, objs),
                      system=KITCHEN_SYSTEM)
    else:
        places = T.load_places(a.places)
        ex = Executor(PORTS_SIMSTACK if a.ports == "simstack" else PORTS_DEFAULT,
                      dry_run=not a.live, places=places)
        agent = Agent(a.model, ex, places, a.host)
    interrupted = False
    try:
        if a.task:
            task = json.loads(a.task.read_text())
            if getattr(ex, "log_event", None):
                ex.log_event("task", f"task {task['name']}: {task['request']}")
            for key in ("opening", "request"):
                if task.get(key):
                    print(f"\nuser> {task[key]}")
                    r = agent.run(task[key]); print(f"gemma> {r}")
            if task.get("clarify") and r.strip().endswith("?"):
                print(f"\nuser> {task['clarify']}")
                r = agent.run(task["clarify"]); print(f"gemma> {r}")
            if task.get("check"):
                print(f"\nuser> {task['check']}")
                print(f"gemma> {agent.run(task['check'])}")
        elif a.say:
            for s in a.say:
                print(f"\nuser> {s}")
                print(f"gemma> {agent.run(s)}")
        else:
            while True:
                try:
                    s = input("\nuser> ").strip()
                except EOFError:
                    break
                if s:
                    print(f"gemma> {agent.run(s, source='typed')}")
    except KeyboardInterrupt:
        interrupted = True
    finally:
        # gesture/dance mode: a normal exit lets the playing clip finish; Ctrl-C stops it
        if not interrupted and hasattr(ex, "wait_idle"):
            try:
                ex.wait_idle()
            except KeyboardInterrupt:
                pass
        ex.call("stop", {})
        ex.close()
        if a.transcript:
            a.transcript.parent.mkdir(parents=True, exist_ok=True)
            a.transcript.write_text(json.dumps(agent.trace, indent=1, default=str))


if __name__ == "__main__":
    main()
