"""Gemma tool vocabulary for the X2 (closed enums only — the model never sees
joint angles, velocities, ports or the shutdown/estop channel).

Each tool maps onto one pc2_kplanner_onnx.py input channel; the executor
(executor.py) owns the translation, clamps and feedback loops.
"""
from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent

# Gesture clips present in ~/x2_cloud_checkpoints/dances_x2m2 (motion_clip_cmd).
GESTURES = {
    "wave": "right_wave_001",
    "high_five": "right_five_001",
    "blow_kiss": "right_kiss_001",
    "handshake": "right_shake_001",
    "heart": "both_heart_002",
    "turn_and_wave_left": "turn_wave_left_001",
    "turn_and_wave_right": "turn_wave_right_001",
    "dance": "victory_dance_macarena_R_001__A306",
    "shadow_box": "shadow_boxing_R_001__A359",
}

# Posture primitives in gear_sonic/data/motions/x2_planner_primitives.pkl
# (planner_cmd intent "primitive": ENTER -> HOLD at last frame -> EXIT).
POSTURES = {
    "crouch_small": "crouch_small", "crouch": "crouch_medium",
    "crouch_deep": "crouch_large",
    "lean_forward": "lean_fwd_medium", "lean_left": "lean_left_medium",
    "lean_right": "lean_right_medium",
    "twist_left": "torso_left_30deg", "twist_right": "torso_right_30deg",
}

MAX_WALK_M = 3.0          # per call; the model re-plans for longer routes
MAX_TURN_DEG = 180


def load_places(path: Path | None) -> dict:
    if path is None or not Path(path).exists():
        return {}
    return json.loads(Path(path).read_text())


def _fn(name, desc, props=None, required=()):
    return {"type": "function", "function": {
        "name": name, "description": desc,
        "parameters": {"type": "object", "properties": props or {},
                       "required": list(required)}}}


def tool_schemas(places: dict) -> list[dict]:
    tools = [
        _fn("stop", "Stop all motion immediately and stand still."),
        _fn("walk", "Walk straight forward or backward by a distance in metres "
            f"(max {MAX_WALK_M} per call).",
            {"direction": {"type": "string", "enum": ["forward", "backward"]},
             "meters": {"type": "number", "description": f"0.1 to {MAX_WALK_M}"}},
            ["direction", "meters"]),
        _fn("sidestep", "Step sideways left or right by a small distance.",
            {"direction": {"type": "string", "enum": ["left", "right"]},
             "meters": {"type": "number", "description": "0.1 to 1.0"}},
            ["direction", "meters"]),
        _fn("turn", "Turn in place left (counter-clockwise) or right by degrees. "
            "'turn around' = 180.",
            {"direction": {"type": "string", "enum": ["left", "right"]},
             "degrees": {"type": "number", "description": f"5 to {MAX_TURN_DEG}"}},
            ["direction", "degrees"]),
        _fn("gesture", "Perform a social gesture or dance clip.",
            {"name": {"type": "string", "enum": sorted(GESTURES)}}, ["name"]),
        _fn("posture", "Hold a body posture (crouch, lean, twist torso). "
            "Use 'stand' to return to normal standing.",
            {"name": {"type": "string", "enum": sorted(POSTURES) + ["stand"]}},
            ["name"]),
        _fn("look", "Turn the head left/right to look, or 'center'.",
            {"direction": {"type": "string", "enum": ["left", "right", "center"]}},
            ["direction"]),
        _fn("get_state", "Report the robot's position, heading, and whether it "
            "is upright."),
    ]
    if places:
        tools.append(_fn(
            "go_to", "Walk to a named place.",
            {"place": {"type": "string", "enum": sorted(places)}}, ["place"]))
    tools.append(_fn(
        "pick", "Pick up an object from a surface at a given height tier "
        "(low=floor/bench, medium=counter/table, high=shelf).",
        {"object": {"type": "string"},
         "height": {"type": "string", "enum": ["low", "medium", "high"]}},
        ["object", "height"]))
    return tools


SYSTEM_PROMPT = """You control a humanoid robot (AgiBot X2) by calling tools.
Rules:
- Translate the user's request into the smallest sequence of tool calls.
- Call tools one step at a time; wait for each result before the next.
- Distances are metres; 1 foot = 0.3 m, one step = 0.3 m. "Turn around" = 180 degrees.
- If the request is unsafe, ambiguous, or impossible with the tools, ask a short
  question or explain instead of guessing. Never invent places or gestures.
- If the user says stop / halt / freeze / wait, call stop immediately.
- After the task is done, reply with one short sentence."""
