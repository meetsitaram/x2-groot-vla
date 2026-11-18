"""Model-agnostic embodiment adapter: make a robot look like the embodiment a frozen VLA was trained on.

    robot obs --[ObsEncoder]--> model obs --> frozen VLA (e.g. GR00T G1) --> model action chunk
              --[ActionDecoder]--> robot command chunk --> robot controller (X2: SONIC)

Every stage is one of:
  analytic  exact maps (joint permutation + offset + sign, velocity scaling)
  learned   a small trained head (e.g. camera-view warp, hand synergy) -- the frozen core never trains
  hybrid    analytic map + learned residual (Any2Any style: frozen core, learned enc/dec heads)

A new VLA release (GR00T 2) only needs a new TargetSpec (keys, dims, ranges, horizon) and, if its
action space changes, a new ActionDecoder. The robot side (RobotState / RobotCommand / the X2 IO)
does not change.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

import numpy as np


# --------------------------------------------------------------------------- the frozen model's side
@dataclass
class TargetSpec:
    """What the frozen VLA expects and returns, read from its checkpoint (no hand-typed numbers)."""
    name: str                          # e.g. "G1-Factory-GR00T-1.7"
    embodiment_tag: str                # tag the checkpoint was trained under
    video_keys: list[str]
    image_hw: tuple[int, int]
    state_keys: dict[str, int]         # key -> dim, in model order
    action_keys: dict[str, int]
    action_horizon: int
    control_hz: float
    state_range: dict[str, tuple[np.ndarray, np.ndarray]] = field(default_factory=dict)
    action_range: dict[str, tuple[np.ndarray, np.ndarray]] = field(default_factory=dict)
    language_key: str = "annotation.human.task_description"

    @classmethod
    def from_gr00t_checkpoint(cls, ckpt: Path, name: str | None = None, image_hw=(480, 640), control_hz=50.0):
        import yaml
        ckpt = Path(ckpt)
        conf = yaml.safe_load((ckpt / "experiment_cfg/conf.yaml").read_text())
        tag = conf["data"]["datasets"][0]["embodiment_tag"] if "datasets" in conf.get("data", {}) else None
        mc = _find_key(conf, "modality_configs")
        tag = tag or next(iter(mc))
        m = mc[tag]
        stats = json.loads((ckpt / "experiment_cfg/dataset_statistics.json").read_text())[tag]
        rng = lambda grp, k: (np.asarray(stats[grp][k]["min"], np.float32), np.asarray(stats[grp][k]["max"], np.float32))
        return cls(
            name=name or ckpt.name, embodiment_tag=tag,
            video_keys=list(m["video"]["modality_keys"]), image_hw=tuple(image_hw),
            state_keys={k: len(stats["state"][k]["min"]) for k in m["state"]["modality_keys"]},
            action_keys={k: len(stats["action"][k]["min"]) for k in m["action"]["modality_keys"]},
            action_horizon=len(m["action"]["delta_indices"]), control_hz=control_hz,
            state_range={k: rng("state", k) for k in m["state"]["modality_keys"]},
            action_range={k: rng("action", k) for k in m["action"]["modality_keys"]},
            language_key=m["language"]["modality_keys"][0])


def _find_key(d, key):
    if isinstance(d, dict):
        if key in d:
            return d[key]
        for v in d.values():
            r = _find_key(v, key)
            if r is not None:
                return r
    return None


# --------------------------------------------------------------------------- the robot's side
@dataclass
class RobotState:
    """One robot observation, in the robot's own conventions."""
    t: float
    joint_q: np.ndarray                # robot joint order (X2: 31, mujoco order)
    left_hand_q: np.ndarray
    right_hand_q: np.ndarray
    images: dict[str, np.ndarray]      # name -> HxWx3 uint8 RGB
    base_quat_wxyz: np.ndarray | None = None


@dataclass
class RobotCommand:
    """One control step for the robot's whole-body controller (X2: SONIC via the kplanner)."""
    left_arm_q: np.ndarray             # 7, robot order
    right_arm_q: np.ndarray            # 7
    left_hand_q: np.ndarray            # robot hand dims (X2 OmniHand: 10)
    right_hand_q: np.ndarray
    base_velocity: np.ndarray          # [forward m/s, left m/s, yaw rad/s]
    hip_height_m: float
    waist_q: np.ndarray | None = None  # 3, robot order (yaw, pitch, roll on X2); None = controller default


# --------------------------------------------------------------------------- the adapter stages
class ObsEncoder(Protocol):
    def __call__(self, state: RobotState, instruction: str) -> dict: ...


class ActionDecoder(Protocol):
    def __call__(self, action: dict[str, np.ndarray], state: RobotState) -> list[RobotCommand]: ...


@dataclass
class EmbodimentAdapter:
    spec: TargetSpec
    encode: ObsEncoder
    decode: ActionDecoder
