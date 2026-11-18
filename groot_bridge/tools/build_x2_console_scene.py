"""Build a sim scene: X2 + OmniHand (finger collisions ON) + a solid console with a red push button.

The gwbc-x2 bridge loads it with --sim-mjcf <xml> --sim-with-omnihand (it accepts a scene that
already holds the canonical OmniHand joints). The same XML is the head-camera render model, so the
model sees the fingers and console exactly as simulated. gwbc-x2's composer is used at run time.

  python -m groot_bridge.tools.build_x2_console_scene --out runs/groot_bridge/scenes/x2_console.xml \
      --console-x 0.75
Console geometry is in the robot's spawn frame (robot at the origin facing +x).
"""
import argparse
import json
import sys
from pathlib import Path

import mujoco

GWBC_X2 = Path.home() / "Projects/gwbc-x2"
sys.path.insert(0, str(GWBC_X2))

# console: desk top at 0.80 m (G1 Factory-ish waist height for a 1.3 m robot), button on top
CONSOLE = dict(half=(0.22, 0.40, 0.40), top_z=0.80)
BUTTON = dict(half=(0.05, 0.05, 0.015), travel=0.012)


def build_pillar(spec, x, y=-0.10, z=0.82):
    """G1-Factory-PushButton-like: grey pillar, yellow box on its face, red button pressed along -x.
    x = button front face distance ahead of spawn. From G1-Factory-Benchmark LMPushButton: the G1 walks
    ~0.83 m, then presses with the right wrist 0.29 m ahead / 0.12 m right / 0.13 m above the pelvis."""
    wb = spec.worldbody
    pil = wb.add_body(); pil.name = "pillar"; pil.pos = (x + 0.06 + 0.15, 0.0, 0.9)
    g = pil.add_geom(); g.name = "pillar_box"; g.type = mujoco.mjtGeom.mjGEOM_BOX
    g.size = (0.15, 0.15, 0.9); g.rgba = (0.6, 0.62, 0.64, 1)
    ybox = wb.add_body(); ybox.name = "button_box"; ybox.pos = (x + 0.03, y, z)
    g = ybox.add_geom(); g.name = "button_box_geom"; g.type = mujoco.mjtGeom.mjGEOM_BOX
    g.size = (0.03, 0.07, 0.09); g.rgba = (0.95, 0.75, 0.1, 1)
    btn = wb.add_body(); btn.name = "button"; btn.pos = (x - 0.012, y, z)
    j = btn.add_joint(); j.name = "button_press"; j.type = mujoco.mjtJoint.mjJNT_SLIDE; j.axis = (1, 0, 0)
    j.range = (0.0, BUTTON["travel"]); j.limited = mujoco.mjtLimited.mjLIMITED_TRUE
    j.stiffness = [400.0, 0.0, 0.0]
    bg = btn.add_geom(); bg.name = "button_cap"; bg.type = mujoco.mjtGeom.mjGEOM_CYLINDER
    bg.size = (0.035, 0.012, 0); bg.quat = (0.7071068, 0, 0.7071068, 0); bg.rgba = (0.85, 0.08, 0.08, 1); bg.mass = 0.05


BOX = dict(half=(0.11, 0.085, 0.09), mass=0.4, friction=1.2)   # 0.22 deep x 0.17 wide x 0.18 tall
TABLE = dict(top_z=0.75, half=(0.30, 0.50, 0.02))


def build_boxlift(spec, box_x):
    """G1-Factory-BoxLift-like: white table, cardboard box (free joint). From G1-Factory-Benchmark
    LMBoxLift (40 episodes): the G1 walks 1.55 m, squeezes the box with open palms, wrists 0.234 m apart,
    0.29 m ahead / 0.16 m above the pelvis, and lifts ~8 cm. box_x = box centre, m ahead of spawn."""
    wb = spec.worldbody
    hx, hy, hz = TABLE["half"]
    tab = wb.add_body(); tab.name = "table"; tab.pos = (box_x + 0.12, 0.0, TABLE["top_z"] - hz)
    g = tab.add_geom(); g.name = "table_top"; g.type = mujoco.mjtGeom.mjGEOM_BOX
    g.size = (hx, hy, hz); g.rgba = (0.93, 0.93, 0.91, 1)
    leg = tab.add_geom(); leg.name = "table_leg"; leg.type = mujoco.mjtGeom.mjGEOM_BOX
    leg.size = (0.04, 0.04, (TABLE["top_z"] - 2 * hz) / 2); leg.pos = (0.15, 0, -(TABLE["top_z"]) / 2)
    leg.rgba = (0.8, 0.8, 0.8, 1)
    bx, by, bz = BOX["half"]
    box = wb.add_body(); box.name = "box"; box.pos = (box_x, 0.0, TABLE["top_z"] + bz + 0.002)
    fj = box.add_freejoint(); fj.name = "box_free"
    bg = box.add_geom(); bg.name = "box_geom"; bg.type = mujoco.mjtGeom.mjGEOM_BOX
    bg.size = (bx, by, bz); bg.rgba = (0.72, 0.55, 0.36, 1); bg.mass = BOX["mass"]
    bg.friction = (BOX["friction"], 0.01, 0.001)


def build(console_x: float, layout: str = "pillar"):
    from gear_sonic.scripts.compose_x2_with_omnihand import build_x2_with_omnihand_spec
    spec, _, _ = build_x2_with_omnihand_spec(disable_hand_collisions=False)
    if layout == "pillar":
        build_pillar(spec, console_x)
        return spec
    if layout == "boxlift":
        build_boxlift(spec, console_x)
        return spec
    wb = spec.worldbody
    hx, hy, hz = CONSOLE["half"]
    desk = wb.add_body(); desk.name = "console"; desk.pos = (console_x + hx, 0.0, CONSOLE["top_z"] - hz)
    g = desk.add_geom(); g.name = "console_box"; g.type = mujoco.mjtGeom.mjGEOM_BOX
    g.size = (hx, hy, hz); g.rgba = (0.55, 0.57, 0.6, 1)
    # button: a body on a vertical slide joint with a spring, sitting on the desk top near its front edge
    bx, by, bz = BUTTON["half"]
    btn = wb.add_body(); btn.name = "button"
    btn.pos = (console_x + 0.12, 0.0, CONSOLE["top_z"] + bz)
    j = btn.add_joint(); j.name = "button_press"; j.type = mujoco.mjtJoint.mjJNT_SLIDE; j.axis = (0, 0, 1)
    j.range = (-BUTTON["travel"], 0.0); j.limited = mujoco.mjtLimited.mjLIMITED_TRUE
    j.stiffness = [400.0, 0.0, 0.0]; j.damping = [5.0, 0.0, 0.0] if hasattr(j.damping, "__len__") else 5.0
    bg = btn.add_geom(); bg.name = "button_cap"; bg.type = mujoco.mjtGeom.mjGEOM_BOX
    bg.size = (bx, by, bz); bg.rgba = (0.85, 0.08, 0.08, 1); bg.mass = 0.05
    return spec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--console-x", type=float, default=1.25, help="button face (pillar) / desk front (desk), m ahead of spawn")
    ap.add_argument("--layout", choices=["pillar", "desk", "boxlift"], default="pillar")
    ap.add_argument("--box-width", type=float, default=None, help="boxlift: box width (m, across the hands)")
    ap.add_argument("--box-mass", type=float, default=None)
    ap.add_argument("--box-friction", type=float, default=None)
    a = ap.parse_args()
    if a.box_width:
        BOX["half"] = (BOX["half"][0], a.box_width / 2, BOX["half"][2])
    if a.box_mass:
        BOX["mass"] = a.box_mass
    if a.box_friction:
        BOX["friction"] = a.box_friction
    spec = build(a.console_x, a.layout)
    model = spec.compile()                                   # check it compiles before writing
    # meshes are relative to gwbc-x2's mjcf dir: pin it so the XML loads from anywhere (docker mounts $HOME)
    rd = GWBC_X2 / "gear_sonic/data/assets/robot_description"
    mjcf_dir = rd / "mjcf"
    index = {}
    for f in rd.rglob("*"):
        if f.suffix.lower() in (".stl", ".obj", ".msh", ".png"):
            index.setdefault(f.name, f)
    bases = [mjcf_dir / (spec.meshdir or ""), mjcf_dir]
    for m in spec.meshes:
        if not m.file or Path(m.file).is_absolute():
            continue
        hit = next((b / m.file for b in bases if (b / m.file).is_file()), None) or index.get(Path(m.file).name)
        if hit is None:
            raise FileNotFoundError(m.file)
        m.file = str(hit.resolve())
    spec.meshdir = ""
    a.out.parent.mkdir(parents=True, exist_ok=True)
    xml = spec.to_xml()
    a.out.write_text(xml)
    m2 = mujoco.MjModel.from_xml_path(str(a.out))            # and that the written file loads
    meta = {"box": BOX, "table": TABLE, "layout": a.layout, "console_x": a.console_x, "console": CONSOLE, "button": BUTTON,
            "nq": int(m2.nq), "njnt": int(m2.njnt), "ngeom": int(m2.ngeom)}
    a.out.with_suffix(".meta.json").write_text(json.dumps(meta, indent=1))
    if a.layout == "boxlift":       # bridge sidecar (<mjcf>.json): publish the box pose on scene_state
        a.out.with_suffix(".json").write_text(json.dumps({"object_freejoint_map": {"box": "box_free"},
                                                          "object_welded_map": {"table": "table"}}, indent=1))
    print(f"wrote {a.out}  nq={model.nq} -> reload nq={m2.nq}, joints {m2.njnt}, geoms {m2.ngeom}")


if __name__ == "__main__":
    main()
