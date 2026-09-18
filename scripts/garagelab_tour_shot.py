"""Garagelab cinematic tour — posed X2 mannequins + waypoint drone camera.

Extends garagelab_orbit_shot.py: spawns several X2s in fixed poses (stand /
crouch / sit — roots pinned every step, no locomotion policy needed), paints
them with the approved MJCF colours (grey 0.03 / white 0.7 / orange feet),
and flies a Catmull-Rom camera path through user-defined waypoints:
roof descent to the center X2 -> couch run -> white table -> computer desk ->
west run to the garage door over the plant -> zoom onto the center X2's head.

Run (env_isaaclab python, from x2-groot-vla):
    /home/stickbot/miniconda3/envs/env_isaaclab/bin/python -u \
        scripts/garagelab_tour_shot.py \
        --scene-usdz assets/garagelab/garagelab_splat.usdz \
        --out-dir /tmp/tour --headless --enable_cameras [--probe]
"""

import argparse
import functools
import math
import os
import sys
import threading

print = functools.partial(print, flush=True)

GROOT_REPO = "/home/stickbot/Projects/GR00T-WholeBodyControl"
sys.path.insert(0, GROOT_REPO)
_LAUNCH_CWD = os.getcwd()
os.chdir(GROOT_REPO)  # gear_sonic asset paths are repo-relative

from isaaclab.app import AppLauncher  # noqa: E402

parser = argparse.ArgumentParser(description="garagelab X2 tour capture")
parser.add_argument("--scene-usdz", type=str, required=True)
parser.add_argument("--collision-usd", type=str, default=None,
                    help="collision mesh USD; robots' limbs rest on it during the "
                         "settle (roots are pinned throughout — they cannot fall)")
parser.add_argument("--out-dir", type=str, required=True)
parser.add_argument("--chores-pkl", type=str, default=None,
                    help="pico chores eval pkl; 'executed' clips are replayed "
                         "kinematically (dof only, root pinned) on the standing X2s")
parser.add_argument("--middle-x2m2", type=str, default=None,
                    help="deploy demo-bank .x2m2 motion looped on the 'middle' X2 "
                         "(e.g. victory_dance_macarena) instead of a chores clip")
parser.add_argument("--sofa-wave-x2m2", type=str, default=None,
                    help=".x2m2 gesture whose RIGHT-ARM channels are looped on the "
                         "sitting sofa X2 (rest of the sit pose kept)")
parser.add_argument("--x2m2", action="append", default=[], metavar="NAME=PATH",
                    help="per-robot demo-bank .x2m2 loop override (repeatable), "
                         "e.g. middle=<macarena.x2m2> south=<shadow_boxing.x2m2>")
parser.add_argument("--free-idle-x2m2", type=str, default=None,
                    help="with --free-roots: FREE robots track this idle loop "
                         "(easier to balance than chores); fallers auto-reset")
parser.add_argument("--focal", type=float, default=15.0,
                    help="camera focal length mm (default aperture 20.955: "
                         "15 = ~70 deg wide-angle, 24 = ~47 deg normal)")
parser.add_argument("--free-roots", action="store_true",
                    help="release the garagelab standers' root locks: floating "
                         "base + stiff PD tracking of the clips on the collision "
                         "mesh (sofa stays seated; grid robots stay fixed-base)")
parser.add_argument("--grid", type=int, default=0,
                    help="N: spawn a (2N+1)^2-1 training-farm grid of chore-"
                         "replaying X2s around the splat on a visible ground "
                         "plane, and prepend a wide dive-in intro shot")
parser.add_argument("--grid-spacing", type=float, default=7.0)
parser.add_argument("--fps", type=int, default=24)
parser.add_argument("--width", type=int, default=1280)
parser.add_argument("--height", type=int, default=720)
parser.add_argument("--accum", type=int, default=12,
                    help="sim.render() calls per frame (RTX sample accumulation)")
parser.add_argument("--probe", action="store_true",
                    help="render one frame per waypoint + midpoints instead of the full path")
parser.add_argument("--eye-z", type=float, default=1.83,
                    help="cruise camera height (waypoints scale to it; 1.83 = 6 ft)")
parser.add_argument("--dome-intensity", type=float, default=300.0)
parser.add_argument("--key-light", type=float, nargs=3, default=[300.0, 40.0, 3000.0],
                    metavar=("AZ_DEG", "EL_DEG", "INTENSITY"))
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import numpy as np  # noqa: E402
import torch  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.assets import Articulation  # noqa: E402
from isaaclab.sim import SimulationContext  # noqa: E402
from isaaclab.sim.spawners.from_files import UsdFileCfg  # noqa: E402
from isaaclab.sensors import Camera, CameraCfg  # noqa: E402
from gear_sonic.envs.manager_env.robots.x2_ultra import (  # noqa: E402
    make_x2_ultra_cfg, X2_ULTRA_ISAACLAB_TO_MUJOCO_DOF)

# ---------------------------------------------------------------- scene cast
# World-frame layout (from the orbit probe): white folding table + door north,
# computer desk NW, shelves/bench west, garage door + plant south(-west),
# couch east facing west, tan rug center.
# User-marked placements (heightmap markup 2026-09-18): all standing, facing
# their nearest wall, except "middle" (the roomiest spot) which faces the room.
# Sofa sitter kept per approved side-view, with a slight backward torso lean.
ROBOTS = [
    # name, (x, y, pelvis_z, yaw_deg), pose overrides {joint regex: rad}
    ("desk", (-1.90, 2.80, 0.66, 90.0), {}),        # computer desk, faces N
    ("table", (-0.55, 2.80, 0.66, 90.0), {}),       # white folding table, faces N
    ("middle", (-0.85, 1.80, 0.66, -90.0), {}),     # roomiest spot, faces the room (S)
    ("bench", (-1.95, 1.60, 0.66, 180.0), {}),      # west bench, faces W
    ("west", (-1.10, 0.05, 0.66, 180.0), {}),       # mid-west floor, faces W
    ("south", (0.83, -0.66, 0.66, -90.0), {}),      # south floor, faces S (garage door)
    ("sofa", (1.00, 1.95, 0.56, 180.0), {           # sits on the sofa's front edge
        ".*_hip_pitch_joint": -1.45, ".*_knee_joint": 1.40,
        ".*_ankle_pitch_joint": 0.05, ".*waist_pitch_joint": -0.10,
    }),
]

# Camera waypoints: (t_seconds, eye_xyz, look_xyz). One smooth counter-clockwise
# perimeter lap (couch east side -> north tables -> west side to the plant ->
# south arc), never staring at the blank garage door. Z values equal to 1.83
# are rescaled to --eye-z so the cruise height is one knob.
WAYPOINTS = [
    (0.0, (-0.30, -1.60, 2.40), (-0.85, 1.75, 1.05)),   # high south, room + middle X2
    (3.7, (-0.30, -1.55, 1.70), (-0.85, 1.80, 1.20)),   # settled, facing middle X2
    (6.7, (0.25, -1.00, 1.70), (1.30, 0.30, 1.00)),     # swung right toward couch south
    (11.0, (0.55, 0.30, 1.70), (1.35, 1.60, 0.95)),     # north along the couch
    (14.0, (-0.35, 1.15, 1.70), (1.00, 1.95, 0.90)),    # sitting (waving) X2, pulled back
    (14.8, (-0.10, 1.60, 1.70), (0.60, 2.90, 1.10)),    # gaze turning LEFT over couch end/door
    (15.6, (0.20, 1.95, 1.70), (-0.75, 1.80, 1.25)),    # ...continuing left onto the dancer
    (17.3, (0.60, 2.60, 1.76), (-0.30, 3.00, 1.20)),    # NE turn, gaze onto the tables
    (19.5, (-0.50, 2.55, 1.70), (-0.70, 3.15, 1.15)),   # over the white table X2
    (23.5, (-1.10, 1.80, 1.70), (-2.10, 3.00, 1.10)),   # computer desk, zoomed out
    (27.0, (-1.50, 1.45, 1.70), (-2.20, 1.65, 1.10)),   # left turn past X2's left, bench X2
    (31.5, (-1.60, -0.30, 1.70), (0.83, -0.66, 1.10)),  # south run, brief pass over south X2
    (33.5, (-1.55, -1.30, 1.70), (-0.10, 0.20, 1.20)),  # at the plant, gaze already inward
    (37.0, (-1.20, -1.10, 1.70), (-0.85, 1.80, 1.30)),  # turned back to the middle X2
    (42.5, (-0.95, 0.60, 1.55), (-0.85, 1.80, 1.35)),   # zoom onto the middle X2's face
]
LOOK_MIN_DIST = 0.9  # m; keeps the gaze from diving at the floor when the look
                     # target's spline path crosses under the camera

X2_LINK_COLORS = {
    "pelvis": (0.298, 0.298, 0.298), "waist_yaw_link": (0.298, 0.298, 0.298),
    "waist_pitch_link": (0.298, 0.298, 0.298), "torso_link": (1.0, 1.0, 1.0),
    "head_yaw_link": (0.867, 0.867, 0.89), "head_pitch_link": (0.2, 0.2, 0.2),
    "imu_in_torso_link": (0.647, 0.62, 0.588), "imu_in_head_link": (0.647, 0.62, 0.588),
    "RGBD_in_head_link": (0.647, 0.62, 0.588), "RGB_binocular_link": (0.267, 0.267, 0.267),
    "RGB_rear_link": (0.867, 0.867, 0.89), "RGB_Forward_link": (0.502, 0.502, 0.502),
}
for _side in ("left", "right"):
    X2_LINK_COLORS.update({
        f"{_side}_hip_pitch_link": (0.298, 0.298, 0.298),
        f"{_side}_hip_roll_link": (1.0, 1.0, 1.0),
        f"{_side}_hip_yaw_link": (1.0, 1.0, 1.0),
        f"{_side}_knee_link": (1.0, 1.0, 1.0),
        f"{_side}_ankle_pitch_link": (0.298, 0.298, 0.298),
        f"{_side}_ankle_roll_link": (1.0, 0.5, 0.0),          # the orange feet
        f"{_side}_shoulder_pitch_link": (0.298, 0.298, 0.298),
        f"{_side}_shoulder_roll_link": (1.0, 1.0, 1.0),
        f"{_side}_shoulder_yaw_link": (1.0, 1.0, 1.0),
        f"{_side}_elbow_link": (1.0, 1.0, 1.0),
        f"{_side}_wrist_yaw_link": (1.0, 1.0, 1.0),
        f"{_side}_wrist_pitch_link": (0.3, 0.3, 0.3),
        f"{_side}_wrist_roll_link": (0.898, 0.918, 0.929),
    })
# Approved garagelab look: grey links + head -> 0.03, pure whites -> 0.7.
_HEAD = ("head_yaw_link", "head_pitch_link", "RGB_rear_link", "RGB_binocular_link",
         "RGB_Forward_link", "RGBD_in_head_link", "imu_in_head_link")
for _k, _v in list(X2_LINK_COLORS.items()):
    if _v == (1.0, 1.0, 1.0):
        X2_LINK_COLORS[_k] = (0.7, 0.7, 0.7)
    elif _v in ((0.298, 0.298, 0.298), (0.3, 0.3, 0.3)) or _k in _HEAD:
        X2_LINK_COLORS[_k] = (0.03, 0.03, 0.03)


def light_quat_from_azel(az_deg: float, el_deg: float):
    """Quaternion (w,x,y,z) that points a DistantLight (shines along local -Z)
    so its rays come FROM azimuth/elevation (deg) toward the scene."""
    az, el = math.radians(az_deg), math.radians(el_deg)
    z = np.array([math.cos(el) * math.cos(az), math.cos(el) * math.sin(az), math.sin(el)])
    x = np.cross([0.0, 0.0, 1.0], z)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    m = np.stack([x, y, z], axis=1)
    w = math.sqrt(max(0.0, 1.0 + m[0, 0] + m[1, 1] + m[2, 2])) / 2.0
    if w > 1e-6:
        qx = (m[2, 1] - m[1, 2]) / (4 * w)
        qy = (m[0, 2] - m[2, 0]) / (4 * w)
        qz = (m[1, 0] - m[0, 1]) / (4 * w)
    else:
        qx, qy, qz = 1.0, 0.0, 0.0
    return (float(w), float(qx), float(qy), float(qz))


def paint_x2(stage, robot_root: str):
    """Bind MJCF-matched UsdPreviewSurface materials to each link's visuals."""
    from pxr import Gf, Sdf, UsdShade

    def ensure_mat(name, rgb):
        path = f"/World/Looks/x2_{name}"
        if stage.GetPrimAtPath(path):
            return UsdShade.Material(stage.GetPrimAtPath(path))
        mat = UsdShade.Material.Define(stage, path)
        sh = UsdShade.Shader.Define(stage, path + "/shader")
        sh.CreateIdAttr("UsdPreviewSurface")
        sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*rgb))
        sh.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.35)
        sh.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.05)
        mat.CreateSurfaceOutput().ConnectToSource(sh.ConnectableAPI(), "surface")
        return mat

    root = stage.GetPrimAtPath(robot_root)
    if not root:
        return 0
    painted = 0
    for link in root.GetChildren():
        rgb = X2_LINK_COLORS.get(link.GetName())
        if rgb is None:
            continue
        vis = stage.GetPrimAtPath(f"{link.GetPath()}/visuals")
        if not vis:
            continue
        api = UsdShade.MaterialBindingAPI.Apply(vis)
        api.Bind(ensure_mat(link.GetName(), rgb), UsdShade.Tokens.strongerThanDescendants)
        painted += 1
    return painted


def quat_mul_arr(q, arr):
    """(w,x,y,z) quaternion q composed onto each row of arr (T,4): q * arr."""
    w1, x1, y1, z1 = q
    w2, x2, y2, z2 = arr[:, 0], arr[:, 1], arr[:, 2], arr[:, 3]
    return np.stack([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ], axis=1)


def anchor_root(rt, rr_wxyz, rx, ry, ryaw_deg, fixed_xy=False):
    """Re-anchor a clip's root track at placement (rx, ry) with heading ryaw."""
    yaw0 = math.atan2(2 * (rr_wxyz[0, 0] * rr_wxyz[0, 3] + rr_wxyz[0, 1] * rr_wxyz[0, 2]),
                      1 - 2 * (rr_wxyz[0, 2] ** 2 + rr_wxyz[0, 3] ** 2))
    d = math.radians(ryaw_deg) - yaw0
    qz = (math.cos(d / 2), 0.0, 0.0, math.sin(d / 2))
    quat_w = quat_mul_arr(qz, rr_wxyz)
    if fixed_xy:
        pos_w = np.column_stack([np.full(len(rt), rx), np.full(len(rt), ry), rt[:, 2]])
    else:
        rel = rt[:, :2] - rt[0, :2]
        c, s = math.cos(d), math.sin(d)
        pos_w = np.column_stack([rx + c * rel[:, 0] - s * rel[:, 1],
                                 ry + s * rel[:, 0] + c * rel[:, 1], rt[:, 2]])
    return pos_w.astype(np.float32), quat_w.astype(np.float32)


def catmull_rom(times, points, t):
    """C1 hermite spline through (times[i], points[i]); finite-difference tangents."""
    times = np.asarray(times)
    pts = np.asarray(points, dtype=np.float64)
    t = min(max(t, times[0]), times[-1])
    i = int(np.searchsorted(times, t, side="right") - 1)
    i = min(i, len(times) - 2)
    t0, t1 = times[i], times[i + 1]
    h = t1 - t0
    s = (t - t0) / h
    p0, p1 = pts[i], pts[i + 1]
    m0 = (pts[i + 1] - pts[i - 1]) / (times[i + 1] - times[i - 1]) if i > 0 \
        else (pts[1] - pts[0]) / (times[1] - times[0])
    m1 = (pts[i + 2] - pts[i]) / (times[i + 2] - times[i]) if i + 2 < len(pts) \
        else (pts[-1] - pts[-2]) / (times[-1] - times[-2])
    h00 = 2 * s**3 - 3 * s**2 + 1
    h10 = s**3 - 2 * s**2 + s
    h01 = -2 * s**3 + 3 * s**2
    h11 = s**3 - s**2
    return h00 * p0 + h10 * m0 * h + h01 * p1 + h11 * m1 * h


def main():
    sim_cfg = sim_utils.SimulationCfg(dt=0.005, render_interval=4)
    # Bridge-proven buffer oversizing: many robots + static trimesh silently
    # drop contacts otherwise (one robot per run sinks/falls).
    sim_cfg.physx.gpu_max_rigid_contact_count = 2 ** 24
    sim_cfg.physx.gpu_max_rigid_patch_count = 2 ** 19
    sim_cfg.physx.gpu_collision_stack_size = 2 ** 28
    sim_cfg.physx.gpu_found_lost_pairs_capacity = 2 ** 23
    sim_cfg.physx.gpu_found_lost_aggregate_pairs_capacity = 2 ** 26
    sim_cfg.physx.gpu_total_aggregate_pairs_capacity = 2 ** 24
    sim = SimulationContext(sim_cfg)

    scene_path = os.path.expanduser(args.scene_usdz)
    if not os.path.isabs(scene_path):
        scene_path = os.path.join(_LAUNCH_CWD, scene_path)
    if not os.path.exists(scene_path):
        sys.exit(f"[tour] world file not found: {scene_path}")
    UsdFileCfg(usd_path=scene_path).func("/World/Splat", UsdFileCfg(usd_path=scene_path))
    print(f"[tour] world: {scene_path}")

    dome = sim_utils.DomeLightCfg(intensity=args.dome_intensity, color=(0.92, 0.92, 0.88))
    dome.func("/World/DomeLight", dome)
    az, el, inten = args.key_light
    key = sim_utils.DistantLightCfg(intensity=inten, color=(1.0, 0.98, 0.94), angle=1.0)
    key.func("/World/KeyLight", key, orientation=light_quat_from_azel(az, el))
    fill = sim_utils.SphereLightCfg(intensity=inten * 0.12, radius=0.3,
                                    color=(0.95, 0.97, 1.0))
    fill.func("/World/FillLight", fill, translation=(-0.2, 0.5, args.eye_z + 0.4))

    # Ground so robots have something under them (splat is visual-only). At
    # -0.02 like the approved bridge launch: at 0.0 it z-fights the splat floor
    # (white shimmer on the dark mats). Visible in grid mode = the classic
    # IsaacLab training-farm plane around the scanned env.
    ground = sim_utils.GroundPlaneCfg(size=(90.0, 90.0), visible=args.grid > 0)
    ground.func("/World/GroundPlane", ground, translation=(0.0, 0.0, -0.02))

    # Collision mesh via TerrainImporter — the bridge's supported path for static
    # trimesh. Roots are pinned every step, so the known >=2-robot root-sinking
    # bug cannot manifest; the mesh just gives feet/seat real support.
    if args.collision_usd:
        col_path = os.path.expanduser(args.collision_usd)
        if not os.path.isabs(col_path):
            col_path = os.path.join(_LAUNCH_CWD, col_path)
        if os.path.exists(col_path):
            from isaaclab.terrains import TerrainImporter, TerrainImporterCfg
            terrain_cfg = TerrainImporterCfg(
                prim_path="/World/Collision",
                terrain_type="usd",
                usd_path=col_path,
                collision_group=-1,
                env_spacing=0.0,
                physics_material=sim_utils.RigidBodyMaterialCfg(
                    static_friction=1.0, dynamic_friction=1.0, restitution=0.0),
            )
            importer = TerrainImporter(terrain_cfg)
            importer.configure_env_origins = lambda *a, **k: None
            print(f"[tour] collision mesh: {col_path}")
        else:
            print(f"[tour] collision mesh not found, skipping: {col_path}")

    base_cfg = make_x2_ultra_cfg(foot="sphere")

    # The URDF spawner writes a ~140 MB converted-USD dir to /tmp/IsaacLab PER
    # SPAWN (1000 robots filled the disk). Convert once via robot #1, then
    # reference that USD for everyone else.
    _shared = {}

    def spawn(name, rx, ry, rz, ryaw, fixed):
        yaw = math.radians(ryaw)
        cfg = base_cfg.replace(prim_path=f"/World/X2_{name}")
        if _shared.get("usd"):
            cfg.spawn = sim_utils.UsdFileCfg(
                usd_path=_shared["usd"],
                activate_contact_sensors=True,
                rigid_props=base_cfg.spawn.rigid_props,
                articulation_props=base_cfg.spawn.articulation_props,
            )
        if fixed:
            ap = cfg.spawn.articulation_props
            ap = (ap.replace(fix_root_link=True) if ap is not None
                  else sim_utils.ArticulationRootPropertiesCfg(fix_root_link=True))
            cfg.spawn = cfg.spawn.replace(articulation_props=ap)
        cfg.init_state.pos = (rx, ry, rz)
        cfg.init_state.rot = (math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2))
        robots.append(Articulation(cfg))
        root_poses.append((rx, ry, rz, yaw))
        if "usd" not in _shared:
            import glob
            cands = sorted(glob.glob("/tmp/IsaacLab/usd_*/**/*.usd", recursive=True),
                           key=os.path.getmtime)
            _shared["usd"] = cands[-1] if cands else None
            print(f"[tour] shared robot USD: {_shared['usd']}")

    robots, root_poses = [], []
    sofa_idx = next(i for i, rb in enumerate(ROBOTS) if rb[0] == "sofa")
    FIXED, FREE = set(), set()
    for ri, (name, (rx, ry, rz, ryaw), overrides) in enumerate(ROBOTS):
        fixed = args.free_roots and ri == sofa_idx  # seated stays anchored
        spawn(name, rx, ry, rz, ryaw, fixed)
        if fixed:
            FIXED.add(ri)
        elif args.free_roots:
            FREE.add(ri)
        print(f"[tour] X2_{name} at ({rx:.2f}, {ry:.2f}, {rz:.2f}) yaw={ryaw:.0f}"
              + (" [fixed]" if fixed else " [free]" if ri in FREE else ""))
    grid_names = []
    if args.grid:
        sp = args.grid_spacing
        for gi in range(-args.grid, args.grid + 1):
            for gj in range(-args.grid, args.grid + 1):
                gx, gy = gi * sp, gj * sp
                if -5.0 < gx < 6.0 and -5.0 < gy < 6.0:
                    continue  # the garagelab splat occupies this cell
                name = f"g{gi + args.grid}_{gj + args.grid}"
                FIXED.add(len(robots))
                grid_names.append(name)
                spawn(name, gx, gy, 0.66,
                      ((gi * 73 + gj * 137) % 360) - 180.0, True)
        print(f"[tour] training-farm grid: {len(grid_names)} X2s at {sp:.0f} m")

    cam = Camera(CameraCfg(
        prim_path="/World/ShotCam",
        width=args.width, height=args.height, data_types=["rgb"], update_period=0.0,
        spawn=sim_utils.PinholeCameraCfg(clipping_range=(0.05, 100.0),
                                         focal_length=args.focal),
    ))

    sim.reset()
    import omni.usd
    stage = omni.usd.get_context().get_stage()
    total = sum(paint_x2(stage, f"/World/X2_{name}") for name, _p, _o in ROBOTS)
    total += sum(paint_x2(stage, f"/World/X2_{n}") for n in grid_names)
    print(f"[tour] painted {total} links")

    # Kinematic chores replay: 'executed' clips (MuJoCo dof order) mapped to
    # IsaacLab order, one per standing robot, staggered so they don't sync.
    IL2MJ = np.asarray(X2_ULTRA_ISAACLAB_TO_MUJOCO_DOF)
    anim = {}  # robot index -> (dof_il float32 (T,31), clip_fps, offset_s)
    if args.chores_pkl:
        import joblib
        clips = joblib.load(os.path.expanduser(args.chores_pkl))
        keys = sorted(k for k in clips if "executed" in k) or sorted(clips)
        ci = 0
        for ri, (name, (rx, ry, rz, ryaw), overrides) in enumerate(ROBOTS):
            if overrides:  # posed robots (the sofa sitter) stay static
                continue
            key = keys[ci % len(keys)]
            dof_il = np.asarray(clips[key]["dof"], dtype=np.float32)[:, IL2MJ]
            fps_c = float(clips[key].get("fps", 50))
            # Full root replay: these 'executed' tracks came out of the SONIC
            # physics stack, so dof+root reproduces SONIC's motion faithfully.
            rt = np.asarray(clips[key]["root_trans_offset"], np.float64)
            rr = np.asarray(clips[key]["root_rot"], np.float64)[:, [3, 0, 1, 2]]  # xyzw->wxyz
            pos_w, quat_w = anchor_root(rt, rr, rx, ry, ryaw)
            anim[ri] = {"dof": dof_il, "fps": fps_c, "off": 7.3 * ci,
                        "pos": pos_w, "quat": quat_w}
            print(f"[tour] X2_{name} replays {key.split('__')[-1]} "
                  f"({dof_il.shape[0]} f @ {fps_c:.0f} fps, offset {7.3 * ci:.1f} s)")
            ci += 1
    def load_x2m2(path):
        import struct
        raw = open(os.path.expanduser(path), "rb").read()
        magic, n_frames, n_dofs = struct.unpack_from("<III", raw, 0)
        fps_m = struct.unpack_from("<d", raw, 12)[0]
        assert magic == 0x58324D32 and n_dofs == 31, "not an X2M2/31-dof file"
        frames = np.frombuffer(raw, dtype=np.float64, offset=20).reshape(n_frames, 35)
        dof_il = frames[:, :31].astype(np.float32)[:, IL2MJ]
        rr = frames[:, 31:35].copy()
        if abs(rr[0, 3]) > abs(rr[0, 0]):  # xyzw on disk -> wxyz
            rr = rr[:, [3, 0, 1, 2]]
        return dof_il, rr, fps_m

    specs = list(args.x2m2)
    if args.middle_x2m2:
        specs.append(f"middle={args.middle_x2m2}")
    for spec in specs:
        nm, _, path = spec.partition("=")
        dof_il, rr, fps_m = load_x2m2(path)
        ri = next(i for i, rb in enumerate(ROBOTS) if rb[0] == nm)
        rx, ry, rz, ryaw = ROBOTS[ri][1]
        rt = np.column_stack([np.zeros((len(dof_il), 2)), np.full(len(dof_il), rz)])
        pos_w, quat_w = anchor_root(rt, rr, rx, ry, ryaw, fixed_xy=True)
        anim[ri] = {"dof": dof_il, "fps": fps_m, "off": 0.0,
                    "pos": pos_w, "quat": quat_w}
        print(f"[tour] X2_{nm} loops {os.path.basename(path)} "
              f"({len(dof_il)} f @ {fps_m:.0f} fps, {len(dof_il) / fps_m:.1f} s)")
    if args.free_roots and args.free_idle_x2m2:
        dof_il, _rr, fps_i = load_x2m2(args.free_idle_x2m2)
        for k, ri in enumerate(sorted(FREE)):
            anim[ri] = {"dof": dof_il, "fps": fps_i, "off": 0.37 * k,
                        "pos": None, "quat": None}
        print(f"[tour] {len(FREE)} FREE robots track idle loop "
              f"{os.path.basename(args.free_idle_x2m2)}")

    # Grid robots: cycle the chores clips, staggered per cell (fixed base ->
    # PD targets alone animate them; no root data needed).
    if grid_names and args.chores_pkl:
        for k, name in enumerate(grid_names):
            ri = len(ROBOTS) + k
            key = keys[k % len(keys)]
            dof_il = np.asarray(clips[key]["dof"], dtype=np.float32)[:, IL2MJ]
            fps_c = float(clips[key].get("fps", 50))
            anim[ri] = {"dof": dof_il, "fps": fps_c, "off": 3.1 * k,
                        "pos": None, "quat": None}

    # Pose the mannequins: joint state written directly, roots pinned each step,
    # PD targets = the pose with boosted gains, a short settle, no gravity drop.
    poses_t = []
    for ri, ((name, _xy, overrides), r) in enumerate(zip(ROBOTS + [
            (n, None, {}) for n in grid_names], robots)):
        r.update(sim.get_physics_dt())
        jp = r.data.default_joint_pos.clone()
        for pattern, val in overrides.items():
            ids, names = r.find_joints(pattern)
            jp[0, ids] = val
            print(f"[tour] X2_{name} {pattern} -> {val:+.2f} rad ({len(names)} joints)")
        if ri in anim:  # start exactly on the clip's first replayed frame
            a = anim[ri]
            f0 = int(a["off"] * a["fps"]) % a["dof"].shape[0]
            jp = torch.tensor(a["dof"][f0:f0 + 1], device=r.device)
        r.write_joint_state_to_sim(jp, torch.zeros_like(jp))
        r.write_joint_stiffness_to_sim(r.data.joint_stiffness * 20.0)
        r.write_joint_damping_to_sim(r.data.joint_damping * math.sqrt(20.0))
        poses_t.append(jp)
    if args.sofa_wave_x2m2:
        # Overlay only the gesture's right-arm channels on the held sit pose.
        dof_w, _rr, fps_w = load_x2m2(args.sofa_wave_x2m2)
        si = next(i for i, rb in enumerate(ROBOTS) if rb[0] == "sofa")
        arm_ids, arm_names = robots[si].find_joints(
            "right_shoulder_.*|right_elbow_.*|right_wrist_.*")
        dof = np.repeat(poses_t[si].cpu().numpy(), len(dof_w), axis=0)
        dof[:, arm_ids] = dof_w[:, arm_ids]
        sx, sy, sz, syaw = ROBOTS[si][1]
        h = math.radians(syaw) / 2
        # off phases the loop so the wave crest (~3 s in) lands at the t=10 s
        # camera pass: (10 + off) mod 7.5 = 3
        anim[si] = {"dof": dof.astype(np.float32), "fps": fps_w, "off": 0.5,
                    "pos": np.tile([sx, sy, sz], (len(dof), 1)).astype(np.float32),
                    "quat": np.tile([math.cos(h), 0, 0, math.sin(h)],
                                    (len(dof), 1)).astype(np.float32)}
        print(f"[tour] X2_sofa waves right arm via "
              f"{os.path.basename(args.sofa_wave_x2m2)} ({len(arm_names)} joints, "
              f"{len(dof) / fps_w:.1f} s loop)")

    hang = [torch.tensor([[rx, ry, rz, math.cos(yw / 2), 0.0, 0.0, math.sin(yw / 2)]],
                         device=robots[0].device) for rx, ry, rz, yw in root_poses]
    zero6 = torch.zeros((1, 6), device=robots[0].device)
    for _ in range(40):  # 0.2 s settle; floating roots pinned, fixed need none
        for ri, (r, hp, jp) in enumerate(zip(robots, hang, poses_t)):
            if ri not in FIXED:
                r.write_root_pose_to_sim(hp)
                r.write_root_velocity_to_sim(zero6)
            r.set_joint_position_target(jp)
            r.write_data_to_sim()
        sim.step()
        for r in robots:
            r.update(sim.get_physics_dt())
    for (name, _xy, _o), r, jp in zip(ROBOTS, robots, poses_t):
        err = float((r.data.joint_pos - jp).abs().max())
        print(f"[tour] X2_{name} pose hold err max {err:.3f} rad")

    for _ in range(60):  # RTX warmup
        sim.render()

    # --- fly the path --------------------------------------------------------
    out_dir = args.out_dir if os.path.isabs(args.out_dir) \
        else os.path.join(_LAUNCH_CWD, args.out_dir)
    os.makedirs(out_dir, exist_ok=True)
    from PIL import Image

    def rescale_z(z):
        return args.eye_z if abs(z - 1.83) < 1e-6 else z

    waypoints = list(WAYPOINTS)
    intro_end = 0.0
    if args.grid:
        # Reference-shot intro: fly THROUGH the crowd at a foot above head
        # height, gaze level, down a corridor between grid columns; rise to
        # show the scale; then dive into the scan through the roof.
        half = args.grid_spacing / 2.0
        waypoints = [
            (0.0, (half, -38.0, 1.75), (half, -30.0, 1.70)),
            (5.5, (half, -17.0, 1.75), (half, -9.0, 1.60)),
            (8.5, (1.10, -10.5, 4.20), (0.40, -3.0, 1.60)),
            (11.0, (0.30, -6.0, 10.5), (-0.30, 0.50, 1.00)),
            (13.5, (-0.20, -3.6, 4.60), (-0.60, 1.00, 1.00)),
        ] + [(t + 15.5, e, l) for t, e, l in WAYPOINTS]
        intro_end = 15.5
    times = [w[0] for w in waypoints]
    eyes = [(w[1][0], w[1][1], rescale_z(w[1][2])) for w in waypoints]
    looks = [w[2] for w in waypoints]

    if os.environ.get("TOUR_DIAG"):
        # fixed diagnostic views of the couch area (side / top / front)
        diag = [((1.55, -0.60, 1.00), (1.55, 1.95, 0.60)),
                ((0.30, 3.50, 2.20), (1.55, 1.85, 0.50)),
                ((-0.50, 1.95, 1.20), (1.55, 1.95, 0.60))]
        for i, (eye, look) in enumerate(diag):
            cam.set_world_poses_from_view(
                torch.tensor([eye], dtype=torch.float32, device=sim.device),
                torch.tensor([look], dtype=torch.float32, device=sim.device))
            for _ in range(args.accum):
                sim.render()
            cam.update(0.0)
            rgb = cam.data.output["rgb"][0].cpu().numpy()
            Image.fromarray(rgb[..., :3].astype(np.uint8)).save(
                os.path.join(out_dir, f"diag_{i}.png"))
            print(f"[tour] diag {i} saved")
        print(f"[tour] DONE (diag) in {out_dir}")
        closer = threading.Thread(target=simulation_app.close, daemon=True)
        closer.start()
        closer.join(timeout=20.0)
        os._exit(0)

    if args.probe:
        sample_ts = []
        for a, b in zip(times[:-1], times[1:]):
            sample_ts += [a, (a + b) / 2]
        sample_ts.append(times[-1])
    else:
        n = int(times[-1] * args.fps)
        sample_ts = [i / args.fps for i in range(n)]

    # Precompute the whole camera path; clamp gaze tilt and low-pass the look
    # DIRECTION so fast pans are angularly smooth. (The old per-frame distance
    # clamp switched on/off as the look spline crossed the eye path — two
    # angular-velocity spikes at the same spot, read as a whip/flip.)
    eyes_a = np.array([catmull_rom(times, eyes, t) for t in sample_ts])
    looks_a = np.array([catmull_rom(times, looks, t) for t in sample_ts])
    dirs = looks_a - eyes_a
    dirs /= np.maximum(np.linalg.norm(dirs, axis=1, keepdims=True), 1e-9)
    # tilt cap: 40 deg in the tour; 60 deg allowed during the intro dive/reveal
    zcap_arr = np.where(np.asarray(sample_ts) < intro_end,
                        -math.sin(math.radians(60.0)),
                        -math.sin(math.radians(40.0)))
    low = dirs[:, 2] < zcap_arr
    if low.any():
        zc = zcap_arr[low]
        hmag = np.sqrt(1 - zc * zc)
        hn = np.maximum(np.linalg.norm(dirs[low, :2], axis=1), 1e-9)
        dirs[low, :2] *= (hmag / hn)[:, None]
        dirs[low, 2] = zc
    if not args.probe:
        w = max(3, int(args.fps * 0.5) | 1)  # ~0.5 s box filter
        pad = np.pad(dirs, ((w // 2, w // 2), (0, 0)), mode="edge")
        ker = np.ones(w) / w
        dirs = np.stack([np.convolve(pad[:, c], ker, mode="valid")
                         for c in range(3)], axis=1)
        dirs /= np.maximum(np.linalg.norm(dirs, axis=1, keepdims=True), 1e-9)

    phys_dt = sim.get_physics_dt()
    steps_done = 0
    fell = set()

    def drive(ts, robot_ids):
        for ri in robot_ids:
            a = anim.get(ri)
            if a is None:
                continue
            r = robots[ri]
            f = int((ts + a["off"]) * a["fps"]) % a["dof"].shape[0]
            jp = torch.tensor(a["dof"][f:f + 1], device=r.device)
            if ri in FREE or ri in FIXED:
                r.set_joint_position_target(jp)  # PD does the rest
            else:  # locked floating mannequin: kinematic replay
                r.write_root_pose_to_sim(torch.tensor(
                    [[*a["pos"][f], *a["quat"][f]]], dtype=torch.float32,
                    device=r.device))
                r.write_root_velocity_to_sim(zero6)
                r.write_joint_state_to_sim(jp, torch.zeros_like(jp))
                r.set_joint_position_target(jp)
            r.write_data_to_sim()

    all_ids = list(range(len(robots)))
    free_ids = sorted(FREE)
    for i, t in enumerate(sample_ts):
        if anim:
            drive(t, all_ids)  # frame-rate target/kinematic updates
            if FREE:  # real-time physics so the released robots track truly
                nsteps = max(1, int(round(t / phys_dt)) - steps_done)
                for _k in range(nsteps):
                    drive(steps_done * phys_dt, free_ids)
                    sim.step(render=False)
                    steps_done += 1
            else:
                sim.step(render=False)
            for r in robots:
                r.update(phys_dt)
            for ri in free_ids:  # falls are data, not surprises — and resettable
                r = robots[ri]
                h = float(r.data.root_pos_w[0, 2])
                if h < 0.35:
                    if ri not in fell:
                        print(f"[tour] WARNING: X2_{ROBOTS[ri][0]} FELL at "
                              f"t={t:.1f}s (pelvis {h:.2f} m) — resetting")
                    fell.add(ri)
                    a = anim.get(ri)
                    jp = poses_t[ri] if a is None else torch.tensor(
                        a["dof"][int((t + a["off"]) * a["fps"])
                                 % a["dof"].shape[0]][None], device=r.device)
                    r.write_root_pose_to_sim(hang[ri])
                    r.write_root_velocity_to_sim(zero6)
                    r.write_joint_state_to_sim(jp, torch.zeros_like(jp))
                    r.write_data_to_sim()
        eye = eyes_a[i]
        look = eye + dirs[i]
        cam.set_world_poses_from_view(
            torch.tensor([eye], dtype=torch.float32, device=sim.device),
            torch.tensor([look], dtype=torch.float32, device=sim.device))
        for _ in range(args.accum):
            sim.render()
        cam.update(0.0)
        rgb = cam.data.output["rgb"][0].cpu().numpy()
        Image.fromarray(rgb[..., :3].astype(np.uint8)).save(
            os.path.join(out_dir, f"frame_{i:04d}.png"))
        if i % 20 == 0 or i == len(sample_ts) - 1:
            print(f"[tour] frame {i + 1}/{len(sample_ts)} t={t:.2f}s "
                  f"eye=({eye[0]:.2f},{eye[1]:.2f},{eye[2]:.2f})")
    print(f"[tour] DONE: {len(sample_ts)} frames in {out_dir} "
          f"({len(fell)} falls)")

    closer = threading.Thread(target=simulation_app.close, daemon=True)
    closer.start()
    closer.join(timeout=20.0)
    if closer.is_alive():
        print("[tour] close() hung — hard exit")
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
