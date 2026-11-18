"""GR00T (G1 checkpoint, frozen) -> G1->X2 adapter -> X2 SONIC in the gwbc-x2 MuJoCo simstack.

Start the simstack first (headless):  SIM_VIEWER=0 ~/Projects/gwbc-x2/gear_sonic/scripts/sim_onnx_planner.sh
Then, in ~/.venvs/groot_n17 (env_isaaclab + transformers 4.57.3) with PYTHONPATH=<GR00T-WBC>/external_dependencies/Isaac-GR00T:
  python -m groot_bridge.run_sim_loop --instruction "Walk to the console and press the red button" --seconds 20
  python -m groot_bridge.run_sim_loop --scripted wave --seconds 8      # plumbing test, no model

Every inference (2.5 Hz) returns a 40-step chunk; the first --exec-steps steps are sent at 50 Hz.
Writes <out>/run.log (heartbeat), <out>/chunks.npz (model actions + X2 commands), <out>/video.mp4
(ego view | chase view, with captions).
"""
from __future__ import annotations

import argparse
import glob
import os
import time
from pathlib import Path

import numpy as np

from .embodiment import TargetSpec
from .g1_x2 import G1NavToX2
from .x2_sim_io import X2Renderer, X2SimIO

CKPT_GLOB = "~/.cache/huggingface/hub/models--nvidia--G1-Factory-GR00T-1.7/snapshots/*"


def log_to(path):
    f = open(path, "a", buffering=1)

    def log(msg):
        line = f"{time.strftime('%H:%M:%S')} {msg}"
        print(line, flush=True); f.write(line + "\n")
    return log


def g1_palm_grasp_point(ep):
    """Midpoint of the G1 palm centres (pelvis frame) at the frame where they are closest (the squeeze)."""
    import mujoco
    from .ee_retarget import ARM, ArmModel, hand_geometry
    from .g1_x2 import G1_XML
    m = mujoco.MjModel.from_xml_path(str(G1_XML)); P = {}
    for side in ("left", "right"):
        gj = [f"{side}_{j}_joint" for j in ARM]
        _, palm = hand_geometry(m, f"{side}_wrist_yaw_link", f"{side}_hand_middle_0_link", f"{side}_hand_thumb_0_link",
                                np.zeros(7), gj)
        arm = ArmModel(m, gj, f"{side}_wrist_yaw_link", tcp_local=palm)
        P[side] = np.array([arm.fk(q)[0] for q in ep[f"{side}_arm"]])
    i = int(np.argmin(np.linalg.norm(P["left"] - P["right"], axis=1)))
    return (P["left"][i] + P["right"][i]) / 2


def load_g1_episode(path):
    """G1-Factory-Benchmark LeRobot episode -> the checkpoint's action keys (43-d joint action layout:
    waist 12:15, left arm 15:22, left hand 22:29, right arm 29:36, right hand 36:43; nav/height in teleop.*)."""
    import pandas as pd
    df = pd.read_parquet(path)
    A = np.stack(df["action"].values).astype(np.float32)
    return {"left_arm": A[:, 15:22], "right_arm": A[:, 29:36], "left_hand": A[:, 22:29], "right_hand": A[:, 36:43],
            "waist": A[:, 12:15],
            "base_height_command": np.stack(df["teleop.base_height_command"].values).astype(np.float32),
            "navigate_command": np.stack(df["teleop.navigate_command"].values).astype(np.float32)}


def json_box_half(scene_xml):
    import json
    meta = json.loads(Path(scene_xml).with_suffix(".meta.json").read_text())
    return meta["box"]["half"]


def box_clearance(q7, half, table_z):
    """Lowest box corner height above the table top, and tilt (deg) of the box's z axis from vertical."""
    import mujoco
    R = np.zeros(9); mujoco.mju_quat2Mat(R, np.asarray(q7[3:7], float)); R = R.reshape(3, 3)
    corners = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)]) * half
    z = q7[2] + (corners @ R.T)[:, 2]
    tilt = float(np.degrees(np.arccos(np.clip(abs(R[2, 2]), -1, 1))))
    return float(z.min() - table_z), tilt


def scripted_chunk(kind, t0, spec, H=40, hz=50.0):
    """G1-space action chunks with no model: exercises the adapter + X2 plumbing end to end."""
    mean = {k: (lo + hi) / 2 for k, (lo, hi) in spec.action_range.items()}
    out = {k: np.tile(np.zeros_like(v) if k in ("navigate_command",) else v, (H, 1)) for k, v in mean.items()}
    ts = t0 + np.arange(H) / hz
    out["base_height_command"][:] = spec.action_range["base_height_command"][1]
    out["left_arm"][:] = 0.0; out["left_arm"][:, 3] = 1.4          # G1 elbow 1.4 = nearly straight
    out["right_arm"][:] = 0.0; out["right_arm"][:, 3] = 1.4
    out["waist"][:] = 0.0
    for k in ("left_hand", "right_hand"):
        out[k][:] = 0.0
    if kind == "wave":                                               # right arm up, forearm swings
        out["right_arm"][:, 0] = -1.2; out["right_arm"][:, 1] = -0.9
        out["right_arm"][:, 3] = 0.3 + 0.5 * np.sin(2 * np.pi * 1.0 * ts)
    elif kind == "walk":
        out["navigate_command"][:, 0] = 0.35
    elif kind == "turn":
        out["navigate_command"][:, 2] = 0.5
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--instruction", default="Walk to the console and press the red button")
    ap.add_argument("--seconds", type=float, default=20.0)
    ap.add_argument("--scripted", choices=["wave", "walk", "turn"], default=None)
    ap.add_argument("--replay-place-box", default="grasp",
                    help="replay only: when the demo's walk ends, move table+box under the robot (removes open-loop "
                         "walk drift). 'grasp' = box centre at the demo's palm grasp point (G1 FK, pelvis frame); "
                         "a number = that far straight ahead at the table height; 'off'")
    ap.add_argument("--replay", type=Path, default=None,
                    help="open-loop replay of a G1 LeRobot episode parquet (G1-Factory-Benchmark) through the adapter")
    ap.add_argument("--exec-steps", type=int, default=20, help="steps of each 40-step chunk to execute")
    ap.add_argument("--blend-steps", type=int, default=0,
                    help="crossfade the leftover of the previous chunk into the new one over N steps (anti-dither)")
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--video-fps", type=int, default=25)
    ap.add_argument("--scene-xml", type=Path, default=None,
                    help="physical scene the sim runs (tools/build_x2_console_scene.py); also the render model")
    ap.add_argument("--arm-mode", choices=["hand", "joint"], default="hand",
                    help="hand = hand-space retarget (G1 FK -> X2 IK); joint = copy joint angles")
    ap.add_argument("--squeeze-offset", type=float, default=0.0,
                    help="hand mode: extra inward wrist offset (m) while the hands close on an object")
    ap.add_argument("--hand-align", choices=["wrist", "geom"], default="wrist",
                    help="hand mode: align wrist frames, or hand geometry (finger dir + palm normal) with palm-centre targets")
    ap.add_argument("--g1-look", action="store_true", help="camera head: render the X2 body G1-grey (render only)")
    ap.add_argument("--ego-fovy", type=float, default=None, help="camera head: virtual head-camera vertical FOV (deg)")
    ap.add_argument("--ego-pitch", type=float, default=None, help="camera head: extra head-camera pitch (deg, + = down)")
    a = ap.parse_args()
    out = a.out or Path(__file__).resolve().parents[1] / f"runs/groot_bridge/{time.strftime('%Y%m%dT%H%M%S')}_{a.scripted or 'groot'}"
    out.mkdir(parents=True, exist_ok=True)
    log = log_to(out / "run.log")
    ckpt = a.ckpt or glob.glob(os.path.expanduser(CKPT_GLOB))[0]
    spec = TargetSpec.from_gr00t_checkpoint(ckpt, name="G1-Factory-GR00T-1.7")
    adapter = G1NavToX2(spec, arm_mode=a.arm_mode, squeeze_offset=a.squeeze_offset, hand_align=a.hand_align)
    log(f"spec {spec.name} tag={spec.embodiment_tag} horizon={spec.action_horizon} out={out}")

    policy = None
    replay = load_g1_episode(a.replay) if a.replay else None
    walk_end = None
    if replay is not None:
        nav_fwd = np.abs(replay["navigate_command"][:, 0]) > 0.05
        walk_end = len(nav_fwd) - int(np.argmax(nav_fwd[::-1])) if nav_fwd.any() else 0
        grasp_pt = g1_palm_grasp_point(replay)
        log(f"replay grasp point (palm midpoint at closest approach, pelvis frame): {np.round(grasp_pt, 3).tolist()}")
        log(f"replay {a.replay.name}: {len(replay['navigate_command'])} frames ({len(replay['navigate_command']) / 50:.1f} s)")
        a.seconds = min(a.seconds, len(replay["navigate_command"]) / 50 + 5.0)   # then hold the final pose 5 s
    if not a.scripted and replay is None:
        from gr00t.policy.gr00t_policy import Gr00tPolicy
        t = time.time()
        policy = Gr00tPolicy(embodiment_tag=spec.embodiment_tag, model_path=ckpt, device="cuda:0")
        log(f"policy loaded in {time.time() - t:.1f}s")

    io = X2SimIO()
    rend = X2Renderer(scene_xml=a.scene_xml, ego_fovy=a.ego_fovy, ego_pitch_deg=a.ego_pitch, g1_look=a.g1_look)
    if not io.wait_ready(60):
        log("ERROR: no x2_debug / robot_pose from the simstack (is it running?)"); return 2
    if rend.props:
        rend.place_props(io.state["root7"])
    log(f"sim ready, root {np.round(io.state['root7'][:3], 3).tolist()}; props placed ahead of the robot")

    import imageio.v2 as imageio
    import cv2
    vw = imageio.get_writer(out / "video.mp4", fps=a.video_fps, codec="libx264", quality=7)
    rec = {"t": [], "root": [], "model": [], "x2_cmd_vel": [], "x2_arm_l": [], "x2_arm_r": [], "infer_s": []}
    dt = 1.0 / spec.control_hz
    best_dist, press_t = 9.9, None
    prev_act = None
    placed = False
    placed_h = False
    box_trace = []
    box0 = io.state["objects"].get("box_free"); box_max_rise, lift_t = 0.0, None
    best_clear, clean_since = -1.0, None
    box_half = np.array(json_box_half(a.scene_xml)) if a.scene_xml else None
    if box0 is not None:
        log(f"box at {np.round(box0[:3], 3).tolist()} (scene_state)")
    t_start = time.time(); next_frame = 0.0; last_hb = 0.0; n_inf = 0; caption = ""
    try:
        while time.time() - t_start < a.seconds:
            rend.set_state(io.state["root7"], io.state["body_q"], io.state["lh"], io.state["rh"], io.state["objects"])
            ego = rend.ego()
            st = io.robot_state({spec.video_keys[0]: ego})
            t_inf = time.time()
            if replay is not None:
                i0 = n_inf * a.exec_steps
                if a.replay_place_box == "grasp" and box0 is not None and not placed_h:
                    # stage 1 (robot still far away): table height so the box centre sits at the grasp height
                    r = io.state["root7"]; tab = rend.m.body_pos[rend.m.body("table").id].copy()
                    bz_w = r[2] + grasp_pt[2]; dz = bz_w - box0[2]
                    tab_new = [tab[0], tab[1], tab[2] + dz]
                    box_new = [box0[0], box0[1], bz_w + 0.002, 1, 0, 0, 0]
                    io.reset_objects({"box_free": box_new}, {"table": tab_new}); rend.move_body("table", tab_new)
                    placed_h = True; box0 = np.array(box_new)
                    log(f"replay: table raised {dz * 100:+.1f} cm, box centre z {bz_w:.3f}")
                if a.replay_place_box != "off" and box0 is not None and not placed and i0 >= walk_end + 10:
                    r = io.state["root7"]; w_, x_, y_, z_ = r[3:7]
                    yaw = np.arctan2(2 * (w_ * z_ + x_ * y_), 1 - 2 * (y_ * y_ + z_ * z_)); c, s_ = np.cos(yaw), np.sin(yaw)
                    tab = rend.m.body_pos[rend.m.body("table").id].copy(); box_cur = io.state["objects"]["box_free"]
                    if a.replay_place_box == "grasp":
                        gx, gy = grasp_pt[0], grasp_pt[1]
                    else:
                        gx, gy = float(a.replay_place_box), 0.0
                    bz_w = box0[2]                                   # height already set (stage 1)
                    t_half = 0.30                                    # table half depth (builder TABLE)
                    gx = max(gx, 0.22 + box_half[0])                 # table front edge >= 0.22 m ahead of the pelvis
                    bx_w = r[:2] + gx * np.array([c, s_]) + gy * np.array([-s_, c])
                    off = t_half - box_half[0] - 0.01                # box at the table's front edge
                    tab_new = [bx_w[0] + off * c, bx_w[1] + off * s_, tab[2]]
                    box_new = [bx_w[0], bx_w[1], bz_w, np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]
                    io.reset_objects({"box_free": box_new}, {"table": tab_new}); rend.move_body("table", tab_new)
                    placed = True; box0 = np.array(box_new)
                    log(f"replay: walk ended (frame {walk_end}); table+box placed, box {np.round(box_new[:3], 3).tolist()} "
                        f"(was {np.round(box_cur[:3], 3).tolist()})")
                act = {k: np.asarray(v[np.clip(np.arange(i0, i0 + spec.action_horizon), 0, len(v) - 1)], np.float32)
                       for k, v in replay.items()}
            elif policy is None:
                act = scripted_chunk(a.scripted, time.time() - t_start, spec)
            else:
                raw, _ = policy.get_action(adapter.encode(st, a.instruction))
                act = {k.removeprefix("action."): np.asarray(v)[0] for k, v in raw.items()}
            infer_s = time.time() - t_inf; n_inf += 1
            if a.blend_steps and prev_act is not None:
                k = a.exec_steps; nb = min(a.blend_steps, spec.action_horizon - k)
                w = (np.arange(nb) + 1)[:, None] / (nb + 1)          # 0 -> 1: old plan fades into the new one
                for key in act:
                    act[key] = act[key].copy()
                    act[key][:nb] = (1 - w) * prev_act[key][k:k + nb] + w * act[key][:nb]
            prev_act = {key: v.copy() for key, v in act.items()}
            cmds = adapter.decode(act, st)[: a.exec_steps]
            rec["model"].append({k: v[: a.exec_steps] for k, v in act.items()}); rec["infer_s"].append(infer_s)
            nav = act["navigate_command"][0]
            caption = (f"GR00T G1 -> X2 | nav fwd {nav[0]:+.2f} side {nav[1]:+.2f} turn {nav[2]:+.2f} | "
                       f"infer {infer_s * 1000:.0f} ms")
            t_chunk = time.time()
            for i, c in enumerate(cmds):
                io.send(c)
                el = time.time() - t_start
                rec["t"].append(el); rec["root"].append(io.state["root7"].copy())
                rec["x2_cmd_vel"].append(c.base_velocity); rec["x2_arm_l"].append(c.left_arm_q); rec["x2_arm_r"].append(c.right_arm_q)
                if el >= next_frame:
                    next_frame += 1.0 / a.video_fps
                    rend.set_state(io.state["root7"], io.state["body_q"], io.state["lh"], io.state["rh"], io.state["objects"])
                    e = cv2.resize(rend.ego(), (480, 360)); ch = cv2.resize(rend.chase_view(), (480, 360))
                    fr = np.concatenate([e, ch], 1).copy()
                    cv2.rectangle(fr, (0, 330), (960, 360), (20, 24, 28), -1)
                    cv2.putText(fr, f"t {el:5.1f}s  {caption}", (8, 351), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (240, 236, 228), 1)
                    cv2.putText(fr, "X2 head camera (model input)", (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (246, 196, 83), 1)
                    cv2.putText(fr, f'"{a.instruction}"' if policy else (f"G1 demo replay: {a.replay.stem}" if a.replay else f"scripted: {a.scripted}"), (488, 18),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (246, 196, 83), 1)
                    bx = io.state["objects"].get("box_free")
                    if box0 is not None and bx is not None:
                        rise = float(bx[2] - box0[2]); box_max_rise = max(box_max_rise, rise)
                        clear, tilt = box_clearance(bx, box_half, box0[2] - box_half[2])
                        box_trace.append((el, rise, clear, tilt))
                        best_clear = max(best_clear, clear) if tilt < 25 else best_clear
                        if clear > 0.02 and tilt < 25:          # clean lift: lowest corner 2 cm off the table, upright
                            clean_since = el if clean_since is None else clean_since
                            if lift_t is None and el - clean_since >= 1.0:
                                lift_t = clean_since
                        else:
                            clean_since = None
                        cv2.putText(fr, f"box rise {rise * 100:+5.1f} cm" + ("  LIFTED" if lift_t is not None else ""),
                                    (488, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (80, 220, 120) if lift_t is not None else (240, 236, 228), 1)
                    dist, pressed = rend.button_check()
                    if dist is not None:
                        best_dist = min(best_dist, dist); press_t = press_t if press_t is not None or not pressed else el
                        cv2.putText(fr, f"tip->button {dist * 100:4.1f} cm" + ("  PRESSED" if press_t is not None else ""),
                                    (488, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (80, 220, 120) if press_t is not None else (240, 236, 228), 1)
                    vw.append_data(fr)
                sleep = t_chunk + (i + 1) * dt - time.time()
                if sleep > 0:
                    time.sleep(sleep)
            if time.time() - last_hb > 5:
                last_hb = time.time()
                r = io.state["root7"]
                log(f"t {time.time() - t_start:5.1f}s inferences {n_inf} last {infer_s * 1000:.0f} ms | root xy "
                    f"({r[0]:+.2f},{r[1]:+.2f}) z {r[2]:.2f} | nav {np.round(nav, 2).tolist()} | clamped {adapter.stats['clamped_joints']}"
                    + (f" | IK err mm x2 {np.mean(adapter.ik_err['x2'][-40:]) * 1e3:.1f} g1 {np.mean(adapter.ik_err['g1'][-4:]) * 1e3:.1f}" if adapter.arm_mode == "hand" else ""))
                if r[2] < 0.45:
                    log("FALL detected (pelvis z < 0.45 m), stopping"); break
    finally:
        io.release(); vw.close(); io.close()
        if box_trace:
            np.save(out / "box_trace.npy", np.array(box_trace))     # t, centre rise, upright clearance, tilt deg
        np.savez(out / "chunks.npz", t=np.array(rec["t"]), root=np.array(rec["root"]),
                 x2_cmd_vel=np.array(rec["x2_cmd_vel"]), x2_arm_l=np.array(rec["x2_arm_l"]),
                 x2_arm_r=np.array(rec["x2_arm_r"]), infer_s=np.array(rec["infer_s"]),
                 **{f"model_{k}": np.concatenate([m[k] for m in rec["model"]]) for k in (rec["model"][0] if rec["model"] else {})})
        r0, r1 = (rec["root"][0], rec["root"][-1]) if rec["root"] else (None, None)
        if r0 is not None:
            log(f"done: {n_inf} inferences, mean {np.mean(rec['infer_s']) * 1000:.0f} ms; root moved "
                f"({r1[0] - r0[0]:+.2f}, {r1[1] - r0[1]:+.2f}) m; final z {r1[2]:.2f}; clamped joints {adapter.stats['clamped_joints']}; squeeze steps {adapter.stats.get('squeeze_steps', 0)}")
        if box0 is not None:
            log(f"box: max centre rise {box_max_rise * 100:.1f} cm; best upright clearance {best_clear * 100:.1f} cm; "
                + (f"CLEAN LIFT (corner >2 cm off table, tilt <25 deg, held 1 s) from t {lift_t:.1f}s" if lift_t is not None else "no clean lift"))
        if best_dist < 9.0:
            log(f"button: closest fingertip {best_dist * 100:.1f} cm; " + (f"PRESSED at t {press_t:.1f}s" if press_t is not None else "NOT pressed"))
        log(f"wrote {out / 'video.mp4'}")


if __name__ == "__main__":
    raise SystemExit(main())
