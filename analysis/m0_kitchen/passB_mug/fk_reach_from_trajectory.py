"""Pass-A reach extraction: FK the executed palm from an IsaacLab trajectory dump.

Input: TrajectoryRecorder pkl (dof_pos in IsaacLab order @25 Hz, root_pos_w
env-origin-relative == kitchen frame, root_quat_w wxyz).
Output: left/right wrist world trajectory, reach event (low wrist speed x
max horizontal excursion), position vs the counter target.
"""
import sys

import importlib.util
import joblib
import mujoco
import numpy as np

REPO = "/home/stickbot/Projects/GR00T-WholeBodyControl"
sys.path.insert(0, REPO)
spec = importlib.util.spec_from_file_location("ev", f"{REPO}/gear_sonic/scripts/eval_x2_mujoco.py")
ev = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ev)

TRAJ = sys.argv[1]
TARGET = np.array([0.283, 2.22, 0.876])  # counter placement spot
HAND = sys.argv[2] if len(sys.argv) > 2 else "left"

t = joblib.load(TRAJ)
dof_il = np.asarray(t["dof_pos"], float)          # (T,31) IL order
root_p = np.asarray(t["root_pos_w"], float)       # (T,3) kitchen frame
root_q = np.asarray(t["root_quat_w"], float)      # (T,4) wxyz
fps = float(t["fps"])
mj2il = np.asarray(ev.MJ_TO_IL_DOF)
dof_mj = dof_il[:, mj2il]                          # (T,31) MJ order

m = mujoco.MjModel.from_xml_path(f"{REPO}/gear_sonic/data/assets/robot_description/mjcf/x2_ultra.xml")
d = mujoco.MjData(m)
bid = m.body(f"{HAND}_wrist_roll_link").id

T = dof_mj.shape[0]
wrist = np.zeros((T, 3))
for i in range(T):
    d.qpos[0:3] = root_p[i]
    d.qpos[3:7] = root_q[i]                        # MuJoCo wants wxyz — matches
    d.qpos[7 : 7 + 31] = dof_mj[i]
    mujoco.mj_forward(m, d)
    wrist[i] = d.xpos[bid]

spd = np.linalg.norm(np.gradient(wrist, 1.0 / fps, axis=0), axis=1)
horiz = np.linalg.norm(wrist[:, :2] - root_p[0, :2], axis=1)
# reach event: among low-speed frames (bottom 30%), max horizontal excursion
low = spd < np.percentile(spd, 30)
cand = np.where(low)[0]
tstar = cand[np.argmax(horiz[cand])]

print(f"frames={T} fps={fps} hand={HAND}")
print(f"t* = frame {tstar} ({tstar/fps:.2f} s)  wrist speed {spd[tstar]:.3f} m/s")
print(f"wrist at t*: {np.round(wrist[tstar], 3)}")
print(f"target spot: {TARGET}")
print(f"delta xy: {np.round(wrist[tstar,:2]-TARGET[:2],3)}  |xy|={np.linalg.norm(wrist[tstar,:2]-TARGET[:2]):.3f} m")
print(f"delta z vs counter 0.876: {wrist[tstar,2]-0.876:+.3f} m")
print(f"wrist z range: {wrist[:,2].min():.3f} .. {wrist[:,2].max():.3f}")
print(f"max horizontal excursion: {horiz.max():.3f} m at frame {horiz.argmax()}")
np.savez(TRAJ.replace(".trajectory.pkl", ".reach.npz"),
         wrist=wrist, tstar=tstar, fps=fps, target=TARGET)
