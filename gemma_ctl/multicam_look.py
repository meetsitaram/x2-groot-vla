"""Gemma boxes on every head camera of one synchronized PC2 grab, with distances, and how much the
cameras disagree about the same object.

Camera geometry comes from the calibrated URDF (docs/CAMERAS.md): every mount on head_pitch_link,
the stereo right eye relative to the left, the Orbbec's physically inverted mount
(rgbd_head_front) and its software-upright child (rgbd_head_front_upright = rolled 180 deg).

  stereo_left / stereo_right  raw ~170 deg fisheye; Gemma sees the raw frame, box centres become rays
                              through each eye's own fisheye calibration (cv2.fisheye.undistortPoints)
  orbbec                      Gemini 335; raw stream is inverted -> rotate 180 = the upright frame;
                              depth is aligned to colour (same K) -> 3D point per box
  front_center                5 MP, no depth (direction only)

Distances come from the Orbbec. For each object it locates in 3D, the other cameras are scored by the
angle between where they SHOULD see it (from their URDF mount) and where Gemma's box says it is.

  python -m gemma_ctl.multicam_look runs/gemma/real/pc2_all_<stamp>
"""
from __future__ import annotations

import json
import math
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from .perceive import gemma_boxes
from .real_robot import LOOK_PROMPT

URDF = Path("/home/stickbot/Projects/GR00T-WBC-gemma/gear_sonic/data/assets/robot_description/urdf/"
            "x2_ultra/x2_ultra_sphere_feet_cams_calibrated.urdf")
FRAME = {"stereo_left": "stereo_head_front_left", "stereo_right": "stereo_head_front_right",
         "front_center": "rgb_head_front_center", "orbbec": "rgbd_head_front_upright"}
MATCH_DEG = 20.0


def urdf_frames():
    """child link -> (R, t) of that link in head_pitch_link (fixed joints only)."""
    joints = {}
    for j in ET.parse(URDF).getroot().findall("joint"):
        o = j.find("origin")
        xyz = [float(v) for v in (o.get("xyz") if o is not None else "0 0 0").split()]
        rpy = [float(v) for v in (o.get("rpy") if o is not None else "0 0 0").split()]
        joints[j.find("child").get("link")] = (j.find("parent").get("link"),
                                               Rotation.from_euler("xyz", rpy).as_matrix(), np.array(xyz))

    def to_head(link):
        R, t = np.eye(3), np.zeros(3)
        while link != "head_pitch_link":
            parent, Rj, tj = joints[link]
            R, t = Rj @ R, Rj @ t + tj
            link = parent
        return R, t
    return {k: to_head(v) for k, v in FRAME.items()}


def info(d, k):
    j = json.loads((d / f"{k}_info.json").read_text())
    return np.array(j["k"], float).reshape(3, 3), np.array(j["d"], float), j["model"]


def box_center(b, W, H):
    y0, x0, y1, x1 = b["box_2d"]
    return (x0 + x1) / 2000 * W, (y0 + y1) / 2000 * H


def normalized(cam, uv, K, D, model):
    p = np.array([[uv]], np.float32)
    if model == "fisheye":
        return cv2.fisheye.undistortPoints(p, K, D[:4])[0, 0]
    return cv2.undistortPoints(p, K, D)[0, 0]


def box_depth(depth, b, W, H, shrink=0.3):
    y0, x0, y1, x1 = b["box_2d"]
    u0, u1, v0, v1 = x0 * W / 1000, x1 * W / 1000, y0 * H / 1000, y1 * H / 1000
    du, dv = (u1 - u0) * shrink / 2, (v1 - v0) * shrink / 2
    patch = depth[int(v0 + dv):int(v1 - dv) + 1, int(u0 + du):int(u1 - du) + 1]
    v = patch[(patch > 0.2) & (patch < 15)]
    return float(np.median(v)) if v.size >= 20 else None


def noun(label):
    w = re.findall(r"[a-z]+", label.lower())
    return w[-1] if w else label


def angle(a, b):
    return math.degrees(math.acos(float(np.clip(np.dot(a, b) / np.linalg.norm(a) / np.linalg.norm(b), -1, 1))))


def run(d):
    d = Path(d); out = d / "multicam"; out.mkdir(exist_ok=True)
    frames = urdf_frames()
    imgs = {"stereo_left": cv2.imread(str(d / "stereo_left.jpg")),
            "stereo_right": cv2.imread(str(d / "stereo_right.jpg")),
            "front_center": cv2.imread(str(d / "front_center.jpg")),
            "orbbec": cv2.rotate(cv2.imread(str(d / "orbbec_rgb.jpg")), cv2.ROTATE_180)}
    odepth = cv2.rotate(np.load(d / "orbbec_depth.npy"), cv2.ROTATE_180).astype(np.float32) / 1000.0
    dets = []
    for cam, img in imgs.items():
        H, W = img.shape[:2]
        K, D, model = info(d, "orbbec_rgb" if cam == "orbbec" else cam)
        if cam == "orbbec":                              # upright image: principal point rotates with it
            K = K.copy(); K[0, 2], K[1, 2] = W - 1 - K[0, 2], H - 1 - K[1, 2]
        p = out / f"{cam}.jpg"
        cv2.imwrite(str(p), img if W <= 2100 else cv2.resize(img, (W // 2, H // 2)), [cv2.IMWRITE_JPEG_QUALITY, 92])
        boxes, dt, _ = gemma_boxes(p, "gemma4:26b", LOOK_PROMPT)
        R, t = frames[cam]
        for b in boxes:
            xn, yn = normalized(cam, box_center(b, W, H), K, D, model)
            ray = R @ np.array([xn, yn, 1.0]); ray /= np.linalg.norm(ray)
            x = dict(cam=cam, label=b.get("label", "?"), box=b["box_2d"], ray=ray.tolist(), origin=t.tolist())
            if cam == "orbbec":
                z = box_depth(odepth, b, W, H)              # depth image = distance along the optical axis
                if z is not None:
                    pt = t + R @ (z * np.array([xn, yn, 1.0]))
                    x.update(xyz=pt.round(3).tolist(), range_m=round(float(np.linalg.norm(pt - t)), 2))
            dets.append(x)
        print(f"{cam:13s} {len(boxes):2d} objects ({dt:.1f}s)", flush=True)
    # anchor on objects the Orbbec placed in 3D; find the same noun in each other camera
    rows = []
    for a in (x for x in dets if x["cam"] == "orbbec" and "xyz" in x):
        row = {"object": a["label"], "range_m": a["range_m"], "xyz": a["xyz"], "cams": {}}
        for cam in ("stereo_left", "stereo_right", "front_center"):
            want = np.array(a["xyz"]) - np.array(frames[cam][1])        # where this camera should see it
            cands = [(angle(want, x["ray"]), x) for x in dets if x["cam"] == cam and noun(x["label"]) == noun(a["label"])]
            if cands:
                err, x = min(cands, key=lambda c: c[0])
                if err < MATCH_DEG:
                    v = np.array(x["ray"]); w = want / np.linalg.norm(want)
                    daz = math.degrees(math.atan2(v[1], v[0]) - math.atan2(w[1], w[0]))
                    del_ = math.degrees(math.asin(v[2]) - math.asin(w[2]))
                    row["cams"][cam] = {"label": x["label"], "err_deg": round(err, 1),
                                        "d_az_deg": round(daz, 1), "d_el_deg": round(del_, 1)}
        rows.append(row)
    (out / "detections.json").write_text(json.dumps(dets, indent=1))
    (out / "comparison.json").write_text(json.dumps(rows, indent=1))
    return dets, rows


if __name__ == "__main__":
    dets, rows = run(sys.argv[1])
    print(f"\nOrbbec placed {len(rows)} objects in 3D. Angle between where each camera SHOULD see it "
          f"(URDF) and where Gemma's box says (az = left+, el = up+):\n")
    for r in sorted(rows, key=lambda r: r["range_m"]):
        cams = "  ".join(f"{c}: {v['err_deg']:4.1f}deg (az {v['d_az_deg']:+.1f} el {v['d_el_deg']:+.1f})"
                         for c, v in r["cams"].items()) or "(only the Orbbec)"
        print(f"{r['object'][:24]:24s} {r['range_m']:4.2f} m   {cams}")
    for cam in ("stereo_left", "stereo_right", "front_center"):
        e = [r["cams"][cam] for r in rows if cam in r["cams"]]
        if e:
            print(f"\n{cam}: {len(e)} matched, median error {np.median([x['err_deg'] for x in e]):.1f} deg, "
                  f"median d_az {np.median([x['d_az_deg'] for x in e]):+.1f}, median d_el {np.median([x['d_el_deg'] for x in e]):+.1f}")
