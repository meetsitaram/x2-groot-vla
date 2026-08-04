# Garage camera-calibration session — 2026-08-04 (robot hanging in gantry)

## Contents

- `calibration.json` — real intrinsics for all four cameras (from PC2
  `camera_info` topics) + IMU samples (torso, chest, stereo-head gravity).
  Stereo left/right + rear are OpenCV **fisheye/equidistant** (k1..k4);
  front_center (Orbbec RGB) is 8-coeff rational. Stereo-head IMU gives raw
  gravity only (no fused orientation).
- `stereo_head_front_left.jpg` etc. — simultaneous captures from all four
  cameras (robot hanging over the mat, door open).
- `pose_match/` — first-light pose search: `real_undistorted.png` (fisheye →
  pinhole via cv2.fisheye), `v*_NN.png` side-by-side panels + `_blend`
  overlays vs the garage splat.

## First-light result

Stereo-left camera pose in the garage/splat frame (mat surface = z0),
bracketed by visual landmark alignment (circle, door frame, couch, boxes):

    x = -0.32 +/- 0.10, y = 0.35 +/- 0.15, z = 1.25 +/- 0.05
    yaw = -98 +/- 5 deg, pitch(down) = 22 +/- 3 deg

Best panels: `v4_00` (slightly far), `v5_00` (slightly near). Tool:
`x2-groot-vla/scripts/splat_pose_match.py` (fast numpy splat projector — no
Isaac needed; splat point cache in `pose_match/splat_pts.npz`).

Known scene difference: the exit door is OPEN in the captures, CLOSED in the
splat scan — align on the door frame, not the panel.

## Next steps

1. Automated refinement: ORB/SIFT match real_undistorted ↔ splat render,
   PnP with splat depth → exact stereo-left pose (goal: <2 cm / <1°).
2. URDF extrinsic validation: from the solved stereo-left pose, predict
   stereo-right / front_center / rear poses via URDF (all on
   head_pitch_link — joint-independent) and check each against its real
   capture; residuals = URDF mount errors.
3. RGBD: remember the mount is physically inverted (URDF encodes it; driver
   un-flips images) — roll the sim camera 180° about the optical axis.
4. Isaac render pass: same pose, `OpenCVCameraCfg`/fisheye with the real
   K/D → pixel-space comparison with distortion in the loop.
5. IMU cross-check: stereo-head gravity (8.30, 1.00, 4.74) vs solved camera
   attitude through the URDF stereo_imu frame.

## Standing verification (same day, robot upright on the mat)

- Fresh captures in `standing/`; torso IMU (-0.62, -0.07, 9.78) — upright,
  ~3.6 deg tilt. Robot turned ~35 deg left vs the hanging heading; exit door
  now closed (matches the scan).
- Renderer upgraded (`scripts/urdf_cam_panels.py`): sim now projects through
  the REAL camera models (OpenCV fisheye k1..k4 / rational 8-coeff) against
  RAW frames — native FOV both sides (fixes the earlier undistort-crop FOV
  mismatch); depth-adaptive point splatting (kills the black speckle, which
  was a point-projector artifact, not sim quality).
- Standing anchor (stereo-left, hand-fit): (-0.25, 0.55, 1.42, yaw -75,
  pitch 15).
- Panel: `pose_match/all_cams_standing.png` —
  * stereo left/right: match (window/door/boxes/circle/couch aligned,
    fisheye curvature agrees).
  * rear: match (garage door, table+chair, workbench all correspond,
    upright).
  * **rgbd/Orbbec: CONFIRMED URDF pitch error** — sim renders ~20-25 deg
    more downward than the real stream, consistent across BOTH robot poses
    (hanging + standing). The URDF rpy (2.2689, 0, 1.5708) encodes ~40 deg
    down; the real (driver-corrected) stream looks ~15-20 deg down.
    Quantify exactly in the PnP pass, then correct the SIM camera frame
    (not the URDF — it may reflect the physical mount with the driver
    correcting more than a flip).

## Automated refinement attempts (2026-08-04 evening) — both failed, lessons logged

- SIFT+PnP (`scripts/pnp_refine.py`): 45 ratio-matches, 0 RANSAC inliers —
  descriptor matching does not survive the point-splat vs photo appearance
  gap.
- Photometric Nelder-Mead (`scripts/photo_refine.py`): blurred-Sobel +
  yellow-mask global NCC is too weak a signal (NCC ~0 even at good poses;
  black undistortion corners poison it) — optimizer drifted AWAY from the
  hand-fit. Do not reuse this metric.
- NEXT (right design): chamfer alignment — distance-transform of the
  yellow-line mask (circle + tape lines lock x/y/z/yaw hard) plus masked
  door-frame edges, coarse grid then simplex; or match against a true NuRec
  render (Isaac) instead of point splats so SIFT works.
- Current best poses remain the HAND-FIT anchors: hanging
  (-0.32, 0.30, 1.25, -100, 22), standing (-0.25, 0.55, 1.42, -75, 15).
  Verified conclusions unchanged: stereo pair + rear placements match;
  rgbd ~20-25 deg more level in reality than URDF's 40-deg-down.
- Shareable panels (under chat-upload size): pose_match/*_small.jpg

## Final standing panel (v3) — IMU-constrained anchor

`pose_match/all_cams_standing_v3.png` (+ `_small.jpg`), anchor
(-0.20, 0.20, 1.42, yaw -71, pitch 9, roll 0):
- Attitude pinned by the standing stereo-head IMU: gravity in IMU frame
  (-0.997, 0.033, -0.067) -> camera ~4 deg from level (IMU x=up,
  z~=optical fwd); best visual fit at 9 deg down => ~5 deg module-vs-IMU
  mount tilt folded in. Height 1.42 from standing kinematics.
- The user's two FOV observations were BOTH the same translation error
  (anchor too far from the front wall): fixed by moving ~0.4 m forward —
  stereo "too wide" and rear "too zoomed" resolved together.
- Row verdicts: stereo L/R match (~5 deg residual yaw), REAR now matches
  convincingly (garage door / chair / table / workbench all correct size +
  position), RGBD confirmed pointing ~25-30 deg further down than the real
  stream — the isolated URDF/driver-chain error to correct in the sim
  camera frame.

## RGBD orientation fitted + URDF variant created

- Fitted the real Orbbec stream's frame against the splat (rgbdfit_25_11):
  pitch ~11 deg down, yaw ~25 deg LEFT of the stereo optical axis — the
  URDF-predicted frame (even after the 180-deg flip) is 45 deg off the
  driver-corrected stream.
- Corrected joint written to a NEW URDF variant (original untouched):
  `gear_sonic/data/assets/robot_description/urdf/x2_ultra/x2_ultra_sphere_feet_cams_calibrated.urdf`
  rgbd_head_front rpy: (2.2689, 0, 1.5708) -> (-1.6202, -0.0662, -1.1424)
  (upright optical convention, matches what the robot actually streams).
  Diff: x2-groot-vla/patches/urdf_rgbd_calibrated.diff. Accuracy ~+/-5 deg
  (limited by the hand-fit anchor); tighten with the chamfer fit.
- The 25-deg yaw offset from the stereo axis is surprisingly large for a
  "forward" module — worth a physical glance at the chin camera's mounting
  angle to confirm it really points left-of-forward.

## CORRECTION (user caught it): FIVE cameras, not four

- `rgb_head_front_center` topic = the 5MP center camera (2688x1944, level
  mount) — this is what the grab script mislabels "rgbd_head_front_orbbec"
  and what all earlier row-4 panels showed. The fitted frame (11 deg down,
  25 deg left) belongs to THIS camera; URDF variant now corrects
  `rgb_head_center` (rpy -1.6202 -0.0662 -1.1424) and rgbd_head_front is
  REVERTED to its original physical mount.
- The TRUE Orbbec RGBD = `/aima/hal/sensor/rgbd_head_front/*` (remapped by
  the orbbec_camera em-app launch; 1280x720 rgb + depth + pointcloud +
  imu; intrinsics in calibration.json as ci_rgbd_head_front_true). Raw
  stream verified DOWN-LOOKING and INVERTED (upside-down box text) —
  matching the URDF's inverted down-forward mount. 5-row panel:
  `pose_match/all_cams_5row.png` (+_small.jpg); row-5 residual aim offset
  ~15 deg pending refinement.
- Patches: x2-groot-vla/patches/urdf_cams_calibrated.diff (rgb_head_center
  only).

## Late-session fixes + remaining residuals (context for the precision pass)

- RENDERER BUG FIXED: the z>0.12 visibility gate culled all content beyond
  ~85 deg off-axis — the fisheye's edge content (corner shelf at ~95 deg)
  could never render, which poisoned every earlier yaw/position judgement.
  Now gated by distance + 100-deg cone (scripts/urdf_cam_panels.py).
- Rear camera: real stream is ~1.3x ZOOMED vs its camera_info
  (--rear-zoom 1.3 matches scale) — likely ISP crop; verify on PC2.
- True Orbbec row: upright-vs-upright fit at anchor+30 deg yaw, 48 deg down
  (--true-fit 30 48 --flip-true-real) — close, ~8 deg residual.
- 5MP center row: fitted anchor+25 yaw, 11 pitch (--rgbd-fit 25 11);
  URDF variant carries it as rgb_head_center.
- REMAINING: stereo anchor has a ~10-15 deg yaw / tens-of-cm position
  ambiguity (real view includes the warm corner-shelf at far left that the
  sim can't yet reach) — manual iteration oscillates; resolve with the
  chamfer/PnP fit, THEN re-derive all dependent fits in one pass.
- Robot has been moved off the standing pose — do NOT recapture; all fits
  reference the existing standing/ set.
- Final panels: all_cams_FINAL.png (pre-edge-fix), yaw_sweep_grid.jpg,
  edgefix.png. Regenerate after the precision fit.

## Circle-center session (2026-08-04, second robot placement)

- User placed the robot at the yellow-circle center: fitted from splat floor
  = (0.325, 1.056), rings 0.725/1.50 m, 4.7 cm RMS -> x/y pinned. Yaw
  fitted -42 deg (warm shelf + window + door + boxes all aligned). Captures
  in `captures_circle/` (renamed convention: rgb_head_front = 5MP center;
  rgbd_front_orbbec = true Orbbec).
- **NuRec (speckle-free) side-by-side**: `pose_match/all_cams_NUREC.png`
  via `scripts/isaac_garage_cams.py` (headless Isaac, per-camera matched
  pinhole intrinsics incl. rear 1.3x, poses from
  `pose_match/isaac_cams.json`). Isaac's simulation_app.close() can hang
  after saves — kill it; renders are already written.
- Residuals: ~10 deg yaw-family offsets per row + exposure differences
  (real frames dark, sim dome-lit). Precision pass (chamfer on the two
  rings — now trivially strong at this pose) remains the path to <1 deg.

## FINAL panel: all_cams_NUREC_v3.png (2026-08-04)

Rules that made it converge (bake into all future sim-camera work):
1. FOV LOCK: undistort real AND render sim at the ORIGINAL K — never
   OpenCV's auto "optimal/estimate" newK (wide-FOV artifacts on both the
   rational 5MP and the fisheyes).
2. PITCH FROM IMU: stereo-head gravity says ~4 deg — trust it over visual
   fits (user's physics check: at circle center the inner ring CANNOT be
   visible to a standing stereo cam; v3 passes).
3. FRONT CAMERAS FACE STRAIGHT AHEAD: the old +25/+30 yaw "fits" were
   old-session anchor artifacts; straight-ahead leveled the horizon and
   centered the rings. rgb_head_front pitch ~6, orbbec pitch ~42 (URDF).
4. Robot-at-circle-center: ring fit (0.325, 1.056) + user placement =
   position to ~5 cm; head cams sit ~8 cm ahead of body center.
Residuals (~5 deg yaw family per row, orbbec ~4 deg pitch, exposure):
chamfer pass when needed. MJCF x2_ultra.xml has NO cameras — add
calibrated camera sites for teleop ego rendering (queued).

## Model-file outcome (the calibration's verdict on URDF/MJCF)

- URDF: **VALIDATED, not corrected** — all five camera mounts match reality
  at nominal values (the earlier rgb_head_center "+25 yaw" edit was an
  anchor artifact; REVERTED). The variant
  `x2_ultra_sphere_feet_cams_calibrated.urdf` now differs from stock only
  by documentation comments (diff: patches/urdf_cams_calibrated.diff).
  Stream-level facts that live OUTSIDE the URDF: rear stream ~1.3x zoomed
  vs camera_info; orbbec raw stream inverted (matches its mount); the
  "rgb_head_front_center" topic is served by the 5MP unit.
- MJCF: `x2_ultra.xml` has NO camera definitions (the recorder builds its
  ego view programmatically). QUEUED: add five <camera> elements on
  head_pitch_link using the URDF poses (convert optical z-fwd/y-down to
  MuJoCo -z-fwd/y-up = 180deg about x) with fovy from real intrinsics
  (stereo/rear: 2*atan(776/685) ~= 97 deg; 5MP: 2*atan(972/1441.7) ~= 68;
  orbbec: 2*atan(360/692.6) ~= 55) — needed for teleop-time ego rendering.

## Camera naming — authoritative table (frame_ids read from the robot)

Source of truth = `header.frame_id` on each camera_info topic (queried
2026-08-04). URDF variant now MATCHES all five, so TF lookups resolve:

| ROS topic (/aima/hal/sensor/…) | robot frame_id | URDF link | our capture file |
|---|---|---|---|
| stereo_head_front_left/camera_info | `stereo_head_front` | `stereo_head_front_left` **(RENAMED)** | stereo_head_front_left.jpg |
| stereo_head_front_right/camera_info | `stereo_head_front_right` | `stereo_head_front_right` **(ADDED)** | stereo_head_front_right.jpg |
| rgb_head_rear/camera_info | `rgb_head_rear` | `rgb_head_rear` | rgb_head_rear.jpg |
| rgb_head_front_center/camera_info | `rgb_head_front_center` | `rgb_head_front_center` **(RENAMED** from rgb_head_center**)** | rgb_head_front.jpg |
| rgbd_head_front/rgb_camera_info | `rgbd_head_front` | `rgbd_head_front` | rgbd_front_orbbec.jpg |

Gotchas that cost time — keep in mind:
- **The robot publishes the LEFT eye's frame_id as `stereo_head_front`**
  (name omits "left"). Our URDF now calls that link `stereo_head_front_left`
  for clarity — when bridging sim<->real TF, treat `stereo_head_front` as an
  ALIAS of `stereo_head_front_left`. This is the only place our link names
  intentionally differ from the robot's.
- `rgb_head_front_center` is the **5 MP** unit (2688x1944, SoC2), NOT the
  Orbbec. The Orbbec RGB is 1280x720 on `rgbd_head_front`.
- `x2_pc2_cameras.sh grab` **mislabels** the 5 MP center capture as
  "rgbd_head_front_orbbec" — fix or ignore, but don't trust that filename.

## URDF changes made (calibrated variant)

1. **Added `stereo_head_front_right`** — absent from stock; its extrinsic
   lived only in the robot's factory `/tf_static`. Baseline 60.2 mm,
   rotation ~0.26 deg. Parent = `stereo_head_front` (left eye), matching
   the factory TF. Verified: 42 links / 41 joints / single root, no
   dangling refs.
2. **Renamed `rgb_head_center` -> `rgb_head_front_center`** to match the
   robot's published frame_id.
3. **Renamed `stereo_head_front` -> `stereo_head_front_left`** for internal
   consistency across the five cameras (robot-side alias noted above; mesh
   filename `stereo_head_front_link.STL` deliberately unchanged). Verified
   no code binds to the old link name — the `stereo_head_front` strings in
   `live_vla_publish_motion_token.py` / recorder are dataset STREAM keys,
   not URDF links. Tree re-validated: 42 links / 41 joints / single root.
Diff: `x2-groot-vla/patches/urdf_cams_calibrated.diff`
