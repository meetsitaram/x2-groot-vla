# X2 cameras — sim/real reference

Canonical reference for the five head cameras: how to reach them, what they
publish, how they're named, and how to place them in sim so renders match
reality. Derived from the garage calibration session (2026-08-04); raw data
and the session narrative live in
[`assets/garage/calib_2026-08-04/`](../assets/garage/calib_2026-08-04/).

## 1. The five cameras

| # | Stream (`/aima/hal/sensor/…`) | robot `frame_id` | device | sensor / native | model |
|---|---|---|---|---|---|
| 1 | `stereo_head_front_left/rgb_image` | `stereo_head_front` | `/dev/cam1` | IMX900, 2064×1552 | fisheye k1–k4 |
| 2 | `stereo_head_front_right/rgb_image` | `stereo_head_front_right` | `/dev/cam2` | IMX900, 2064×1552 | fisheye k1–k4 |
| 3 | `rgb_head_rear/rgb_image` | `rgb_head_rear` | `/dev/cam0` | IMX900, 2064×1552 | fisheye k1–k4 |
| 4 | `rgb_head_front_center/rgb_image` | `rgb_head_front_center` | SoC2 | 5 MP, 2688×1944 | rational (8-coeff) |
| 5 | `rgbd_head_front/rgb_image` | `rgbd_head_front` | USB (Gemini 335) | 1280×720 + depth + IMU | plumb_bob |

Also on the head: the stereo IMU (`stereo_head_front/imu`) and the chest
lidar (`lidar_chest_front/*`, ethernet `10.11.1.100` on PC2's `sensor0`).

Intrinsics for all five: [`calibration.json`](../assets/garage/calib_2026-08-04/calibration.json)
(K + D as published by each camera's `camera_info`).

**Naming traps** — each of these cost real debugging time:

- `stereo_head_front` **is the LEFT eye.** The robot's frame_id omits
  "left". Our URDF calls the link `stereo_head_front_left` for clarity, so
  when bridging sim↔real TF treat the two as aliases (only intentional
  divergence).
- `rgb_head_front_center` is the **5 MP** unit — **not** the Orbbec. The
  Orbbec RGB is 1280×720 on `rgbd_head_front`.
- `gear_sonic_deploy/scripts/x2_pc2_cameras.sh grab` **mislabels** the 5 MP
  capture as `rgbd_head_front_orbbec`. Don't trust that filename; the true
  Orbbec must be grabbed from `rgbd_head_front/rgb_image/compressed`.

## 2. Reaching the cameras (PC2)

```sh
# WiFi leg (wired is run@10.0.1.41)
export X2_PC2_HOST=192.168.86.32

# Are the streams up? IMX900s are lost to a boot-time Argus race ~every boot.
bash gear_sonic_deploy/scripts/x2_pc2_cameras.sh status     # stereo/rear pubs=0 ?
bash gear_sonic_deploy/scripts/x2_pc2_cameras.sh restart-hal  # deterministic fix
bash gear_sonic_deploy/scripts/x2_pc2_cameras.sh grab       # 1 JPEG per camera
```

**Mandatory for any custom subscriber on PC2** — without this the image
topics deliver nothing (silently):

```sh
export FASTRTPS_DEFAULT_PROFILES_FILE=/agibot/software/entry/cfg/ros_dds_configuration.xml
```

Other gotchas: images/IMU need **best-effort** QoS, `camera_info` is
**transient-local**, and the IMX900 `camera_info` only publishes while the
image pipeline is streaming. Raw `v4l2-ctl` / `nvarguscamerasrc` capture
fails while the HAL owns the sensors — always go through the topics.
`restart-hal` also brings up the chest-lidar topics.

## 3. Model files

**URDF** — `gear_sonic/data/assets/robot_description/urdf/x2_ultra/x2_ultra_sphere_feet_cams_calibrated.urdf`
(calibrated variant; stock file untouched. Diff:
[`patches/urdf_cams_calibrated.diff`](../patches/urdf_cams_calibrated.diff)).

All five mounts sit on `head_pitch_link` — so camera extrinsics are
**independent of leg/waist pose**, and any robot pose validates them.
Calibration verdict: **the stock mounts were correct**; the variant's
changes are naming/completeness, not geometry.

| change | why |
|---|---|
| **added** `stereo_head_front_right` | absent from stock — the extrinsic existed only in the robot's factory `/tf_static` (60.2 mm baseline, ~0.26° rotation) |
| **renamed** `rgb_head_center` → `rgb_head_front_center` | match the robot's published frame_id |
| **renamed** `stereo_head_front` → `stereo_head_front_left` | the stock name reads as the rig centre; see naming traps |

**MJCF** — `x2_ultra.xml` has **no camera definitions** (the dataset
recorder builds its ego view programmatically). TODO for teleop-time ego
rendering: add five `<camera>` elements on `head_pitch_link` using the URDF
poses, converting optical (z-fwd, y-down) → MuJoCo (−z-fwd, y-up) with a
180° rotation about x, and `fovy` from the real intrinsics —
stereo/rear ≈ 97°, 5 MP ≈ 68°, Orbbec ≈ 55°.

## 4. Placing cameras in sim so renders match reality

Four rules, each learned the hard way (see the session README for the
failures that produced them):

1. **FOV lock — use the camera's original `K` on both sides.** Undistort the
   real frame *and* render sim at the published intrinsics. OpenCV's
   `getOptimalNewCameraMatrix` / `estimateNewCameraMatrixForUndistortRectify`
   silently widen the FOV and make sim look zoomed-out.
2. **Take pitch from the IMU, not from eyeballing.** The stereo-head IMU's
   gravity vector gives head attitude directly (~4° at stand).
3. **Front cameras face straight ahead.** Any fitted yaw offset on a front
   camera is almost certainly an error in the *robot's* pose, not the
   camera's mount.
4. **Pin the robot pose with ground truth before trusting any camera fit.**
   A wrong body pose makes every derived camera look mis-mounted — this
   produced two phantom "URDF errors" during calibration. In the garage,
   standing the robot at the yellow-circle centre fixes x/y to ±5 cm
   (circle fitted from the splat floor: centre (0.325, 1.056), radii
   0.725 / 1.500 m).

Also worth knowing: the **rear stream is ~1.3× zoomed** relative to its own
`camera_info` (ISP crop — scale `fx`/`fy` when rendering), and the
**Orbbec's raw stream is inverted** (its mount genuinely is; the driver does
not un-flip on these topics).

## 5. Tools

| script | purpose |
|---|---|
| [`scripts/isaac_garage_cams.py`](../scripts/isaac_garage_cams.py) | render all five views of a splat world in Isaac (NuRec, artifact-free) from poses/intrinsics in `isaac_cams.json` |
| [`scripts/urdf_cam_panels.py`](../scripts/urdf_cam_panels.py) | fast point-projector real-vs-sim panels for pose iteration (no Isaac boot; speckle is a projector artifact, not splat quality) |
| [`scripts/splat_pose_match.py`](../scripts/splat_pose_match.py) | single-camera pose search against a splat |

Result of the campaign:
[`all_cams_NUREC_v3.png`](../assets/garage/calib_2026-08-04/pose_match/all_cams_NUREC_v3.png)
— all five cameras, real vs sim, in the garage splat.

Open: ~5°/row residual (chamfer fit on the circle would close it), exposure
matching, and the MJCF camera block.
