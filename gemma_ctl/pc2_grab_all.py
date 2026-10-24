"""One-shot, READ-ONLY grab on PC2: one frame from every head camera + Orbbec depth + camera_info.
Writes /tmp/x2_gemma_grab/ and exits. Publishes nothing."""
import json, os, sys, time
import numpy as np, rclpy
from rclpy.qos import qos_profile_sensor_data, QoSProfile, DurabilityPolicy, ReliabilityPolicy
from sensor_msgs.msg import CompressedImage, Image, CameraInfo
OUT = "/tmp/x2_gemma_grab"; os.makedirs(OUT, exist_ok=True)
for f in os.listdir(OUT): os.remove(os.path.join(OUT, f))
B = "/aima/hal/sensor/"
JPG = {"stereo_left": B + "stereo_head_front_left/rgb_image/compressed",
       "stereo_right": B + "stereo_head_front_right/rgb_image/compressed",
       "front_center": B + "rgb_head_front_center/rgb_image/compressed",
       "orbbec_rgb": B + "rgbd_head_front/rgb_image/compressed"}
INFO = {"stereo_left": B + "stereo_head_front_left/camera_info", "stereo_right": B + "stereo_head_front_right/camera_info",
        "front_center": B + "rgb_head_front_center/camera_info", "orbbec_rgb": B + "rgbd_head_front/rgb_camera_info",
        "orbbec_depth": B + "rgbd_head_front/depth_camera_info"}
got = {}
rclpy.init(); n = rclpy.create_node("gemma_oneshot_grab")
def jpg_cb(k):
    def cb(m):
        if k in got: return
        open(f"{OUT}/{k}.jpg", "wb").write(bytes(m.data)); got[k] = m.header.stamp.sec + m.header.stamp.nanosec * 1e-9
    return cb
def depth_cb(m):
    if "orbbec_depth" in got: return
    a = np.frombuffer(bytes(m.data), dtype=np.uint16 if "16" in m.encoding else np.float32).reshape(m.height, m.width)
    np.save(f"{OUT}/orbbec_depth.npy", a); got["orbbec_depth"] = m.header.stamp.sec + m.header.stamp.nanosec * 1e-9
    json.dump({"encoding": m.encoding, "w": m.width, "h": m.height}, open(f"{OUT}/orbbec_depth_meta.json", "w"))
def info_cb(k):
    def cb(m):
        if "info_" + k in got: return
        json.dump({"w": m.width, "h": m.height, "k": list(m.k), "d": list(m.d), "model": m.distortion_model,
                   "frame_id": m.header.frame_id}, open(f"{OUT}/{k}_info.json", "w")); got["info_" + k] = 1
    return cb
subs = [n.create_subscription(CompressedImage, t, jpg_cb(k), qos_profile_sensor_data) for k, t in JPG.items()]
subs.append(n.create_subscription(Image, B + "rgbd_head_front/depth_image", depth_cb, qos_profile_sensor_data))
for k, t in INFO.items():
    for q in (qos_profile_sensor_data, QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL,
                                                   reliability=ReliabilityPolicy.RELIABLE)):
        subs.append(n.create_subscription(CameraInfo, t, info_cb(k), q))
t0 = time.time()
while time.time() - t0 < 15 and len(got) < len(JPG) + 1 + len(INFO):
    rclpy.spin_once(n, timeout_sec=0.2)
json.dump(got, open(f"{OUT}/stamps.json", "w"))
print("GOT", sorted(got), f"{time.time() - t0:.1f}s")
