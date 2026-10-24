"""Runs ON PC2. READ-ONLY live Orbbec feed over plain HTTP (no installs: rclpy + numpy + cv2 + stdlib).

Subscribes to the Orbbec colour (compressed) and depth topics and serves the newest frames:
  GET /rgb.jpg     latest colour, resized to --width (default 640), JPEG
  GET /depth.bin   latest depth, downsampled to 1/4 (320x180), uint16 mm, zlib; header JSON line first
  GET /health      stamps + frame counts
Publishes nothing on ROS; touches no robot control. Stop with: kill $(cat /tmp/x2_gemma_live.pid)

  source /opt/ros/humble/setup.bash
  export FASTRTPS_DEFAULT_PROFILES_FILE=/agibot/software/entry/cfg/ros_dds_configuration.xml
  python3 /tmp/x2_gemma_live.py --port 8793
"""
import argparse, json, os, threading, time, zlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2
import numpy as np
import rclpy
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CompressedImage, Image

B = "/aima/hal/sensor/rgbd_head_front/"
ap = argparse.ArgumentParser()
ap.add_argument("--port", type=int, default=8793)
ap.add_argument("--width", type=int, default=640)
a = ap.parse_args()
open("/tmp/x2_gemma_live.pid", "w").write(str(os.getpid()))

lock = threading.Lock()
S = {"rgb": None, "rgb_t": 0.0, "depth": None, "depth_t": 0.0, "n_rgb": 0, "n_depth": 0, "rgb_raw": None}


def on_rgb(m):
    with lock:
        S["rgb_raw"] = bytes(m.data); S["rgb_t"] = m.header.stamp.sec + m.header.stamp.nanosec * 1e-9
        S["n_rgb"] += 1


def on_depth(m):
    d = np.frombuffer(bytes(m.data), dtype=np.uint16).reshape(m.height, m.width)[::4, ::4]
    blob = zlib.compress(np.ascontiguousarray(d).tobytes(), 1)
    hdr = json.dumps({"w": d.shape[1], "h": d.shape[0], "full_w": m.width, "full_h": m.height,
                      "t": m.header.stamp.sec + m.header.stamp.nanosec * 1e-9}).encode() + b"\n"
    with lock:
        S["depth"] = hdr + blob; S["n_depth"] += 1


def encode_latest():
    """Resize + re-encode only when asked (keeps PC2 CPU near zero when nobody watches)."""
    with lock:
        raw = S["rgb_raw"]
    if raw is None:
        return None
    img = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
    h, w = img.shape[:2]
    img = cv2.resize(img, (a.width, int(h * a.width / w)), interpolation=cv2.INTER_AREA)
    return cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 80])[1].tobytes()


class H(BaseHTTPRequestHandler):
    def log_message(self, *x):
        pass

    def _send(self, body, ctype):
        self.send_response(200); self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body))); self.end_headers(); self.wfile.write(body)

    def do_GET(self):
        if self.path.startswith("/rgb.jpg"):
            b = encode_latest()
            return self._send(b, "image/jpeg") if b else self.send_error(503)
        if self.path.startswith("/depth.bin"):
            with lock:
                b = S["depth"]
            return self._send(b, "application/octet-stream") if b else self.send_error(503)
        if self.path.startswith("/health"):
            with lock:
                h = {k: S[k] for k in ("rgb_t", "depth_t", "n_rgb", "n_depth")}
            return self._send(json.dumps(h).encode(), "application/json")
        self.send_error(404)


rclpy.init()
node = rclpy.create_node("gemma_live_readonly")
node.create_subscription(CompressedImage, B + "rgb_image/compressed", on_rgb, qos_profile_sensor_data)
node.create_subscription(Image, B + "depth_image", on_depth, qos_profile_sensor_data)
threading.Thread(target=ThreadingHTTPServer(("0.0.0.0", a.port), H).serve_forever, daemon=True).start()
print(f"[gemma_live] serving :{a.port} (read-only Orbbec rgb + depth)", flush=True)
try:
    rclpy.spin(node)
finally:
    os.path.exists("/tmp/x2_gemma_live.pid") and os.remove("/tmp/x2_gemma_live.pid")
