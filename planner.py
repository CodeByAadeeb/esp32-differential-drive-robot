"""
Single Source of Truth (SSOT) robot backend.

Architecture
------------
Raw ESP32 telemetry (wheel ticks, gyro_z) + raw camera frames
                    |
                    v
            TelemetryHub  (thread/coroutine-safe ingestion, no pose math here)
                    |
                    v
          OdometryProvider  (the ONLY place pose is ever computed)
             /            \\
            v              v
   WebSocket Publisher   SlamGraphManager
   (Web UI browsers)     (GTSAM pose graph)

Both downstream consumers read the *same* PoseDelta / global pose objects
produced by OdometryProvider.process(). Neither the UI broadcaster nor the
SLAM graph builder is allowed to independently recompute x/y/theta from raw
ticks or ORB matches — that duplication was the original bug.

REQUIRED FIRMWARE CHANGE
-------------------------
The ESP32 must send raw sensor data, not its own pre-computed pose, e.g.:
    {"ticks_left": 1234, "ticks_right": 1198, "gyro_z_dps": 0.42,
     "kp": 0.4, "error": 0.0, "lpwm": 200, "rpwm": 200, ...}
If the ESP32 keeps sending its own x/y/theta and this backend keeps using
them, you still have two independently-drifting pose sources wearing one
JSON payload — that defeats the point of an SSOT pipeline.
"""

import asyncio
import json
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import numpy as np
import cv2
import requests
import websockets
import gtsam
from gtsam import Pose2
from gtsam.symbol_shorthand import X

# ── Constants ────────────────────────────────────────────────────────────
PYTHON_WS_PORT = 8765
PHONE_URL = "http://192.168.1.15:8080/shot.jpg"

WHEEL_DIAMETER_MM = 68.0
TRACK_WIDTH_MM = 317.0
TICKS_PER_REVOLUTION = 2340.0

FRAME_WIDTH, FRAME_HEIGHT = 1920, 1080
FOCAL_LENGTH = float(FRAME_WIDTH * 0.8)
PRINCIPAL_POINT = (float(FRAME_WIDTH / 2.0), float(FRAME_HEIGHT / 2.0))

KEYFRAME_MIN_DIST_MM = 150.0
KEYFRAME_MIN_ANGLE_DEG = 11.0
LOOP_CLOSURE_MIN_HISTORY = 10
LOOP_CLOSURE_SKIP_RECENT = 8
MIN_LOOP_MATCHES = 30
MIN_INLIER_COUNT = 20
MIN_INLIER_RATIO = 0.50


class OdometryMode(Enum):
    PURE_ENCODER = "PURE_ENCODER"
    FUSED_VISUAL = "FUSED_VISUAL"


# ── Pose primitives ──────────────────────────────────────────────────────
@dataclass
class PoseDelta:
    delta_x: float = 0.0
    delta_y: float = 0.0
    delta_theta: float = 0.0


@dataclass
class GlobalPose:
    x: float = 0.0
    y: float = 0.0
    theta: float = 0.0

    def apply(self, delta: PoseDelta):
        self.x += delta.delta_x
        self.y += delta.delta_y
        self.theta += delta.delta_theta

    def as_dict(self):
        return {"x": self.x, "y": self.y, "theta": self.theta}


# ── Thread/coroutine-safe ingestion point ───────────────────────────────
class TelemetryHub:
    """
    The only place raw ESP32 telemetry and raw camera frames are stored.
    Nothing downstream reads raw sensor data directly except OdometryProvider.
    """

    def __init__(self):
        self._lock = asyncio.Lock()
        self.latest_telemetry: Optional[dict] = None
        self.latest_frame: Optional[np.ndarray] = None

    async def ingest_telemetry(self, data: dict):
        async with self._lock:
            self.latest_telemetry = data

    async def ingest_frame(self, frame: np.ndarray):
        async with self._lock:
            self.latest_frame = frame

    async def snapshot_frame(self) -> Optional[np.ndarray]:
        async with self._lock:
            return None if self.latest_frame is None else self.latest_frame.copy()


# ── Encoder-only kinematics ──────────────────────────────────────────────
class EncoderOdometryEngine:
    """Differential-drive kinematics from raw wheel tick deltas. No camera."""

    def __init__(self, wheel_diameter_mm, track_width_mm, ticks_per_rev):
        self.dist_per_tick = np.pi * wheel_diameter_mm / ticks_per_rev
        self.track_width_mm = track_width_mm

    def compute_delta(self, d_ticks_left: int, d_ticks_right: int, heading_rad: float) -> PoseDelta:
        s_left = d_ticks_left * self.dist_per_tick
        s_right = d_ticks_right * self.dist_per_tick
        delta_s = (s_left + s_right) / 2.0
        delta_theta = (s_right - s_left) / self.track_width_mm
        return PoseDelta(
            delta_x=delta_s * np.cos(heading_rad),
            delta_y=delta_s * np.sin(heading_rad),
            delta_theta=delta_theta,
        )


# ── Visual + encoder fusion ──────────────────────────────────────────────
class VisualOdometryEngine:
    """
    Direction/rotation come from ORB + essential-matrix decomposition;
    absolute SCALE always comes from the wheel encoders (monocular vision
    has no metric scale on its own). Gyro is used as a sanity gate: if the
    visual rotation estimate disagrees wildly with gyro, the fused rotation
    falls back to gyro instead of trusting a likely-degenerate visual solve.
    """

    def __init__(self, focal_length, principal_point, scale_gate_ratio=0.4):
        self.focal = focal_length
        self.pp = principal_point
        self.scale_gate_ratio = scale_gate_ratio

    def compute_delta(self, prev_kf, new_keypoints, new_descriptors,
                       encoder_delta_s, gyro_delta_theta, heading_rad):
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        matches = bf.knnMatch(new_descriptors, prev_kf.descriptors, k=2)

        pts_new, pts_prev = [], []
        for pair in matches:
            if len(pair) == 2:
                m, n = pair
                if m.distance < 0.75 * n.distance:
                    pts_new.append(new_keypoints[m.queryIdx].pt)
                    pts_prev.append(prev_kf.keypoints[m.trainIdx].pt)

        if len(pts_new) <= 8:
            return self._encoder_fallback(encoder_delta_s, gyro_delta_theta, heading_rad), False

        pts_new_arr = np.float32(pts_new)
        pts_prev_arr = np.float32(pts_prev)

        E, mask = cv2.findEssentialMat(
            pts_new_arr, pts_prev_arr, focal=self.focal, pp=self.pp,
            method=cv2.RANSAC, prob=0.99, threshold=1.0,
        )
        if E is None:
            return self._encoder_fallback(encoder_delta_s, gyro_delta_theta, heading_rad), False

        _, R, t, _ = cv2.recoverPose(E, pts_new_arr, pts_prev_arr, mask=mask)
        visual_theta = float(np.arctan2(R[1, 0], R[0, 0]))

        disagreement = abs(visual_theta - gyro_delta_theta)
        gate = max(self.scale_gate_ratio * abs(gyro_delta_theta), np.radians(5))
        if disagreement > gate:
            fused_theta = gyro_delta_theta
            agreed = False
        else:
            fused_theta = 0.5 * (visual_theta + gyro_delta_theta)
            agreed = True

        return PoseDelta(
            delta_x=encoder_delta_s * float(t[0, 0]),
            delta_y=encoder_delta_s * float(t[1, 0]),
            delta_theta=fused_theta,
        ), agreed

    def _encoder_fallback(self, encoder_delta_s, gyro_delta_theta, heading_rad):
        return PoseDelta(
            delta_x=encoder_delta_s * np.cos(heading_rad),
            delta_y=encoder_delta_s * np.sin(heading_rad),
            delta_theta=gyro_delta_theta,
        )


# ── The single source of pose truth ─────────────────────────────────────
class OdometryProvider:
    """
    THE only object allowed to compute x/y/theta. The WebSocket publisher and
    the SLAM graph builder both call process() and read its return value —
    neither computes pose on its own.
    """

    def __init__(self, mode: OdometryMode, encoder_engine: EncoderOdometryEngine,
                 visual_engine: VisualOdometryEngine):
        self.mode = mode
        self.encoder_engine = encoder_engine
        self.visual_engine = visual_engine
        self.pose = GlobalPose()
        self._prev_ticks_left: Optional[int] = None
        self._prev_ticks_right: Optional[int] = None
        self._prev_kf = None  # last Keyframe used for visual matching

    def set_mode(self, mode: OdometryMode):
        self.mode = mode

    def _consume_tick_delta(self, telemetry: dict):
        tl, tr = telemetry["ticks_left"], telemetry["ticks_right"]
        if self._prev_ticks_left is None:
            self._prev_ticks_left, self._prev_ticks_right = tl, tr
            return 0, 0
        d_left = tl - self._prev_ticks_left
        d_right = tr - self._prev_ticks_right
        self._prev_ticks_left, self._prev_ticks_right = tl, tr
        return d_left, d_right

    def process(self, telemetry: dict, new_kf=None) -> PoseDelta:
        """
        Call once per telemetry tick. Pass `new_kf` only when a fresh camera
        keyframe was captured this tick (FUSED_VISUAL mode); otherwise pose
        falls back to encoder-only for that tick even in FUSED_VISUAL mode.
        """
        d_left, d_right = self._consume_tick_delta(telemetry)

        if self.mode == OdometryMode.PURE_ENCODER or new_kf is None or self._prev_kf is None:
            if self.mode == OdometryMode.FUSED_VISUAL and new_kf is not None and self._prev_kf is None:
                self._prev_kf = new_kf  # bootstrap: first frame has nothing to match against yet
            delta = self.encoder_engine.compute_delta(d_left, d_right, self.pose.theta)
            self.pose.apply(delta)
            return delta

        s_left = d_left * self.encoder_engine.dist_per_tick
        s_right = d_right * self.encoder_engine.dist_per_tick
        encoder_delta_s = (s_left + s_right) / 2.0
        gyro_delta_theta = np.radians(telemetry.get("gyro_z_dps", 0.0)) * telemetry.get("dt_s", 0.1)

        delta, _agreed = self.visual_engine.compute_delta(
            self._prev_kf, new_kf.keypoints, new_kf.descriptors,
            encoder_delta_s, gyro_delta_theta, self.pose.theta,
        )
        self._prev_kf = new_kf
        self.pose.apply(delta)
        return delta

    def current_pose_dict(self):
        return self.pose.as_dict()


# ── Keyframes & SLAM graph ───────────────────────────────────────────────
class Keyframe:
    def __init__(self, kf_id: int, frame: np.ndarray, pose_at_capture: dict):
        self.id = kf_id
        self.pose_at_capture = pose_at_capture       # SSOT pose when this frame was taken
        self.visualpose = dict(pose_at_capture)      # corrected on loop closure; starts equal to SSOT pose
        orb = cv2.ORB_create(nfeatures=750)
        self.keypoints, self.descriptors = orb.detectAndCompute(frame, mask=None)


def should_create_keyframe(current_pose: dict, last_kf_pose: dict,
                            min_dist=KEYFRAME_MIN_DIST_MM, min_angle_deg=KEYFRAME_MIN_ANGLE_DEG) -> bool:
    dx = current_pose["x"] - last_kf_pose["x"]
    dy = current_pose["y"] - last_kf_pose["y"]
    delta_d = (dx ** 2 + dy ** 2) ** 0.5
    delta_theta_deg = abs(np.degrees(current_pose["theta"] - last_kf_pose["theta"]))
    return delta_d >= min_dist or delta_theta_deg >= min_angle_deg


class SlamGraphManager:
    def __init__(self):
        self.values = gtsam.Values()
        self.graph = gtsam.NonlinearFactorGraph()
        self.odom_noise = gtsam.noiseModel.Diagonal.Sigmas(np.array([5.0, 5.0, np.radians(2.0)]))
        self.loop_noise = gtsam.noiseModel.Diagonal.Sigmas(np.array([1.0, 1.0, np.radians(0.5)]))

    def add_node(self, node_id: int, pose: dict, is_prior=False):
        gtsam_pose = Pose2(pose["x"], pose["y"], pose["theta"])
        self.values.insert(X(node_id), gtsam_pose)
        if is_prior:
            self.graph.add(gtsam.PriorFactorPose2(
                X(node_id), gtsam_pose,
                gtsam.noiseModel.Diagonal.Sigmas(np.array([0.001, 0.001, np.radians(0.01)])),
            ))

    def add_odometry_factor(self, from_id: int, to_id: int, delta: PoseDelta):
        rel = Pose2(delta.delta_x, delta.delta_y, delta.delta_theta)
        self.graph.add(gtsam.BetweenFactorPose2(X(from_id), X(to_id), rel, self.odom_noise))

    def add_loop_closure_factor(self, from_id: int, to_id: int, drift: dict):
        rel = Pose2(drift["x"], drift["y"], drift["theta"])
        self.graph.add(gtsam.BetweenFactorPose2(X(from_id), X(to_id), rel, self.loop_noise))

    def optimize(self):
        params = gtsam.LevenbergMarquardtParams()
        optimizer = gtsam.LevenbergMarquardtOptimizer(self.graph, self.values, params)
        result = optimizer.optimize()
        self.values = result
        return result


class KeyframeManager:
    """Owns the keyframe list, keyframe creation, and loop-closure search."""

    def __init__(self, graph: SlamGraphManager):
        self.keyframes: list[Keyframe] = []
        self.graph = graph

    def maybe_add_keyframe(self, frame: np.ndarray, ssot_pose: dict, delta: PoseDelta):
        """
        Returns the new Keyframe if one was created this tick, else None.
        Adds the corresponding node + odometry factor to the SLAM graph using
        the SAME delta the OdometryProvider just computed — no separate math.
        """
        if not self.keyframes:
            kf = Keyframe(0, frame, ssot_pose)
            self.keyframes.append(kf)
            self.graph.add_node(0, kf.visualpose, is_prior=True)
            print("[KEYFRAME] Initial Keyframe #0 added.")
            return kf

        last_kf = self.keyframes[-1]
        if not should_create_keyframe(ssot_pose, last_kf.pose_at_capture):
            return None

        new_id = len(self.keyframes)
        kf = Keyframe(new_id, frame, ssot_pose)
        self.keyframes.append(kf)

        self.graph.add_node(new_id, kf.visualpose)
        self.graph.add_odometry_factor(last_kf.id, new_id, delta)
        print(f"[KEYFRAME] #{new_id} added at ({ssot_pose['x']:.1f}, {ssot_pose['y']:.1f})")
        return kf

    def detect_loop_closure(self, new_kf: Keyframe) -> Optional[dict]:
        if len(self.keyframes) < LOOP_CLOSURE_MIN_HISTORY:
            return None

        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        candidates = self.keyframes[:-LOOP_CLOSURE_SKIP_RECENT] if len(self.keyframes) > LOOP_CLOSURE_SKIP_RECENT else []

        best_count, best_idx = 0, -1
        best_pts_new, best_pts_old = [], []

        for idx, old_kf in enumerate(candidates):
            matches = bf.knnMatch(new_kf.descriptors, old_kf.descriptors, k=2)
            good, pts_new, pts_old = [], [], []
            for pair in matches:
                if len(pair) == 2:
                    m, n = pair
                    if m.distance < 0.75 * n.distance:
                        good.append(m)
                        pts_new.append(new_kf.keypoints[m.queryIdx].pt)
                        pts_old.append(old_kf.keypoints[m.trainIdx].pt)
            if len(good) > best_count:
                best_count, best_idx = len(good), idx
                best_pts_new, best_pts_old = pts_new, pts_old

        if best_count < MIN_LOOP_MATCHES:
            return None

        pts_new_arr = np.float32(best_pts_new)
        pts_old_arr = np.float32(best_pts_old)
        E, mask = cv2.findEssentialMat(
            pts_new_arr, pts_old_arr, focal=FOCAL_LENGTH, pp=PRINCIPAL_POINT,
            method=cv2.RANSAC, prob=0.99, threshold=1.0,
        )
        if mask is None:
            return None

        inlier_count = int(np.sum(mask))
        inlier_ratio = inlier_count / best_count
        print(f"[LOOP CANDIDATE] #{best_idx} | matches={best_count} inliers={inlier_count} ({inlier_ratio*100:.1f}%)")

        if inlier_count < MIN_INLIER_COUNT or inlier_ratio < MIN_INLIER_RATIO:
            print("❌ [LOOP CLOSURE REJECTED] geometric verification failed.")
            return None

        matched_kf = candidates[best_idx]
        drift = {
            "x": new_kf.visualpose["x"] - matched_kf.visualpose["x"],
            "y": new_kf.visualpose["y"] - matched_kf.visualpose["y"],
            "theta": new_kf.visualpose["theta"] - matched_kf.visualpose["theta"],
        }
        print(f"✅ [LOOP CLOSURE] drift x={drift['x']:.1f} y={drift['y']:.1f} theta={np.degrees(drift['theta']):.1f}°")
        return {"matched_kf_id": matched_kf.id, "drift": drift}


# ── Wiring: hub -> odometry provider -> UI + SLAM ───────────────────────
class RobotBackend:
    def __init__(self, mode: OdometryMode = OdometryMode.PURE_ENCODER):
        self.hub = TelemetryHub()
        self.encoder_engine = EncoderOdometryEngine(WHEEL_DIAMETER_MM, TRACK_WIDTH_MM, TICKS_PER_REVOLUTION)
        self.visual_engine = VisualOdometryEngine(FOCAL_LENGTH, PRINCIPAL_POINT)
        self.odometry = OdometryProvider(mode, self.encoder_engine, self.visual_engine)
        self.graph = SlamGraphManager()
        self.keyframes = KeyframeManager(self.graph)

        self.browser_clients: set = set()
        self.esp32_socket = None
        self._last_gyro_time = time.time()

    def set_mode(self, mode: OdometryMode):
        self.odometry.set_mode(mode)

    async def handle_telemetry(self, data: dict):
        """One raw ESP32 telemetry message -> SSOT pose update -> broadcast + SLAM."""
        now = time.time()
        data["dt_s"] = max(now - self._last_gyro_time, 1e-3)
        self._last_gyro_time = now

        await self.hub.ingest_telemetry(data)

        frame = None
        new_kf = None
        if self.odometry.mode == OdometryMode.FUSED_VISUAL:
            frame = await self.hub.snapshot_frame()

        # 1. SSOT pose update — the ONLY place x/y/theta is computed.
        delta = self.odometry.process(data, new_kf=None)  # encoder-only tick update first
        pose = self.odometry.current_pose_dict()

        # 2. Keyframe + SLAM graph, only when a frame is available and mode is visual.
        if frame is not None:
            kf = self.keyframes.maybe_add_keyframe(frame, pose, delta)
            if kf is not None and kf.id > 0:
                loop = self.keyframes.detect_loop_closure(kf)
                if loop is not None:
                    self.graph.add_loop_closure_factor(kf.id, loop["matched_kf_id"], loop["drift"])
                    result = self.graph.optimize()
                    for stored_kf in self.keyframes.keyframes:
                        if result.exists(X(stored_kf.id)):
                            p = result.atPose2(X(stored_kf.id))
                            stored_kf.visualpose = {"x": p.x(), "y": p.y(), "theta": p.theta()}
                    print("✅ Global trajectory corrected via graph optimization.")
                    await self._broadcast_trajectory(kf.id, loop["matched_kf_id"])

        # 3. Web UI publisher — reads the SAME pose dict the graph just used.
        await self._broadcast_telemetry(data, pose)

    async def _broadcast_telemetry(self, raw_data: dict, pose: dict):
        if not self.browser_clients:
            return
        payload = dict(raw_data)
        payload["x"], payload["y"], payload["theta"] = pose["x"], pose["y"], np.degrees(pose["theta"])
        websockets.broadcast(self.browser_clients, json.dumps(payload))

    async def _broadcast_trajectory(self, loop_from: int, loop_to: int):
        if not self.browser_clients:
            return
        payload = json.dumps({
            "type": "TRAJECTORY_UPDATE",
            "loop_from": loop_from,
            "loop_to": loop_to,
            "keyframes": [
                {"id": kf.id, "x": kf.visualpose["x"], "y": kf.visualpose["y"], "theta": kf.visualpose["theta"]}
                for kf in self.keyframes.keyframes
            ],
        })
        websockets.broadcast(self.browser_clients, payload)

    async def handle_scan(self, message: str, data: dict):
        # Raw ToF sweep — forwarded as-is to browsers (already own dedicated visualization).
        if self.browser_clients:
            websockets.broadcast(self.browser_clients, message)

    async def handle_client(self, websocket):
        print(f"New connection from {websocket.remote_address}")
        try:
            async for message in websocket:
                if message == "ESP32_READY":
                    self.esp32_socket = websocket
                    print("[ESP32] Identified and registered")
                    continue

                self.browser_clients.add(websocket)

                try:
                    data = json.loads(message)
                except json.JSONDecodeError:
                    print(f"[BROWSER→ESP32] {message}")
                    if self.esp32_socket:
                        await self.esp32_socket.send(message)
                    continue

                if data.get("type") == "SCAN":
                    await self.handle_scan(message, data)
                elif "ticks_left" in data and "ticks_right" in data:
                    await self.handle_telemetry(data)
                # else: unrecognized message shape — ignored rather than silently mis-parsed.

        except websockets.exceptions.ConnectionClosed:
            pass
        finally:
            if websocket == self.esp32_socket:
                self.esp32_socket = None
                print("[ESP32] Disconnected")
            else:
                self.browser_clients.discard(websocket)
                print(f"[BROWSER] Disconnected. Total: {len(self.browser_clients)}")

    async def fetch_camera_frames(self):
        """Continuously pulls the newest camera frame into the hub (non-blocking)."""
        loop = asyncio.get_running_loop()
        while True:
            try:
                resp = await loop.run_in_executor(None, lambda: requests.get(PHONE_URL, timeout=1))
                if resp.status_code == 200:
                    img_array = np.array(bytearray(resp.content), dtype=np.uint8)
                    frame = cv2.imdecode(img_array, cv2.IMREAD_GRAYSCALE)
                    if frame is not None:
                        await self.hub.ingest_frame(frame)
            except Exception:
                pass
            await asyncio.sleep(0.03)


async def main():
    backend = RobotBackend(mode=OdometryMode.PURE_ENCODER)  # flip to FUSED_VISUAL when ready
    asyncio.create_task(backend.fetch_camera_frames())
    async with websockets.serve(backend.handle_client, "0.0.0.0", PYTHON_WS_PORT):
        print(f"[PYTHON] SSOT server running on port {PYTHON_WS_PORT}")
        await asyncio.Future()


if __name__ == "__main__":
    asyncio.run(main())