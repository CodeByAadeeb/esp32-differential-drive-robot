"""
Single Source of Truth (SSOT) robot backend, with multi-modal loop closure.

Architecture
------------
Raw ESP32 telemetry (wheel ticks, gyro_z) + raw camera frames + ToF sweeps
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

Loop closure voting
--------------------
1. Camera (ORB + essential-matrix + RANSAC) is checked first, against ALL
   sufficiently old keyframes. If it passes strict geometric verification,
   that's accepted immediately — it's the strongest evidence.
2. If camera does NOT confirm a closure, keyframes within ODOM_PROXIMITY_RADIUS_MM
   of the current (drifted) SSOT position are checked with a lightweight ToF
   sweep alignment (ScanMatcher). If the sweep shapes agree well after a small
   drift-correction search, that's accepted as a probable closure — odometry
   proximity + matching scan shape together are treated as sufficient evidence
   even without a camera confirmation.

REQUIRED FIRMWARE
------------------
The ESP32 must send raw sensor data, not its own pre-computed pose:
    {"ticks_left": 1234, "ticks_right": 1198, "gyro_z_dps": 0.42,
     "kp": 0.4, "error": 0.0, "lpwm": 200, "rpwm": 200, "tick_reset": false, ...}
"""

import asyncio
import json
import time
from dataclasses import dataclass
from enum import Enum
from typing import Optional

import numpy as np
import cv2
import requests
import websockets
import gtsam
from gtsam import Pose2
from gtsam.symbol_shorthand import X
from collections import deque

# ── Constants ────────────────────────────────────────────────────────────
PYTHON_WS_PORT = 8765
PHONE_IP = "10.129.54.209"
PHONE_STREAM_URL = f"http://{PHONE_IP}:8080/video"

WHEEL_DIAMETER_MM = 68.0
TRACK_WIDTH_MM = 317.0
TICKS_PER_REVOLUTION = 2340.0

# Original phone stream resolution (before the resize applied in fetch_camera_frames).
FRAME_WIDTH, FRAME_HEIGHT = 1920, 1080
_RESIZE_TARGET_WIDTH = 640
_RESIZE_SCALE = _RESIZE_TARGET_WIDTH / FRAME_WIDTH

# Calibrated intrinsics at ORIGINAL resolution, then scaled to match the resize
# applied to every frame before ORB runs on it — intrinsics MUST scale with the
# image or essential-matrix decomposition becomes systematically wrong, not just fast.
_FOCAL_LENGTH_ORIG = 525.0
_PRINCIPAL_POINT_ORIG = (323.033, 240.073)
FOCAL_LENGTH = _FOCAL_LENGTH_ORIG * _RESIZE_SCALE
PRINCIPAL_POINT = (_PRINCIPAL_POINT_ORIG[0] * _RESIZE_SCALE, _PRINCIPAL_POINT_ORIG[1] * _RESIZE_SCALE)

CAMERA_OFFSET_X_MM = -25.0  # forward(+)/behind(-) robot center
CAMERA_OFFSET_Y_MM = 70.0   # left(+)/right(-) of robot center

KEYFRAME_MIN_DIST_MM = 150.0
KEYFRAME_MIN_ANGLE_DEG = 11.0

# Camera loop-closure thresholds — kept internally consistent: inlier requirement
# stays BELOW the match requirement (inliers can never exceed total matches), and
# history exceeds skip-recent so there's ever actually a non-empty candidate set.
LOOP_CLOSURE_MIN_HISTORY = 6
LOOP_CLOSURE_SKIP_RECENT = 3
MIN_LOOP_MATCHES = 15
MIN_INLIER_COUNT = 8
MIN_INLIER_RATIO = 0.50

# Odometry-proximity + ToF scan-matching loop closure (fallback when camera doesn't confirm).
ODOM_PROXIMITY_RADIUS_MM = 800.0
SCAN_MIN_RANGE_MM = 20.0
SCAN_MAX_RANGE_MM = 1000.0
SCAN_MATCH_SEARCH_XY_MM = 300.0
SCAN_MATCH_XY_STEP_MM = 50.0
SCAN_MATCH_SEARCH_THETA_DEG = 20.0
SCAN_MATCH_THETA_STEP_DEG = 5.0
SCAN_MATCH_MAX_RESIDUAL_MM = 80.0
SCAN_MATCH_MIN_INLIER_POINTS = 6


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


def scan_to_world_points(pose_rad: dict, measures: list,
                          min_range=SCAN_MIN_RANGE_MM, max_range=SCAN_MAX_RANGE_MM) -> list:
    """Project a raw ToF sweep (list of distances across a 0-180deg arc) into
    world-frame (x, y) mm points, using the SSOT pose (theta in RADIANS) at the
    time the sweep was captured. Mirrors the same geometry index.html uses."""
    n = len(measures)
    if n < 2:
        return []
    step = np.pi / (n - 1)
    theta = pose_rad["theta"]
    points = []
    for i, dist in enumerate(measures):
        if min_range < dist < max_range:
            servo_angle = i * step - (np.pi / 2)
            world_angle = theta + servo_angle
            x = pose_rad["x"] + dist * np.cos(world_angle)
            y = pose_rad["y"] + dist * np.sin(world_angle)
            points.append((x, y))
    return points


# ── Thread/coroutine-safe ingestion point ───────────────────────────────
class TelemetryHub:
    """
    The only place raw ESP32 telemetry and raw camera frames are stored.
    Nothing downstream reads raw sensor data directly except OdometryProvider.
    """

    def __init__(self):
        self._lock = asyncio.Lock()
        self.latest_telemetry: Optional[dict] = None
        self.frame_buffer: deque = deque(maxlen=60)  # ~2 seconds of frames at 30fps

    async def ingest_telemetry(self, data: dict):
        async with self._lock:
            self.latest_telemetry = data

    async def ingest_frame(self, frame: np.ndarray, timestamp: float):
        async with self._lock:
            self.frame_buffer.append((timestamp, frame))

    async def get_closest_frame(self, target_ts: float, max_age_s: float = 0.15):
        """Return the buffered frame whose capture time is nearest target_ts,
        or None if nothing in the buffer is within max_age_s of it (better to
        skip visual odometry for a tick than silently pair a stale frame)."""
        async with self._lock:
            if not self.frame_buffer:
                return None
            closest_ts, closest_frame = min(self.frame_buffer, key=lambda item: abs(item[0] - target_ts))
            if abs(closest_ts - target_ts) > max_age_s:
                return None
            return closest_frame.copy()


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
        delta_theta = -(s_right - s_left) / self.track_width_mm
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

    def __init__(self, focal_length, principal_point, scale_gate_ratio=0.4,
                 camera_offset=(0.0, 0.0)):
        self.focal = focal_length
        self.pp = principal_point
        self.scale_gate_ratio = scale_gate_ratio
        self.camera_offset = np.array(camera_offset)  # (x, y) mm, robot frame

    def compute_delta(self, prev_kf, new_keypoints, new_descriptors,
                       encoder_delta_s, gyro_delta_theta, heading_rad):
        previous_descriptors = prev_kf.descriptors
        descriptors_usable = (
            new_descriptors is not None
            and previous_descriptors is not None
            and new_descriptors.ndim == 2
            and previous_descriptors.ndim == 2
            and new_descriptors.shape[1] == previous_descriptors.shape[1]
            and new_descriptors.dtype == previous_descriptors.dtype
        )
        if not descriptors_usable:
            return self._encoder_fallback(encoder_delta_s, gyro_delta_theta, heading_rad), False

        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        matches = bf.knnMatch(new_descriptors, previous_descriptors, k=2)

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

        # OpenCV camera convention: X=right, Y=down, Z=forward. For a forward-facing
        # camera, forward motion is camera-Z, and lateral (robot-left) is -camera-X.
        camera_dx = encoder_delta_s * float(t[2, 0])
        camera_dy = encoder_delta_s * float(-t[0, 0])

        # Correct for the camera not being at the rotation center: subtract how much
        # the offset point moved due to rotation alone, leaving pure robot-center motion.
        cos_before, sin_before = np.cos(heading_rad), np.sin(heading_rad)
        cos_after, sin_after = np.cos(heading_rad + fused_theta), np.sin(heading_rad + fused_theta)
        ox, oy = self.camera_offset
        offset_shift_x = (cos_after * ox - sin_after * oy) - (cos_before * ox - sin_before * oy)
        offset_shift_y = (sin_after * ox + cos_after * oy) - (sin_before * ox + cos_before * oy)

        return PoseDelta(
            delta_x=camera_dx - offset_shift_x,
            delta_y=camera_dy - offset_shift_y,
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
        if telemetry.get("tick_reset"):
            self._prev_ticks_left = 0
            self._prev_ticks_right = 0
            return 0, 0
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
        Call once per telemetry tick. Pass `new_kf` (a Keyframe with
        keypoints/descriptors already computed) only on ticks where a fresh
        camera keyframe was actually captured this tick — that's what makes
        visual fusion actually engage, instead of always falling back silently.
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

    def reset(self):
        self.pose = GlobalPose()
        self._prev_ticks_left = None
        self._prev_ticks_right = None
        self._prev_kf = None


# ── Keyframes ─────────────────────────────────────────────────────────
class Keyframe:
    def __init__(self, kf_id: int, frame: np.ndarray, pose_at_capture: dict, scan_points: Optional[list] = None):
        self.id = kf_id
        self.pose_at_capture = pose_at_capture       # SSOT pose when this frame was taken (radians theta)
        self.visualpose = dict(pose_at_capture)      # corrected on loop closure; starts equal to SSOT pose
        self.scan_points = scan_points                # world-frame ToF sweep points near this capture, or None
        orb = cv2.ORB_create(nfeatures=750)
        self.keypoints, self.descriptors = orb.detectAndCompute(frame, mask=None)


def should_create_keyframe(current_pose: dict, last_kf_pose: dict,
                            min_dist=KEYFRAME_MIN_DIST_MM, min_angle_deg=KEYFRAME_MIN_ANGLE_DEG) -> bool:
    dx = current_pose["x"] - last_kf_pose["x"]
    dy = current_pose["y"] - last_kf_pose["y"]
    delta_d = (dx ** 2 + dy ** 2) ** 0.5
    delta_theta_deg = abs(np.degrees(current_pose["theta"] - last_kf_pose["theta"]))
    return delta_d >= min_dist or delta_theta_deg >= min_angle_deg


# ── SLAM graph ────────────────────────────────────────────────────────
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

    def reset(self):
        self.values = gtsam.Values()
        self.graph = gtsam.NonlinearFactorGraph()


# ── Lightweight ToF sweep alignment (not a full ICP, deliberately simple) ─
class ScanMatcher:
    """
    Both point sets are already expressed in odometry world-frame, so only a
    SMALL drift-correction search is needed (not an arbitrary large transform).
    Grid-searches (dx, dy, dtheta) around zero, picks the alignment with the
    most nearest-neighbor inliers, and reports whether that's good enough to
    call it a physical revisit.
    """

    def __init__(self, search_xy_mm=SCAN_MATCH_SEARCH_XY_MM, xy_step_mm=SCAN_MATCH_XY_STEP_MM,
                 search_theta_deg=SCAN_MATCH_SEARCH_THETA_DEG, theta_step_deg=SCAN_MATCH_THETA_STEP_DEG,
                 max_residual_mm=SCAN_MATCH_MAX_RESIDUAL_MM, min_inlier_points=SCAN_MATCH_MIN_INLIER_POINTS):
        self.xy_offsets = np.arange(-search_xy_mm, search_xy_mm + 1e-6, xy_step_mm)
        self.theta_offsets = np.radians(np.arange(-search_theta_deg, search_theta_deg + 1e-6, theta_step_deg))
        self.max_residual_mm = max_residual_mm
        self.min_inlier_points = min_inlier_points

    def match(self, points_a: list, points_b: list):
        """Returns (matched: bool, transform: dict|None, mean_residual_mm: float|None)."""
        if len(points_a) < self.min_inlier_points or len(points_b) < self.min_inlier_points:
            return False, None, None

        pts_a = np.array(points_a, dtype=np.float64)
        pts_b = np.array(points_b, dtype=np.float64)

        best_inliers = 0
        best_residual = None
        best_transform = None

        # Deliberately simple nested search (not fully vectorized) — point counts
        # are small (<40) so this stays well under real-time budget; optimize later
        # if scan history grows large enough to matter.
        for dtheta in self.theta_offsets:
            c, s = np.cos(dtheta), np.sin(dtheta)
            rot = np.array([[c, -s], [s, c]])
            rotated = pts_a @ rot.T
            for dx in self.xy_offsets:
                for dy in self.xy_offsets:
                    shifted = rotated + np.array([dx, dy])
                    diffs = shifted[:, None, :] - pts_b[None, :, :]
                    dists = np.sqrt((diffs ** 2).sum(axis=2))
                    nearest = dists.min(axis=1)
                    inlier_mask = nearest <= self.max_residual_mm
                    inlier_count = int(inlier_mask.sum())
                    if inlier_count > best_inliers:
                        best_inliers = inlier_count
                        best_residual = float(nearest[inlier_mask].mean())
                        best_transform = {"dx": float(dx), "dy": float(dy), "dtheta": float(dtheta)}

        if best_inliers >= self.min_inlier_points:
            return True, best_transform, best_residual
        return False, None, None


# ── Keyframe store + multi-modal loop closure ───────────────────────────
class KeyframeManager:
    def __init__(self, graph: SlamGraphManager, scan_matcher: ScanMatcher):
        self.keyframes: list[Keyframe] = []
        self.graph = graph
        self.scan_matcher = scan_matcher

    def reset(self):
        self.keyframes.clear()

    def add_keyframe(self, kf: Keyframe, prev_kf_id: Optional[int], delta: Optional[PoseDelta]):
        self.keyframes.append(kf)
        if prev_kf_id is None:
            self.graph.add_node(kf.id, kf.visualpose, is_prior=True)
            print("[KEYFRAME] Initial Keyframe #0 added.")
        else:
            self.graph.add_node(kf.id, kf.visualpose)
            self.graph.add_odometry_factor(prev_kf_id, kf.id, delta)
            print(f"[KEYFRAME] #{kf.id} added at ({kf.pose_at_capture['x']:.1f}, {kf.pose_at_capture['y']:.1f})")

    def odometry_proximate_candidates(self, current_pose: dict, radius_mm: float, exclude_last_n: int) -> list:
        if len(self.keyframes) <= exclude_last_n:
            return []
        out = []
        pool = self.keyframes[:-exclude_last_n] if exclude_last_n > 0 else self.keyframes
        for kf in pool:
            dx = current_pose["x"] - kf.pose_at_capture["x"]
            dy = current_pose["y"] - kf.pose_at_capture["y"]
            if (dx * dx + dy * dy) ** 0.5 <= radius_mm:
                out.append(kf)
        return out

    def detect_camera_loop_closure(self, new_kf: Keyframe) -> Optional[dict]:
        if len(self.keyframes) < LOOP_CLOSURE_MIN_HISTORY:
            return None
        if new_kf.descriptors is None:
            return None

        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        candidates = self.keyframes[:-LOOP_CLOSURE_SKIP_RECENT] if len(self.keyframes) > LOOP_CLOSURE_SKIP_RECENT else []
        if not candidates:
            return None

        best_count, best_idx = 0, -1
        best_pts_new, best_pts_old = [], []

        for idx, old_kf in enumerate(candidates):
            if (old_kf.descriptors is None
                    or old_kf.descriptors.ndim != 2
                    or new_kf.descriptors.ndim != 2
                    or old_kf.descriptors.shape[1] != new_kf.descriptors.shape[1]
                    or old_kf.descriptors.dtype != new_kf.descriptors.dtype):
                continue
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
        print(f"[LOOP CANDIDATE:camera] #{best_idx} | matches={best_count} inliers={inlier_count} ({inlier_ratio*100:.1f}%)")

        if inlier_count < MIN_INLIER_COUNT or inlier_ratio < MIN_INLIER_RATIO:
            print("❌ [LOOP CLOSURE:camera] geometric verification failed.")
            return None

        matched_kf = candidates[best_idx]
        drift = {
            "x": new_kf.visualpose["x"] - matched_kf.visualpose["x"],
            "y": new_kf.visualpose["y"] - matched_kf.visualpose["y"],
            "theta": new_kf.visualpose["theta"] - matched_kf.visualpose["theta"],
        }
        return {"matched_kf_id": matched_kf.id, "drift": drift, "source": "camera"}

    def detect_scan_loop_closure(self, new_kf: Keyframe, candidates: list) -> Optional[dict]:
        if not new_kf.scan_points or len(new_kf.scan_points) < self.scan_matcher.min_inlier_points:
            return None

        best = None
        for cand in candidates:
            if not cand.scan_points:
                continue
            matched, transform, residual = self.scan_matcher.match(new_kf.scan_points, cand.scan_points)
            if matched and (best is None or residual < best["residual"]):
                drift = {
                    "x": new_kf.visualpose["x"] - cand.visualpose["x"] + transform["dx"],
                    "y": new_kf.visualpose["y"] - cand.visualpose["y"] + transform["dy"],
                    "theta": new_kf.visualpose["theta"] - cand.visualpose["theta"] + transform["dtheta"],
                }
                best = {"matched_kf_id": cand.id, "drift": drift, "residual": residual, "source": "odom+scan"}
        return best

    def detect_loop_closure(self, new_kf: Keyframe, current_pose: dict) -> Optional[dict]:
        camera_result = self.detect_camera_loop_closure(new_kf)
        if camera_result is not None:
            print("✅ [LOOP CLOSURE] confirmed by camera (ORB + RANSAC).")
            return camera_result

        proximate = self.odometry_proximate_candidates(current_pose, ODOM_PROXIMITY_RADIUS_MM, exclude_last_n=2)
        if not proximate:
            return None

        scan_result = self.detect_scan_loop_closure(new_kf, proximate)
        if scan_result is not None:
            print(f"🔶 [LOOP CLOSURE] no camera match, but odometry-proximity + ToF scan agree "
                  f"(residual={scan_result['residual']:.1f}mm, kf#{scan_result['matched_kf_id']}) "
                  f"— accepting as probable closure.")
            return scan_result

        return None


# ── Wiring: hub -> odometry provider -> UI + SLAM ───────────────────────
class RobotBackend:
    def __init__(self, mode: OdometryMode = OdometryMode.FUSED_VISUAL):
        self.hub = TelemetryHub()
        self.encoder_engine = EncoderOdometryEngine(WHEEL_DIAMETER_MM, TRACK_WIDTH_MM, TICKS_PER_REVOLUTION)
        self.visual_engine = VisualOdometryEngine(
            FOCAL_LENGTH, PRINCIPAL_POINT,
            camera_offset=(CAMERA_OFFSET_X_MM, CAMERA_OFFSET_Y_MM),
        )
        self.odometry = OdometryProvider(mode, self.encoder_engine, self.visual_engine)
        self.graph = SlamGraphManager()
        self.scan_matcher = ScanMatcher()
        self.keyframes = KeyframeManager(self.graph, self.scan_matcher)

        self.browser_clients: set = set()
        self.esp32_socket = None
        self._last_gyro_time = time.time()
        self.latest_scan_points: Optional[list] = None

    def set_mode(self, mode: OdometryMode):
        self.odometry.set_mode(mode)

    async def handle_telemetry(self, data: dict):
        """One raw ESP32 telemetry message -> SSOT pose update -> broadcast + SLAM."""
        now = time.time()
        data["dt_s"] = max(now - self._last_gyro_time, 1e-3)
        self._last_gyro_time = now

        await self.hub.ingest_telemetry(data)

        new_kf_candidate = None
        if self.odometry.mode == OdometryMode.FUSED_VISUAL:
            frame = await self.hub.get_closest_frame(now)
            if frame is not None:
                last_kf = self.keyframes.keyframes[-1] if self.keyframes.keyframes else None
                approx_pose = self.odometry.current_pose_dict()  # pose as of end of previous tick
                due = (last_kf is None) or should_create_keyframe(approx_pose, last_kf.pose_at_capture)
                if due:
                    candidate_id = len(self.keyframes.keyframes)
                    new_kf_candidate = Keyframe(candidate_id, frame, approx_pose)

        # 1. SSOT pose update — the ONLY place x/y/theta is computed. Visual fusion
        #    engages here whenever a candidate keyframe is due this tick (fixes the
        #    old hardcoded new_kf=None that silently disabled visual fusion).
        delta = self.odometry.process(data, new_kf=new_kf_candidate)
        pose = self.odometry.current_pose_dict()

        # 2. Finalize keyframe with the corrected pose + attach the latest ToF sweep
        #    as its scan signature, add graph node/factor, then run multi-modal
        #    loop-closure detection (camera first, odometry+scan fallback second).
        if new_kf_candidate is not None:
            new_kf_candidate.pose_at_capture = pose
            new_kf_candidate.visualpose = dict(pose)
            new_kf_candidate.scan_points = self.latest_scan_points

            prev_id = self.keyframes.keyframes[-1].id if self.keyframes.keyframes else None
            self.keyframes.add_keyframe(new_kf_candidate, prev_id, delta)

            if new_kf_candidate.id > 0:
                loop = self.keyframes.detect_loop_closure(new_kf_candidate, pose)
                if loop is not None:
                    self.graph.add_loop_closure_factor(new_kf_candidate.id, loop["matched_kf_id"], loop["drift"])
                    result = self.graph.optimize()
                    for stored_kf in self.keyframes.keyframes:
                        if result.exists(X(stored_kf.id)):
                            p = result.atPose2(X(stored_kf.id))
                            stored_kf.visualpose = {"x": p.x(), "y": p.y(), "theta": p.theta()}
                    print(f"✅ Global trajectory corrected via graph optimization (source={loop['source']}).")
                    await self._broadcast_trajectory(new_kf_candidate.id, loop["matched_kf_id"])

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
        """Re-stamp the ToF sweep with the current SSOT pose, store it (in world
        frame) for the next keyframe's scan signature, then forward to browsers."""
        pose_rad = self.odometry.current_pose_dict()  # radians — internal representation
        self.latest_scan_points = scan_to_world_points(pose_rad, data.get("measure", []))

        if self.browser_clients:
            data["x"] = pose_rad["x"]
            data["y"] = pose_rad["y"]
            data["theta"] = np.degrees(pose_rad["theta"])
            websockets.broadcast(self.browser_clients, json.dumps(data))

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
                    if message == "RESET":
                        self.odometry.reset()
                        self.keyframes.reset()
                        self.graph.reset()
                        self.latest_scan_points = None
                        print("[SSOT] Pose provider + SLAM graph reset alongside ESP32.")
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
        """Continuously pulls frames from a persistent MJPEG stream, resizing
        each one to match the intrinsics calibrated for _RESIZE_TARGET_WIDTH."""
        loop = asyncio.get_running_loop()

        def read_mjpeg_stream():
            print(f"🔄 [CAMERA] Connecting to persistent stream at {PHONE_STREAM_URL}...")
            while True:
                try:
                    response = requests.get(PHONE_STREAM_URL, stream=True, timeout=5)
                    if response.status_code != 200:
                        print(f"❌ [CAMERA] HTTP {response.status_code} error. Retrying...")
                        time.sleep(1)
                        continue

                    print("✅ [CAMERA] Persistent stream connected successfully!")
                    bytes_buffer = b""

                    for chunk in response.iter_content(chunk_size=4096):
                        bytes_buffer += chunk
                        start = bytes_buffer.find(b'\xff\xd8')
                        end = bytes_buffer.find(b'\xff\xd9')

                        if start != -1 and end != -1:
                            jpg_data = bytes_buffer[start:end + 2]
                            bytes_buffer = bytes_buffer[end + 2:]

                            img_array = np.frombuffer(jpg_data, dtype=np.uint8)
                            frame = cv2.imdecode(img_array, cv2.IMREAD_GRAYSCALE)

                            if frame is not None:
                                scale = _RESIZE_TARGET_WIDTH / frame.shape[1]
                                frame = cv2.resize(frame, (_RESIZE_TARGET_WIDTH, int(frame.shape[0] * scale)))
                                asyncio.run_coroutine_threadsafe(
                                    self.hub.ingest_frame(frame, time.time()),
                                    loop,
                                )

                except Exception as e:
                    print(f"❌ [CAMERA] Connection error: {e}. Reconnecting in 1s...")
                    time.sleep(1)

        await loop.run_in_executor(None, read_mjpeg_stream)


async def main():
    backend = RobotBackend(mode=OdometryMode.FUSED_VISUAL)
    async with websockets.serve(backend.handle_client, "0.0.0.0", PYTHON_WS_PORT):
        print(f"[PYTHON] SSOT server running on port {PYTHON_WS_PORT}")
        # Camera task created AFTER server is listening — both run concurrently
        await asyncio.gather(
            backend.fetch_camera_frames(),
            asyncio.Future(),   # keeps the server alive forever
        )


if __name__ == "__main__":
    asyncio.run(main())