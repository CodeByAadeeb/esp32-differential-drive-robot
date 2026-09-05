
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
import math
import torch
from PIL import Image
from transformers import pipeline
import sys

# ── Constants ────────────────────────────────────────────────────────────
PYTHON_WS_PORT = 8765
PHONE_IP = "10.80.123.123"
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
LOOP_CLOSURE_MIN_HISTORY = 16 # 6
LOOP_CLOSURE_SKIP_RECENT = 8 # 3
MIN_LOOP_MATCHES = 8 # was 15
MIN_INLIER_COUNT = 8
MIN_INLIER_RATIO = 0.30 # was 0.5

# Odometry-proximity + ToF scan-matching loop closure (fallback when camera doesn't confirm).
ODOM_PROXIMITY_RADIUS_MM = 1000.0 #800.0
SCAN_LOOP_SKIP_RECENT = 5 # 10
SCAN_MIN_RANGE_MM = 20.0
SCAN_MAX_RANGE_MM = 1000.0
SCAN_MATCH_SEARCH_XY_MM = 300.0
SCAN_MATCH_XY_STEP_MM = 50.0
SCAN_MATCH_SEARCH_THETA_DEG = 20.0
SCAN_MATCH_THETA_STEP_DEG = 5.0
SCAN_MATCH_MAX_RESIDUAL_MM = 100.0 #80.0
SCAN_MATCH_MIN_INLIER_POINTS = 4 #6
OFFSET = 10.0



class OdometryMode(Enum):
    PURE_ENCODER = "PURE_ENCODER"
    FUSED_VISUAL = "FUSED_VISUAL"

@dataclass
class PoseDelta:
    delta_x : float = 0.0
    delta_y : float = 0.0
    delta_theta : float = 0.0

@dataclass
class GlobalPose:
    x : float = 0.0
    y : float = 0.0
    theta : float = 0.0

    def apply(self, delta: PoseDelta):
        self.x += delta.delta_x
        self.y += delta.delta_y
        self.theta += delta.delta_theta
    
    def as_dict(self):
        return {'x': self.x, 'y': self.y, 'theta': self.theta} # had math.degrees so check if theta is in degrees or radians when we pass argument

class WSLogger:
    def __init__(self):
        self.terminal = sys.stdout
        self.queue = asyncio.Queue()

    def write(self, text):
        self.terminal.write(text)  # Still prints on robot terminal if connected
        msg = text.strip()
        if msg:  # Filter out empty newlines
            try:
                loop = asyncio.get_running_loop()
                loop.call_soon_threadsafe(self.queue.put_nowait, msg)
            except RuntimeError:
                pass

    def flush(self):
        self.terminal.flush()

# Redirect python print() statements globally to our logger
ws_logger = WSLogger()
sys.stdout = ws_logger

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
            servo_angle = i * step - (np.pi / 2) # --- doubt --- why subtract pi/2 won't it be exactly opp of the actual angle?  
            world_angle = theta + servo_angle
            x = pose_rad["x"] + dist * np.cos(world_angle)
            y = pose_rad["y"] + dist * np.sin(world_angle)
            points.append((x, y))
    return points  

class TelemetryHub:
    """
    The only place raw ESP32 telemetry and raw camera frames are stored.
    Nothing downstream reads raw sensor data directly except OdometryProvider.
    """

    def __init__(self):
        self.latest_telemetry : dict = None  # just the most recent ESP32 JSON dict (only one, always overwritten)
        self.frame_buffer = deque(maxlen=60)
        self.lock = asyncio.Lock()

    async def ingest_telemetry(self, data: dict):
        async with self.lock:
            self.latest_telemetry = data

    async def ingest_frame(self, frame, timestamp: float):
        async with self.lock:
            self.frame_buffer.append((timestamp, frame))

    async def get_closest_frame(self, target_ts: float, max_age_s: float = 0.15):
        async with self.lock:
            if not self.frame_buffer:
                return None
            closest_ts, closest_frame = min(self.frame_buffer, key=lambda item: abs(item[0] - target_ts))
            if abs(closest_ts - target_ts) <= max_age_s:
                return closest_frame.copy()
            else:
                return None  

class EncoderOdometryEngine:

    def __init__(self, wheel_diameter_mm, track_width_mm, ticks_per_rev):
        self.dist_per_tick = np.pi * wheel_diameter_mm / ticks_per_rev
        self.track_width_mm = track_width_mm

    def compute_delta(self, d_ticks_left: int, d_ticks_right: int, heading_rad: float) -> PoseDelta:
        s_left = d_ticks_left * self.dist_per_tick
        s_right = d_ticks_right * self.dist_per_tick
        delta_s = (s_left + s_right) / 2.0
        delta_theta = -(s_right - s_left) / self.track_width_mm
        theta_mid = heading_rad + (delta_theta / 2.0) # to calculate x,y using the theta we get while moving not theta of the previous pose 
        # though i don't understand the equation
        return PoseDelta(
            delta_x=delta_s * np.cos(theta_mid),
            delta_y=delta_s * np.sin(theta_mid),
            delta_theta=delta_theta,
        )
    
        # ---use this if there are sharper turns while moving---
        # --doubt-- don't understand the equations
        #if abs(delta_theta) > 1e-6:
        #    radius = delta_s / delta_theta
        #    delta_x = radius * (np.sin(heading_rad + delta_theta) - np.sin(heading_rad))
        #    delta_y = radius * (np.cos(heading_rad) - np.cos(heading_rad + delta_theta))
        #else:
        #    delta_x = delta_s * np.cos(heading_rad)
        #    delta_y = delta_s * np.sin(heading_rad)
        

class VisualOdometryEngine:
    def __init__(self, focal_length: float, principal_point: tuple, scale_gate_ratio=0.4, camera_offset=(0.0, 0.0)):
        self.focal_length = focal_length
        self.principal_point = principal_point
        self.scale_gate_ratio = scale_gate_ratio
        self.camera_offset = np.array(camera_offset) 

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
            pts_new_arr, pts_prev_arr, focal=self.focal_length, pp=self.principal_point,
            method=cv2.RANSAC, prob=0.99, threshold=1.0,
        )
        if E is None:
            return self._encoder_fallback(encoder_delta_s, gyro_delta_theta, heading_rad), False

        _, R, t, _ = cv2.recoverPose(E, pts_new_arr, pts_prev_arr, mask=mask, focal=self.focal_length, pp=self.principal_point)
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
        cos_before, sin_before = np.cos(heading_rad), np.sin(heading_rad) # could we also use gyro_delta_theta here? --- No, we need the heading before the rotation to compute the offset shift correctly.
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
    
class OdometryProvider:
    def __init__(self,mode: OdometryMode,encoder_engine: EncoderOdometryEngine, visual_engine: VisualOdometryEngine):
        self.pose = GlobalPose()
        self.mode = mode
        self.encoder_engine = encoder_engine
        self.visual_engine = visual_engine
        self._prev_left_ticks : int = None
        self._prev_right_ticks : int = None
        self._prev_kf = None

    def set_mode(self, mode: OdometryMode):
        self.mode = mode

    def reset(self):
        self.pose = GlobalPose()
        self._prev_left_ticks = None
        self._prev_right_ticks = None
        self.mode = None

    def current_pose_dict(self):
        return self.pose.as_dict() if self.pose else None

    def _consume_tick_delta(self, telemetry: dict):
        if telemetry.get("tick_reset"):
            self._prev_left_ticks = 0
            self._prev_right_ticks = 0
            return 0, 0
        tl, tr = telemetry["ticks_left"], telemetry["ticks_right"]
        if self._prev_left_ticks is None:
            self._prev_left_ticks, self._prev_right_ticks = tl, tr
            return 0, 0
        d_left = tl - self._prev_left_ticks
        d_right = tr - self._prev_right_ticks
        self._prev_left_ticks, self._prev_right_ticks = tl, tr
        return d_left, d_right

    def process(self, telemetry: dict, new_kf=None) -> PoseDelta:

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

class MetricDepthEstimator:
    """
    Monocular metric depth estimation wrapper using Depth Anything V2.
    Predicts per-pixel depth in millimeters for unprojecting 2D keypoints to 3D.
    """
    def __init__(self, model_id="depth-anything/Depth-Anything-V2-Small-hf"):
        device = 0 if torch.cuda.is_available() else -1
        self.pipe = pipeline(task="depth-estimation", model=model_id, device=device)

    def estimate_depth(self, frame_gray_or_bgr: np.ndarray) -> np.ndarray:
        if len(frame_gray_or_bgr.shape) == 2:
            frame_rgb = cv2.cvtColor(frame_gray_or_bgr, cv2.COLOR_GRAY2RGB)
        else:
            frame_rgb = cv2.cvtColor(frame_gray_or_bgr, cv2.COLOR_BGR2RGB)

        pil_img = Image.fromarray(frame_rgb)
        result = self.pipe(pil_img)
        depth_map = np.array(result["depth"], dtype=np.float32)

        # Resize depth map to strictly match camera frame size if necessary
        h, w = frame_gray_or_bgr.shape[:2]
        if depth_map.shape[:2] != (h, w):
            depth_map = cv2.resize(depth_map, (w, h), interpolation=cv2.INTER_LINEAR)

        return depth_map


class CameraPnPResolver:
    """
    Converts 2D visual matches into 3D-to-2D PnP pose estimates.
    Calculates exact local spatial offset (dx, dy, dtheta) between loop-closure keyframes.
    """
    def __init__(self, focal_length: float, principal_point: tuple):
        self.focal_length = focal_length
        self.principal_point = principal_point
        self.camera_matrix = np.array([
            [focal_length, 0, principal_point[0]],
            [0, focal_length, principal_point[1]],
            [0, 0, 1]
        ], dtype=np.float64)
        self.dist_coeffs = np.zeros((4, 1), dtype=np.float64)  # Assuming rectified image

    def compute_metric_transform(self, matched_kf, new_kf, pts_matched_2d: list, pts_new_2d: list, matched_depth_map: np.ndarray):
        """
        pts_matched_2d: 2D pixel coordinates in the old matched keyframe.
        pts_new_2d: 2D pixel coordinates in the current new keyframe.
        matched_depth_map: Per-pixel metric depth map of matched_kf.
        """
        pts_3d_matched = []
        valid_pts_new = []

        fx = fy = self.focal_length
        cx, cy = self.principal_point

        # 1. Back-project 2D keypoints in matched_kf into 3D spatial points
        for pt_old, pt_new in zip(pts_matched_2d, pts_new_2d):
            u, v = int(round(pt_old[0])), int(round(pt_old[1]))
            
            # Boundary check
            if 0 <= v < matched_depth_map.shape[0] and 0 <= u < matched_depth_map.shape[1]:
                z = float(matched_depth_map[v, u])
                if z > 50.0:  # Ignore zero/noise depth values (< 50mm)
                    x = (u - cx) * z / fx
                    y = (v - cy) * z / fy
                    pts_3d_matched.append([x, y, z])
                    valid_pts_new.append(pt_new)

        if len(pts_3d_matched) < 8:
            return False, None

        pts_3d_arr = np.array(pts_3d_matched, dtype=np.float64)
        pts_2d_arr = np.array(valid_pts_new, dtype=np.float64)

        # 2. Solve Perspective-n-Point with RANSAC
        success, rvec, tvec, inliers = cv2.solvePnPRansac(
            objectPoints=pts_3d_arr,
            imagePoints=pts_2d_arr,
            cameraMatrix=self.camera_matrix,
            distCoeffs=self.dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE,
            reprojectionError=2.0,
            confidence=0.99
        )

        if not success or inliers is None or len(inliers) < 6:
            return False, None

        # 3. Convert OpenCV camera frame to Robot Local Frame
        # OpenCV Camera: +X right, +Y down, +Z forward
        # Robot Frame:   +X forward, +Y left, +dtheta counter-clockwise
        t_x, t_y, t_z = tvec.flatten()
        dx = float(t_z)
        dy = float(-t_x)

        # Extract yaw angle (rotation around camera Y-axis)
        R, _ = cv2.Rodrigues(rvec)
        dtheta = float(-np.arctan2(R[0, 2], R[2, 2]))

        return True, {"x": dx, "y": dy, "theta": dtheta}

class Keyframe:
    def __init__(self, kf_id: int, frame: np.ndarray, pose_at_capture: dict, scan_points: Optional[list] = None):
        self.id = kf_id
        self.frame = frame
        self.pose_at_capture = pose_at_capture       # SSOT pose when this frame was taken (radians theta) 
        self.visualpose = dict(pose_at_capture)      # corrected on loop closure; starts equal to SSOT pose
        self.scan_points = scan_points
        orb = cv2.ORB_create(nfeatures=750)
        self.keypoints, self.descriptors = orb.detectAndCompute(frame, mask=None)

class ScanFrame:
    def __init__(self, scan_id: int, raw_sweep: list, pose_at_capture: dict):
        self.id = scan_id
        self.raw_sweep = raw_sweep  # 1D array of millimeter distances
        self.corrected_scan_points = None
        self.pose_at_capture = pose_at_capture
        self.corrected_pose = dict(pose_at_capture)
        base_loop_noise = gtsam.noiseModel.Diagonal.Sigmas(np.array([1.0, 1.0, np.radians(0.5)]))
        self.loop_noise = gtsam.noiseModel.Robust.Create(
            gtsam.noiseModel.mEstimator.Huber.Create(1.345), 
            base_loop_noise
        )


def should_create_keyframe(current_pose: dict, last_pose: dict,
                            min_dist=KEYFRAME_MIN_DIST_MM, min_angle_deg=KEYFRAME_MIN_ANGLE_DEG) -> bool:
    dx = current_pose["x"] - last_pose["x"]
    dy = current_pose["y"] - last_pose["y"]
    delta_d = (dx ** 2 + dy ** 2) ** 0.5
    delta_theta_deg = abs(np.degrees(current_pose["theta"] - last_pose["theta"]))
    return delta_d >= min_dist or delta_theta_deg >= min_angle_deg


class SlamGraphManager:
    def __init__(self): # Nodes are poses and factors are constraints between them (odometry, loop closure, etc.)
        self.values = gtsam.Values() # stores nodes
        self.graph = gtsam.NonlinearFactorGraph() # stores factors 
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
    def add_odom_fac(self, from_id, to_id, from_pose: dict, to_pose: dict):
        p_from = Pose2(from_pose["x"], from_pose["y"], from_pose["theta"])
        p_to = Pose2(to_pose["x"], to_pose["y"], to_pose["theta"])
        
        # .between() rigorously calculates the relative transform in the local frame of p_from
        relative_pose = p_from.between(p_to)
        self.graph.add(gtsam.BetweenFactorPose2(X(from_id), X(to_id), relative_pose, self.odom_noise))

    def add_lpenc_fac(self, current_id, loop_matched_id, measured_relative: dict):
        # constraint is the LOCAL measured pose difference, not global drift
        lp_meas = Pose2(measured_relative['x'], measured_relative['y'], measured_relative['theta'])
        self.graph.add(gtsam.BetweenFactorPose2(X(current_id), X(loop_matched_id), lp_meas, self.loop_noise))

    def run_optimization(self):
        params = gtsam.LevenbergMarquardtParams()
        optimizer = gtsam.LevenbergMarquardtOptimizer(self.graph, self.values, params) # Fixed self.values
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
                 max_residual_mm=SCAN_MATCH_MAX_RESIDUAL_MM, min_inlier_points=SCAN_MATCH_MIN_INLIER_POINTS, offset=OFFSET):
        self.xy_offsets = np.arange(-search_xy_mm, search_xy_mm + 1e-6, xy_step_mm)
        self.theta_offsets = np.radians(np.arange(-search_theta_deg, search_theta_deg + 1e-6, theta_step_deg))
        self.max_residual_mm = max_residual_mm
        self.min_inlier_points = min_inlier_points
        self.offset = offset

    def match(self, raw_sweep_a: list, raw_sweep_b: list):
            """
            Local frame ToF shape matching.
            Compares raw 1D distance profiles first (independent of global pose),
            then computes local metric transform (dx, dy, dtheta) using centroids.
            """
            arr_a = np.array(raw_sweep_a, dtype=np.float64)
            arr_b = np.array(raw_sweep_b, dtype=np.float64)

            # Mask out invalid zero or out-of-range readings
            valid_mask = (arr_a > 20.0) & (arr_b > 20.0) & (arr_a < 1000.0) & (arr_b < 1000.0)
            if np.sum(valid_mask) < self.min_inlier_points:
                return False, None, None

            # ── Phase 1: Shape Similarity (1D Profile Matching) ─────────────
            max_shift_indices = 5  # Search circular index shifts (~ +/- 10 degrees)
            best_mae = float("inf")
            best_shift = 0

            for shift in range(-max_shift_indices, max_shift_indices + 1):
                shifted_a = np.roll(arr_a, shift)
                mae = float(np.mean(np.abs(shifted_a[valid_mask] - arr_b[valid_mask])))
                if mae < best_mae:
                    best_mae = mae
                    best_shift = shift

            # Reject if shapes differ too much on average
            if best_mae > self.max_residual_mm:
                return False, None, None

            # ── Phase 2: Local 2D Metric Alignment ───────────────────────────
            # Convert raw sweeps into 2D Cartesian points in local robot frame
            angles = np.linspace(0.0, np.pi, num=len(arr_a))
            shifted_arr_a = np.roll(arr_a, best_shift)

            pts_a = np.column_stack((
                shifted_arr_a[valid_mask] * np.cos(angles[valid_mask]),
                shifted_arr_a[valid_mask] * np.sin(angles[valid_mask])
            ))
            pts_b = np.column_stack((
                arr_b[valid_mask] * np.cos(angles[valid_mask]),
                arr_b[valid_mask] * np.sin(angles[valid_mask])
            ))

            # Compute geometric centroids
            centroid_a = np.mean(pts_a, axis=0)
            centroid_b = np.mean(pts_b, axis=0)

            # Initial translation offset between local centroids
            delta_x = float(centroid_a[0] - centroid_b[0])
            delta_y = float(centroid_a[1] - centroid_b[1])

            # Center point clouds at origin
            centered_a = pts_a - centroid_a
            centered_b = pts_b - centroid_b

            # Fine-tune rotation around the initial index shift angle
            step_angle = np.pi / (len(arr_a) - 1)
            base_dtheta = best_shift * step_angle
            fine_search_angles = np.radians(np.arange(-5.0, 5.5, 1.0))

            best_rot_error = float("inf")
            best_dtheta = base_dtheta

            for dtheta_offset in fine_search_angles:
                dtheta_cand = base_dtheta + dtheta_offset
                c, s = np.cos(dtheta_cand), np.sin(dtheta_cand)
                rot_matrix = np.array([[c, -s], [s, c]])

                rotated_a = centered_a @ rot_matrix.T
                mean_err = float(np.mean(np.linalg.norm(rotated_a - centered_b, axis=1)))

                if mean_err < best_rot_error:
                    best_rot_error = mean_err
                    best_dtheta = float(dtheta_cand)

            transform = {
                "dx": delta_x,
                "dy": delta_y,
                "dtheta": best_dtheta
            }

            return True, transform, best_mae
    

# ── Keyframe store + multi-modal loop closure ───────────────────────────
class KeyframeManager:
    def __init__(self, graph: SlamGraphManager):
        self.keyframes: list[Keyframe] = []
        self.graph = graph
        self.depth_estimator = MetricDepthEstimator()
        self.pnp_resolver = CameraPnPResolver(FOCAL_LENGTH, PRINCIPAL_POINT)

    def reset(self):
        self.keyframes.clear()

    def add_graph_node(self, kf: Keyframe, prev_kf: Optional[Keyframe]):
        self.keyframes.append(kf)
        if prev_kf is None:
            self.graph.add_node(kf.id, kf.visualpose, is_prior=True)
            print("[KEYFRAME] Initial Keyframe #0 added.")
        else:
            self.graph.add_node(kf.id, kf.visualpose)
            self.graph.add_odom_fac(prev_kf.id, kf.id, prev_kf.visualpose, kf.visualpose)
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

    async def detect_camera_loop_closure(self, new_kf: Keyframe, candidates: list) -> Optional[dict]:
        if len(self.keyframes) < LOOP_CLOSURE_MIN_HISTORY:
            return None
        if new_kf.descriptors is None:
            return None

        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        if not candidates:
            return None

        best_count, best_idx = 0, -1
        best_pts_new, best_pts_old = [], []

        for idx, old_kf in enumerate(candidates):
            if (old_kf.descriptors is None
                    or old_kf.descriptors.ndim != 2 # 
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

        # Inside KeyframeManager.detect_camera_loop_closure:
        if inlier_count >= MIN_INLIER_COUNT and inlier_ratio >= MIN_INLIER_RATIO:
            matched_kf = candidates[best_idx]
            
            # 1. Estimate metric depth of matched keyframe frame
            matched_depth_map = await asyncio.get_running_loop().run_in_executor(
                    None, self.depth_estimator.estimate_depth, matched_kf.frame
                )
            
            # 2. Run 3D-2D PnP RANSAC to get physical offset
            success, transform = self.pnp_resolver.compute_metric_transform(
                matched_kf, new_kf, best_pts_old, best_pts_new, matched_depth_map
            )
            
            if success:
                return {"matched_kf_id": matched_kf.id, "drift": transform, "source": "camera_pnp"}

    async def detect_loop_closure(self, new_kf: Keyframe, current_pose: dict) -> Optional[dict]:
            # 🟢 UPDATED: Use LOOP_CLOSURE_SKIP_RECENT instead of hardcoded 5
            '''proximate = self.odometry_proximate_candidates(current_pose, ODOM_PROXIMITY_RADIUS_MM, exclude_last_n=LOOP_CLOSURE_SKIP_RECENT)
            if not proximate:
                return None

            camera_result = await self.detect_camera_loop_closure(new_kf, proximate)
            if camera_result is not None:
                print("✅ [LOOP CLOSURE] confirmed by camera (ORB + RANSAC).")
                print(f"[DEBUG] Camera proximate candidates count: {len(proximate)}")
                return camera_result
'''
            return None

    
class ScanManager:
    def __init__(self, slam_graph: SlamGraphManager ,scan_matcher):
        self.scans = []
        self.scan_matcher = scan_matcher
        self.slam_graph = slam_graph
        
        # Buffer management
        self.latest_raw_sweep = None   # Stores raw JSON from ESP32
        self.has_new_sweep = False     # Flag indicating an UNPROCESSED sweep exists

    def on_esp32_sweep_received(self, sweep_points: list):
        """Callback executed ONLY when ESP32 sends a completed sweep JSON."""
        self.latest_raw_sweep = sweep_points
        self.has_new_sweep = True
        print("📥 [ESP32] Fresh ToF sweep received and buffered.")

    '''def add_scan_keyframe(self, current_pose: dict) -> Optional[ScanFrame]:
        """Creates a SLAM ScanFrame node."""
        
        # CRITICAL FIX: If driving straight without a sweep, do NOT clone old data!
        if self.has_new_sweep and self.latest_raw_sweep:
            raw_sweep = self.latest_raw_sweep
            self.has_new_sweep = False      # Reset flag
            self.latest_raw_sweep = None    # FLUSH BUFFER immediately
        else:
            # Driving straight / distance trigger only — no ToF point cloud attached
            raw_sweep = []

        scan_frame = ScanFrame(
            id=len(self.scans),
            pose_at_capture=current_pose,
            raw_sweep=raw_sweep
        )
        self.scans.append(scan_frame)
        return scan_frame'''

    def reset(self):
        self.scans.clear()

    def add_graph_node(self, scan: ScanFrame, prev_scan: Optional[ScanFrame]):
        self.scans.append(scan)
        if prev_scan is None:
            self.slam_graph.add_node(scan.id, scan.corrected_pose, is_prior=True)
            print("[SCAN] Initial Scan #0 added.")
        else:
            self.slam_graph.add_node(scan.id, scan.corrected_pose)
            self.slam_graph.add_odom_fac(prev_scan.id, scan.id, prev_scan.corrected_pose, scan.corrected_pose)
            print(f"[SCAN] #{scan.id} added at ({scan.pose_at_capture['x']:.1f}, {scan.pose_at_capture['y']:.1f})")

    def odometry_proximate_candidates(self, current_pose: dict, radius_mm: float, exclude_last_n: int) -> list:
        if len(self.scans) <= exclude_last_n:
            return []
        out = []
        pool = self.scans[:-exclude_last_n] if exclude_last_n > 0 else self.scans
        for scan in pool:
            dx = current_pose["x"] - scan.pose_at_capture["x"]
            dy = current_pose["y"] - scan.pose_at_capture["y"]
            if (dx * dx + dy * dy) ** 0.5 <= radius_mm:
                out.append(scan)
        return out

    def detect_scan_loop_closure(self, new_scan: ScanFrame, candidates: list) -> Optional[dict]:
            # Rule 1: Reject if current frame was created while driving (no fresh ToF sweep)
            if not new_scan.raw_sweep or len(new_scan.raw_sweep) < 15:
                return None

            best = None
            for cand in candidates:
                # Rule 2: Ignore candidate frames that have empty sweeps
                if not cand.raw_sweep or len(cand.raw_sweep) < 15:
                    continue

                # Rule 3: Enforce index gap (must be at least SCAN_LOOP_SKIP_RECENT scans old)
                if (new_scan.id - cand.id) < SCAN_LOOP_SKIP_RECENT:
                    continue

                # Perform ToF Scan ICP Alignment
                matched, transform, residual = self.scan_matcher.match(new_scan.raw_sweep, cand.raw_sweep)
                print(f"🔍 [SCAN MATCH] #{new_scan.id} vs #{cand.id} | matched={matched} residual={residual:.1f}mm" if residual is not None else f"🔍 [SCAN MATCH] #{new_scan.id} vs #{cand.id} | matched={matched} residual=None")
                if matched and residual <= SCAN_MATCH_MAX_RESIDUAL_MM:
                    # Sanity check: Discard absurd physical jumps (> 250mm)
                    scan_shift = (transform["dx"] ** 2 + transform["dy"] ** 2) ** 0.5
                    if scan_shift > 250.0:
                        continue

                    if best is None or residual < best["residual"]:
                        measured_relative = {
                            "x": transform["dx"], 
                            "y": transform["dy"], 
                            "theta": transform["dtheta"]
                        }
                        best = {
                            "matched_kf_id": cand.id, 
                            "drift": measured_relative, 
                            "residual": residual, 
                            "source": "odom+scan"
                        }
                        
            return best
'''
    def detect_loop_closure(self, current_scan, current_pose):
        # Exclude recent scans to avoid self-matching
        candidate_scans = self.scans[:-4] 
        
        for candidate in candidate_scans:
            # Extract pose safely regardless of attribute name
            cand_pose = getattr(candidate, 'corrected_pose', None) or getattr(candidate, 'pose_dict', None) or getattr(candidate, 'raw_pose', None)
            if cand_pose:
                cand_x, cand_y, cand_theta = cand_pose['x'], cand_pose['y'], cand_pose['theta']
            else:
                cand_x, cand_y, cand_theta = candidate.x, candidate.y, candidate.theta

            # Distance calculation
            dist = np.hypot(current_pose['x'] - cand_x, current_pose['y'] - cand_y)
            
            # Heading difference calculation
            yaw_diff = abs(np.arctan2(
                np.sin(current_pose['theta'] - cand_theta),
                np.cos(current_pose['theta'] - cand_theta)
            ))

            print(f"🔍 [LOOP CHECK] Scan #{current_scan.id} vs #{candidate.id} | Dist: {dist:.1f} mm | YawDiff: {np.degrees(yaw_diff):.1f}°")

            if dist > 500.0:  # Distance threshold in mm
                continue

            if yaw_diff > np.radians(60.0):
                print(f"⚠️ [LOOP REJECTED] Scan #{candidate.id} facing difference ({np.degrees(yaw_diff):.1f}°).")
                continue

            match_result = self.scan_matcher.align(current_scan, candidate)
            print(f"📊 [ICP RESULT] Score: {getattr(match_result, 'score', 0):.3f}")

            if getattr(match_result, 'success', False):
                return {"matched_kf_id": candidate.id, "drift": match_result.transform, "source": "tof"}

        return None
'''
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
            self.camera_graph = SlamGraphManager()
            self.tof_graph = SlamGraphManager()
            self.scan_matcher = ScanMatcher()
            self.keyframes = KeyframeManager(self.camera_graph)
            self.tof_scans = ScanManager(self.tof_graph, self.scan_matcher)

            self.browser_clients: set = set()
            self.esp32_socket = None
            self._last_gyro_time = time.time()
            self.latest_scan_points: Optional[list] = None
            self.latest_raw_sweep: Optional[list] = None
            self.has_new_sweep: bool = False 

    def set_mode(self, mode: OdometryMode): # --- doubt --- who is calling this function in this code right now?
        self.odometry.set_mode(mode)

    async def handle_telemetry(self, data: dict):
            """One raw ESP32 telemetry message -> SSOT pose update -> broadcast + SLAM."""
            now = time.time()
            data["dt_s"] = max(now - self._last_gyro_time, 1e-3)
            self._last_gyro_time = now

            await self.hub.ingest_telemetry(data)
            
            # 🟢 1. UNCOMMENTED: SSOT pose update from encoder/gyro data
            self.odometry.process(data, new_kf=None)
            pose = self.odometry.current_pose_dict()

            # 🟢 2. ONLY create a ToF ScanFrame when a fresh ESP32 sweep JSON has arrived
            new_scan_candidate = None
            if self.has_new_sweep and self.latest_raw_sweep:
                candidate_id = len(self.tof_scans.scans)
                new_scan_candidate = ScanFrame(candidate_id, self.latest_raw_sweep, pose)
                
                # FLUSH BUFFER: Prevent driving straight from re-using this sweep
                self.has_new_sweep = False
                self.latest_raw_sweep = None

            if new_scan_candidate is not None:
                prev_scan = self.tof_scans.scans[-1] if self.tof_scans.scans else None
                self.tof_scans.add_graph_node(new_scan_candidate, prev_scan)

                if new_scan_candidate.id > 0:
                    candidates = self.tof_scans.odometry_proximate_candidates(
                        pose, ODOM_PROXIMITY_RADIUS_MM, exclude_last_n=SCAN_LOOP_SKIP_RECENT
                    )
                    loop = self.tof_scans.detect_scan_loop_closure(new_scan_candidate, candidates)
                    if loop is not None:
                        self.tof_graph.add_lpenc_fac(new_scan_candidate.id, loop["matched_kf_id"], loop["drift"])
                        result = self.tof_graph.run_optimization()
                        for stored_scan in self.tof_scans.scans:
                            if result.exists(X(stored_scan.id)):
                                p = result.atPose2(X(stored_scan.id))
                                stored_scan.corrected_pose = {"x": p.x(), "y": p.y(), "theta": p.theta()}
                                for stored_scan in self.tof_scans.scans:
                                    if result.exists(X(stored_scan.id)):
                                        p = result.atPose2(X(stored_scan.id))
                                        stored_scan.corrected_pose = {
                                            "x": p.x(), "y": p.y(), "theta": p.theta()
                                        }
                                        # Reproject scan points using corrected pose
                                        if stored_scan.raw_sweep:
                                            stored_scan.corrected_scan_points = scan_to_world_points(
                                                stored_scan.corrected_pose, stored_scan.raw_sweep
                                            )
                        print(f"✅ ToF trajectory corrected via graph optimization (source={loop['source']}).")
                        await self._broadcast_trajectory(new_scan_candidate.id, loop["matched_kf_id"], source="tof")

            # 3. Web UI publisher — reads the updated pose dict
            await self._broadcast_telemetry(data, pose)

    async def _broadcast_telemetry(self, raw_data: dict, pose: dict):
        if not self.browser_clients:
            return
        payload = dict(raw_data)
        payload["x"], payload["y"], payload["theta"] = pose["x"], pose["y"], np.degrees(pose["theta"])
        websockets.broadcast(self.browser_clients, json.dumps(payload))

    async def _broadcast_trajectory(self, loop_from: int, loop_to: int, source: str = "camera"): 
        if not self.browser_clients:
            return
        
        # Choose which scan/keyframe list to send
        if source == "tof":
            nodes = [
                {"id": s.id, "x": s.corrected_pose["x"], "y": s.corrected_pose["y"], "theta": s.corrected_pose["theta"], "scan_points": s.corrected_scan_points if s.corrected_scan_points else []}
                for s in self.tof_scans.scans
            ]
        else:
            nodes = [
                {"id": kf.id, "x": kf.visualpose["x"], "y": kf.visualpose["y"], "theta": kf.visualpose["theta"]}
                for kf in self.keyframes.keyframes
            ]

        payload = json.dumps({
            "type": "TRAJECTORY_UPDATE",
            "loop_from": loop_from,
            "loop_to": loop_to,
            "keyframes": nodes,
        })
        print(f"[TRAJECTORY] Broadcasting {len(nodes)} nodes, first node keys: {list(nodes[0].keys()) if nodes else 'empty'}")
        total_pts = sum(len(s.corrected_scan_points) for s in self.tof_scans.scans if s.corrected_scan_points)
        print(f"[TRAJECTORY] Total corrected scan points across all nodes: {total_pts}")
        websockets.broadcast(self.browser_clients, payload)

    async def handle_scan(self, message: str, data: dict):
            """Re-stamp the ToF sweep with the current SSOT pose, store it (in world
            frame) for the next keyframe's scan signature, then forward to browsers."""
            pose_rad = self.odometry.current_pose_dict()  # radians — internal representation
            self.latest_raw_sweep = data.get("measure", [])
            self.has_new_sweep = True  # 🟢 MARK SWEEP AS FRESH
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
                        self.camera_graph.reset()
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

    async def start_log_broadcaster(self):
            """Continuously streams intercepted print statements to browser clients."""
            while True:
                msg = await ws_logger.queue.get()
                if self.browser_clients:
                    payload = json.dumps({"type": "LOG", "message": msg})
                    websockets.broadcast(self.browser_clients, payload)
                ws_logger.queue.task_done()


async def main():
    backend = RobotBackend(mode=OdometryMode.PURE_ENCODER)
    async with websockets.serve(backend.handle_client, "0.0.0.0", PYTHON_WS_PORT):
        print(f"[PYTHON] SSOT server running on port {PYTHON_WS_PORT}")
        # Camera task created AFTER server is listening — both run concurrently
        await asyncio.gather(
            backend.fetch_camera_frames(),
            backend.start_log_broadcaster(),
            asyncio.Future(),   # keeps the server alive forever
        )


if __name__ == "__main__":
    asyncio.run(main())