import asyncio
import websockets
import json
import requests
from math import sqrt, pow
import numpy as np
import cv2
import gtsam
from gtsam import Pose2
from gtsam.symbol_shorthand import X

# --- Constants ---
PYTHON_WS_PORT = 8765
PHONE_URL = "http://192.168.1.15:8080/shot.jpg"# shot.jpeg used for only using realtime frames not from buffers
# the ip is for Airtel_9945484809, 192.168.1.29:8080 for Airtel_Aerial


browser_clients = set()
esp32_websocket = None  # holds the ESP32 connection specifically
slam_data = []

# Global buffer to keep ONLY the newest image frame from phone
latest_frame = None
frame_lock = asyncio.Lock()

Keyframes = []
Nodes = {}
odom_noise = gtsam.noiseModel.Diagonal.Sigmas(np.array([5.0, 5.0, np.radians(2.0)]))
lpenc_noise = gtsam.noiseModel.Diagonal.Sigmas(np.array([1.0, 1.0, np.radians(0.5)]))

# --- Camera Intrinsics (Phone Camera) ---
# Assuming 1920x1080 stream resolution. Update if your resolution differs!
FRAME_WIDTH = 1920
FRAME_HEIGHT = 1080

# Quick approximation for phone cameras: f ≈ width * 0.8
focal_length = float(FRAME_WIDTH * 0.8)

# Principal point is the optical center of the image
principal_point = (float(FRAME_WIDTH / 2.0), float(FRAME_HEIGHT / 2.0))

current_master_pose = {'x': 0.0, 'y': 0.0, 'theta': 0.0}
ODOMETRY_MODE = "ENCODER"  # Default mode; can be switched to "VISUAL" if needed

class EncoderOdometryEngine:
    def compute_step(self, telemetry_data, frame=None):
        # Calculates pure wheel differential kinematics
        # Returns: (delta_x, delta_y, delta_theta)
        
        pass

class FusedVisualOdometryEngine:
    def __init__(self, camera_intrinsics):
        self.camera_intrinsics = camera_intrinsics

    def compute_step(self, telemetry_data, frame):
        # ORB recoverPose + Scale Agreement check against telemetry_data['delta_s']
        # Returns: (delta_x, delta_y, delta_theta)
        pass

class Keyframe:
    def __init__(self,kf_id,frame,frame_pose):
        self.id = kf_id
        self.encoderpose = {'x': frame_pose['x'], 'y': frame_pose['y'], 'theta': frame_pose['theta']}
        self.visualpose = {}
        self.orb = cv2.ORB_create(nfeatures=750)
        self.keypoints, self.descriptors = self.orb.detectAndCompute(frame, mask=None)

class GraphValuesManager:
    def __init__(self):
        self.values = gtsam.Values()
        self.graph = gtsam.NonlinearFactorGraph()

    def add_node_value(self, node_id, pose_dict):
        gtsam_pose = Pose2(pose_dict['x'], pose_dict['y'], pose_dict['theta'])
        self.values.insert(X(node_id), gtsam_pose)
        if node_id == 0:
            self.graph.add(gtsam.PriorFactorPose2(
                X(node_id), 
                gtsam_pose, 
                gtsam.noiseModel.Diagonal.Sigmas(np.array([0.001, 0.001, np.radians(0.01)]))
            ))

    def add_odom_fac(self, from_id, to_id, delta_x, delta_y, yaw, noise_model):
        relative_pose = Pose2(delta_x, delta_y, yaw)
        self.graph.add(gtsam.BetweenFactorPose2(X(from_id), X(to_id), relative_pose, noise_model)) # Note: BetweenFactorPose2

    def add_lpenc_fac(self, current_id, loop_matched_id, drift, noise_model):
        lp_drift = Pose2(drift['x'], drift['y'], np.radians(drift['theta']))
        self.graph.add(gtsam.BetweenFactorPose2(X(current_id), X(loop_matched_id), lp_drift, noise_model))

    def run_optimization(self):
        params = gtsam.LevenbergMarquardtParams()
        optimizer = gtsam.LevenbergMarquardtOptimizer(self.graph, self.values, params) # Fixed self.values
        return optimizer.optimize()

# Instantiate globally so it maintains persistent state!
graph_manager = GraphValuesManager()

# ── CAMERA FETCH TASK ─────────────────────────────────────────────────────────
async def fetch_camera_frames():
    """
    Continuously pulls the latest snapshot from IP Webcam without blocking
    the WebSocket event loop.
    """
    global latest_frame
    loop = asyncio.get_running_loop()

    while True:
        try:
            # Run blocking HTTP request in thread pool so asyncio server never freezes
            img_resp = await loop.run_in_executor(None, lambda: requests.get(PHONE_URL, timeout=1))
            
            if img_resp.status_code == 200:
                img_array = np.array(bytearray(img_resp.content), dtype=np.uint8)
                frame = cv2.imdecode(img_array, cv2.IMREAD_GRAYSCALE) # it was cv2.IMREAD_COLOR check with gemini if this is correct

                async with frame_lock:
                    latest_frame = frame
                    
        except Exception as e:
            # Handles momentary Wi-Fi drops or timeouts gracefully
            pass

        # Brief pause to cap grab rate at ~30 FPS and avoid hogging CPU
        await asyncio.sleep(0.03)

async def handle_client(websocket):
    global esp32_websocket
    print(f"New connection from {websocket.remote_address}")

    try:
        async for message in websocket:
            # First message from ESP32 is "ESP32_READY" — identify it
            if message == "ESP32_READY":
                esp32_websocket = websocket
                print("[ESP32] Identified and registered")
                continue

            # Treat any other connection as a browser client
            browser_clients.add(websocket)

            # Try to parse as JSON
            try:
                data = json.loads(message)


                # If it came from ESP32 (has robot telemetry fields) → forward to browsers
                #if "current_servo_angle" in data:
                 #   slam_data.append(data)
                  #  print(slam_data)
                    # Capture the synchronized camera frame matching this scan


                    #print(f"[SLAM] Received scan #{len(slam_data)} | Servo Angle: {data['current_servo_angle']}°")
                # Broadcast raw ToF sweeps received from ESP32 directly to web clients

                if data.get("type") == "SCAN":
                    scan_measures = data.get("measure", [])
                    angle_step = 180.0 / (len(scan_measures) - 1) if len(scan_measures) > 1 else 5.0
                    
                    print(f"[SCAN] Received ToF Sweep: {len(scan_measures)} points at {angle_step:.1f}° step")

                    # Save scan with robot pose context
                    slam_data.append({
                        'pose': {'x': data['x'], 'y': data['y'], 'theta': data['theta']},
                        'measures': scan_measures,
                        'step': angle_step
                    })

                    # Forward scan data to all connected browser web clients
                    if browser_clients:
                        websockets.broadcast(browser_clients, message)

                
                
                elif "error" in data and "kp" in data:
                    if browser_clients:
                        websockets.broadcast(browser_clients, message)

                    async with frame_lock:
                        current_img = latest_frame.copy() if latest_frame is not None else None

                    if current_img is not None:
                        current_frame_pose = {'x': data['x'], 'y': data['y'], 'theta': data['theta']}

                        # 1. Handle initial keyframe
                        if len(Keyframes) == 0:
                            first_kf = Keyframe(0, current_img,current_frame_pose)
                            first_kf.visualpose = {'x': 0.0, 'y': 0.0, 'theta': 0.0}
                            Keyframes.append(first_kf)

                            # Add initial node & prior factor to global graph
                            graph_manager.add_node_value(0, first_kf.visualpose)
                            print(f"[KEYFRAME] Initial Keyframe #0 added.")

                        # 2. Handle subsequent keyframes   
                        elif should_create_keyframe(current_frame_pose, Keyframes[-1].encoderpose):
                            prev_kf = Keyframes[-1]
                            new_kf_id = len(Keyframes)
                            new_kf = Keyframe(new_kf_id, current_img, current_frame_pose)

                            # Localized arrays for this specific frame pair
                            local_matches = []
                            pts1_local = []
                            pts2_local = []

                            bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
                            matches = bf.knnMatch(new_kf.descriptors, Keyframes[-1].descriptors, k=2)

                            for match_pair in matches:
                                if len(match_pair) == 2:
                                    m, n = match_pair
                                    if m.distance < 0.75 * n.distance:
                                        local_matches.append(m)
                                        pts1_local.append(new_kf.keypoints[m.queryIdx].pt)
                                        pts2_local.append(Keyframes[-1].keypoints[m.trainIdx].pt)

                            if len(local_matches) > 8:
                                pts1_arr = np.float32(pts1_local)
                                pts2_arr = np.float32(pts2_local)

                                E, mask = cv2.findEssentialMat(pts1_arr, pts2_arr, focal=focal_length, pp=principal_point)
                                _, R, t, mask = cv2.recoverPose(E, pts1_arr, pts2_arr)

                                yaw = np.arctan2(R[1, 0], R[0, 0])

                                # Scaled Visual Odometry
                                Delta_x_fused = data['delta_s'] * t[0, 0]
                                Delta_y_fused = data['delta_s'] * t[1, 0]
                                
                                # Integrate from previous keyframe's visual pose
                                prev_vis = Keyframes[-1].visualpose
                                x_kf = prev_vis['x'] + Delta_x_fused
                                y_kf = prev_vis['y'] + Delta_y_fused
                                theta_kf = prev_vis['theta'] + yaw

                                # Correct for Camera Offset to get Robot Center Pose
                                theta = np.radians(data['current_angle'])
                                x_offset, y_offset = 5.0, 6.0  # offset in mm
                                
                                camera_pos = np.array([x_kf, y_kf])
                                offset = np.array([x_offset, y_offset])
                                
                                cos_t, sin_t = np.cos(theta), np.sin(theta)
                                rotation_matrix = np.array([
                                    [cos_t, -sin_t],
                                    [sin_t,  cos_t]
                                ])
                                
                                # SUBTRACT rotated offset to map from Camera Position -> Robot Center
                                robot_center_pos = camera_pos - (rotation_matrix @ offset)

                                new_kf.visualpose = {
                                    'x': robot_center_pos[0], 
                                    'y': robot_center_pos[1], 
                                    'theta': theta_kf
                                }

                            Keyframes.append(new_kf)
                            graph_manager.add_node_value(new_kf_id, new_kf.visualpose)
                            graph_manager.add_odom_fac(prev_kf.id, new_kf_id, Delta_x_fused, Delta_y_fused, yaw, odom_noise)

                            # 3. Handle loop enclosure
                            lp_enc_kf = await detect_loop_closure(new_kf)
                            if lp_enc_kf is not None:
                                graph_manager.add_lpenc_fac(new_kf_id, lp_enc_kf['matched_kf_id'], lp_enc_kf['drift'], lpenc_noise)
                                result_values = graph_manager.run_optimization()

                                # Update visual pose for ALL keyframes in memory
                                for kf in Keyframes:
                                    if result_values.exists(X(kf.id)):
                                        p = result_values.atPose2(X(kf.id))
                                        kf.visualpose = {'x': p.x(), 'y': p.y(), 'theta': p.theta()}
                                print("✅ Global Trajectory Corrected via Graph Optimization!")

                                trajectory_payload = json.dumps({
                                    "type": "TRAJECTORY_UPDATE",
                                    "loop_from": new_kf_id,
                                    "loop_to": lp_enc_kf['matched_kf_id'],
                                    "keyframes": [{'id': kf.id, 'x': kf.visualpose['x'], 'y': kf.visualpose['y'], 'theta': kf.visualpose['theta']} for kf in Keyframes]
                                })
                                if browser_clients:
                                    websockets.broadcast(browser_clients, trajectory_payload)


            except json.JSONDecodeError:
                # Not JSON — must be a plain command string from browser (FORWARD, STOP etc.)
                print(f"[BROWSER→ESP32] {message}")
                if esp32_websocket:
                    await esp32_websocket.send(message)

    except websockets.exceptions.ConnectionClosed:
        pass
    finally:
        # Clean up whichever type disconnecte
        if websocket == esp32_websocket:
            esp32_websocket = None
            print("[ESP32] Disconnected")
        else:
            browser_clients.discard(websocket)
            print(f"[BROWSER] Disconnected. Total: {len(browser_clients)}")

def should_create_keyframe(current_frame_pose, last_keyframe_pose, min_dist=150, min_angle=11):
    delta_d = sqrt(pow(current_frame_pose['x']-last_keyframe_pose['x'],2) + pow(current_frame_pose['y']-last_keyframe_pose['y'],2))
    delta_theta = abs(current_frame_pose['theta'] - last_keyframe_pose['theta'])
    return delta_d >= min_dist or delta_theta >= min_angle 


async def detect_loop_closure(new_kf):
    if len(Keyframes) < 10:
        return None  # Not enough history yet

    best_match_count = 0
    best_candidate_idx = -1
    best_pts_current = []
    best_pts_candidate = []

    MIN_LOOP_MATCHES = 30
    MIN_INLIER_COUNT = 20
    MIN_INLIER_RATIO = 0.50  # 50% of matches must fit 3D geometry

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    possible_loop_enc = Keyframes[:-8]  # Skip recent 8 frames

    for idx, old_kf in enumerate(possible_loop_enc):
        matches = bf.knnMatch(new_kf.descriptors, old_kf.descriptors, k=2)
        
        good_matches = []
        pts_curr = []
        pts_old = []

        for match_pair in matches:
            if len(match_pair) == 2:
                m, n = match_pair
                if m.distance < 0.75 * n.distance:
                    good_matches.append(m)
                    pts_curr.append(new_kf.keypoints[m.queryIdx].pt)
                    pts_old.append(old_kf.keypoints[m.trainIdx].pt)

        match_count = len(good_matches)

        if match_count > best_match_count:
            best_match_count = match_count
            best_candidate_idx = idx
            best_pts_current = pts_curr
            best_pts_candidate = pts_old

    # Check 1: Initial match count threshold
    if best_match_count < MIN_LOOP_MATCHES:
        return None

    # Convert point lists to numpy arrays for OpenCV RANSAC
    pts_curr_arr = np.float32(best_pts_current)
    pts_old_arr = np.float32(best_pts_candidate)

    # Check 2: Geometric Verification using Essential Matrix + RANSAC
    E, mask = cv2.findEssentialMat(
        pts_curr_arr, 
        pts_old_arr, 
        focal=focal_length, 
        pp=principal_point, 
        method=cv2.RANSAC, 
        prob=0.99, 
        threshold=1.0
    )

    if mask is None:
        return None

    # Count inliers
    inlier_count = int(np.sum(mask))
    inlier_ratio = inlier_count / best_match_count

    print(f"[LOOP CANDIDATE] Frame #{best_candidate_idx} | Matches: {best_match_count} | Inliers: {inlier_count} ({inlier_ratio*100:.1f}%)")

    # Check 3: Dual Inlier Verification Threshold
    if inlier_count >= MIN_INLIER_COUNT and inlier_ratio >= MIN_INLIER_RATIO:
        matched_kf = Keyframes[best_candidate_idx]

        # Calculate pure drift between current frame visual pose and matched historical frame visual pose
        drift_x = new_kf.visualpose['x'] - matched_kf.visualpose['x']
        drift_y = new_kf.visualpose['y'] - matched_kf.visualpose['y']
        drift_theta = new_kf.visualpose['theta'] - matched_kf.visualpose['theta']

        print(f"✅ [LOOP CLOSURE DETECTED] Drift -> X: {drift_x:.2f}mm, Y: {drift_y:.2f}mm, Theta: {np.degrees(drift_theta):.2f}°")

        return {
            'matched_kf_id': matched_kf.id,
            'drift': {'x': drift_x, 'y': drift_y, 'theta': drift_theta}
        }

    else:
        print("❌ [LOOP CLOSURE REJECTED] Geometric verification failed.")
        return None
    
        
    

async def main():
    async with websockets.serve(handle_client, "0.0.0.0", PYTHON_WS_PORT):
        print(f"[PYTHON] Server running on port {PYTHON_WS_PORT}")
        await asyncio.Future()  # run foreveror 



if __name__ == "__main__":
    asyncio.run(main())