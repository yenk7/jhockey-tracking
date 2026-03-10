import cv2
import numpy as np
import time
import asyncio
import base64
import os
import json
import threading
import platform
from collections import deque

import tracker_config as cfg


# ──────────────────────────────────────────────────────────────────────────────
# Threaded camera capture — hides USB I/O latency behind detection work
# ──────────────────────────────────────────────────────────────────────────────

class ThreadedCamera:
    """Continuously grabs frames in a background thread so the main loop
    always gets the latest frame without blocking on USB I/O (~20-30ms saved)."""

    def __init__(self, cap):
        self.cap = cap
        self._frame = None
        self._ret = False
        self._lock = threading.Lock()
        self._running = True
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()

    def _capture_loop(self):
        while self._running:
            ret, frame = self.cap.read()
            with self._lock:
                self._ret = ret
                self._frame = frame

    def read(self):
        with self._lock:
            if self._frame is not None:
                return self._ret, self._frame.copy()
            return False, None

    def release(self):
        self._running = False
        self._thread.join(timeout=2)
        self.cap.release()

    # Delegate property setters so callers can still do cap.set(...)
    def set(self, prop, val):
        return self.cap.set(prop, val)

    def get(self, prop):
        return self.cap.get(prop)

# Store both corner positions and their pixel locations when locked
locked_corners = None  # World coordinates of corners
locked_pixel_positions = None  # Pixel positions of corners when locked
lock_state = False  # Indicates whether corners are locked

# History for marker filtering
marker_history = {}  # Stores history of marker positions
position_kalman_filters = {}  # Stores Kalman filters for each marker

# Movement stability filtering
previous_robot_positions = {}  # Stores last sent positions for each robot
movement_threshold = {'x': 0.0, 'y': 0.0}  # Minimum movement required to send update (in cm)

# Frame processing optimization
frame_count = 0
last_stream_frame_time = 0

# Simple recovery tracking
marker_last_seen = {}  # Track when each marker was last detected
marker_timestamps = {}  # Track last-seen timestamps per marker
markers_seen_before = set()  # Track markers seen at least once
recovery_mode = False  # Recovery flag
recovery_missing_streak = 0
recovery_found_streak = 0


def reset_tracking_state():
    """Reset runtime tracking state for a fresh tracking session."""
    global locked_corners, locked_pixel_positions, lock_state
    global marker_history, position_kalman_filters, previous_robot_positions
    global marker_last_seen, marker_timestamps, markers_seen_before
    global recovery_mode, recovery_missing_streak, recovery_found_streak

    marker_history.clear()
    position_kalman_filters.clear()
    previous_robot_positions.clear()
    marker_last_seen.clear()
    marker_timestamps.clear()
    markers_seen_before.clear()
    recovery_mode = False
    recovery_missing_streak = 0
    recovery_found_streak = 0
    locked_corners = None
    locked_pixel_positions = None
    lock_state = False


def prune_stale_tracking_state(now):
    """Remove stale marker and robot state that should not survive indefinitely."""
    stale_history_ids = {
        mid for mid, ts in marker_last_seen.items()
        if now - ts > cfg.MARKER_HISTORY_EXPIRY_S
    }

    for mid in stale_history_ids:
        marker_history.pop(mid, None)

    stale_state_ids = {
        mid for mid, ts in marker_timestamps.items()
        if now - ts > cfg.SEEN_MARKER_PRUNE_TIMEOUT_S
    }

    for mid in stale_state_ids:
        marker_last_seen.pop(mid, None)
        marker_timestamps.pop(mid, None)
        marker_history.pop(mid, None)
        position_kalman_filters.pop(mid, None)
        markers_seen_before.discard(mid)
        previous_robot_positions.pop(mid, None)

    return len(stale_history_ids), len(stale_state_ids)

def make_parameters_recovery_friendly(base_params):
    """Return a much more permissive parameter set for recovery mode."""
    recovery_params = base_params.copy()

    # Adaptive thresholding: widen windows, increase constant
    recovery_params['adaptiveThreshWinSizeMin'] = 3
    recovery_params['adaptiveThreshWinSizeMax'] = max(31, base_params.get('adaptiveThreshWinSizeMax', 31))
    recovery_params['adaptiveThreshConstant'] = max(7, base_params.get('adaptiveThreshConstant', 7))

    # Size constraints: allow smaller and larger perimeters
    recovery_params['minMarkerPerimeterRate'] = max(0.005, base_params.get('minMarkerPerimeterRate', 0.03) * cfg.RECOVERY_PERIMETER_MIN_SCALE)
    recovery_params['maxMarkerPerimeterRate'] = min(5.0, base_params.get('maxMarkerPerimeterRate', 4.0) * cfg.RECOVERY_PERIMETER_MAX_SCALE)

    # Shape/approximation tolerance
    recovery_params['polygonalApproxAccuracyRate'] = min(0.12, base_params.get('polygonalApproxAccuracyRate', 0.03) * cfg.RECOVERY_APPROX_SCALE)
    recovery_params['minCornerDistanceRate'] = max(0.005, base_params.get('minCornerDistanceRate', 0.05) * cfg.RECOVERY_CORNER_DIST_SCALE)

    # Border & refinement
    recovery_params['minDistanceToBorder'] = 0
    recovery_params['cornerRefinementMethod'] = 0  # CORNER_REFINE_NONE for speed

    # Error correction tolerance
    recovery_params['errorCorrectionRate'] = max(cfg.RECOVERY_ERROR_CORRECTION_MIN, base_params.get('errorCorrectionRate', 0.35))

    print("🔍 Recovery parameters prepared: permissive detection")
    return recovery_params

def _ensure_odd(n: int, fallback: int) -> int:
    try:
        n = int(n)
    except Exception:
        return fallback
    return n if n % 2 == 1 else n + 1

def _build_detector(dictionary, params_dict):
    """Helper to build an ArUco detector from a params dict with sanitization."""
    detector_params = cv2.aruco.DetectorParameters()

    # Sanitize critical parameters before applying
    # markerBorderBits must be >= 1
    mbb = int(params_dict.get('markerBorderBits', 1))
    params_dict['markerBorderBits'] = max(1, mbb)

    # perspectiveRemoveIgnoredMarginPerCell in [0, 0.49]
    # (>=0.5 can produce zero/negative inner ROI per cell in OpenCV internals)
    margin = float(params_dict.get('perspectiveRemoveIgnoredMarginPerCell', 0.13))
    params_dict['perspectiveRemoveIgnoredMarginPerCell'] = min(0.49, max(0.0, margin))

    # maxErroneousBitsInBorderRate in [0, 1]
    mebr = float(params_dict.get('maxErroneousBitsInBorderRate', 0.35))
    params_dict['maxErroneousBitsInBorderRate'] = min(1.0, max(0.0, mebr))

    # minSideLengthCanonicalImg sufficiently large to avoid zero cell size; floor at 24
    msl = int(params_dict.get('minSideLengthCanonicalImg', 24))
    params_dict['minSideLengthCanonicalImg'] = max(24, msl)

    # Adaptive threshold windows must be positive odd and min <= max
    if 'adaptiveThreshWinSizeMin' in params_dict:
        params_dict['adaptiveThreshWinSizeMin'] = max(3, _ensure_odd(params_dict['adaptiveThreshWinSizeMin'], 3))
    if 'adaptiveThreshWinSizeMax' in params_dict:
        params_dict['adaptiveThreshWinSizeMax'] = max(5, _ensure_odd(params_dict['adaptiveThreshWinSizeMax'], 31))
    if params_dict.get('adaptiveThreshWinSizeMax', 31) < params_dict.get('adaptiveThreshWinSizeMin', 3):
        params_dict['adaptiveThreshWinSizeMax'] = params_dict['adaptiveThreshWinSizeMin'] + 2

    # errorCorrectionRate in [0,1]
    ecr = float(params_dict.get('errorCorrectionRate', 0.5))
    params_dict['errorCorrectionRate'] = min(1.0, max(0.0, ecr))

    # minDistanceToBorder >= 0
    if 'minDistanceToBorder' in params_dict:
        try:
            params_dict['minDistanceToBorder'] = max(0, int(params_dict['minDistanceToBorder']))
        except Exception:
            params_dict['minDistanceToBorder'] = 0

    for key, value in params_dict.items():
        if hasattr(detector_params, key):
            setattr(detector_params, key, value)
    return cv2.aruco.ArucoDetector(dictionary, detector_params)

def get_detector_for_mode(dictionary, base_params, in_recovery_mode):
    """Get a detector configured for normal or recovery mode."""
    if in_recovery_mode:
        params_dict = make_parameters_recovery_friendly(base_params)
        print("🔍 Using recovery detector parameters")
    else:
        params_dict = make_parameters_dynamic_friendly(base_params)
    # Ensure a few required parameters exist (values will be sanitized in _build_detector)
    params_dict.setdefault('markerBorderBits', 1)
    params_dict.setdefault('minSideLengthCanonicalImg', 24 if in_recovery_mode else 24)
    params_dict.setdefault('perspectiveRemoveIgnoredMarginPerCell', 0.13)
    return _build_detector(dictionary, params_dict)

def check_recovery_needed(raw_ids, now):
    """Determine whether recovery mode should be active based on visibility history."""
    global recovery_mode, marker_timestamps, markers_seen_before
    global recovery_missing_streak, recovery_found_streak

    current_markers = set()
    if raw_ids is not None and len(raw_ids) > 0:
        current_markers = {int(mid[0]) for mid in raw_ids}

    current_known_markers = current_markers & markers_seen_before

    # Success conditions: if previously seen markers are visible again
    if recovery_mode:
        if current_known_markers:
            recovery_found_streak += 1
            recovery_missing_streak = 0
            if recovery_found_streak >= cfg.RECOVERY_EXIT_CONSECUTIVE_HITS:
                print(f"✅ Recovery complete - found markers: {sorted(list(current_known_markers))}")
                recovery_mode = False
                recovery_found_streak = 0
                return False
        else:
            recovery_found_streak = 0
        return True

    # Decide if we've been without known markers for long enough
    if markers_seen_before:
        if not current_known_markers:
            last_seen_times = [now - marker_timestamps.get(mid, 0) for mid in markers_seen_before]
            if last_seen_times and min(last_seen_times) > cfg.RECOVERY_TIMEOUT_S:
                recovery_missing_streak += 1
                if recovery_missing_streak >= cfg.RECOVERY_ENTER_CONSECUTIVE_MISSES:
                    print(f"🔄 Recovery mode activated - looking for lost markers: {sorted(list(markers_seen_before))}")
                    recovery_mode = True
                    recovery_missing_streak = 0
                    return True
        else:
            recovery_missing_streak = 0
            recovery_found_streak = 0

    return recovery_mode

def filter_robot_movement(new_positions, threshold_x=1.0, threshold_y=1.0):
    """
    Filter robot position updates based on movement threshold to reduce jitter
    
    Args:
        new_positions: Dict of {tag_id: [x, y]} with new calculated positions
        threshold_x: Minimum X movement in cm to send update
        threshold_y: Minimum Y movement in cm to send update
        
    Returns:
        Dict of filtered positions that exceed movement threshold
    """
    global previous_robot_positions
    
    filtered_positions = {}
    
    for tag_id, new_pos in new_positions.items():
        tag_id = int(tag_id)
        new_x, new_y = new_pos[0], new_pos[1]
        
        # If this is the first position for this robot, always include it
        if tag_id not in previous_robot_positions:
            filtered_positions[tag_id] = new_pos
            previous_robot_positions[tag_id] = new_pos.copy()
            if cfg.LOG_LEVEL == "DEBUG":
                print(f"🤖 Robot {tag_id}: First position recorded at ({new_x:.1f}, {new_y:.1f})")
            continue
        
        # Calculate movement from previous position
        prev_x, prev_y = previous_robot_positions[tag_id]
        delta_x = abs(new_x - prev_x)
        delta_y = abs(new_y - prev_y)
        
        # Only include if movement exceeds threshold in either direction
        if delta_x >= threshold_x or delta_y >= threshold_y:
            filtered_positions[tag_id] = new_pos
            previous_robot_positions[tag_id] = new_pos.copy()
            if cfg.LOG_LEVEL == "DEBUG":
                print(f"🔄 Robot {tag_id}: Position updated ({prev_x:.1f},{prev_y:.1f}) → ({new_x:.1f},{new_y:.1f}) [Δx:{delta_x:.1f}, Δy:{delta_y:.1f}]")
        else:
            # Keep previous position (no significant movement)
            filtered_positions[tag_id] = previous_robot_positions[tag_id]
    
    return filtered_positions

def validate_marker_geometry(corner):
    """
    Validate that a marker has reasonable geometry (roughly square, reasonable size)
    
    Args:
        corner: Marker corner points as numpy array
        
    Returns:
        bool: True if geometry is valid, False otherwise
    """
    try:
        if corner is None or len(corner) == 0:
            return False
            
        corner_pts = corner[0]
        
        # Ensure we have exactly 4 corner points
        if len(corner_pts) != 4:
            return False
    except (IndexError, TypeError, AttributeError) as e:
        # Handle malformed corner data gracefully
        return False
    
    # Calculate side lengths
    side_lengths = []
    for i in range(4):
        p1 = corner_pts[i]
        p2 = corner_pts[(i + 1) % 4]
        side_length = np.linalg.norm(p2 - p1)
        side_lengths.append(side_length)
    
    if len(side_lengths) != 4 or min(side_lengths) <= 0:
        return False
    
    min_side = min(side_lengths)
    max_side = max(side_lengths)
    aspect_ratio = max_side / min_side
    
    # Reject if too elongated (config-driven, more lenient in recovery)
    ar_limit = cfg.GEOMETRY_ASPECT_RATIO_MAX_RECOVERY if recovery_mode else cfg.GEOMETRY_ASPECT_RATIO_MAX
    if aspect_ratio > ar_limit:
        return False
    
    # Check if marker is too small (likely noise)
    min_px = cfg.GEOMETRY_MIN_SIDE_PX_RECOVERY if recovery_mode else cfg.GEOMETRY_MIN_SIDE_PX
    if min_side < min_px:
        return False
    
    # Check if marker is unreasonably large
    if max_side > cfg.GEOMETRY_MAX_SIDE_PX:
        return False
    
    # Calculate area to check for reasonable polygon
    try:
        x = corner_pts[:, 0]
        y = corner_pts[:, 1]
        area = 0.5 * abs(sum(x[i] * y[(i + 1) % 4] - x[(i + 1) % 4] * y[i] for i in range(4)))
        expected_area = min_side * min_side
        area_ratio = area / expected_area if expected_area > 0 else 0
        
        ar_range = cfg.GEOMETRY_AREA_RATIO_RANGE_RECOVERY if recovery_mode else cfg.GEOMETRY_AREA_RATIO_RANGE
        if area_ratio < ar_range[0] or area_ratio > ar_range[1]:
            return False
            
    except (ValueError, ZeroDivisionError):
        return False
    
    return True

def filter_markers(marker_ids, marker_corners, history_frames=5, min_detections=3, max_jitter=20):
    """
    Filter out false positive markers by applying temporal consistency and geometry checks.

    Movement vs. jitter distinction:
    - Small displacement (< max_jitter): normal inter-frame noise — accumulate history as usual.
    - Large displacement (>= max_jitter but < large_move_threshold): likely real movement — 
      reset history so the tag can re-qualify quickly rather than being silently dropped.
    - Very large displacement (>= large_move_threshold): treat as teleport / ID collision noise
      and skip this detection entirely.

    Args:
        marker_ids: IDs of detected markers
        marker_corners: Corners of detected markers
        history_frames: Number of frames to keep in history
        min_detections: Minimum number of detections required to accept a marker
        max_jitter: Pixel threshold below which displacement is considered noise

    Returns:
        filtered_ids: List of IDs that passed the filtering
        filtered_corners: List of corners that passed the filtering
    """
    global marker_history

    if marker_ids is None or len(marker_ids) == 0:
        return None, None

    # A displacement this large in a single frame is almost certainly a false detection
    # (misidentified ID), not genuine movement.  Scale with max_jitter so it stays
    # proportional across resolutions.
    large_move_threshold = max_jitter * 8

    # Calculate marker centers
    centers = [np.mean(corner[0], axis=0) for corner in marker_corners]

    # First pass: Validate geometry and update history
    valid_indices = []

    for i, marker_id in enumerate(marker_ids):
        marker_id = int(marker_id[0])

        # Skip markers with invalid geometry
        if not validate_marker_geometry(marker_corners[i]):
            continue

        # Initialize history for new markers
        if marker_id not in marker_history:
            marker_history[marker_id] = deque(maxlen=history_frames)

        # Check displacement from last known position
        if len(marker_history[marker_id]) > 0:
            last_pos = marker_history[marker_id][-1]
            distance = np.linalg.norm(centers[i] - last_pos)

            if distance >= large_move_threshold:
                # Almost certainly a spurious detection with a colliding ID — skip.
                if cfg.LOG_LEVEL == "DEBUG":
                    print(f"⚠️  Marker {marker_id}: teleport ({distance:.0f}px) ignored")
                continue
            elif distance >= max_jitter:
                # Real movement: reset history so re-qualification is fast (not instant,
                # but just min_detections frames rather than never).
                marker_history[marker_id].clear()
                if cfg.LOG_LEVEL == "DEBUG":
                    print(f"🏃 Marker {marker_id}: moved {distance:.0f}px — history reset for fast reacquire")

        # Add current position to history
        marker_history[marker_id].append(centers[i])
        valid_indices.append(i)

    # Second pass: Check detection consistency
    filtered_indices = []
    for i in valid_indices:
        marker_id = int(marker_ids[i][0])

        detection_count = len(marker_history[marker_id])
        detection_rate = detection_count / history_frames

        # Robot tags need consistent detection to filter false positives
        if marker_id > cfg.CORNER_TAG_MAX_ID:
            if detection_count >= cfg.FILTER_MIN_DETECTIONS_ROBOT and detection_rate >= cfg.FILTER_DETECTION_RATE_ROBOT:
                filtered_indices.append(i)
                if cfg.LOG_LEVEL == "DEBUG":
                    print(f"🤖 Robot {marker_id} passed filter (detections: {detection_count}/{history_frames}, rate: {detection_rate:.0%})")
            elif detection_count >= 1 and cfg.LOG_LEVEL == "DEBUG":
                print(f"⏳ Robot {marker_id} pending ({detection_count}/{cfg.FILTER_MIN_DETECTIONS_ROBOT} frames)")
        # Corner tags
        elif detection_count >= min_detections and detection_rate >= cfg.FILTER_DETECTION_RATE_CORNER:
            filtered_indices.append(i)

    # Create filtered lists
    if filtered_indices:
        filtered_ids = np.array([marker_ids[i] for i in filtered_indices])
        filtered_corners = [marker_corners[i] for i in filtered_indices]
        return filtered_ids, filtered_corners
    else:
        return None, None

def load_calibration_parameters(filename="best_aruco_params.json"):
    """Load calibration parameters from a JSON file if it exists"""
    try:
        with open(filename, 'r') as f:
            params = json.load(f)
        print(f"✅ Loaded calibration parameters from {filename}")
        return params
    except FileNotFoundError:
        print(f"ℹ️ No calibration file found at {filename}, using default parameters")
        return None
    except Exception as e:
        print(f"⚠️ Error loading parameters: {e}")
        return None

def make_parameters_dynamic_friendly(static_params):
    """
    Adjust calibration parameters to be more tolerant for moving markers.
    Uses offsets/scales from tracker_config.DYNAMIC_ADJUSTMENTS.
    """
    adj = cfg.DYNAMIC_ADJUSTMENTS
    dynamic_params = static_params.copy()

    if 'adaptiveThreshWinSizeMin' in dynamic_params:
        dynamic_params['adaptiveThreshWinSizeMin'] = max(3, dynamic_params['adaptiveThreshWinSizeMin'] + adj['adaptiveThreshWinSizeMin_offset'])
    if 'adaptiveThreshWinSizeMax' in dynamic_params:
        dynamic_params['adaptiveThreshWinSizeMax'] = min(50, dynamic_params['adaptiveThreshWinSizeMax'] + adj['adaptiveThreshWinSizeMax_offset'])
    if 'adaptiveThreshConstant' in dynamic_params:
        dynamic_params['adaptiveThreshConstant'] = max(3, dynamic_params['adaptiveThreshConstant'] + adj['adaptiveThreshConstant_offset'])
    if 'minMarkerPerimeterRate' in dynamic_params:
        dynamic_params['minMarkerPerimeterRate'] = dynamic_params['minMarkerPerimeterRate'] * adj['minMarkerPerimeterRate_scale']
    if 'maxMarkerPerimeterRate' in dynamic_params:
        dynamic_params['maxMarkerPerimeterRate'] = dynamic_params['maxMarkerPerimeterRate'] * adj['maxMarkerPerimeterRate_scale']
    if 'errorCorrectionRate' in dynamic_params:
        dynamic_params['errorCorrectionRate'] = min(1.0, dynamic_params['errorCorrectionRate'] + adj['errorCorrectionRate_offset'])
    if 'polygonalApproxAccuracyRate' in dynamic_params:
        dynamic_params['polygonalApproxAccuracyRate'] = min(0.1, dynamic_params['polygonalApproxAccuracyRate'] + adj['polygonalApproxAccuracyRate_offset'])
    if 'minCornerDistanceRate' in dynamic_params:
        dynamic_params['minCornerDistanceRate'] = max(0.01, dynamic_params['minCornerDistanceRate'] + adj['minCornerDistanceRate_offset'])

    return dynamic_params

# Use resolution configs from tracker_config
RESOLUTION_CONFIGS = cfg.RESOLUTION_CONFIGS

def crop_frame(frame, resolution_config):
    """
    Crop the given frame using resolution-specific crop settings.
    :param frame: The input frame to crop.
    :param resolution_config: Resolution configuration dict with crop settings.
    :return: The cropped frame.
    """
    crop = resolution_config['crop']
    # Ensure crop doesn't exceed frame boundaries
    frame_h, frame_w = frame.shape[:2]
    x = min(crop['x'], frame_w - 1)
    y = min(crop['y'], frame_h - 1)
    w = min(crop['w'], frame_w - x)
    h = min(crop['h'], frame_h - y)
    
    return frame[y:y+h, x:x+w]

async def track_aruco_tags(lock_queue, scale_factor=0.7, resolution='4k', movement_threshold_cm=1.0):
    
    global locked_corners, locked_pixel_positions, lock_state, movement_threshold
    global recovery_mode
    print(f"🟢 Initial lock state: {lock_state}", flush=True)
    print(f"🏷️  Tag system: {cfg.TAG_SYSTEM} ({cfg.TAG_DICTIONARY_NAME})")

    reset_tracking_state()
    print("🧹 Tracking state reset")
    
    # Set movement threshold
    movement_threshold['x'] = movement_threshold_cm
    movement_threshold['y'] = movement_threshold_cm
    print(f"🎯 Movement stability threshold: {movement_threshold_cm} cm (reduces jitter)")
    
    # Get resolution configuration
    if resolution not in RESOLUTION_CONFIGS:
        print(f"⚠️ Unknown resolution '{resolution}', defaulting to 4k")
        resolution = '4k'
    
    config = RESOLUTION_CONFIGS[resolution]
    print(f"📹 Using {config['display_name']} resolution with crop settings: {config['crop']}")
    print(f"📹 Streaming at {config['stream_resolution']} resolution")

    # Get dictionary from config (supports ArUco and AprilTag)
    dictionary = cfg.get_dictionary()
    detector_params = cv2.aruco.DetectorParameters()

    # Try to load calibrated parameters, but make them more tolerant for moving targets
    calibration = load_calibration_parameters()
    if calibration:
        base_params = make_parameters_dynamic_friendly(calibration)
        print("🔄 Parameters optimized for moving target detection")
    else:
        base_params = cfg.DEFAULT_DETECTION_PARAMS.copy()

    # Build initial detector (normal mode)
    detector = get_detector_for_mode(dictionary, base_params, False)
    print("🔄 Detector created (normal mode)")
    if platform.system() == "Darwin" and hasattr(cv2, "CAP_AVFOUNDATION"):
        raw_cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)
    else:
        raw_cap = cv2.VideoCapture(0)

    # Keep capture buffer shallow so processing uses the newest frame.
    raw_cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    
    raw_cap.set(cv2.CAP_PROP_FRAME_WIDTH, config['width'])
    raw_cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config['height'])
    
    actual_width = raw_cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    actual_height = raw_cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    print(f"📹 Camera resolution set to: {actual_width}x{actual_height}")

    # Wrap in threaded camera if enabled
    if cfg.THREADED_CAPTURE:
        cap = ThreadedCamera(raw_cap)
        print("🧵 Threaded camera capture enabled")
        time.sleep(0.3)  # let the thread grab a first frame
    else:
        cap = raw_cap
    
    # Use per-resolution scale factor from config (overrides CLI if present)
    scale_factor = config.get('detection_scale', scale_factor)
    print(f"🔬 Detection scale factor: {scale_factor}")
    
    # Prepare lens correction calibration; maps are built lazily for the actual
    # post-crop frame size to avoid map/frame dimension mismatches.
    undistort_map1 = None
    undistort_map2 = None
    undistort_map_size = None
    undistort_cam_mtx = None
    undistort_dist_coeffs = None
    if cfg.UNDISTORT_ENABLED:
        undistort_cam_mtx, undistort_dist_coeffs = cfg.load_camera_calibration(resolution)
        print("📷 Undistortion enabled (maps will be generated for cropped frame size)")
    else:
        print("📷 Undistortion disabled (set UNDISTORT_ENABLED = True in tracker_config.py)")
    
    print(f"🔍 Using {config['display_name']} for detection, streaming at {config['stream_resolution']}")
    
    prev_time = time.time()
    stream_fps_limit = cfg.STREAM_FPS_LIMIT
    stream_frame_interval = 1.0 / stream_fps_limit
    last_stream_frame_time = 0
    frame_count = 0
    last_history_cleanup = time.time()
    
    # Corner tag positions from config
    corner_tag_positions = cfg.CORNER_TAG_POSITIONS

    while True:
        # Check for lock state updates
        try:
            new_lock_state = lock_queue.get_nowait()
            # If lock state changed from unlocked to locked, save current corners
            if new_lock_state and not lock_state:
                if detected_corner_positions and pixel_positions:
                    # When locking, store both world coordinates and pixel positions
                    locked_corners = {int(k): v.copy() for k, v in detected_corner_positions.items()}
                    locked_pixel_positions = {int(k): v.copy() for k, v in pixel_positions.items() if int(k) <= 3}
                    print(f"🔒 Corners locked: {len(locked_corners)} corners saved")
                else:
                    print("⚠️ No corners to lock!")
                    # Don't update lock state if there are no corners to lock
                    continue
            # If unlocking, clear locked corners
            elif not new_lock_state and lock_state:
                print("🔓 Corners unlocked")
                locked_corners = None
                locked_pixel_positions = None
            
            lock_state = new_lock_state
                
        except asyncio.QueueEmpty:
            pass

        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            await asyncio.sleep(0.1)
            continue

        # Apply cropping with resolution-specific settings
        frame = crop_frame(frame, config)

        # Conditional rotation (skip if camera is mounted right-side-up)
        if cfg.CAMERA_ROTATE_180:
            frame = cv2.rotate(frame, cv2.ROTATE_180)

        # Apply lens distortion correction (LUT generated for current frame size)
        if undistort_cam_mtx is not None and undistort_dist_coeffs is not None:
            h_frame, w_frame = frame.shape[:2]
            current_size = (w_frame, h_frame)
            if undistort_map1 is None or undistort_map_size != current_size:
                new_mtx, _ = cv2.getOptimalNewCameraMatrix(
                    undistort_cam_mtx,
                    undistort_dist_coeffs,
                    current_size,
                    1,
                    current_size,
                )
                undistort_map1, undistort_map2 = cv2.initUndistortRectifyMap(
                    undistort_cam_mtx,
                    undistort_dist_coeffs,
                    None,
                    new_mtx,
                    current_size,
                    cv2.CV_16SC2,
                )
                undistort_map_size = current_size
                print(f"📷 Undistortion maps generated ({w_frame}×{h_frame})")

            frame = cv2.remap(frame, undistort_map1, undistort_map2, cv2.INTER_LINEAR)

        curr_time = time.time()
        fps = 1.0 / (curr_time - prev_time)
        prev_time = curr_time
        
        frame_count += 1

        small_frame = cv2.resize(frame, (0, 0), fx=scale_factor, fy=scale_factor)
        gray = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)

        if gray is None or gray.size == 0:
            print("⚠️ Grayscale image is empty or invalid")
            await asyncio.sleep(0.1)
            continue

        try:
            # Detect with current detector
            corners, ids, _ = detector.detectMarkers(gray)

            now = time.time()

            # Update marker timestamps and seen set
            if ids is not None and len(ids) > 0:
                for marker_id in ids.flatten():
                    mid = int(marker_id)
                    marker_last_seen[mid] = now
                    marker_timestamps[mid] = now
                    markers_seen_before.add(mid)

            # Decide if we need to switch modes
            prev_mode = recovery_mode
            recovery_mode = check_recovery_needed(ids, now)
            if recovery_mode != prev_mode:
                detector = get_detector_for_mode(dictionary, base_params, recovery_mode)
                print("🔁 Detector switched:", "recovery" if recovery_mode else "normal")

            # Periodic cleanup of stale marker_history entries
            if curr_time - last_history_cleanup > cfg.MARKER_HISTORY_EXPIRY_S:
                stale_history_count, stale_state_count = prune_stale_tracking_state(curr_time)
                if stale_history_count and cfg.LOG_LEVEL == "DEBUG":
                    print(f"🧹 Cleaned {stale_history_count} stale marker history entries")
                if stale_state_count:
                    print(f"🧹 Pruned {stale_state_count} stale marker state entries")
                last_history_cleanup = curr_time

            # Filter out false positives
            if ids is not None and len(ids) > 0:
                max_jitter = cfg.FILTER_MAX_JITTER_4K if resolution == '4k' else cfg.FILTER_MAX_JITTER_1080P
                min_detections = cfg.FILTER_MIN_DETECTIONS_ROBOT if resolution == '4k' else cfg.FILTER_MIN_DETECTIONS_CORNER

                if recovery_mode:
                    max_jitter *= 2
                    min_detections = max(1, min_detections - 2)

                ids, corners = filter_markers(ids, corners, history_frames=cfg.FILTER_HISTORY_FRAMES, min_detections=min_detections, max_jitter=max_jitter)
        except cv2.error as e:
            print(f"⚠️ OpenCV error during detectMarkers: {e}")
            await asyncio.sleep(0.1)
            continue

        detected_corner_positions = {}  # For new corners
        pixel_positions = {}  # Stores pixel positions of all markers
        estimated_robot_positions = {}  # Stores estimated real-world positions

        if ids is not None:
            if cfg.LOG_LEVEL == "DEBUG":
                print(f"🔎 Detected marker IDs after filtering: {[int(id[0]) for id in ids]}")
            corners = [corner / scale_factor for corner in corners]

            # Determine if we need to draw on frame (only when streaming)
            should_draw = (curr_time - last_stream_frame_time >= stream_frame_interval)
            if should_draw:
                cv2.aruco.drawDetectedMarkers(frame, corners, ids)

            for i, corner in enumerate(corners):
                marker_id = int(ids[i][0])  # Convert NumPy int to Python int

                if marker_id > cfg.MAX_VALID_TAG_ID:
                    continue

                center = np.mean(corner[0], axis=0)

                if marker_id in corner_tag_positions:
                    detected_corner_positions[marker_id] = corner_tag_positions[marker_id] * cfg.FEET_TO_CM
                    pixel_positions[marker_id] = center

                if marker_id > cfg.CORNER_TAG_MAX_ID:
                    pixel_positions[marker_id] = center

                center_int = tuple(center.astype(int))
                if should_draw:
                    cv2.circle(frame, center_int, 5, (0, 255, 0), -1)
                    cv2.putText(frame, f"ID: {marker_id}", (center_int[0] + 10, center_int[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # If locked, work with locked corners
            if lock_state and locked_corners:
                # We use locked world coordinates for all available corners
                combined_corners = locked_corners.copy()
                
                # For pixel positions, use the currently visible corners and 
                # supplement with stored positions for invisible corners
                combined_pixels = {}
                
                # First, add all currently visible corners
                for tag_id in pixel_positions:
                    if int(tag_id) <= cfg.CORNER_TAG_MAX_ID:
                        combined_pixels[int(tag_id)] = pixel_positions[tag_id]
                
                # Then, for any locked corner that's not currently visible,
                # add its stored pixel position if we have one
                if locked_pixel_positions:
                    for tag_id in locked_corners:
                        tag_id = int(tag_id)
                        if tag_id <= cfg.CORNER_TAG_MAX_ID and tag_id not in combined_pixels and tag_id in locked_pixel_positions:
                            combined_pixels[tag_id] = locked_pixel_positions[tag_id]
                            if should_draw:
                                # Mark this as a "ghost" corner in the frame for visualization
                                center_int = tuple(locked_pixel_positions[tag_id].astype(int))
                                cv2.circle(frame, center_int, 8, (0, 0, 255), 2)  # Red circle for ghost corners
                                cv2.putText(frame, f"ID: {tag_id} (locked)", 
                                          (center_int[0] + 10, center_int[1] - 10), 
                                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                
                # Add all robot markers
                for tag_id in pixel_positions:
                    if int(tag_id) > cfg.CORNER_TAG_MAX_ID:
                        combined_pixels[int(tag_id)] = pixel_positions[tag_id]
                
                # Use the combined information
                detected_corner_positions = combined_corners
                pixel_positions = combined_pixels
            
            elif not lock_state:
                # When not locked, just use what we see
                # But save current state in case we lock soon
                if detected_corner_positions:
                    locked_corners = detected_corner_positions.copy() 
                if pixel_positions:
                    locked_pixel_positions = {int(k): v.copy() for k, v in pixel_positions.items() if int(k) <= cfg.CORNER_TAG_MAX_ID}
            
            if cfg.LOG_LEVEL == "DEBUG":
                robot_pixels = {k: v for k, v in pixel_positions.items() if int(k) > cfg.CORNER_TAG_MAX_ID}
                corner_pixels = {k: v for k, v in pixel_positions.items() if int(k) <= cfg.CORNER_TAG_MAX_ID}
                if robot_pixels:
                    print(f"📍 Robot pixel positions: {robot_pixels}")
                if corner_pixels and frame_count % 30 == 0:
                    print(f"🏁 Corner pixel positions: {corner_pixels}")
                    for cid, ppos in corner_pixels.items():
                        world_pos = detected_corner_positions.get(cid, [0,0])
                        print(f"   Corner {cid}: pixel=({ppos[0]:.0f}, {ppos[1]:.0f}) -> world=({world_pos[0]:.1f}, {world_pos[1]:.1f})")
                print(f"🏁 Corner count: {len(detected_corner_positions)}, Lock state: {lock_state}")
            
            # Only proceed if we have enough corners for coordinate transformation
            if len(detected_corner_positions) >= 3:
                # Prepare source and destination points for transformation
                src_points = []
                dst_points = []
                available_corners = 0

                for tag_id, world_pos in detected_corner_positions.items():
                    tag_id = int(tag_id)
                    # Skip if tag is not in pixel_positions
                    if tag_id not in pixel_positions:
                        continue
                    
                    src_points.append(pixel_positions[tag_id])
                    dst_points.append(world_pos)
                    available_corners += 1

                # Only proceed if we have enough corners
                if available_corners >= 3:
                    src_points = np.array(src_points, dtype=np.float32)
                    dst_points = np.array(dst_points, dtype=np.float32)
                    
                    # Choose appropriate transformation method based on available corners
                    try:
                        if available_corners >= 4:
                            # Use homography for 4+ corners
                            H, status = cv2.findHomography(src_points, dst_points)
                            transform_type = "homography"
                        else:
                            # Use affine transform for exactly 3 corners
                            H = cv2.getAffineTransform(src_points[:3], dst_points[:3])
                            transform_type = "affine"
                        
                        # Calculate position for all robot tags
                        for tag_id, pixel_pos in pixel_positions.items():
                            if int(tag_id) > cfg.CORNER_TAG_MAX_ID:  # Only process robot tags
                                tag_id = int(tag_id)
                                pixel_tag = np.array([pixel_pos], dtype=np.float32)
                                
                                try:
                                    if transform_type == "affine":
                                        # For affine transform, use transform directly
                                        world_tag = cv2.transform(pixel_tag[None, :, :], H)
                                        estimated_position = world_tag[0, 0]
                                    else:  # homography
                                        # For homography, use homogeneous coordinates
                                        pixel_tag_homogeneous = np.array([pixel_pos[0], pixel_pos[1], 1])
                                        world_tag_homogeneous = np.dot(H, pixel_tag_homogeneous)
                                        # Protect against division by zero
                                        if abs(world_tag_homogeneous[2]) > 1e-10:
                                            world_tag_homogeneous /= world_tag_homogeneous[2]
                                            estimated_position = world_tag_homogeneous[:2]
                                        else:
                                            # If division is unsafe, fall back to affine transform
                                            H_affine = cv2.getAffineTransform(src_points[:3], dst_points[:3])
                                            world_tag = cv2.transform(pixel_tag[None, :, :], H_affine)
                                            estimated_position = world_tag[0, 0]
                                    
                                    estimated_robot_positions[tag_id] = np.round(estimated_position)
                                except Exception as e:
                                    print(f"⚠️ Error calculating position for tag {tag_id}: {e}")
                    except Exception as e:
                        print(f"⚠️ Error computing transformation: {e}")
            
            # Simple recovery tracking - no complex movement speed needed
            
            # Apply movement threshold filtering to reduce jitter
            if estimated_robot_positions:
                if cfg.LOG_LEVEL == "DEBUG":
                    print(f"📍 Raw robot positions before filter: {estimated_robot_positions}")
                estimated_robot_positions = filter_robot_movement(
                    estimated_robot_positions, 
                    movement_threshold['x'], 
                    movement_threshold['y']
                )
                if cfg.LOG_LEVEL == "DEBUG":
                    print(f"📍 Robot positions after filter: {estimated_robot_positions}")

        # Optimize frame encoding - only encode for streaming at limited FPS
        base64_frame = None
        
        # Only encode frame if enough time has passed (limit streaming FPS)
        if curr_time - last_stream_frame_time >= stream_frame_interval:
            encode_params = [cv2.IMWRITE_JPEG_QUALITY, cfg.STREAM_JPEG_QUALITY_4K if resolution == '4k' else cfg.STREAM_JPEG_QUALITY_1080P]
            
            # Create streaming version using configuration
            stream_frame = cv2.resize(frame, config['stream_resolution'])
            _, buffer = cv2.imencode(".jpg", stream_frame, encode_params)
            base64_frame = base64.b64encode(buffer).decode("utf-8")
            last_stream_frame_time = curr_time

        # Add additional information about lock state and resolution
        output_dict = {
            'robot_tags': {int(tag_id): estimated_robot_positions[tag_id].tolist() for tag_id in estimated_robot_positions},
            'corner_tags': {int(tag_id): detected_corner_positions[tag_id].tolist() for tag_id in detected_corner_positions},
            'fps': round(fps, 2),
            'lock_state': lock_state,
            'available_corners': available_corners if 'available_corners' in locals() else 0,
            'resolution': config['display_name'],
            'stream_size': f"{config['stream_resolution'][0]}x{config['stream_resolution'][1]}",
            'frame_count': frame_count,
            'movement_threshold': f"{movement_threshold['x']}cm"
        }
        
        if cfg.LOG_LEVEL == "DEBUG" and output_dict['robot_tags']:
            print(f"📤 OUTPUT robot_tags: {output_dict['robot_tags']}")
            x_max = cfg.CORNER_TAG_POSITIONS[2][0] * cfg.FEET_TO_CM
            y_max = cfg.CORNER_TAG_POSITIONS[2][1] * cfg.FEET_TO_CM
            for tag_id, pos in output_dict['robot_tags'].items():
                if pos[0] < 0 or pos[0] > x_max or pos[1] < 0 or pos[1] > y_max:
                    print(f"⚠️ Robot {tag_id} OUT OF BOUNDS: ({pos[0]:.1f}, {pos[1]:.1f})")
        
        # Only include frame data when we have a new encoded frame
        if base64_frame is not None:
            output_dict['frame'] = base64_frame

        # Dynamic sleep based on processing time to maintain target FPS
        processing_time = time.time() - curr_time
        target_frame_time = 1.0 / cfg.TARGET_DETECTION_FPS
        sleep_time = max(0.001, target_frame_time - processing_time)
        
        await asyncio.sleep(sleep_time)
        yield output_dict

    cap.release()
