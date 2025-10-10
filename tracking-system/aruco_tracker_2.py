import cv2
import numpy as np
import time
import asyncio
import base64
import os
import json
from collections import deque

# Store both corner positions and their pixel locations when locked
locked_corners = None  # World coordinates of corners
locked_pixel_positions = None  # Pixel positions of corners when locked
lock_state = False  # Indicates whether corners are locked

# History for marker filtering
marker_history = {}  # Stores history of marker positions
position_kalman_filters = {}  # Stores Kalman filters for each marker

# Movement stability filtering
previous_robot_positions = {}  # Stores last sent positions for each robot
movement_threshold = {'x': 1.0, 'y': 1.0}  # Minimum movement required to send update (in cm)

# Frame processing optimization
frame_count = 0
last_stream_frame_time = 0

# Simple recovery tracking
marker_last_seen = {}  # Track when each marker was last detected
marker_timestamps = {}  # Track last-seen timestamps per marker
markers_seen_before = set()  # Track markers seen at least once
recovery_mode = False  # Recovery flag

def make_parameters_recovery_friendly(base_params):
    """Return a much more permissive parameter set for recovery mode."""
    recovery_params = base_params.copy()

    # Adaptive thresholding: widen windows, increase constant
    recovery_params['adaptiveThreshWinSizeMin'] = 3
    recovery_params['adaptiveThreshWinSizeMax'] = max(31, base_params.get('adaptiveThreshWinSizeMax', 31))
    recovery_params['adaptiveThreshConstant'] = max(7, base_params.get('adaptiveThreshConstant', 7))

    # Size constraints: allow smaller and larger perimeters
    recovery_params['minMarkerPerimeterRate'] = max(0.005, base_params.get('minMarkerPerimeterRate', 0.03) * 0.4)
    recovery_params['maxMarkerPerimeterRate'] = min(5.0, base_params.get('maxMarkerPerimeterRate', 4.0) * 1.7)

    # Shape/approximation tolerance
    recovery_params['polygonalApproxAccuracyRate'] = min(0.12, base_params.get('polygonalApproxAccuracyRate', 0.03) * 2.5)
    recovery_params['minCornerDistanceRate'] = max(0.005, base_params.get('minCornerDistanceRate', 0.05) * 0.4)

    # Border & refinement
    recovery_params['minDistanceToBorder'] = 0
    recovery_params['cornerRefinementMethod'] = 0  # CORNER_REFINE_NONE for speed

    # Error correction tolerance
    recovery_params['errorCorrectionRate'] = max(0.7, base_params.get('errorCorrectionRate', 0.35))

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

    # perspectiveRemoveIgnoredMarginPerCell in [0, 1]
    margin = float(params_dict.get('perspectiveRemoveIgnoredMarginPerCell', 0.13))
    params_dict['perspectiveRemoveIgnoredMarginPerCell'] = min(1.0, max(0.0, margin))

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

    current_markers = set()
    if raw_ids is not None and len(raw_ids) > 0:
        current_markers = {int(mid[0]) for mid in raw_ids}

    # Success conditions: if previously seen markers are visible again
    if recovery_mode and (current_markers & markers_seen_before):
        print(f"✅ Partial recovery - found markers: {sorted(list(current_markers & markers_seen_before))}")
        recovery_mode = False
        return False

    # Decide if we've been without known markers for long enough
    if markers_seen_before:
        # If we see none of the known markers now
        if not (current_markers & markers_seen_before):
            # Compute time since any known marker was last seen
            last_seen_times = [now - marker_timestamps.get(mid, 0) for mid in markers_seen_before]
            if last_seen_times and min(last_seen_times) > 1.0:
                # Need recovery if we're not already in it
                if not recovery_mode:
                    print(f"🔄 Recovery mode activated - looking for lost markers: {sorted(list(markers_seen_before))}")
                recovery_mode = True
                return True

    # Also if we've never seen any marker and raw_ids is empty for a while, don't spam
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
    
    # Check aspect ratio (should be roughly square)
    min_side = min(side_lengths)
    max_side = max(side_lengths)
    aspect_ratio = max_side / min_side
    
    # Reject if too elongated (more lenient in recovery)
    if aspect_ratio > (3.0 if recovery_mode else 2.5):
        return False
    
    # Check if marker is too small (likely noise)
    if min_side < (10 if recovery_mode else 15):  # Allow smaller in recovery
        return False
    
    # Check if marker is unreasonably large (likely false positive)
    if max_side > 600:  # Allow slightly larger
        return False
    
    # Calculate area to check for reasonable polygon
    try:
        # Use shoelace formula for polygon area
        x = corner_pts[:, 0]
        y = corner_pts[:, 1]
        area = 0.5 * abs(sum(x[i] * y[(i + 1) % 4] - x[(i + 1) % 4] * y[i] for i in range(4)))
        
        # Expected area for a square
        expected_area = min_side * min_side
        area_ratio = area / expected_area if expected_area > 0 else 0
        
        # Reject if area is too different from expected square area
        if area_ratio < (0.4 if recovery_mode else 0.5) or area_ratio > (2.2 if recovery_mode else 2.0):
            return False
            
    except (ValueError, ZeroDivisionError):
        return False
    
    return True

def filter_markers(marker_ids, marker_corners, history_frames=5, min_detections=3, max_jitter=20):
    """
    Filter out false positive markers by applying temporal consistency and geometry checks
    
    Args:
        marker_ids: IDs of detected markers
        marker_corners: Corners of detected markers
        history_frames: Number of frames to keep in history
        min_detections: Minimum number of detections required to accept a marker
        max_jitter: Maximum allowed position change between frames (in pixels)
        
    Returns:
        filtered_ids: List of IDs that passed the filtering
        filtered_corners: List of corners that passed the filtering
    """
    global marker_history
    
    if marker_ids is None or len(marker_ids) == 0:
        return None, None
    
    # Calculate marker centers
    centers = [np.mean(corner[0], axis=0) for corner in marker_corners]
    
    # First pass: Validate geometry and update history
    valid_indices = []
    current_markers = {}
    
    for i, marker_id in enumerate(marker_ids):
        marker_id = int(marker_id[0])
        
        # Skip markers with invalid geometry
        if not validate_marker_geometry(marker_corners[i]):
            continue
            
        current_markers[marker_id] = centers[i]
        
        # Initialize history for new markers
        if marker_id not in marker_history:
            marker_history[marker_id] = deque(maxlen=history_frames)
        
        # Check position jitter (temporal consistency)
        if len(marker_history[marker_id]) > 0:
            last_pos = marker_history[marker_id][-1]
            current_pos = centers[i]
            distance = np.linalg.norm(current_pos - last_pos)
            
            # If jitter is too high, don't add to history
            if distance > max_jitter:
                continue
        
        # Add current position to history
        marker_history[marker_id].append(centers[i])
        valid_indices.append(i)
    
    # Second pass: Check detection consistency
    filtered_indices = []
    for i in valid_indices:
        marker_id = int(marker_ids[i][0])
        
        # Only include markers that have been seen consistently
        detection_count = len(marker_history[marker_id])
        detection_rate = detection_count / history_frames
        
        # Require minimum detections and reasonable detection rate
        if detection_count >= min_detections and detection_rate >= 0.4:
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
    Adjust calibration parameters to be more tolerant for moving ArUco markers
    Static calibration often over-optimizes for stationary targets
    """
    dynamic_params = static_params.copy()
    
    # Make adaptive thresholding more tolerant
    if 'adaptiveThreshWinSizeMin' in dynamic_params:
        dynamic_params['adaptiveThreshWinSizeMin'] = max(3, dynamic_params['adaptiveThreshWinSizeMin'] - 2)
    
    if 'adaptiveThreshWinSizeMax' in dynamic_params:
        dynamic_params['adaptiveThreshWinSizeMax'] = min(50, dynamic_params['adaptiveThreshWinSizeMax'] + 10)
    
    if 'adaptiveThreshConstant' in dynamic_params:
        dynamic_params['adaptiveThreshConstant'] = max(3, dynamic_params['adaptiveThreshConstant'] - 2)
    
    # Make perimeter detection more lenient 
    if 'minMarkerPerimeterRate' in dynamic_params:
        dynamic_params['minMarkerPerimeterRate'] = dynamic_params['minMarkerPerimeterRate'] * 0.7  # 30% more lenient
    
    if 'maxMarkerPerimeterRate' in dynamic_params:
        dynamic_params['maxMarkerPerimeterRate'] = dynamic_params['maxMarkerPerimeterRate'] * 1.3  # 30% more lenient
    
    # Increase error correction for moving targets
    if 'errorCorrectionRate' in dynamic_params:
        dynamic_params['errorCorrectionRate'] = min(1.0, dynamic_params['errorCorrectionRate'] + 0.1)
    
    # Make corner detection more forgiving
    if 'polygonalApproxAccuracyRate' in dynamic_params:
        dynamic_params['polygonalApproxAccuracyRate'] = min(0.1, dynamic_params['polygonalApproxAccuracyRate'] + 0.02)
    
    if 'minCornerDistanceRate' in dynamic_params:
        dynamic_params['minCornerDistanceRate'] = max(0.01, dynamic_params['minCornerDistanceRate'] - 0.02)
    
    return dynamic_params

# Removed complex adaptive functions - keeping it simple for reliability

# Resolution settings with crop configurations
RESOLUTION_CONFIGS = {
    '4k': {
        'width': 3840,
        'height': 2160,
        'crop': {
            'x': 500,
            'y': 400,
            'w': 3840,
            'h': 2160
        },
        'stream_resolution': (1280, 720),  # 720p for streaming
        'display_name': '4K (3840x2160)'
    },
    '1080p': {
        'width': 1920,
        'height': 1080,
        'crop': {
            'x': 10,  # Adjusted for 1080p
            'y': 10,  # Adjusted for 1080p
            'w': 1920,
            'h': 1080
        },
        'stream_resolution': (960, 540),  # Half resolution for streaming
        'display_name': 'Full HD (1920x1080)'
    }
}

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
    print(f"🟢 Initial lock state: {lock_state}", flush=True)
    
    # Initialize recovery tracking variables
    recovery_mode = False
    marker_last_seen = {}
    
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

    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_250)
    detector_params = cv2.aruco.DetectorParameters()

    # Try to load calibrated parameters, but make them more tolerant for moving targets
    calibration = load_calibration_parameters()
    if calibration:
        # Apply loaded parameters but make them more dynamic-friendly
        base_params = make_parameters_dynamic_friendly(calibration)
        print("🔄 Parameters optimized for moving target detection")
    else:
        # Use default parameters
        base_params = {
            'adaptiveThreshWinSizeMin': 3,
            'adaptiveThreshWinSizeMax': 30,
            'minMarkerPerimeterRate': 0.03,
            'maxMarkerPerimeterRate': 4.0,
            'polygonalApproxAccuracyRate': 0.02,
            'minCornerDistanceRate': 0.05,
            'cornerRefinementMethod': cv2.aruco.CORNER_REFINE_CONTOUR
        }

    # Build initial detector (normal mode)
    detector = get_detector_for_mode(dictionary, base_params, False)
    print("🔄 Detector created (normal mode)")
    cap = cv2.VideoCapture(0)
    
    # Set camera resolution based on configuration
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, config['width'])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config['height'])
    
    # Verify the resolution was set correctly
    actual_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    actual_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    print(f"📹 Camera resolution set to: {actual_width}x{actual_height}")
    
    # Only run resolution debug in 4K mode to avoid spamming logs
    if resolution == '4k':
        print("\n[Camera Resolution Debug - 4K Mode]")
        test_resolutions = [
            (640, 480), (1280, 720), (1920, 1080), (2560, 1440), (3840, 2160)
        ]
        for w, h in test_resolutions:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
            actual_w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
            actual_h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
            print(f"Tried {w}x{h} -> Camera reports: {actual_w}x{actual_h}")
        # Reset to intended resolution
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, config['width'])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config['height'])
    
    print(f"🔍 Using {config['display_name']} for detection, streaming at {config['stream_resolution']}")
    
    prev_time = time.time()
    stream_fps_limit = 20  # Limit streaming to 20 FPS to reduce bandwidth
    stream_frame_interval = 1.0 / stream_fps_limit
    last_stream_frame_time = 0  # Initialize stream frame timing
    frame_count = 0  # Initialize frame counter
    
    corner_tag_positions = {
        0: np.array([0, 0]),
        1: np.array([0, 7.75]),
        2: np.array([3.74, 7.75]),
        3: np.array([3.74, 0])
    }

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

        frame = cv2.rotate(frame, cv2.ROTATE_180)

        curr_time = time.time()
        fps = 1.0 / (curr_time - prev_time)
        prev_time = curr_time
        
        frame_count += 1

        small_frame = cv2.resize(frame, (0, 0), fx=scale_factor, fy=scale_factor)
        gray = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)

        # Debug: Save or display the grayscale image
        cv2.imwrite("debug_gray_image.jpg", gray)

        # Ensure the grayscale image has enough contours
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

            # Filter out false positives
            if ids is not None and len(ids) > 0:
                # Use normal filtering
                max_jitter = 60 if resolution == '4k' else 30
                min_detections = 4 if resolution == '4k' else 3

                # In recovery mode, be more lenient
                if recovery_mode:
                    max_jitter *= 2
                    min_detections = max(1, min_detections - 2)

                ids, corners = filter_markers(ids, corners, history_frames=6, min_detections=min_detections, max_jitter=max_jitter)
        except cv2.error as e:
            print(f"⚠️ OpenCV error during detectMarkers: {e}")
            await asyncio.sleep(0.1)
            continue

        detected_corner_positions = {}  # For new corners
        pixel_positions = {}  # Stores pixel positions of all markers
        estimated_robot_positions = {}  # Stores estimated real-world positions

        if ids is not None:
            corners = [corner / scale_factor for corner in corners]
            cv2.aruco.drawDetectedMarkers(frame, corners, ids)

            for i, corner in enumerate(corners):
                marker_id = int(ids[i][0])  # Convert NumPy int to Python int

                if marker_id >= 30:
                    continue

                center = np.mean(corner[0], axis=0)

                if marker_id in corner_tag_positions:
                    detected_corner_positions[marker_id] = corner_tag_positions[marker_id] * 30.48
                    pixel_positions[marker_id] = center

                if marker_id > 3:
                    pixel_positions[marker_id] = center

                center_int = tuple(center.astype(int))
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
                    if int(tag_id) <= 3:
                        combined_pixels[int(tag_id)] = pixel_positions[tag_id]
                
                # Then, for any locked corner that's not currently visible,
                # add its stored pixel position if we have one
                if locked_pixel_positions:
                    for tag_id in locked_corners:
                        tag_id = int(tag_id)
                        if tag_id <= 3 and tag_id not in combined_pixels and tag_id in locked_pixel_positions:
                            combined_pixels[tag_id] = locked_pixel_positions[tag_id]
                            # Mark this as a "ghost" corner in the frame for visualization
                            center_int = tuple(locked_pixel_positions[tag_id].astype(int))
                            cv2.circle(frame, center_int, 8, (0, 0, 255), 2)  # Red circle for ghost corners
                            cv2.putText(frame, f"ID: {tag_id} (locked)", 
                                      (center_int[0] + 10, center_int[1] - 10), 
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                
                # Add all robot markers
                for tag_id in pixel_positions:
                    if int(tag_id) > 3:
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
                    locked_pixel_positions = {int(k): v.copy() for k, v in pixel_positions.items() if int(k) <= 3}
            
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
                            if int(tag_id) > 3:  # Only process robot tags
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
                estimated_robot_positions = filter_robot_movement(
                    estimated_robot_positions, 
                    movement_threshold['x'], 
                    movement_threshold['y']
                )

        # Optimize frame encoding - only encode for streaming at limited FPS
        base64_frame = None
        
        # Only encode frame if enough time has passed (limit streaming FPS)
        if curr_time - last_stream_frame_time >= stream_frame_interval:
            encode_params = [cv2.IMWRITE_JPEG_QUALITY, 70 if resolution == '4k' else 75]
            
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
        
        # Only include frame data when we have a new encoded frame
        if base64_frame is not None:
            output_dict['frame'] = base64_frame

        # Dynamic sleep based on processing time to maintain target FPS
        processing_time = time.time() - curr_time
        target_fps = 30  # Target detection FPS
        target_frame_time = 1.0 / target_fps
        sleep_time = max(0.001, target_frame_time - processing_time)  # Minimum 1ms sleep
        
        await asyncio.sleep(sleep_time)
        yield output_dict

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
