import cv2
import numpy as np
import time
import argparse
import os
import json
import itertools
import concurrent.futures
from collections import deque
from typing import Dict, List, Tuple, Optional, Any

# Global variables for trackbars
param_values = {
    'adaptiveThreshWinSizeMin': 3,
    'adaptiveThreshWinSizeMax': 30,
    'adaptiveThreshWinSizeStep': 10,
    'adaptiveThreshConstant': 7,
    'minMarkerPerimeterRate': 3,  # Multiplied by 100 for trackbar
    'maxMarkerPerimeterRate': 400,  # Multiplied by 100 for trackbar
    'polygonalApproxAccuracyRate': 2,  # Multiplied by 100 for trackbar
    'minCornerDistanceRate': 5,  # Multiplied by 100 for trackbar
    'minMarkerDistanceRate': 5,  # Multiplied by 100 for trackbar
    'cornerRefinementMethod': 1,  # 0-None, 1-Subpix, 2-Contour
    'cornerRefinementWinSize': 5,
    'cornerRefinementMaxIterations': 30,
    'cornerRefinementMinAccuracy': 10,  # Multiplied by 100 for trackbar
    'minDistanceToBorder': 3,
    'adaptiveThreshConstant': 7,
    'perspectiveRemovePixelPerCell': 4,
    'perspectiveRemoveIgnoredMarginPerCell': 13,  # Multiplied by 100 for trackbar
    'maxErroneousBitsInBorderRate': 60,  # Multiplied by 100 for trackbar
    'minOtsuStdDev': 5.0,  # Multiplied by 100 for trackbar
}

# Dictionary to store detection counts for each parameter set
detection_history = {}
best_params = None
best_detection_count = 0
frame_count = 0
detection_count = 0
param_change_count = 0

# Parameters for automatic calibration
auto_param_ranges = {
    'adaptiveThreshWinSizeMin': [3, 5, 7],
    'adaptiveThreshWinSizeMax': [23, 30, 45],
    'adaptiveThreshWinSizeStep': [4, 10, 16],
    'adaptiveThreshConstant': [5, 7, 9, 11],
    'minMarkerPerimeterRate': [0.01, 0.02, 0.03, 0.04],
    'maxMarkerPerimeterRate': [3.0, 4.0, 5.0],
    'polygonalApproxAccuracyRate': [0.01, 0.02, 0.03, 0.05],
    'cornerRefinementMethod': [0, 1, 2],  # 0-None, 1-Subpix, 2-Contour
}

# Resolution settings for calibration (same as aruco_tracker_2.py)
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
        'display_name': '4K (3840x2160)'
    },
    '1080p': {
        'width': 1920,
        'height': 1080,
        'crop': {
            'x': 250,
            'y': 200,
            'w': 1920,
            'h': 1080
        },
        'display_name': 'Full HD (1920x1080)'
    }
}

def crop_frame(frame, resolution='4k'):
    """
    Crop the given frame using resolution-specific crop settings.
    :param frame: The input frame to crop.
    :param resolution: Resolution configuration key ('4k' or '1080p').
    :return: The cropped frame.
    """
    if resolution not in RESOLUTION_CONFIGS:
        resolution = '4k'
    
    config = RESOLUTION_CONFIGS[resolution]
    crop = config['crop']
    
    # Ensure crop doesn't exceed frame boundaries
    frame_h, frame_w = frame.shape[:2]
    x = min(crop['x'], frame_w - 1)
    y = min(crop['y'], frame_h - 1)
    w = min(crop['w'], frame_w - x)
    h = min(crop['h'], frame_h - y)
    
    return frame[y:y+h, x:x+w]

def create_window_and_trackbars():
    """Create a window with trackbars for each parameter"""
    global param_values
    
    cv2.namedWindow("ArUco Calibration")
    
    # Add trackbars for the main parameters that affect detection
    cv2.createTrackbar("adaptiveThreshWinSizeMin", "ArUco Calibration", param_values['adaptiveThreshWinSizeMin'], 20, lambda v: update_param('adaptiveThreshWinSizeMin', v))
    cv2.createTrackbar("adaptiveThreshWinSizeMax", "ArUco Calibration", param_values['adaptiveThreshWinSizeMax'], 100, lambda v: update_param('adaptiveThreshWinSizeMax', v))
    cv2.createTrackbar("adaptiveThreshWinSizeStep", "ArUco Calibration", param_values['adaptiveThreshWinSizeStep'], 20, lambda v: update_param('adaptiveThreshWinSizeStep', v))
    cv2.createTrackbar("adaptiveThreshConstant", "ArUco Calibration", param_values['adaptiveThreshConstant'], 40, lambda v: update_param('adaptiveThreshConstant', v))
    cv2.createTrackbar("minMarkerPerimeter*100", "ArUco Calibration", param_values['minMarkerPerimeterRate'], 10, lambda v: update_param('minMarkerPerimeterRate', v))
    cv2.createTrackbar("maxMarkerPerimeter*100", "ArUco Calibration", param_values['maxMarkerPerimeterRate'], 1000, lambda v: update_param('maxMarkerPerimeterRate', v))
    cv2.createTrackbar("polygonalApproxAccuracy*100", "ArUco Calibration", param_values['polygonalApproxAccuracyRate'], 10, lambda v: update_param('polygonalApproxAccuracyRate', v))
    cv2.createTrackbar("cornerRefinementMethod", "ArUco Calibration", param_values['cornerRefinementMethod'], 2, lambda v: update_param('cornerRefinementMethod', v))

    # Create a separate window for advanced parameters
    cv2.namedWindow("Advanced Parameters")
    cv2.createTrackbar("minCornerDistance*100", "Advanced Parameters", param_values['minCornerDistanceRate'], 20, lambda v: update_param('minCornerDistanceRate', v))
    cv2.createTrackbar("minMarkerDistance*100", "Advanced Parameters", param_values['minMarkerDistanceRate'], 20, lambda v: update_param('minMarkerDistanceRate', v))
    cv2.createTrackbar("cornerRefinementWinSize", "Advanced Parameters", param_values['cornerRefinementWinSize'], 20, lambda v: update_param('cornerRefinementWinSize', v))
    cv2.createTrackbar("cornerRefinementMaxIter", "Advanced Parameters", param_values['cornerRefinementMaxIterations'], 100, lambda v: update_param('cornerRefinementMaxIterations', v))
    cv2.createTrackbar("cornerRefinementMinAcc*100", "Advanced Parameters", param_values['cornerRefinementMinAccuracy'], 100, lambda v: update_param('cornerRefinementMinAccuracy', v))

def update_param(param_name, value):
    """Update parameter value when trackbar is moved"""
    global param_values, param_change_count
    
    # Only record a parameter change if the value actually changes
    if param_values[param_name] != value:
        param_values[param_name] = value
        param_change_count += 1

def update_detector_parameters(detector_params, params=None):
    """Apply current parameter values to the detector parameters"""
    global param_values
    
    # Use provided params or global param_values
    p = params if params is not None else param_values
    
    # Basic parameters
    detector_params.adaptiveThreshWinSizeMin = int(p.get('adaptiveThreshWinSizeMin', 3))
    detector_params.adaptiveThreshWinSizeMax = int(p.get('adaptiveThreshWinSizeMax', 30))
    detector_params.adaptiveThreshWinSizeStep = int(p.get('adaptiveThreshWinSizeStep', 10))
    detector_params.adaptiveThreshConstant = float(p.get('adaptiveThreshConstant', 7))
    
    # Convert trackbar integer values to actual parameter values if needed
    if 'minMarkerPerimeterRate' in p and p['minMarkerPerimeterRate'] > 1:
        detector_params.minMarkerPerimeterRate = float(p['minMarkerPerimeterRate']) / 100.0
    else:
        detector_params.minMarkerPerimeterRate = float(p.get('minMarkerPerimeterRate', 0.03))
        
    if 'maxMarkerPerimeterRate' in p and p['maxMarkerPerimeterRate'] > 10:
        detector_params.maxMarkerPerimeterRate = float(p['maxMarkerPerimeterRate']) / 100.0
    else:
        detector_params.maxMarkerPerimeterRate = float(p.get('maxMarkerPerimeterRate', 4.0))
        
    if 'polygonalApproxAccuracyRate' in p and p['polygonalApproxAccuracyRate'] > 0.5:
        detector_params.polygonalApproxAccuracyRate = float(p['polygonalApproxAccuracyRate']) / 100.0
    else:
        detector_params.polygonalApproxAccuracyRate = float(p.get('polygonalApproxAccuracyRate', 0.02))
        
    if 'minCornerDistanceRate' in p and p['minCornerDistanceRate'] > 0.5:
        detector_params.minCornerDistanceRate = float(p['minCornerDistanceRate']) / 100.0
    else:
        detector_params.minCornerDistanceRate = float(p.get('minCornerDistanceRate', 0.05))
        
    if 'minMarkerDistanceRate' in p and p['minMarkerDistanceRate'] > 0.5:
        detector_params.minMarkerDistanceRate = float(p['minMarkerDistanceRate']) / 100.0
    else:
        detector_params.minMarkerDistanceRate = float(p.get('minMarkerDistanceRate', 0.05))
    
    # Corner refinement parameters
    detector_params.cornerRefinementMethod = int(p.get('cornerRefinementMethod', 1))
    if detector_params.cornerRefinementMethod > 0:
        detector_params.cornerRefinementWinSize = int(p.get('cornerRefinementWinSize', 5))
        detector_params.cornerRefinementMaxIterations = int(p.get('cornerRefinementMaxIterations', 30))
        
        if 'cornerRefinementMinAccuracy' in p and p['cornerRefinementMinAccuracy'] > 0.5:
            detector_params.cornerRefinementMinAccuracy = float(p['cornerRefinementMinAccuracy']) / 100.0
        else:
            detector_params.cornerRefinementMinAccuracy = float(p.get('cornerRefinementMinAccuracy', 0.1))
    
    return detector_params

def draw_text_info(frame, ids, fps, detection_rate):
    """Draw text info on the frame"""
    global param_change_count, best_params, best_detection_count
    
    cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(frame, f"Detection rate: {detection_rate:.1f}%", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    if ids is not None:
        cv2.putText(frame, f"Tags detected: {len(ids)}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    else:
        cv2.putText(frame, "No tags detected", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    # Draw current parameter set
    y_pos = 120
    cv2.putText(frame, f"Parameter changes: {param_change_count}", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
    
    # Show the best parameters if we have them
    if best_params is not None:
        y_pos += 30
        cv2.putText(frame, f"Best detection rate: {best_detection_count}%", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 200, 0), 2)

def save_parameters(params, filename="best_aruco_params.json"):
    """Save parameters to a JSON file"""
    # Convert from trackbar integer values to actual parameter values for saves
    save_params = params.copy()
    
    # Convert only if these are integers from trackbars
    if save_params.get('minMarkerPerimeterRate', 0) > 1:
        save_params['minMarkerPerimeterRate'] /= 100.0
    if save_params.get('maxMarkerPerimeterRate', 0) > 10:
        save_params['maxMarkerPerimeterRate'] /= 100.0
    if save_params.get('polygonalApproxAccuracyRate', 0) > 0.5:
        save_params['polygonalApproxAccuracyRate'] /= 100.0
    if save_params.get('minCornerDistanceRate', 0) > 0.5:
        save_params['minCornerDistanceRate'] /= 100.0
    if save_params.get('minMarkerDistanceRate', 0) > 0.5:
        save_params['minMarkerDistanceRate'] /= 100.0
    if save_params.get('cornerRefinementMinAccuracy', 0) > 0.5:
        save_params['cornerRefinementMinAccuracy'] /= 100.0
    
    with open(filename, 'w') as f:
        json.dump(save_params, f, indent=4)
    print(f"✅ Parameters saved to {filename}")

def load_parameters(filename="best_aruco_params.json"):
    """Load parameters from a JSON file"""
    global param_values
    
    try:
        with open(filename, 'r') as f:
            loaded_params = json.load(f)
            
        # Convert from actual parameter values to trackbar integer values for loads
        for param in loaded_params:
            if param in param_values:
                if param in ['minMarkerPerimeterRate', 'maxMarkerPerimeterRate', 
                           'polygonalApproxAccuracyRate', 'minCornerDistanceRate',
                           'minMarkerDistanceRate', 'cornerRefinementMinAccuracy']:
                    param_values[param] = int(loaded_params[param] * 100)
                else:
                    param_values[param] = int(loaded_params[param])
                    
        print(f"✅ Parameters loaded from {filename}")
        return True
    except FileNotFoundError:
        print(f"ℹ️ Parameter file {filename} not found")
        return False
    except Exception as e:
        print(f"⚠️ Error loading parameters: {e}")
        return False

def run_calibration(scale_factor=0.7, resolution='4k', load_previous=True):
    """Run the calibration process"""
    global detection_history, best_params, best_detection_count, frame_count, detection_count, param_values, param_change_count
    
    # Get resolution configuration
    if resolution not in RESOLUTION_CONFIGS:
        print(f"⚠️ Unknown resolution '{resolution}', defaulting to 4k")
        resolution = '4k'
    
    config = RESOLUTION_CONFIGS[resolution]
    print(f"📹 Calibrating for {config['display_name']} resolution")
    
    # Try to load previous best parameters if requested
    if load_previous:
        load_parameters()
    
    # Initialize detector
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_250)
    detector_params = cv2.aruco.DetectorParameters()
    detector_params = update_detector_parameters(detector_params)
    detector = cv2.aruco.ArucoDetector(dictionary, detector_params)
    
    # Open camera
    cap = cv2.VideoCapture(0)
    
    # Set resolution based on configuration
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, config['width'])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config['height'])
    
    # Get actual resolution
    actual_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    actual_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    print(f"📹 Camera resolution: {actual_width}x{actual_height}")
    
    # Create window and trackbars
    create_window_and_trackbars()
    
    # Performance tracking
    prev_time = time.time()
    detection_window = []  # For calculating detection rate over time
    evaluation_interval = 30  # Evaluate parameters every X frames
    auto_tuning_active = False
    
    # Help text
    help_text = [
        "Press 'h' to show/hide this help",
        "Press 's' to save current parameters",
        "Press 'l' to load saved parameters",
        "Press 'r' to reset to default parameters",
        "Press 'a' to toggle auto-tuning mode",
        "Press 'q' to quit calibration"
    ]
    show_help = True
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        # Calculate FPS
        current_time = time.time()
        fps = 1.0 / (current_time - prev_time)
        prev_time = current_time
        # Apply cropping with resolution-specific settings
        frame = crop_frame(frame, resolution)

        # Resize for better performance 
        small_frame = cv2.resize(frame, (0, 0), fx=scale_factor, fy=scale_factor)
        gray = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)
        
        # Update detector with current parameter values
        detector_params = update_detector_parameters(detector_params)
        detector = cv2.aruco.ArucoDetector(dictionary, detector_params)
        
        # Detect markers
        corners, ids, rejected = detector.detectMarkers(gray)
        
        # Track detection rate
        frame_count += 1
        if ids is not None:
            detection_count += 1
            detection_window.append(1)
        else:
            detection_window.append(0)
            
        # Keep detection window at a reasonable size
        if len(detection_window) > 100:
            detection_window.pop(0)
            
        # Calculate detection rate
        detection_rate = sum(detection_window) / len(detection_window) * 100
        
        # Draw the detected markers
        if ids is not None:
            # Scale corners back to original frame size
            corners = [corner / scale_factor for corner in corners]
            cv2.aruco.drawDetectedMarkers(frame, corners, ids)
            
            # Draw center and ID for each marker
            for i, corner in enumerate(corners):
                center = np.mean(corner[0], axis=0).astype(int)
                cv2.circle(frame, tuple(center), 5, (0, 255, 0), -1)
                cv2.putText(frame, f"ID: {ids[i][0]}", 
                            (center[0] + 10, center[1] - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Draw text information
        draw_text_info(frame, ids, fps, detection_rate)
        
        # Display parameter values
        param_y = 150
        cv2.putText(frame, "Current Parameters:", (500, param_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        param_y += 30
        
        for i, (param, value) in enumerate(param_values.items()):
            if i < 8:  # Display only the most important parameters
                display_value = value
                if param in ['minMarkerPerimeterRate', 'maxMarkerPerimeterRate', 
                           'polygonalApproxAccuracyRate', 'minCornerDistanceRate',
                           'minMarkerDistanceRate', 'cornerRefinementMinAccuracy']:
                    display_value = value / 100.0
                    
                cv2.putText(frame, f"{param}: {display_value}", 
                            (500, param_y + i*25), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        
        # Save parameter evaluation
        if frame_count % evaluation_interval == 0:
            param_key = str(param_values)  # Convert dict to string for dictionary key
            if param_key in detection_history:
                detection_history[param_key] += detection_rate
            else:
                detection_history[param_key] = detection_rate
                
            # Update best parameters if current ones are better
            if detection_rate > best_detection_count:
                best_detection_count = detection_rate
                best_params = param_values.copy()
        
        # Display help text
        if show_help:
            help_y = 400
            for line in help_text:
                cv2.putText(frame, line, (10, help_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                help_y += 25
        
        # Show the frame
        cv2.imshow("ArUco Calibration", frame)
        
        # Handle keyboard input
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            save_parameters(param_values)
        elif key == ord('l'):
            load_parameters()
        elif key == ord('h'):
            show_help = not show_help
        elif key == ord('r'):
            # Reset to default parameters
            param_values = {
                'adaptiveThreshWinSizeMin': 3,
                'adaptiveThreshWinSizeMax': 30,
                'adaptiveThreshWinSizeStep': 10,
                'adaptiveThreshConstant': 7,
                'minMarkerPerimeterRate': 3,
                'maxMarkerPerimeterRate': 400,
                'polygonalApproxAccuracyRate': 2,
                'minCornerDistanceRate': 5,
                'minMarkerDistanceRate': 5,
                'cornerRefinementMethod': 1,
                'cornerRefinementWinSize': 5,
                'cornerRefinementMaxIterations': 30,
                'cornerRefinementMinAccuracy': 10,
                'minDistanceToBorder': 3,
                'perspectiveRemovePixelPerCell': 4,
                'perspectiveRemoveIgnoredMarginPerCell': 13,
                'maxErroneousBitsInBorderRate': 60,
                'minOtsuStdDev': 500,
            }
            # Update trackbars to reflect reset values
            for param, value in param_values.items():
                if param in ['minMarkerPerimeterRate', 'maxMarkerPerimeterRate', 
                           'polygonalApproxAccuracyRate', 'minCornerDistanceRate',
                           'minMarkerDistanceRate', 'cornerRefinementMinAccuracy']:
                    value = int(value * 100)
                cv2.setTrackbarPos(param, "ArUco Calibration", value)

        elif key == ord('a'):
            # Toggle auto-tuning mode
            auto_tuning_active = not auto_tuning_active
            print(f"Auto-tuning mode {'enabled' if auto_tuning_active else 'disabled'}")
        
    # Save best parameters before exiting
    if best_params is not None:
        save_parameters(best_params, "best_aruco_params.json")
        
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()

def evaluate_parameters(params: Dict[str, Any], frames: List[np.ndarray], dictionary, scale_factor: float = 0.7) -> Tuple[float, int]:
    """
    Evaluate a set of parameters on the provided frames
    Returns (detection_rate, num_detections)
    """
    detector_params = cv2.aruco.DetectorParameters()
    detector_params = update_detector_parameters(detector_params, params)
    detector = cv2.aruco.ArucoDetector(dictionary, detector_params)
    
    total_detections = 0
    total_marker_count = 0
    
    for frame in frames:
        # Resize for better performance
        small_frame = cv2.resize(frame, (0, 0), fx=scale_factor, fy=scale_factor)
        gray = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)
        
        # Detect markers
        corners, ids, rejected = detector.detectMarkers(gray)
        
        if ids is not None:
            total_detections += 1
            total_marker_count += len(ids)
    
    detection_rate = (total_detections / len(frames)) * 100 if frames else 0
    
    return detection_rate, total_marker_count

def run_auto_calibration(scale_factor=0.7, num_frames=20, max_combinations=50, resolution='4k') -> Dict[str, Any]:
    """
    Run automated calibration to find the best ArUco detection parameters
    Returns the best parameter set
    """
    print("🔍 Starting automated ArUco calibration process...")
    
    # Get resolution configuration
    if resolution not in RESOLUTION_CONFIGS:
        print(f"⚠️ Unknown resolution '{resolution}', defaulting to 4k")
        resolution = '4k'
    
    config = RESOLUTION_CONFIGS[resolution]
    print(f"📹 Calibrating for {config['display_name']} resolution")
    
    # Initialize detector with default parameters
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_250)
    
    # Open camera
    cap = cv2.VideoCapture(0)
    
    # Set resolution based on configuration
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, config['width'])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config['height'])
    
    # Get actual resolution
    actual_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    actual_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    print(f"📹 Camera resolution: {actual_width}x{actual_height}")
    
    # Collect sample frames
    print(f"📸 Collecting {num_frames} sample frames for calibration...")
    frames = []
    for _ in range(num_frames):
        ret, frame = cap.read()
        if not ret:
            print("⚠️ Failed to grab frame")
            continue

        # Apply cropping with resolution-specific settings
        frame = crop_frame(frame, resolution)

        frames.append(frame)
        time.sleep(0.1)  # Short delay to get varied frames
        
    print(f"✅ Collected {len(frames)} frames for testing")
    
    # Generate parameter combinations
    param_keys = list(auto_param_ranges.keys())
    param_values_list = list(auto_param_ranges.values())
    
    # Generate all possible combinations
    all_combinations = list(itertools.product(*param_values_list))
    
    # Limit to a reasonable number
    if len(all_combinations) > max_combinations:
        print(f"⚠️ Too many combinations ({len(all_combinations)}), selecting {max_combinations} random samples")
        import random
        all_combinations = random.sample(all_combinations, max_combinations)
    
    total_combinations = len(all_combinations)
    print(f"🧪 Testing {total_combinations} parameter combinations...")
    
    # Create a progress window
    cv2.namedWindow("Calibration Progress")
    progress_img = np.ones((100, 500, 3), dtype=np.uint8) * 255
    
    # Test all parameter combinations
    best_score = 0
    best_marker_count = 0
    best_params = None
    
    progress_counter = 0
    
    # Test parameters
    for combo_values in all_combinations:
        # Create parameter dictionary
        test_params = {param_keys[i]: combo_values[i] for i in range(len(param_keys))}
        
        # Add default values for params not being tested
        for param, value in param_values.items():
            if param not in test_params:
                test_params[param] = value
        
        # Evaluate parameters
        detection_rate, marker_count = evaluate_parameters(test_params, frames, dictionary, scale_factor)
        
        # Update progress display
        progress_counter += 1
        progress_percent = (progress_counter / total_combinations) * 100
        
        progress_img.fill(255)
        # Draw progress bar
        cv2.rectangle(progress_img, (10, 40), (10 + int(480 * progress_percent / 100), 60), (0, 255, 0), -1)
        cv2.putText(progress_img, f"Testing parameters: {progress_counter}/{total_combinations}", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
        cv2.putText(progress_img, f"Best detection rate: {best_score:.1f}%", 
                   (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)
        
        cv2.imshow("Calibration Progress", progress_img)
        cv2.waitKey(1)
        
        # We prioritize detection rate, but if equal, choose the one with more markers detected
        if detection_rate > best_score or (detection_rate == best_score and marker_count > best_marker_count):
            best_score = detection_rate
            best_marker_count = marker_count
            best_params = test_params
            print(f"📈 New best parameters: {best_score:.1f}% detection rate with {best_marker_count} markers")
            
            # Update progress display with new best params
            cv2.putText(progress_img, f"Best detection rate: {best_score:.1f}% ({best_marker_count} markers)", 
                       (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)
            
            cv2.imshow("Calibration Progress", progress_img)
            cv2.waitKey(1)
    
    # Clean up
    cv2.destroyAllWindows()
    cap.release()
    
    if best_params is not None:
        print(f"🎉 Calibration complete! Best detection rate: {best_score:.1f}% with {best_marker_count} markers")
        print(f"📊 Best parameters: {best_params}")
        save_parameters(best_params, "best_aruco_params.json")
        return best_params
    else:
        print("⚠️ No good parameters found!")
        return {}

def run_live_adaptive_calibration(scale_factor=0.7, resolution='4k', target_detection_rate=100, max_duration=300):
    """
    Run live adaptive calibration that dynamically adjusts parameters until target detection rate is achieved
    
    Args:
        scale_factor: Processing scale factor
        resolution: Camera resolution ('4k' or '1080p')  
        target_detection_rate: Target detection rate percentage (default: 100%)
        max_duration: Maximum calibration time in seconds (default: 300s = 5 minutes)
    
    Returns:
        Best parameter set that achieved the target or closest to it
    """
    print(f"🔍 Starting LIVE ADAPTIVE calibration targeting {target_detection_rate}% detection rate...")
    
    # Get resolution configuration
    if resolution not in RESOLUTION_CONFIGS:
        print(f"⚠️ Unknown resolution '{resolution}', defaulting to 4k")
        resolution = '4k'
    
    config = RESOLUTION_CONFIGS[resolution]
    print(f"📹 Calibrating for {config['display_name']} resolution")
    
    # Initialize detector with default parameters
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_250)
    
    # Open camera
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, config['width'])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config['height'])
    
    # Get actual resolution
    actual_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    actual_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    print(f"📹 Camera resolution: {actual_width}x{actual_height}")
    
    # Define progressive parameter adjustment strategies
    adjustment_strategies = [
        {
            'name': 'Conservative Detection',
            'params': {
                'adaptiveThreshWinSizeMin': 5,
                'adaptiveThreshWinSizeMax': 23,
                'adaptiveThreshConstant': 7,
                'minMarkerPerimeterRate': 0.04,
                'maxMarkerPerimeterRate': 3.0,
                'cornerRefinementMethod': 1,  # Subpixel
                'polygonalApproxAccuracyRate': 0.03,
                'minCornerDistanceRate': 0.05,
                'errorCorrectionRate': 0.6
            }
        },
        {
            'name': 'High Sensitivity',
            'params': {
                'adaptiveThreshWinSizeMin': 3,
                'adaptiveThreshWinSizeMax': 45,
                'adaptiveThreshConstant': 9,
                'minMarkerPerimeterRate': 0.02,
                'maxMarkerPerimeterRate': 5.0,
                'cornerRefinementMethod': 2,  # Contour
                'polygonalApproxAccuracyRate': 0.05,
                'minCornerDistanceRate': 0.03,
                'errorCorrectionRate': 0.8
            }
        },
        {
            'name': 'Ultra High Sensitivity',
            'params': {
                'adaptiveThreshWinSizeMin': 3,
                'adaptiveThreshWinSizeMax': 50,
                'adaptiveThreshConstant': 11,
                'minMarkerPerimeterRate': 0.01,
                'maxMarkerPerimeterRate': 6.0,
                'cornerRefinementMethod': 2,
                'polygonalApproxAccuracyRate': 0.07,
                'minCornerDistanceRate': 0.02,
                'errorCorrectionRate': 0.9
            }
        },
        {
            'name': 'Lighting Adaptive',
            'params': {
                'adaptiveThreshWinSizeMin': 7,
                'adaptiveThreshWinSizeMax': 35,
                'adaptiveThreshConstant': 5,
                'minMarkerPerimeterRate': 0.025,
                'maxMarkerPerimeterRate': 4.5,
                'cornerRefinementMethod': 1,
                'polygonalApproxAccuracyRate': 0.02,
                'minCornerDistanceRate': 0.04,
                'errorCorrectionRate': 0.7
            }
        }
    ]
    
    start_time = time.time()
    best_params = None
    best_detection_rate = 0
    best_marker_count = 0
    strategy_index = 0
    frames_tested = 0
    detection_history = deque(maxlen=50)  # Store last 50 frames of detection results
    
    print(f"🎯 Target: {target_detection_rate}% detection rate")
    print(f"⏱️ Maximum duration: {max_duration} seconds")
    print(f"🔄 Will try {len(adjustment_strategies)} different strategies")
    
    # Create progress window
    cv2.namedWindow("Live Adaptive Calibration", cv2.WINDOW_AUTOSIZE)
    
    while time.time() - start_time < max_duration and strategy_index < len(adjustment_strategies):
        current_strategy = adjustment_strategies[strategy_index]
        
        print(f"\n🧪 Testing Strategy {strategy_index + 1}/{len(adjustment_strategies)}: {current_strategy['name']}")
        
        # Apply current strategy parameters
        detector_params = cv2.aruco.DetectorParameters()
        detector_params = update_detector_parameters(detector_params, current_strategy['params'])
        
        # Required parameters to prevent errors
        detector_params.markerBorderBits = 1
        detector_params.minSideLengthCanonicalImg = 16
        detector_params.perspectiveRemoveIgnoredMarginPerCell = 0.13
        
        detector = cv2.aruco.ArucoDetector(dictionary, detector_params)
        
        strategy_start_time = time.time()
        strategy_frames = 0
        strategy_detections = []
        
        # Test current strategy for up to 30 seconds
        while (time.time() - strategy_start_time < 30 and 
               time.time() - start_time < max_duration and 
               len(strategy_detections) < 100):  # Max 100 frames per strategy
            
            ret, frame = cap.read()
            if not ret:
                continue
            
            # Apply cropping with resolution-specific settings
            frame = crop_frame(frame, resolution)
            
            # Resize for processing
            small_frame = cv2.resize(frame, (0, 0), fx=scale_factor, fy=scale_factor)
            gray = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)
            
            # Detect markers
            corners, ids, rejected = detector.detectMarkers(gray)
            
            # Count detections (looking for corner markers 0,1,2,3)
            corner_detections = 0
            if ids is not None:
                for marker_id in ids.flatten():
                    if 0 <= marker_id <= 3:  # Only count corner markers
                        corner_detections += 1
            
            strategy_detections.append(corner_detections)
            detection_history.append(corner_detections)
            
            # Calculate current detection rates
            strategy_rate = (sum(1 for d in strategy_detections if d >= 3) / len(strategy_detections)) * 100
            overall_rate = (sum(1 for d in detection_history if d >= 3) / len(detection_history)) * 100 if detection_history else 0
            
            # Draw visualization
            if corners is not None and ids is not None:
                # Scale corners back to original frame size
                corners = [corner / scale_factor for corner in corners]
                cv2.aruco.drawDetectedMarkers(frame, corners, ids)
                
                # Draw centers and IDs
                for i, corner in enumerate(corners):
                    center = np.mean(corner[0], axis=0).astype(int)
                    cv2.circle(frame, tuple(center), 8, (0, 255, 0), -1)
                    cv2.putText(frame, f"ID: {ids[i][0]}", 
                              (center[0] + 15, center[1] - 10), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Draw status information
            elapsed_time = time.time() - start_time
            cv2.putText(frame, f"Strategy: {current_strategy['name']}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
            cv2.putText(frame, f"Current Rate: {strategy_rate:.1f}% ({corner_detections}/4 corners)", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f"Best Rate: {best_detection_rate:.1f}%", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.putText(frame, f"Target: {target_detection_rate}%", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
            cv2.putText(frame, f"Time: {elapsed_time:.1f}s / {max_duration}s", (10, 190), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Progress bar for strategy
            bar_width = 400
            bar_progress = min(1.0, len(strategy_detections) / 50.0)
            cv2.rectangle(frame, (10, 210), (10 + int(bar_width * bar_progress), 230), (0, 255, 0), -1)
            cv2.rectangle(frame, (10, 210), (10 + bar_width, 230), (255, 255, 255), 2)
            
            cv2.imshow("Live Adaptive Calibration", frame)
            
            strategy_frames += 1
            frames_tested += 1
            
            # Check if we've achieved target with current strategy
            if len(strategy_detections) >= 20 and strategy_rate >= target_detection_rate:
                print(f"🎉 TARGET ACHIEVED! {strategy_rate:.1f}% detection rate with {current_strategy['name']}")
                best_params = current_strategy['params'].copy()
                best_detection_rate = strategy_rate
                break
            
            # Update best if this strategy is better
            if strategy_rate > best_detection_rate:
                best_detection_rate = strategy_rate
                best_params = current_strategy['params'].copy()
                best_marker_count = max(strategy_detections) if strategy_detections else 0
                print(f"📈 New best: {best_detection_rate:.1f}% with {current_strategy['name']}")
            
            # ESC to quit early
            if cv2.waitKey(1) & 0xFF == 27:
                print("🛑 Calibration stopped by user")
                break
        
        print(f"✅ Strategy '{current_strategy['name']}' completed: {strategy_rate:.1f}% detection rate over {len(strategy_detections)} frames")
        
        # Move to next strategy if target not achieved
        if best_detection_rate < target_detection_rate:
            strategy_index += 1
        else:
            break
    
    # Cleanup
    cv2.destroyAllWindows()
    cap.release()
    
    # Results
    total_time = time.time() - start_time
    if best_params is not None:
        if best_detection_rate >= target_detection_rate:
            print(f"🎉 SUCCESS! Achieved {best_detection_rate:.1f}% detection rate in {total_time:.1f}s")
        else:
            print(f"📊 PARTIAL SUCCESS: Best achieved {best_detection_rate:.1f}% detection rate in {total_time:.1f}s")
            print(f"   (Target was {target_detection_rate}%)")
        
        print(f"🔧 Best parameters: {best_params}")
        save_parameters(best_params, "best_aruco_params.json")
        return best_params
    else:
        print(f"❌ No improvement found in {total_time:.1f}s")
        return {}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='ArUco tag detection calibration tool')
    parser.add_argument('--no-load', action='store_false', dest='load_previous',
                        help='Do not load previous best parameters')
    parser.add_argument('--scale', type=float, default=0.7, 
                        help='Scale factor for processing (default: 0.7)')
    parser.add_argument('--auto', action='store_true',
                        help='Run automated calibration instead of manual')
    parser.add_argument('--live', action='store_true',
                        help='Run live adaptive calibration targeting 100% detection')
    parser.add_argument('--frames', type=int, default=20,
                        help='Number of frames to capture for automated calibration (default: 20)')
    parser.add_argument('--max-combinations', type=int, default=50,
                        help='Maximum number of parameter combinations to test (default: 50)')
    parser.add_argument('--target-rate', type=int, default=100,
                        help='Target detection rate percentage for live calibration (default: 100)')
    parser.add_argument('--max-duration', type=int, default=300,
                        help='Maximum calibration duration in seconds (default: 300)')
    parser.add_argument('--resolution', choices=['4k', '1080p'], default='4k',
                        help='Camera resolution for calibration (default: 4k)')
                        
    args = parser.parse_args()
    
    if args.live:
        # Run live adaptive calibration
        run_live_adaptive_calibration(args.scale, args.resolution, args.target_rate, args.max_duration)
    elif args.auto:
        # Run automated calibration
        run_auto_calibration(args.scale, args.frames, args.max_combinations, args.resolution)
    else:
        # Run manual calibration
        run_calibration(args.scale, args.resolution, args.load_previous)