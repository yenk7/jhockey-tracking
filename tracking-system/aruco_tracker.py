import cv2
import numpy as np
import time
import asyncio
import base64
import platform
from collections import deque

locked_corners = None  # Stores the locked corner positions
lock_state = False  # Indicates whether corners are locked
auto_lock_active = False  # Flag to indicate if auto-lock is active
auto_lock_start_time = 0  # When auto-lock process started
auto_lock_timeout = 15  # Auto-lock timeout in seconds
last_frame_base64 = ""  # Store the last encoded frame to prevent feed interruption
current_corner_detections = {}  # Store the most recent corner detections for auto-lock

# Dictionary to store position history for each marker (for temporal filtering)
position_history = {}
# Number of frames to use for temporal filtering
HISTORY_LENGTH = 3

# New: Add focus zone history to track detection quality in different regions
focus_zone_history = {}
ZONE_GRID = (2, 2)  # Divide the image into 2x2 zones for multi-zone processing

async def track_aruco_tags(lock_queue, scale_factor=0.7):
    
    global locked_corners, lock_state, position_history, focus_zone_history, auto_lock_active, auto_lock_start_time, last_frame_base64
    print(f"🟢 Initial lock state: {lock_state}", flush=True)

    # Optimize CV2 parameters for M1 Mac
    is_mac_arm = platform.system() == 'Darwin' and 'arm' in platform.machine().lower()
    if is_mac_arm:
        print("Optimizing OpenCV parameters for Apple Silicon")
        # For Apple Silicon, we'll use Metal acceleration if available
        cv2.setUseOptimized(True)
        
    # Load the ArUco dictionary - Use 4x4 for better performance
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_250)
    detector_params = cv2.aruco.DetectorParameters()
    
    # Enhanced detector parameters for more consistent tracking
    detector_params.adaptiveThreshWinSizeMin = 3
    detector_params.adaptiveThreshWinSizeMax = 23
    detector_params.adaptiveThreshWinSizeStep = 10
    detector_params.adaptiveThreshConstant = 7
    detector_params.minMarkerPerimeterRate = 0.02
    detector_params.maxMarkerPerimeterRate = 4.0
    detector_params.polygonalApproxAccuracyRate = 0.03
    detector_params.minCornerDistanceRate = 0.05
    detector_params.minMarkerDistanceRate = 0.04
    detector_params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    detector_params.cornerRefinementWinSize = 5
    detector_params.cornerRefinementMaxIterations = 30
    detector_params.cornerRefinementMinAccuracy = 0.1
    detector_params.minDistanceToBorder = 3

    # Create multiple detectors with different parameters for focus zones
    detector = cv2.aruco.ArucoDetector(dictionary, detector_params)
    
    # For sharper areas (closer to camera)
    sharp_detector_params = detector_params.clone() if hasattr(detector_params, 'clone') else cv2.aruco.DetectorParameters()
    sharp_detector_params.adaptiveThreshConstant = 9
    sharp_detector_params.cornerRefinementMinAccuracy = 0.05
    sharp_detector = cv2.aruco.ArucoDetector(dictionary, sharp_detector_params)
    
    # For blurrier areas (farther from camera)
    blur_detector_params = detector_params.clone() if hasattr(detector_params, 'clone') else cv2.aruco.DetectorParameters()
    blur_detector_params.adaptiveThreshConstant = 5
    blur_detector_params.cornerRefinementMinAccuracy = 0.15
    blur_detector_params.minMarkerPerimeterRate = 0.015  # Allow smaller markers
    blur_detector = cv2.aruco.ArucoDetector(dictionary, blur_detector_params)
    
    # Try to open the camera with hardware acceleration if available
    cap = cv2.VideoCapture(0)
    
    # Get available resolutions by testing common high resolutions
    print("📊 Detecting available camera resolutions...")
    
    # Common resolutions to test, ordered from highest to lowest
    test_resolutions = [
        (4096, 2160),  # 4K
        (3840, 2160),  # 4K UHD
        (2560, 1440),  # 2K QHD
        (1920, 1080),  # Full HD
        (1600, 1200),  # UXGA
        (1280, 720),   # HD
        (640, 480)     # VGA
    ]
    
    # Get current default values
    default_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    default_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    print(f"📹 Default camera resolution: {default_width}x{default_height}")
    
    # Test each resolution
    available_resolutions = []
    print("Testing supported resolutions:")
    for width, height in test_resolutions:
        # Try to set resolution
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        
        # Check what we actually got
        actual_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        actual_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        
        # Check if the resolution was actually set (with some tolerance)
        if abs(width - actual_width) < 100 and abs(height - actual_height) < 100:
            print(f"  ✅ {actual_width}x{actual_height}")
            available_resolutions.append((width, height, actual_width, actual_height))
        else:
            print(f"  ❌ {width}x{height} (not supported)")
    
    # Set to the highest available resolution
    if available_resolutions:
        # Sort by total pixels (width * height) in descending order
        available_resolutions.sort(key=lambda res: res[0] * res[1], reverse=True)
        best_width, best_height, actual_width, actual_height = available_resolutions[0]
        
        print(f"🔍 Setting to maximum resolution: {actual_width}x{actual_height}")
        
        # Try to set the highest resolution again to ensure it's applied
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, best_width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, best_height)
        
        # Adjust scale factor based on resolution for optimal performance
        if best_width >= 3840:  # 4K
            scale_factor = 0.2
            print(f"⚡ Using scale factor of {scale_factor} for 4K resolution")
        elif best_width >= 2560:  # 2K/QHD
            scale_factor = 0.25
            print(f"⚡ Using scale factor of {scale_factor} for 2K resolution")
        elif best_width >= 1920:  # Full HD
            scale_factor = 0.3
            print(f"⚡ Using scale factor of {scale_factor} for Full HD resolution")
        elif best_width >= 1280:  # HD
            scale_factor = 0.5
            print(f"⚡ Using scale factor of {scale_factor} for HD resolution")
    else:
        print("⚠️ No resolution change detected, using default settings")
    
    # Enhanced camera settings for better ArUco detection
    print("Optimizing camera parameters...")
    
    # Try setting various camera parameters, ignoring errors if not supported
    try:
        cap.set(cv2.CAP_PROP_FPS, 30)
        print("✓ Set target FPS: 30")
    except:
        print("⚠️ Could not set FPS")
    
    # Note: Since the camera has a fixed position and focus issues,
    # we intentionally avoid disabling autofocus and auto exposure
    
    try:
        # For cameras that support it, try to increase the sharpness
        cap.set(cv2.CAP_PROP_SHARPNESS, 50)
        print("✓ Increased camera sharpness")
    except:
        print("⚠️ Could not set camera sharpness")
    
    # For Apple Silicon, attempt to use optimized settings
    if is_mac_arm:
        try:
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc('M', 'J', 'P', 'G'))
            print("✓ Set MJPG codec for Apple Silicon")
        except:
            print("⚠️ Could not set video codec")
    
    if not cap.isOpened():
        print("⚠️ Failed to open camera, trying alternate methods")
        cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)  # Try macOS specific API
        
        # Try setting camera parameters again with macOS API
        if available_resolutions:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, best_width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, best_height)
    
    # Get final camera parameters
    final_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    final_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    final_fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"🎥 Camera initialized at {final_width}x{final_height} @ {final_fps}fps")
    
    prev_time = time.time()
    frame_count = 0
    fps_update_interval = 10  # Update FPS every 10 frames
    fps = 0.0  # Initialize fps to avoid reference before assignment
    
    corner_tag_positions = {
        0: np.array([0, 0]),
        1: np.array([0, 7.75]),
        2: np.array([3.74, 7.75]),
        3: np.array([3.74, 0])
    }
    
    # Create kernels for different image enhancements
    kernel_sharpen = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    kernel_stronger_sharpen = np.array([[-2,-2,-2], [-2,17,-2], [-2,-2,-2]])
    kernel_edge_enhance = np.array([[-1,-1,-1,-1,-1],
                                    [-1,2,2,2,-1],
                                    [-1,2,8,2,-1],
                                    [-1,2,2,2,-1],
                                    [-1,-1,-1,-1,-1]]) / 8.0

    # Helper function for auto-locking corners with improved error handling
    def check_auto_lock_status(detected_corners, current_time):
        global auto_lock_active, auto_lock_start_time, lock_state, locked_corners, current_corner_detections
        
        if not auto_lock_active:
            return None
            
        try:
            # Add frame processing timeout protection
            frame_process_start = time.time()
            
            # Store detected corners in the global variable to persist between frames
            # Only update for corner tags (0-3)
            for tag_id, position in detected_corners.items():
                # Limit processing time per tag to prevent freezing
                if time.time() - frame_process_start > 0.05:  # 50ms max processing time
                    return {"status": "searching", "message": "Processing frame... please wait"}
                    
                tag_id = int(tag_id)
                if tag_id <= 3:
                    current_corner_detections[tag_id] = position
            
            elapsed = current_time - auto_lock_start_time
            
            # Convert all tag IDs to standard Python integers
            corner_ids = set(int(k) for k in current_corner_detections.keys())
            
            # Check if we have all four corners (0, 1, 2, 3)
            if len(corner_ids) == 4 and all(i in corner_ids for i in range(4)):
                # Success - found all corners!
                auto_lock_active = False
                lock_state = True
                # Create a clean copy without references to the original dict
                locked_corners = {int(k): v.copy() if hasattr(v, 'copy') else v 
                                for k, v in current_corner_detections.items()}
                print("🎯 Auto-lock successful! All four corners found.")
                # Reset the detection cache
                current_corner_detections = {}
                return {"status": "success", "message": "All corners found and locked! 🎯"}
            
            # Check for timeout
            if elapsed >= auto_lock_timeout:
                auto_lock_active = False
                current_corner_detections = {}  # Reset detection cache
                print("❌ Auto-lock failed: timeout after 15 seconds.")
                return {"status": "failure", "message": "Auto-lock failed after 15 seconds ☹️"}
            
            # Still searching - report how many corners found so far
            found_corners = sum(1 for i in range(4) if i in corner_ids)
            remaining_time = int(auto_lock_timeout - elapsed)
            return {"status": "searching", "message": f"Finding corners: {found_corners}/4 (⏱️ {remaining_time}s)"}
            
        except Exception as e:
            # Catch any exception, log it, but don't let it crash the system
            print(f"Non-critical error in auto-lock process: {e}")
            # Clear auto-lock if we get a critical exception to prevent system hang
            if str(e).lower().find("memory") >= 0 or str(e).lower().find("resource") >= 0:
                auto_lock_active = False
                current_corner_detections = {}
                return {"status": "failure", "message": "Auto-lock failed: system resources exceeded"}
            return {"status": "searching", "message": f"Finding corners... (⏱️ {int(auto_lock_timeout - elapsed)}s)"}

    # Helper function to divide image into zones and process each with adaptive parameters
    def process_image_by_zones(image, grid_size=(2, 2)):
        h, w = image.shape[:2]
        zone_h, zone_w = h // grid_size[0], w // grid_size[1]
        all_corners, all_ids = [], []
        
        for y in range(grid_size[0]):
            for x in range(grid_size[1]):
                # Extract zone
                zone_y1, zone_y2 = y * zone_h, (y + 1) * zone_h
                zone_x1, zone_x2 = x * zone_w, (x + 1) * zone_w
                
                # Ensure we don't exceed image boundaries
                zone_y2 = min(zone_y2, h)
                zone_x2 = min(zone_x2, w)
                
                zone = image[zone_y1:zone_y2, zone_x1:zone_x2]
                
                # Skip if zone is too small
                if zone.size == 0:
                    continue
                
                # Apply zone-specific enhancements based on past detection success
                zone_key = f"{y},{x}"
                if zone_key not in focus_zone_history:
                    focus_zone_history[zone_key] = deque([0], maxlen=5)  # Track detection success rate
                
                success_rate = sum(focus_zone_history[zone_key]) / max(1, len(focus_zone_history[zone_key]))
                
                # Enhanced preprocessing specific to this zone's historical performance
                zone_gray = cv2.cvtColor(zone, cv2.COLOR_BGR2GRAY) if len(zone.shape) == 3 else zone
                
                # Different processing strategies based on past detection success
                if success_rate < 0.3:  # Low success rate - try aggressive enhancement
                    # Apply stronger enhancements for challenging zones
                    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
                    zone_enhanced = clahe.apply(zone_gray)
                    zone_sharpened = cv2.filter2D(zone_enhanced, -1, kernel_stronger_sharpen)
                    zone_bilateral = cv2.bilateralFilter(zone_sharpened, 5, 75, 75)
                    
                    # Try multiple thresholding approaches
                    zone_thresh1 = cv2.adaptiveThreshold(zone_bilateral, 255, 
                                                        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                                        cv2.THRESH_BINARY_INV, 11, 2)
                    zone_thresh2 = cv2.threshold(zone_bilateral, 0, 255, 
                                                cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
                    
                    # Use blurrier detector optimized for difficult zones
                    corners1, ids1, _ = blur_detector.detectMarkers(zone_bilateral)
                    corners2, ids2, _ = blur_detector.detectMarkers(zone_thresh1)
                    corners3, ids3, _ = blur_detector.detectMarkers(zone_thresh2)
                    
                    # Combine results
                    zone_corners = []
                    zone_ids = []
                    
                    if ids1 is not None:
                        zone_corners.extend(corners1)
                        zone_ids.extend(ids1)
                    if ids2 is not None and ids2.size > 0:
                        zone_corners.extend(corners2)
                        zone_ids.extend(ids2)
                    if ids3 is not None and ids3.size > 0:
                        zone_corners.extend(corners3)
                        zone_ids.extend(ids3)
                        
                    if len(zone_corners) > 0:
                        zone_ids = np.vstack(zone_ids) if len(zone_ids) > 0 else np.array([])
                    else:
                        zone_ids = None
                        
                elif success_rate > 0.7:  # High success rate - use sharper detector
                    # For zones that work well, use standard enhancements
                    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
                    zone_enhanced = clahe.apply(zone_gray)
                    zone_sharpened = cv2.filter2D(zone_enhanced, -1, kernel_sharpen)
                    
                    # Use sharper detector for already successful zones
                    zone_corners, zone_ids, _ = sharp_detector.detectMarkers(zone_sharpened)
                else:  # Medium success rate - use standard detector
                    # Balanced approach
                    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8,8))
                    zone_enhanced = clahe.apply(zone_gray)
                    zone_bilateral = cv2.bilateralFilter(zone_enhanced, 5, 65, 65)
                    zone_sharpened = cv2.filter2D(zone_bilateral, -1, kernel_edge_enhance)
                    
                    # Use standard detector
                    zone_corners, zone_ids, _ = detector.detectMarkers(zone_sharpened)
                
                # Update success history for this zone
                if zone_ids is not None and len(zone_ids) > 0:
                    focus_zone_history[zone_key].append(1)  # Success
                else:
                    focus_zone_history[zone_key].append(0)  # Failure
                
                # If markers found, adjust coordinates to global frame
                if zone_ids is not None and len(zone_ids) > 0:
                    for i in range(len(zone_corners)):
                        # Adjust coordinates
                        zone_corners[i][0][:, 0] += zone_x1
                        zone_corners[i][0][:, 1] += zone_y1
                        all_corners.append(zone_corners[i])
                        all_ids.append(zone_ids[i])
        
        # Combine results from all zones
        if len(all_corners) > 0:
            all_ids = np.array(all_ids)
            return all_corners, all_ids
        else:
            return [], None
    
    while True:
        try:
            # Non-blocking check for lock state changes or auto-lock commands
            try:
                command = lock_queue.get_nowait()
                
                # If the command is a boolean, it's the normal lock_state toggle
                if isinstance(command, bool):
                    lock_state = command
                    if not lock_state:
                        locked_corners = None
                # If command is a dict with "auto_lock": True, start the auto-lock process
                elif isinstance(command, dict) and command.get("auto_lock") is True:
                    auto_lock_active = True
                    auto_lock_start_time = time.time()
                    # Clear previous corner detections to start fresh
                    global current_corner_detections
                    current_corner_detections = {}
                    print("🔄 Auto-lock process started. Looking for all 4 corners...")
                
            except asyncio.QueueEmpty:
                pass

            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame")
                await asyncio.sleep(0.1)
                continue

            frame = cv2.rotate(frame, cv2.ROTATE_180)

            # Update FPS calculation
            frame_count += 1
            if frame_count == 1:
                start_time = time.time()
            elif frame_count % fps_update_interval == 0:
                curr_time = time.time()
                fps = fps_update_interval / (curr_time - start_time)
                start_time = curr_time
            else:
                curr_time = time.time()
                fps = 1.0 / (curr_time - prev_time)
            
            prev_time = time.time()

            # Resize frame for performance
            small_frame = cv2.resize(frame, (0, 0), fx=scale_factor, fy=scale_factor)
            
            # Apply multi-zone processing to handle variable focus across the image
            corners, ids = process_image_by_zones(small_frame, ZONE_GRID)
            
            # If multi-zone approach failed, fallback to traditional methods
            if ids is None or len(ids) == 0:
                # Enhanced image preprocessing
                gray = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)
                
                # Apply adaptive histogram equalization
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
                enhanced_gray = clahe.apply(gray)
                
                # Apply slight sharpening
                sharpened = cv2.filter2D(enhanced_gray, -1, kernel_sharpen)
                
                # Apply mild bilateral filtering to reduce noise while preserving edges
                filtered = cv2.bilateralFilter(sharpened, 5, 75, 75)
                
                # Adaptive thresholding for better marker detection in variable lighting
                thresh = cv2.adaptiveThreshold(filtered, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                              cv2.THRESH_BINARY_INV, 11, 2)
                
                # Try detecting markers with multiple processing methods
                corners, ids, _ = detector.detectMarkers(filtered)
                
                # If no markers found, try with original grayscale
                if ids is None or len(ids) == 0:
                    corners, ids, _ = detector.detectMarkers(gray)
                    
                # And if still no markers found, try with threshold image
                if ids is None or len(ids) == 0:
                    corners, ids, _ = detector.detectMarkers(thresh)

            detected_corner_positions = {}  # For new corners
            pixel_positions = {}  # Stores pixel positions of all markers
            estimated_robot_positions = {}  # Stores estimated real-world positions

            if ids is not None:
                # Scale corners back to original frame size
                corners = [corner / scale_factor for corner in corners]
                cv2.aruco.drawDetectedMarkers(frame, corners, ids)

                for i, corner in enumerate(corners):
                    # <<< FIX: Explicitly convert NumPy int to standard Python int >>>
                    # <<< Apply this conversion *here* in the main loop >>>
                    marker_id = int(ids[i][0])

                    if marker_id >= 30:
                        continue

                    center = np.mean(corner[0], axis=0)

                    # Apply temporal filtering to smooth out jitter
                    # Use the standard Python int 'marker_id' as the key
                    if marker_id not in position_history:
                        position_history[marker_id] = deque(maxlen=HISTORY_LENGTH)
                    
                    position_history[marker_id].append(center)
                    
                    # Calculate smoothed position using temporal averaging
                    if len(position_history[marker_id]) == HISTORY_LENGTH:
                        smoothed_center = np.mean(position_history[marker_id], axis=0)
                    else:
                        smoothed_center = center
                    
                    # Use the standard Python int 'marker_id' for comparisons and keys
                    if marker_id in corner_tag_positions:
                        detected_corner_positions[marker_id] = corner_tag_positions[marker_id] * 30.48
                        pixel_positions[marker_id] = smoothed_center

                    # Use the standard Python int 'marker_id' for comparisons and keys
                    if marker_id > 3:
                        pixel_positions[marker_id] = smoothed_center

                    center_int = tuple(smoothed_center.astype(int))
                    cv2.circle(frame, center_int, 5, (0, 255, 0), -1)
                    cv2.putText(frame, f"ID: {marker_id}", (center_int[0] + 10, center_int[1] - 10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                # Check auto-lock status if active
                auto_lock_info = None
                if auto_lock_active:
                    # This block now uses detected_corner_positions which was populated 
                    # using standard Python int keys above.
                    auto_lock_start_time_local = time.time()
                    auto_lock_info = check_auto_lock_status(detected_corner_positions, time.time())
                    
                    # If auto-lock processing takes too long, yield control temporarily
                    if time.time() - auto_lock_start_time_local > 0.1:  # 100ms 
                        print("Auto-lock processing taking too long, yielding control")
                        # Force a quick yield to ensure UI stays responsive
                        await asyncio.sleep(0.05)

                # If locked, use locked corners instead of newly detected ones
                if lock_state and locked_corners:
                    try:
                        # Ensure comparisons use standard Python ints
                        pixel_positions_corners = {
                            tag_id: detected_pixel_positions[tag_id]
                            for tag_id in detected_pixel_positions if int(tag_id) <= 3
                        }

                        pixel_positions_robots = {
                            tag_id: pixel_positions[tag_id]
                            for tag_id in pixel_positions if int(tag_id) > 3
                        }

                        pixel_positions = {**pixel_positions_corners, **pixel_positions_robots}
                        # locked_corners should already have standard int keys from check_auto_lock_status
                        detected_corner_positions = locked_corners.copy()
                    except NameError:
                        # Handle the case if detected_pixel_positions is not defined yet
                        print("Warning: Corner positions not yet defined for locking")
                    except TypeError as te:
                        # Catch potential type errors during key access/comparison
                        print(f"Type error during locking logic: {te}")
                
                elif not lock_state:
                    # Ensure keys are standard ints when copying
                    locked_corners = {int(k): v for k, v in detected_corner_positions.items()} if detected_corner_positions else None
                    detected_pixel_positions = {int(k): v for k, v in pixel_positions.items()}
                
                # Only proceed if we have at least 3 corner markers
                # Ensure keys are standard ints when checking length and iterating
                if len(detected_corner_positions) >= 3:
                    src_points = []
                    dst_points = []

                    for tag_id, world_pos in detected_corner_positions.items():
                        # Ensure pixel_positions uses standard int keys
                        src_points.append(pixel_positions[int(tag_id)]) 
                        dst_points.append(world_pos)

                    src_points = np.array(src_points, dtype=np.float32)
                    dst_points = np.array(dst_points, dtype=np.float32)

                    if len(detected_corner_positions) == 3:
                        H = cv2.getAffineTransform(src_points[:3], dst_points[:3])
                    else:
                        H, _ = cv2.findHomography(src_points, dst_points)

                    for tag_id, pixel_pos in pixel_positions.items():
                        if tag_id > 3:
                            pixel_tag = np.array([pixel_pos], dtype=np.float32)

                            if len(detected_corner_positions) == 3:
                                world_tag = cv2.transform(pixel_tag[None, :, :], H)
                                estimated_position = world_tag[0, 0]
                            else:
                                pixel_tag_homogeneous = np.array([pixel_pos[0], pixel_pos[1], 1])
                                world_tag_homogeneous = np.dot(H, pixel_tag_homogeneous)
                                world_tag_homogeneous /= world_tag_homogeneous[2]
                                estimated_position = world_tag_homogeneous[:2]

                            estimated_robot_positions[tag_id] = np.round(estimated_position)

                # Draw focus zones on debug frame for visualization
                debug_frame = frame.copy()
                h, w = debug_frame.shape[:2]
                zone_h, zone_w = h // ZONE_GRID[0], w // ZONE_GRID[1]
                
                # Display auto-lock status message if active
                if auto_lock_active:
                    # Update status based on accumulated detections, even if none this frame
                    auto_lock_info = check_auto_lock_status(current_corner_detections, time.time())
                    message = auto_lock_info.get("message", "")
                    status = auto_lock_info.get("status", "searching")
                    # Choose color based on status
                    if status == "success":
                        color = (0, 255, 0)
                    elif status == "failure":
                        color = (0, 0, 255)
                    else:
                        color = (0, 165, 255)
                    cv2.putText(debug_frame, message, (int(w/2)-200, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
                
                for y in range(ZONE_GRID[0]):
                    for x in range(ZONE_GRID[1]):
                        zone_key = f"{y},{x}"
                        # Calculate success rate for this zone
                        if zone_key in focus_zone_history and len(focus_zone_history[zone_key]) > 0:
                            success_rate = sum(focus_zone_history[zone_key]) / len(focus_zone_history[zone_key])
                            
                            # Color code the zones based on detection success rate
                            zone_y1, zone_y2 = y * zone_h, min((y + 1) * zone_h, h)
                            zone_x1, zone_x2 = x * zone_w, min((x + 1) * zone_w, w)
                            
                            if success_rate > 0.7:
                                color = (0, 255, 0)  # Green for good zones
                            elif success_rate > 0.3:
                                color = (0, 165, 255)  # Orange for medium zones
                            else:
                                color = (0, 0, 255)  # Red for bad zones
                            
                            # Draw zone border with color indicating detection quality
                            cv2.rectangle(debug_frame, (zone_x1, zone_y1), (zone_x2, zone_y2), color, 2)

                # Reduce the frame quality for faster transmission
                if is_mac_arm:
                    # Use a lower resolution encoding for better performance
                    frame_small = cv2.resize(debug_frame, (0, 0), fx=0.5, fy=0.5)
                    encode_quality = [int(cv2.IMWRITE_JPEG_QUALITY), 70]
                    _, buffer = cv2.imencode(".jpg", frame_small, encode_quality)
                else:
                    _, buffer = cv2.imencode(".jpg", debug_frame)
                
                # Store the last successful frame encoding
                last_frame_base64 = base64.b64encode(buffer).decode("utf-8")

                output_dict = {
                    'robot_tags': {int(tag_id): estimated_robot_positions[tag_id].tolist() for tag_id in estimated_robot_positions},
                    'corner_tags': {int(tag_id): detected_corner_positions[tag_id].tolist() for tag_id in detected_corner_positions},
                    'fps': round(fps, 2),
                    "frame": last_frame_base64  # Always include the latest frame
                }

                # Add auto-lock status to output if active
                if auto_lock_info:
                    output_dict["auto_lock_info"] = auto_lock_info

                # Control the frame rate to prevent overwhelming the CPU
                await asyncio.sleep(0.01)  # Helps prevent locking up the event loop
                yield output_dict

            else:
                # If no markers detected, still yield empty data with current fps
                # But still show the zone visualization
                debug_frame = frame.copy()
                
                # Display auto-lock status message if active
                if auto_lock_active:
                    # Update status based on accumulated detections, even if none this frame
                    auto_lock_info = check_auto_lock_status(current_corner_detections, time.time())
                    message = auto_lock_info.get("message", "")
                    status = auto_lock_info.get("status", "searching")
                    # Choose color based on status
                    if status == "success":
                        color = (0, 255, 0)
                    elif status == "failure":
                        color = (0, 0, 255)
                    else:
                        color = (0, 165, 255)
                    cv2.putText(debug_frame, message, (int(w/2)-200, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
                    
                h, w = debug_frame.shape[:2]
                zone_h, zone_w = h // ZONE_GRID[0], w // ZONE_GRID[1]
                
                for y in range(ZONE_GRID[0]):
                    for x in range(ZONE_GRID[1]):
                        zone_key = f"{y},{x}"
                        zone_y1, zone_y2 = y * zone_h, min((y + 1) * zone_h, h)
                        zone_x1, zone_x2 = x * zone_w, min((x + 1) * zone_w, w)
                        
                        if zone_key in focus_zone_history and len(focus_zone_history[zone_key]) > 0:
                            success_rate = sum(focus_zone_history[zone_key]) / len(focus_zone_history[zone_key])
                            
                            if success_rate > 0.7:
                                color = (0, 255, 0)  # Green
                            elif success_rate > 0.3:
                                color = (0, 165, 255)  # Orange
                            else:
                                color = (0, 0, 255)  # Red
                        else:
                            color = (128, 128, 128)  # Gray for no data
                            
                        cv2.rectangle(debug_frame, (zone_x1, zone_y1), (zone_x2, zone_y2), color, 2)
                
                if is_mac_arm:
                    frame_small = cv2.resize(debug_frame, (0, 0), fx=0.5, fy=0.5)
                    encode_quality = [int(cv2.IMWRITE_JPEG_QUALITY), 70]
                    _, buffer = cv2.imencode(".jpg", frame_small, encode_quality)
                else:
                    _, buffer = cv2.imencode(".jpg", debug_frame)
                    
                base64_frame = base64.b64encode(buffer).decode("utf-8")
                
                output_dict = {
                    'robot_tags': {},
                    'corner_tags': {},
                    'fps': round(fps, 2),
                    "frame": base64_frame
                }
                
                # Add auto-lock status to output if active
                if auto_lock_active:
                    output_dict["auto_lock_info"] = auto_lock_info
                    
                await asyncio.sleep(0.01)
                yield output_dict

        except Exception as e:
            print(f"Error in tracking loop: {e}")
            await asyncio.sleep(0.1)

    cap.release()
    cv2.destroyAllWindows()
