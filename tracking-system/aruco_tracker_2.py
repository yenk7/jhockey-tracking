import cv2
import numpy as np
import time
import asyncio
import base64
import os
import json

# Store both corner positions and their pixel locations when locked
locked_corners = None  # World coordinates of corners
locked_pixel_positions = None  # Pixel positions of corners when locked
lock_state = False  # Indicates whether corners are locked

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

def crop_frame(frame, x=300, y=110, w=1350, h=690):
    """
    Crop the given frame to the specified region.
    :param frame: The input frame to crop.
    :param x: The x-coordinate of the top-left corner of the crop region.
    :param y: The y-coordinate of the top-left corner of the crop region.
    :param w: The width of the crop region.
    :param h: The height of the crop region.
    :return: The cropped frame.
    """
    return frame[y:y+h, x:x+w]

async def track_aruco_tags(lock_queue, scale_factor=0.7):
    
    global locked_corners, locked_pixel_positions, lock_state
    print(f"🟢 Initial lock state: {lock_state}", flush=True)

    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_250)
    detector_params = cv2.aruco.DetectorParameters()
    
    # Try to load calibrated parameters if they exist
    calibration = load_calibration_parameters()
    if calibration:
        # Apply loaded parameters
        for param_name, param_value in calibration.items():
            if hasattr(detector_params, param_name):
                setattr(detector_params, param_name, param_value)
                print(f"  - Applied {param_name}: {param_value}")
    else:
        # Use default parameters
        detector_params.adaptiveThreshWinSizeMin = 3
        detector_params.adaptiveThreshWinSizeMax = 30
        detector_params.adaptiveThreshWinSizeStep = 10
        detector_params.minMarkerPerimeterRate = 0.03
        detector_params.maxMarkerPerimeterRate = 4.0
        detector_params.polygonalApproxAccuracyRate = 0.02
        detector_params.minCornerDistanceRate = 0.05
        detector_params.minMarkerDistanceRate = 0.05
        detector_params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_CONTOUR

    # Set required parameters that are causing the error
    detector_params.markerBorderBits = 1  # Default value, must be greater than 0
    detector_params.minSideLengthCanonicalImg = 16  # Determines the cell size
    detector_params.perspectiveRemoveIgnoredMarginPerCell = 0.13  # Between 0 and 1
    
    detector = cv2.aruco.ArucoDetector(dictionary, detector_params)
    cap = cv2.VideoCapture(0)
    # Set camera resolution to 1920x1080
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    
    # Verify the resolution was set correctly
    actual_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    actual_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    print(f"📹 Camera resolution set to: {actual_width}x{actual_height}")
    prev_time = time.time()

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

        # Apply cropping
        frame = crop_frame(frame)

        frame = cv2.rotate(frame, cv2.ROTATE_180)

        curr_time = time.time()
        fps = 1.0 / (curr_time - prev_time)
        prev_time = curr_time

        small_frame = cv2.resize(frame, (0, 0), fx=scale_factor, fy=scale_factor)
        gray = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)

        # Debug: Save or display the grayscale image
        cv2.imwrite("debug_gray_image.jpg", gray)

        # Ensure the grayscale image has enough contours
        if gray is None or gray.size == 0:
            print("⚠️ Grayscale image is empty or invalid")
            await asyncio.sleep(0.1)
            continue

        # Adjust detector parameters if needed
        detector_params.adaptiveThreshWinSizeMin = max(3, detector_params.adaptiveThreshWinSizeMin)
        detector_params.adaptiveThreshWinSizeMax = max(23, detector_params.adaptiveThreshWinSizeMax)
        detector_params.minMarkerPerimeterRate = max(0.03, detector_params.minMarkerPerimeterRate)
        detector_params.maxMarkerPerimeterRate = min(4.0, detector_params.maxMarkerPerimeterRate)

        try:
            corners, ids, _ = detector.detectMarkers(gray)
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

        # Encode frame as JPEG and convert to base64
        _, buffer = cv2.imencode(".jpg", frame)
        base64_frame = base64.b64encode(buffer).decode("utf-8")

        # Add additional information about lock state
        output_dict = {
            'robot_tags': {int(tag_id): estimated_robot_positions[tag_id].tolist() for tag_id in estimated_robot_positions},
            'corner_tags': {int(tag_id): detected_corner_positions[tag_id].tolist() for tag_id in detected_corner_positions},
            'fps': round(fps, 2),
            'frame': base64_frame,
            'lock_state': lock_state,
            'available_corners': available_corners if 'available_corners' in locals() else 0
        }

        await asyncio.sleep(0.01)  # Helps prevent locking up the event loop
        yield output_dict

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
