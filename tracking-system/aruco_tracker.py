import cv2
import numpy as np
import time
import asyncio
import base64

locked_corners = None  # Stores the locked corner positions
lock_state = False  # Indicates whether corners are locked

async def track_aruco_tags(lock_queue, scale_factor=1):
    
    global locked_corners, lock_state
    print(f"🟢 Initial lock state: {lock_state}", flush=True)

    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_250)
    detector_params = cv2.aruco.DetectorParameters()
    
    detector_params.adaptiveThreshWinSizeMin = 3
    detector_params.adaptiveThreshWinSizeMax = 23
    detector_params.adaptiveThreshWinSizeStep = 10
    detector_params.minMarkerPerimeterRate = 0.03
    detector_params.maxMarkerPerimeterRate = 4.0
    detector_params.polygonalApproxAccuracyRate = 0.02
    detector_params.minCornerDistanceRate = 0.05
    detector_params.minMarkerDistanceRate = 0.05
    detector_params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_CONTOUR

    detector = cv2.aruco.ArucoDetector(dictionary, detector_params)
    cap = cv2.VideoCapture(0)
    prev_time = time.time()

    corner_tag_positions = {
        0: np.array([0, 0]),
        1: np.array([0, 4]),
        2: np.array([4, 4]),
        3: np.array([4, 0])
    }

    while True:
        try:
            lock_state = lock_queue.get_nowait()
        except asyncio.QueueEmpty:
            pass

        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        frame = cv2.rotate(frame, cv2.ROTATE_180)

        curr_time = time.time()
        fps = 1.0 / (curr_time - prev_time)
        prev_time = curr_time

        small_frame = cv2.resize(frame, (0, 0), fx=scale_factor, fy=scale_factor)
        gray = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)

        corners, ids, _ = detector.detectMarkers(gray)

        detected_corner_positions = {}  # For new corners
        pixel_positions = {}  # Stores pixel positions of all markers
        estimated_robot_positions = {}  # Stores estimated real-world positions

        if ids is not None:
            corners = [corner / scale_factor for corner in corners]
            cv2.aruco.drawDetectedMarkers(frame, corners, ids)

            for i, corner in enumerate(corners):
                marker_id = ids[i][0]

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

            # If locked, use locked corners instead of newly detected ones
            if lock_state and locked_corners:   
                pixel_positions_corners = {
                    tag_id: detected_pixel_positions[tag_id]
                    for tag_id in detected_pixel_positions if tag_id <= 3
                }

                pixel_positions_robots = {
                    tag_id: pixel_positions[tag_id]
                    for tag_id in pixel_positions if tag_id > 3
                }

                pixel_positions = {**pixel_positions_corners, **pixel_positions_robots}
                detected_corner_positions = locked_corners.copy()

            elif not lock_state:
                locked_corners = detected_corner_positions.copy() if detected_corner_positions else None
                detected_pixel_positions = pixel_positions.copy()
            
            if len(detected_corner_positions) >= 3:
                src_points = []
                dst_points = []

                for tag_id, world_pos in detected_corner_positions.items():
                    src_points.append(pixel_positions[tag_id])
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

            # Encode frame as JPEG and convert to base64
            _, buffer = cv2.imencode(".jpg", frame)
            base64_frame = base64.b64encode(buffer).decode("utf-8")

            output_dict = {
                'robot_tags': {int(tag_id): estimated_robot_positions[tag_id].tolist() for tag_id in estimated_robot_positions},
                'corner_tags': {int(tag_id): detected_corner_positions[tag_id].tolist() for tag_id in detected_corner_positions},
                'fps': round(fps, 2),
                "frame": base64_frame
            }

            # cv2.imshow("ArUco Tracker", frame)

            await asyncio.sleep(0.01)  # Helps prevent locking up the event loop
            yield output_dict

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
