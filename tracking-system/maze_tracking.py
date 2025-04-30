import cv2
import numpy as np
import time

# Load the ArUco dictionary
dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_250)

# Set up detector parameters
detector_params = cv2.aruco.DetectorParameters()

# Initialize detector
detector = cv2.aruco.ArucoDetector(dictionary, detector_params)

# Open webcam
cap = cv2.VideoCapture(0)

prev_time = time.time()
scale_factor = 0.5  # Adjust as needed

# Define known positions of corner tags in feet
corner_tag_positions = {
    0: np.array([0, 0]),
    1: np.array([0, 5]),
    2: np.array([5, 5]),
    3: np.array([5, 0])
}

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    curr_time = time.time()
    fps = 1.0 / (curr_time - prev_time)
    prev_time = curr_time

    # Downsample frame
    small_frame = cv2.resize(frame, (0, 0), fx=scale_factor, fy=scale_factor)
    gray = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)

    # Detect markers
    corners, ids, rejected = detector.detectMarkers(gray)

    detected_corner_positions = {}
    pixel_positions = {}
    estimated_robot_positions = {}

    if ids is not None:
        # Scale corners back to original frame size
        corners = [corner / scale_factor for corner in corners]
        cv2.aruco.drawDetectedMarkers(frame, corners, ids)

        for i, corner in enumerate(corners):
            marker_id = ids[i][0]
            center = np.mean(corner[0], axis=0)

            if marker_id in corner_tag_positions:
                detected_corner_positions[marker_id] = corner_tag_positions[marker_id] * 30.48  # Convert feet to cm
                pixel_positions[marker_id] = center

            if marker_id > 3:
                pixel_positions[marker_id] = center

            # Draw center and ID
            center_int = tuple(center.astype(int))
            cv2.circle(frame, center_int, 5, (0, 255, 0), -1)
            cv2.putText(frame, f"ID: {marker_id}", (center_int[0] + 10, center_int[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        if len(detected_corner_positions) >= 3:
            # Prepare the source and destination points for transformation
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

            output_dict = {
                'robot_tags': {tag_id: estimated_robot_positions[tag_id].tolist() for tag_id in estimated_robot_positions},
                'corner_tags': {tag_id: detected_corner_positions[tag_id].tolist() for tag_id in detected_corner_positions}
            }

            print(output_dict)

    cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    cv2.imshow("ArUco Tracker", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()