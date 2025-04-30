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

    if ids is not None:
        # Scale corners back to original frame size
        corners = [corner / scale_factor for corner in corners]
        cv2.aruco.drawDetectedMarkers(frame, corners, ids)

        for i, corner in enumerate(corners):
            center = np.mean(corner[0], axis=0).astype(int)
            cv2.circle(frame, tuple(center), 5, (0, 255, 0), -1)
            cv2.putText(frame, f"ID: {ids[i][0]}", (center[0] + 10, center[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0, 255, 0), 2)

    # Display FPS
    cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    cv2.imshow("ArUco Tracker", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
