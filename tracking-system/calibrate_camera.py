#!/usr/bin/env python3
"""
calibrate_camera.py — Camera calibration tool for the tracking system.

Supports BOTH ChArUco boards (recommended) and plain checkerboards.
Computes camera matrix and distortion coefficients, saves to
camera_calibration.json which tracker_config.py loads automatically.

Usage (ChArUco board — recommended):
    python3 calibrate_camera.py --charuco                          # 8x11 board, 25mm squares
    python3 calibrate_camera.py --charuco --resolution 4k          # 4K mode
    python3 calibrate_camera.py --charuco --square-size 25 --marker-size 18

Usage (plain checkerboard):
    python3 calibrate_camera.py                                    # 9x6 checkerboard
    python3 calibrate_camera.py --resolution 4k
    python3 calibrate_camera.py --load-images ./imgs/
"""

import cv2
import numpy as np
import json
import os
import argparse
import glob
import time
import platform

try:
    import tracker_config as cfg
except ImportError:
    cfg = None


# ──────────────────────────────────────────────────────────────────────────────
# ChArUco board calibration (recommended — works with partial visibility)
# ──────────────────────────────────────────────────────────────────────────────

def create_charuco_board(cols=8, rows=11, square_size_mm=25.0, marker_size_mm=18.0,
                         dictionary_name=cv2.aruco.DICT_4X4_250):
    """Create a ChArUco board matching the user's printed board."""
    dictionary = cv2.aruco.getPredefinedDictionary(dictionary_name)
    board = cv2.aruco.CharucoBoard(
        (cols, rows),
        square_size_mm / 1000.0,  # OpenCV expects meters
        marker_size_mm / 1000.0,
        dictionary
    )
    return board, dictionary


def detect_aruco_dictionary(frame):
    """Auto-detect which ArUco dictionary a board uses by testing all common ones."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    candidates = [
        (cv2.aruco.DICT_4X4_50,   "DICT_4X4_50"),
        (cv2.aruco.DICT_4X4_100,  "DICT_4X4_100"),
        (cv2.aruco.DICT_4X4_250,  "DICT_4X4_250"),
        (cv2.aruco.DICT_4X4_1000, "DICT_4X4_1000"),
        (cv2.aruco.DICT_5X5_50,   "DICT_5X5_50"),
        (cv2.aruco.DICT_5X5_100,  "DICT_5X5_100"),
        (cv2.aruco.DICT_5X5_250,  "DICT_5X5_250"),
        (cv2.aruco.DICT_5X5_1000, "DICT_5X5_1000"),
        (cv2.aruco.DICT_6X6_50,   "DICT_6X6_50"),
        (cv2.aruco.DICT_6X6_250,  "DICT_6X6_250"),
        (cv2.aruco.DICT_7X7_50,   "DICT_7X7_50"),
        (cv2.aruco.DICT_7X7_250,  "DICT_7X7_250"),
    ]
    best_count = 0
    best_dict = cv2.aruco.DICT_4X4_250
    best_name = "DICT_4X4_250"
    for dict_id, name in candidates:
        dictionary = cv2.aruco.getPredefinedDictionary(dict_id)
        detector = cv2.aruco.ArucoDetector(dictionary)
        corners, ids, _ = detector.detectMarkers(gray)
        count = len(corners) if corners is not None else 0
        if count > best_count:
            best_count = count
            best_dict = dict_id
            best_name = name
    if best_count > 0:
        print(f"🔎 Auto-detected dictionary: {best_name} ({best_count} markers found)")
    else:
        print(f"⚠️  No markers detected with any dictionary — defaulting to DICT_4X4_250")
    return best_dict


def calibrate_charuco_from_frames(frames, board, dictionary):
    """
    Run ChArUco calibration on a list of frames.
    Returns (camera_matrix, dist_coeffs, reprojection_error) or (None, None, None).
    """
    detector_params = cv2.aruco.DetectorParameters()
    charuco_detector = cv2.aruco.CharucoDetector(board)

    all_charuco_corners = []
    all_charuco_ids = []

    print(f"\n🔍 Processing {len(frames)} frames for ChArUco board...")

    for i, frame in enumerate(frames):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        charuco_corners, charuco_ids, marker_corners, marker_ids = charuco_detector.detectBoard(gray)

        if charuco_corners is not None and charuco_ids is not None and len(charuco_ids) >= 4:
            all_charuco_corners.append(charuco_corners)
            all_charuco_ids.append(charuco_ids)
            print(f"  ✅ Frame {i+1}: {len(charuco_ids)} charuco corners found")
        else:
            n = len(charuco_ids) if charuco_ids is not None else 0
            print(f"  ❌ Frame {i+1}: only {n} corners (need 4+)")

    if len(all_charuco_corners) < 5:
        print(f"\n⚠️ Only {len(all_charuco_corners)} valid frames — need at least 5")
        if len(all_charuco_corners) < 3:
            print("❌ Not enough data for calibration")
            return None, None, None

    print(f"\n📊 Calibrating with {len(all_charuco_corners)} valid frames...")
    h, w = frames[0].shape[:2]

    # OpenCV 4.7+ removed cv2.aruco.calibrateCameraCharuco — use new API
    try:
        all_obj = []
        all_img = []
        for corners, ids in zip(all_charuco_corners, all_charuco_ids):
            op, ip = board.matchImagePoints(corners, ids)
            if op is not None and len(op) >= 4:
                all_obj.append(op)
                all_img.append(ip)

        ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
            all_obj, all_img, (w, h), None, None
        )
    except AttributeError:
        # Fallback for OpenCV < 4.7
        ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.aruco.calibrateCameraCharuco(
            all_charuco_corners, all_charuco_ids, board, (w, h), None, None
        )

    print(f"✅ ChArUco calibration complete!")
    print(f"   Reprojection error: {ret:.4f} px (lower = better, aim for < 0.5)")
    print(f"\n   Camera matrix:\n{camera_matrix}")
    print(f"\n   Distortion coefficients:\n{dist_coeffs.flatten()}")

    return camera_matrix, dist_coeffs, ret


# ──────────────────────────────────────────────────────────────────────────────
# Plain checkerboard calibration (fallback)
# ──────────────────────────────────────────────────────────────────────────────

def calibrate_checkerboard_from_frames(frames, board_size, square_size_mm=25.0):
    """Run plain checkerboard calibration on a list of frames."""
    objp = np.zeros((board_size[0] * board_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:board_size[0], 0:board_size[1]].T.reshape(-1, 2)
    objp *= square_size_mm

    obj_points = []
    img_points = []
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    print(f"\n🔍 Processing {len(frames)} frames for checkerboard ({board_size[0]}×{board_size[1]})...")

    for i, frame in enumerate(frames):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        found, corners = cv2.findChessboardCorners(
            gray, board_size,
            cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE
        )
        if found:
            corners_refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            obj_points.append(objp)
            img_points.append(corners_refined)
            print(f"  ✅ Frame {i+1}: checkerboard found")
        else:
            print(f"  ❌ Frame {i+1}: no checkerboard detected")

    if len(obj_points) < 3:
        print("❌ Not enough data for calibration")
        return None, None, None

    print(f"\n📊 Calibrating with {len(obj_points)} valid frames...")
    h, w = frames[0].shape[:2]
    ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        obj_points, img_points, (w, h), None, None
    )

    print(f"✅ Checkerboard calibration complete!")
    print(f"   Reprojection error: {ret:.4f} px")
    print(f"\n   Camera matrix:\n{camera_matrix}")
    print(f"\n   Distortion coefficients:\n{dist_coeffs.flatten()}")

    return camera_matrix, dist_coeffs, ret


# ──────────────────────────────────────────────────────────────────────────────
# Shared utilities
# ──────────────────────────────────────────────────────────────────────────────

def save_calibration(camera_matrix, dist_coeffs, resolution, error, filename="camera_calibration.json"):
    """Save calibration data to JSON."""
    data = {}
    if os.path.exists(filename):
        try:
            with open(filename, "r") as f:
                data = json.load(f)
        except Exception:
            pass

    data[resolution] = {
        "camera_matrix": camera_matrix.tolist(),
        "distortion_coefficients": dist_coeffs.flatten().tolist(),
        "reprojection_error": float(error),
    }

    with open(filename, "w") as f:
        json.dump(data, f, indent=4)

    print(f"\n💾 Saved to {filename} (resolution: {resolution})")
    print(f"   Set UNDISTORT_ENABLED = True in tracker_config.py to activate")


def interactive_capture(resolution="1080p", use_charuco=False, board_size=(9, 6),
                        square_size_mm=25.0, marker_size_mm=11.0, num_captures=20):
    """
    Interactive mode: show live camera feed and capture frames on spacebar press.
    Works with both ChArUco and plain checkerboard.
    """
    res_configs = {"4k": (3840, 2160), "1080p": (1920, 1080)}
    w, h = res_configs.get(resolution, (1920, 1080))

    if platform.system() == "Darwin" and hasattr(cv2, "CAP_AVFOUNDATION"):
        cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)
    else:
        cap = cv2.VideoCapture(0)

    # Keep only the newest frame to reduce visible lag on live preview.
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"📹 Camera: {actual_w}×{actual_h}")

    # Set up board detection
    charuco_detector = None
    board = None
    dictionary = None
    if use_charuco:
        # Auto-detect which ArUco dictionary the printed board uses
        print("🔎 Auto-detecting ArUco dictionary from camera feed...")
        ret_test, frame_test = cap.read()
        if ret_test:
            detected_dict = detect_aruco_dictionary(frame_test)
        else:
            print("⚠️  Could not read frame for auto-detection, using DICT_4X4_250")
            detected_dict = cv2.aruco.DICT_4X4_250
        board, dictionary = create_charuco_board(
            cols=board_size[0], rows=board_size[1],
            square_size_mm=square_size_mm, marker_size_mm=marker_size_mm,
            dictionary_name=detected_dict
        )
        charuco_detector = cv2.aruco.CharucoDetector(board)
        board_label = f"ChArUco {board_size[0]}×{board_size[1]}"
    else:
        board_label = f"Checkerboard {board_size[0]}×{board_size[1]}"

    frames = []
    print(f"\n🎯 Hold your {board_label} in view")
    print(f"   Press SPACE to capture ({num_captures} needed)")
    print(f"   Press 'c' to calibrate with current captures")
    print(f"   Press 'q' to quit\n")

    cv2.namedWindow("Camera Calibration", cv2.WINDOW_NORMAL)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("⚠️ Failed to grab frame")
            continue

        display = frame.copy()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        found = False

        if use_charuco:
            charuco_corners, charuco_ids, marker_corners, marker_ids = charuco_detector.detectBoard(gray)
            if charuco_corners is not None and charuco_ids is not None and len(charuco_ids) >= 4:
                cv2.aruco.drawDetectedCornersCharuco(display, charuco_corners, charuco_ids)
                if marker_corners is not None:
                    cv2.aruco.drawDetectedMarkers(display, marker_corners, marker_ids)
                found = True
                status_text = f"CHARUCO FOUND ({len(charuco_ids)} corners) — press SPACE"
            else:
                n = len(charuco_ids) if charuco_ids is not None else 0
                status_text = f"Looking for ChArUco... ({n} corners, need 4+)"
        else:
            found_board, corners = cv2.findChessboardCorners(
                gray, board_size,
                cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK
            )
            if found_board:
                cv2.drawChessboardCorners(display, board_size, corners, found_board)
                found = True
                status_text = "CHECKERBOARD FOUND — press SPACE"
            else:
                status_text = "No checkerboard — adjust position"

        status_color = (0, 255, 0) if found else (0, 0, 255)

        # HUD
        cv2.putText(display, status_text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
        cv2.putText(display, f"Captures: {len(frames)}/{num_captures}", (10, 65),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        # Resize for display if too big
        if display.shape[1] > 1280:
            scale = 1280 / display.shape[1]
            display = cv2.resize(display, (0, 0), fx=scale, fy=scale)

        cv2.imshow("Camera Calibration", display)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord(' ') and found:
            frames.append(frame.copy())
            print(f"📸 Captured frame {len(frames)}/{num_captures}")
            if len(frames) >= num_captures:
                print(f"\n✅ All {num_captures} frames captured!")
                break
        elif key == ord('c') and len(frames) >= 3:
            print(f"\n🔧 Calibrating with {len(frames)} frames...")
            break

    cv2.destroyAllWindows()
    cap.release()

    if len(frames) < 3:
        print("❌ Not enough captures for calibration")
        return

    # Run calibration
    if use_charuco:
        mtx, dist, error = calibrate_charuco_from_frames(frames, board, dictionary)
    else:
        mtx, dist, error = calibrate_checkerboard_from_frames(frames, board_size, square_size_mm)

    if mtx is not None:
        save_calibration(mtx, dist, resolution, error)


def calibrate_from_directory(image_dir, resolution="1080p", use_charuco=False,
                             board_size=(9, 6), square_size_mm=25.0, marker_size_mm=11.0):
    """Load images from a directory and calibrate."""
    patterns = ["*.jpg", "*.jpeg", "*.png", "*.bmp"]
    image_files = []
    for p in patterns:
        image_files.extend(glob.glob(os.path.join(image_dir, p)))

    if not image_files:
        print(f"❌ No images found in {image_dir}")
        return

    image_files.sort()
    print(f"📂 Found {len(image_files)} images in {image_dir}")

    frames = [cv2.imread(path) for path in image_files if cv2.imread(path) is not None]
    if not frames:
        print("❌ Could not load any images")
        return

    if use_charuco:
        board, dictionary = create_charuco_board(
            cols=board_size[0], rows=board_size[1],
            square_size_mm=square_size_mm, marker_size_mm=marker_size_mm
        )
        mtx, dist, error = calibrate_charuco_from_frames(frames, board, dictionary)
    else:
        mtx, dist, error = calibrate_checkerboard_from_frames(frames, board_size, square_size_mm)

    if mtx is not None:
        save_calibration(mtx, dist, resolution, error)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Camera calibration tool for ArUco/AprilTag tracking",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # ChArUco board (calib.io 8x11, 15mm squares, 11mm markers):
  python3 calibrate_camera.py --charuco --square-size 15 --marker-size 11

  # Plain checkerboard (9x6 inner corners):
  python3 calibrate_camera.py --rows 6 --cols 9 --square-size 25

  # From saved images:
  python3 calibrate_camera.py --charuco --load-images ./calibration_imgs/
        """
    )
    parser.add_argument("--charuco", action="store_true",
                        help="Use ChArUco board (recommended). Default board: 8x11, 15mm, DICT_4X4")
    parser.add_argument("--resolution", choices=["4k", "1080p"], default="1080p",
                        help="Camera resolution (default: 1080p)")
    parser.add_argument("--rows", type=int, default=None,
                        help="Board rows (default: 11 for charuco, 6 for checkerboard)")
    parser.add_argument("--cols", type=int, default=None,
                        help="Board columns (default: 8 for charuco, 9 for checkerboard)")
    parser.add_argument("--square-size", type=float, default=None,
                        help="Square size in mm (default: 25 for charuco, 25 for checkerboard)")
    parser.add_argument("--marker-size", type=float, default=18.0,
                        help="ArUco marker size in mm, charuco only (default: 18)")
    parser.add_argument("--captures", type=int, default=20,
                        help="Number of frames to capture (default: 20)")
    parser.add_argument("--load-images", type=str, default=None,
                        help="Load images from directory instead of live capture")
    args = parser.parse_args()

    # Set defaults based on board type
    if args.charuco:
        cols = args.cols or 8
        rows = args.rows or 11
        square_size = args.square_size or 25.0
    else:
        cols = args.cols or 9
        rows = args.rows or 6
        square_size = args.square_size or 25.0

    board_size = (cols, rows)

    if args.load_images:
        calibrate_from_directory(args.load_images, args.resolution, args.charuco,
                                 board_size, square_size, args.marker_size)
    else:
        interactive_capture(args.resolution, args.charuco, board_size,
                            square_size, args.marker_size, args.captures)
