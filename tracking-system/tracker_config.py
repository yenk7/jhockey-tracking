"""
tracker_config.py — Centralized configuration for the ArUco/AprilTag tracking system.

Edit the variables below to tune detection, streaming, and field parameters.
Both aruco_tracker_2.py and aruco_tracker_calibration.py import from here.
"""

import cv2
import numpy as np

# ──────────────────────────────────────────────────────────────────────────────
# TAG SYSTEM  –  switch between "aruco" and "apriltag"
# ──────────────────────────────────────────────────────────────────────────────
TAG_SYSTEM = "apriltag"   # "aruco" or "apriltag"

# Available dictionaries for each tag system
TAG_DICTIONARIES = {
    "aruco": {
        "4x4_50":   cv2.aruco.DICT_4X4_50,
        "4x4_100":  cv2.aruco.DICT_4X4_100,
        "4x4_250":  cv2.aruco.DICT_4X4_250,
        "4x4_1000": cv2.aruco.DICT_4X4_1000,
        "5x5_50":   cv2.aruco.DICT_5X5_50,
        "5x5_100":  cv2.aruco.DICT_5X5_100,
        "5x5_250":  cv2.aruco.DICT_5X5_250,
        "6x6_50":   cv2.aruco.DICT_6X6_50,
        "6x6_250":  cv2.aruco.DICT_6X6_250,
    },
    "apriltag": {
        "16h5":  cv2.aruco.DICT_APRILTAG_16h5,
        "25h9":  cv2.aruco.DICT_APRILTAG_25h9,
        "36h10": cv2.aruco.DICT_APRILTAG_36h10,
        "36h11": cv2.aruco.DICT_APRILTAG_36h11,
    },
}

# Which specific dictionary to use within the chosen tag system
#TAG_DICTIONARY_NAME = "4x4_250"      # for aruco
TAG_DICTIONARY_NAME = "25h9"      # uncomment for apriltag


def get_dictionary():
    """Return the cv2.aruco dictionary object for the current TAG_SYSTEM."""
    family = TAG_DICTIONARIES.get(TAG_SYSTEM)
    if family is None:
        raise ValueError(f"Unknown TAG_SYSTEM '{TAG_SYSTEM}'. Use 'aruco' or 'apriltag'.")
    dict_id = family.get(TAG_DICTIONARY_NAME)
    if dict_id is None:
        raise ValueError(
            f"Unknown dictionary '{TAG_DICTIONARY_NAME}' for {TAG_SYSTEM}. "
            f"Available: {list(family.keys())}"
        )
    return cv2.aruco.getPredefinedDictionary(dict_id)


# ──────────────────────────────────────────────────────────────────────────────
# RESOLUTION & CAMERA
# ──────────────────────────────────────────────────────────────────────────────
RESOLUTION_CONFIGS = {
    "4k": {
        "width": 3840,
        "height": 2160,
        "crop": {"x": 500, "y": 400, "w": 3840, "h": 2160},
        "stream_resolution": (1280, 720),
        "display_name": "4K (3840x2160)",
        "detection_scale": 0.5,    # 4K → 1920×1080 for detection (markers are big enough)
    },
    "1080p": {
        "width": 1920,
        "height": 1080,
        "crop": {"x": 10, "y": 10, "w": 1920, "h": 1080},
        "stream_resolution": (960, 540),
        "display_name": "Full HD (1920x1080)",
        "detection_scale": 0.7,    # 1080p → 1344×756 for detection
    },
}

# Set to True if the camera is mounted upside-down (rotates frame 180°)
CAMERA_ROTATE_180 = True

# Use a background thread for camera capture (hides USB I/O latency)
THREADED_CAPTURE = True


# ──────────────────────────────────────────────────────────────────────────────
# FIELD / CORNER TAG POSITIONS  (in feet — converted to cm at runtime × 30.48)
# ──────────────────────────────────────────────────────────────────────────────
CORNER_TAG_POSITIONS = {
    0: np.array([0,    0]),
    1: np.array([0,    7.75]),
    2: np.array([3.74, 7.75]),
    3: np.array([3.74, 0]),
}

FEET_TO_CM = 30.48

# Maximum valid tag ID (IDs above this are discarded as noise)
MAX_VALID_TAG_ID = 29

# Corner tag IDs are 0–3; robot tags are > 3
CORNER_TAG_MAX_ID = 3


# ──────────────────────────────────────────────────────────────────────────────
# DETECTION PARAMETERS  (defaults loaded when no calibration JSON exists)
# ──────────────────────────────────────────────────────────────────────────────
DEFAULT_DETECTION_PARAMS = {
    "adaptiveThreshWinSizeMin": 3,
    "adaptiveThreshWinSizeMax": 30,
    "adaptiveThreshWinSizeStep": 10,
    "adaptiveThreshConstant": 7,
    "minMarkerPerimeterRate": 0.03,
    "maxMarkerPerimeterRate": 4.0,
    "polygonalApproxAccuracyRate": 0.02,
    "minCornerDistanceRate": 0.05,
    "cornerRefinementMethod": cv2.aruco.CORNER_REFINE_CONTOUR,
    "errorCorrectionRate": 0.6,
}


# ──────────────────────────────────────────────────────────────────────────────
# DYNAMIC-FRIENDLY ADJUSTMENTS  (applied on top of calibrated params at runtime)
# ──────────────────────────────────────────────────────────────────────────────
DYNAMIC_ADJUSTMENTS = {
    "adaptiveThreshWinSizeMin_offset": -2,       # subtract from calibrated value (floor 3)
    "adaptiveThreshWinSizeMax_offset": +10,      # add to calibrated value (cap 50)
    "adaptiveThreshConstant_offset": -2,         # subtract (floor 3)
    "minMarkerPerimeterRate_scale": 0.7,         # multiply calibrated value
    "maxMarkerPerimeterRate_scale": 1.3,         # multiply calibrated value
    "errorCorrectionRate_offset": +0.1,          # add (cap 1.0)
    "polygonalApproxAccuracyRate_offset": +0.02, # add (cap 0.1)
    "minCornerDistanceRate_offset": -0.02,       # subtract (floor 0.01)
}


# ──────────────────────────────────────────────────────────────────────────────
# RECOVERY MODE  (permissive params when markers are lost)
# ──────────────────────────────────────────────────────────────────────────────
RECOVERY_TIMEOUT_S = 1.0          # seconds without a known marker before entering recovery
RECOVERY_ENTER_CONSECUTIVE_MISSES = 3
RECOVERY_EXIT_CONSECUTIVE_HITS = 2
RECOVERY_PERIMETER_MIN_SCALE = 0.4
RECOVERY_PERIMETER_MAX_SCALE = 1.7
RECOVERY_APPROX_SCALE = 2.5
RECOVERY_CORNER_DIST_SCALE = 0.4
RECOVERY_ERROR_CORRECTION_MIN = 0.7


# ──────────────────────────────────────────────────────────────────────────────
# MARKER FILTERING & MOVEMENT
# ──────────────────────────────────────────────────────────────────────────────
FILTER_HISTORY_FRAMES = 6         # rolling window for temporal consistency
FILTER_MIN_DETECTIONS_ROBOT = 2   # min detections to accept a robot tag (lowered: 4 was too slow after movement)
FILTER_MIN_DETECTIONS_CORNER = 3  # min detections to accept a corner tag
FILTER_DETECTION_RATE_ROBOT = 0.3 # lowered from 0.5 so re-acquisition after movement is fast
FILTER_DETECTION_RATE_CORNER = 0.4
FILTER_MAX_JITTER_4K = 60        # px
FILTER_MAX_JITTER_1080P = 30     # px

# Geometry validation
GEOMETRY_ASPECT_RATIO_MAX = 2.5       # normal mode
GEOMETRY_ASPECT_RATIO_MAX_RECOVERY = 3.0
GEOMETRY_MIN_SIDE_PX = 15             # normal
GEOMETRY_MIN_SIDE_PX_RECOVERY = 10    # recovery
GEOMETRY_MAX_SIDE_PX = 600
GEOMETRY_AREA_RATIO_RANGE = (0.5, 2.0)
GEOMETRY_AREA_RATIO_RANGE_RECOVERY = (0.4, 2.2)


# ──────────────────────────────────────────────────────────────────────────────
# STREAMING
# ──────────────────────────────────────────────────────────────────────────────
STREAM_FPS_LIMIT = 30
STREAM_JPEG_QUALITY_4K = 70
STREAM_JPEG_QUALITY_1080P = 75
TARGET_DETECTION_FPS = 30


# ──────────────────────────────────────────────────────────────────────────────
# AUTO-CALIBRATION SEARCH RANGES  (used by aruco_tracker_calibration.py)
# ──────────────────────────────────────────────────────────────────────────────
AUTO_CALIBRATION_PARAM_RANGES = {
    "adaptiveThreshWinSizeMin": [3, 5, 7],
    "adaptiveThreshWinSizeMax": [23, 30, 45],
    "adaptiveThreshWinSizeStep": [4, 10, 16],
    "adaptiveThreshConstant": [5, 7, 9, 11],
    "minMarkerPerimeterRate": [0.01, 0.02, 0.03, 0.04],
    "maxMarkerPerimeterRate": [3.0, 4.0, 5.0],
    "polygonalApproxAccuracyRate": [0.01, 0.02, 0.03, 0.05],
    "cornerRefinementMethod": [0, 1, 2],
}

# How many expected markers the calibration should look for
CALIBRATION_EXPECTED_MARKERS = 4

# Train/validation split ratio for calibration
CALIBRATION_VALIDATION_RATIO = 0.3

# If winner's validation score drops more than this below train score, reject it
CALIBRATION_OVERFIT_TOLERANCE = 15.0   # percentage points


# ──────────────────────────────────────────────────────────────────────────────
# CAMERA DISTORTION CORRECTION
# ──────────────────────────────────────────────────────────────────────────────
# Set to True to apply lens distortion correction (requires calibration data).
# When False, no undistortion is applied — useful if you haven't calibrated yet.
UNDISTORT_ENABLED = False

# Path to the JSON file produced by calibrate_camera.py
# If this file exists and UNDISTORT_ENABLED is True, the intrinsics are loaded
# automatically at startup.
CAMERA_CALIBRATION_FILE = "camera_calibration.json"

# Fallback intrinsics used ONLY if no calibration file exists.
# These are approximate values for the Logitech Brio 4K at 90° FOV.
# Run calibrate_camera.py with a checkerboard to get accurate values for YOUR camera.
CAMERA_INTRINSICS = {
    "4k": {
        # fx, fy ≈ width / (2 * tan(hfov/2))  — rough estimate for 90° diagonal FOV
        "camera_matrix": np.array([
            [1800.0,    0.0, 1920.0],
            [   0.0, 1800.0, 1080.0],
            [   0.0,    0.0,    1.0],
        ], dtype=np.float64),
        "distortion_coefficients": np.zeros(5, dtype=np.float64),
    },
    "1080p": {
        "camera_matrix": np.array([
            [ 900.0,   0.0,  960.0],
            [   0.0, 900.0,  540.0],
            [   0.0,   0.0,    1.0],
        ], dtype=np.float64),
        "distortion_coefficients": np.zeros(5, dtype=np.float64),
    },
}


def load_camera_calibration(resolution: str = "4k"):
    """
    Load camera intrinsics from the calibration JSON file, falling back to
    the hardcoded estimates in CAMERA_INTRINSICS.

    Returns (camera_matrix, distortion_coefficients) as numpy arrays.
    """
    import json, os
    if os.path.exists(CAMERA_CALIBRATION_FILE):
        try:
            with open(CAMERA_CALIBRATION_FILE, "r") as f:
                data = json.load(f)
            res_data = data.get(resolution, data.get("4k", {}))
            mtx = np.array(res_data["camera_matrix"], dtype=np.float64)
            dist = np.array(res_data["distortion_coefficients"], dtype=np.float64)
            print(f"📷 Camera calibration loaded from {CAMERA_CALIBRATION_FILE} ({resolution})")
            return mtx, dist
        except Exception as e:
            print(f"⚠️ Failed to load {CAMERA_CALIBRATION_FILE}: {e}")

    # Fallback to built-in estimates
    fallback = CAMERA_INTRINSICS.get(resolution, CAMERA_INTRINSICS["4k"])
    print(f"📷 Using estimated camera intrinsics for {resolution} (run calibrate_camera.py for accuracy)")
    return fallback["camera_matrix"].copy(), fallback["distortion_coefficients"].copy()


# ──────────────────────────────────────────────────────────────────────────────
# LOGGING
# ──────────────────────────────────────────────────────────────────────────────
# "INFO"  = startup messages + errors + mode changes only  (recommended for production)
# "DEBUG" = all per-frame prints (robot positions, corner positions, filter status, etc.)
LOG_LEVEL = "INFO"

# Marker history cleanup — remove entries for markers not seen in this many seconds
MARKER_HISTORY_EXPIRY_S = 60.0

# Prune long-lived marker/robot state so stale IDs don't accumulate forever.
SEEN_MARKER_PRUNE_TIMEOUT_S = 30.0
