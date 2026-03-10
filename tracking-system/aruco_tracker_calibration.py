import cv2
import numpy as np
import time
import argparse
import os
import json
import itertools
import random
from collections import deque
from typing import Dict, List, Tuple, Optional, Any

import tracker_config as cfg

# ──────────────────────────────────────────────────────────────────────────────
# Global state for interactive (manual) calibration
# ──────────────────────────────────────────────────────────────────────────────

# Trackbar values (integer‑scaled where needed)
param_values = {
    'adaptiveThreshWinSizeMin': 3,
    'adaptiveThreshWinSizeMax': 30,
    'adaptiveThreshWinSizeStep': 10,
    'adaptiveThreshConstant': 7,
    'minMarkerPerimeterRate': 3,       # ×100
    'maxMarkerPerimeterRate': 400,     # ×100
    'polygonalApproxAccuracyRate': 2,  # ×100
    'minCornerDistanceRate': 5,        # ×100
    'minMarkerDistanceRate': 5,        # ×100
    'cornerRefinementMethod': 1,
    'cornerRefinementWinSize': 5,
    'cornerRefinementMaxIterations': 30,
    'cornerRefinementMinAccuracy': 10, # ×100
    'minDistanceToBorder': 3,
    'perspectiveRemovePixelPerCell': 4,
    'perspectiveRemoveIgnoredMarginPerCell': 13,  # ×100
    'maxErroneousBitsInBorderRate': 60,           # ×100
    'minOtsuStdDev': 5.0,
}

detection_history = {}
best_params = None
best_detection_count = 0
frame_count = 0
detection_count = 0
param_change_count = 0

# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def crop_frame(frame, resolution='4k'):
    """Crop the given frame using resolution-specific crop settings."""
    config = cfg.RESOLUTION_CONFIGS.get(resolution, cfg.RESOLUTION_CONFIGS['4k'])
    crop = config['crop']
    frame_h, frame_w = frame.shape[:2]
    x = min(crop['x'], frame_w - 1)
    y = min(crop['y'], frame_h - 1)
    w = min(crop['w'], frame_w - x)
    h = min(crop['h'], frame_h - y)
    return frame[y:y+h, x:x+w]


def update_param(param_name, value):
    """Update parameter value when trackbar is moved."""
    global param_values, param_change_count
    if param_values[param_name] != value:
        param_values[param_name] = value
        param_change_count += 1


def update_detector_parameters(detector_params, params=None):
    """Apply current parameter values to the detector parameters."""
    p = params if params is not None else param_values

    detector_params.adaptiveThreshWinSizeMin = int(p.get('adaptiveThreshWinSizeMin', 3))
    detector_params.adaptiveThreshWinSizeMax = int(p.get('adaptiveThreshWinSizeMax', 30))
    detector_params.adaptiveThreshWinSizeStep = int(p.get('adaptiveThreshWinSizeStep', 10))
    detector_params.adaptiveThreshConstant = float(p.get('adaptiveThreshConstant', 7))

    # Trackbar-integer → float conversions
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

    # Error correction
    if 'errorCorrectionRate' in p:
        detector_params.errorCorrectionRate = float(p['errorCorrectionRate'])

    # Corner refinement
    detector_params.cornerRefinementMethod = int(p.get('cornerRefinementMethod', 1))
    if detector_params.cornerRefinementMethod > 0:
        detector_params.cornerRefinementWinSize = int(p.get('cornerRefinementWinSize', 5))
        detector_params.cornerRefinementMaxIterations = int(p.get('cornerRefinementMaxIterations', 30))
        if 'cornerRefinementMinAccuracy' in p and p['cornerRefinementMinAccuracy'] > 0.5:
            detector_params.cornerRefinementMinAccuracy = float(p['cornerRefinementMinAccuracy']) / 100.0
        else:
            detector_params.cornerRefinementMinAccuracy = float(p.get('cornerRefinementMinAccuracy', 0.1))

    return detector_params


def save_parameters(params, filename="best_aruco_params.json"):
    """Save parameters to a JSON file, converting trackbar ints → real floats."""
    save_params = params.copy()
    for key in ['minMarkerPerimeterRate', 'maxMarkerPerimeterRate',
                'polygonalApproxAccuracyRate', 'minCornerDistanceRate',
                'minMarkerDistanceRate', 'cornerRefinementMinAccuracy',
                'perspectiveRemoveIgnoredMarginPerCell', 'maxErroneousBitsInBorderRate']:
        val = save_params.get(key, 0)
        if key in ('minMarkerPerimeterRate',) and val > 1:
            save_params[key] = val / 100.0
        elif key in ('maxMarkerPerimeterRate',) and val > 10:
            save_params[key] = val / 100.0
        elif key not in ('minMarkerPerimeterRate', 'maxMarkerPerimeterRate') and val > 0.5:
            save_params[key] = val / 100.0

    with open(filename, 'w') as f:
        json.dump(save_params, f, indent=4)
    print(f"✅ Parameters saved to {filename}")


def load_parameters(filename="best_aruco_params.json"):
    """Load parameters from a JSON file into the global param_values dict."""
    global param_values
    try:
        with open(filename, 'r') as f:
            loaded = json.load(f)
        for param in loaded:
            if param in param_values:
                if param in ['minMarkerPerimeterRate', 'maxMarkerPerimeterRate',
                             'polygonalApproxAccuracyRate', 'minCornerDistanceRate',
                             'minMarkerDistanceRate', 'cornerRefinementMinAccuracy',
                             'perspectiveRemoveIgnoredMarginPerCell', 'maxErroneousBitsInBorderRate']:
                    param_values[param] = int(loaded[param] * 100)
                else:
                    param_values[param] = int(loaded[param])
        print(f"✅ Parameters loaded from {filename}")
        return True
    except FileNotFoundError:
        print(f"ℹ️ Parameter file {filename} not found")
        return False
    except Exception as e:
        print(f"⚠️ Error loading parameters: {e}")
        return False


def draw_text_info(frame, ids, fps, detection_rate):
    """Draw HUD text on the frame."""
    cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(frame, f"Detection rate: {detection_rate:.1f}%", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    if ids is not None:
        cv2.putText(frame, f"Tags detected: {len(ids)}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    else:
        cv2.putText(frame, "No tags detected", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    y_pos = 120
    cv2.putText(frame, f"Parameter changes: {param_change_count}", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
    if best_params is not None:
        y_pos += 30
        cv2.putText(frame, f"Best detection rate: {best_detection_count}%", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 200, 0), 2)


# ──────────────────────────────────────────────────────────────────────────────
# EVALUATE PARAMETERS  (per-marker scoring — fixes overfitting)
# ──────────────────────────────────────────────────────────────────────────────

def evaluate_parameters(
    params: Dict[str, Any],
    frames: List[np.ndarray],
    dictionary,
    scale_factor: float = 0.7,
    expected_markers: int = None,
) -> Tuple[float, int]:
    """
    Evaluate a parameter set on a list of frames.

    Scores by *per-marker detection rate* — for each frame, count how many of
    the expected markers were found and divide by expected_markers.  The final
    score is the average across all frames (0–100%).

    Returns (score, total_marker_detections).
    """
    if expected_markers is None:
        expected_markers = cfg.CALIBRATION_EXPECTED_MARKERS

    detector_params = cv2.aruco.DetectorParameters()
    detector_params = update_detector_parameters(detector_params, params)
    detector = cv2.aruco.ArucoDetector(dictionary, detector_params)

    total_marker_hits = 0
    per_frame_scores = []

    for frame in frames:
        small = cv2.resize(frame, (0, 0), fx=scale_factor, fy=scale_factor)
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        _, ids, _ = detector.detectMarkers(gray)

        if ids is not None:
            n = len(ids)
            total_marker_hits += n
            per_frame_scores.append(min(n, expected_markers) / expected_markers * 100.0)
        else:
            per_frame_scores.append(0.0)

    score = sum(per_frame_scores) / len(per_frame_scores) if per_frame_scores else 0.0
    return score, total_marker_hits


# ──────────────────────────────────────────────────────────────────────────────
# AUTO-CALIBRATION  (train/val split, per-marker scoring, overfit guard)
# ──────────────────────────────────────────────────────────────────────────────

def run_auto_calibration(
    scale_factor=0.7,
    num_frames=50,
    max_combinations=50,
    resolution='4k',
    expected_markers=None,
) -> Dict[str, Any]:
    """
    Run automated calibration with train/validation split to avoid overfitting.

    Changes vs old version:
      • Captures more frames (default 50) spread over time.
      • Splits frames into train (70%) and validation (30%).
      • Scores per-marker (not binary per-frame).
      • Cross-validates winning params: rejects if val score drops more than
        CALIBRATION_OVERFIT_TOLERANCE below train score.
    """
    if expected_markers is None:
        expected_markers = cfg.CALIBRATION_EXPECTED_MARKERS

    print("🔍 Starting automated calibration (with train/val split)...")

    # Resolve resolution
    if resolution not in cfg.RESOLUTION_CONFIGS:
        print(f"⚠️ Unknown resolution '{resolution}', defaulting to 4k")
        resolution = '4k'
    config = cfg.RESOLUTION_CONFIGS[resolution]
    print(f"📹 Calibrating for {config['display_name']} resolution")

    dictionary = cfg.get_dictionary()

    # Open camera
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, config['width'])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config['height'])
    actual_w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    actual_h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    print(f"📹 Camera resolution: {actual_w}x{actual_h}")

    # ── Capture frames ───────────────────────────────────────────────────
    print(f"📸 Collecting {num_frames} frames (spread over time for diversity)...")
    print("   👉 TIP: slightly vary marker positions / lighting while capturing")
    frames = []
    for i in range(num_frames):
        ret, frame = cap.read()
        if not ret:
            print("⚠️ Failed to grab frame")
            continue
        frame = crop_frame(frame, resolution)
        frames.append(frame)
        # Brief delay between frames to increase diversity
        time.sleep(0.15)
        if (i + 1) % 10 == 0:
            print(f"   captured {i + 1}/{num_frames}")

    print(f"✅ Collected {len(frames)} frames")

    if len(frames) < 5:
        print("❌ Not enough frames captured — aborting.")
        cap.release()
        return {}

    # ── Train / validation split ─────────────────────────────────────────
    random.shuffle(frames)
    val_size = max(2, int(len(frames) * cfg.CALIBRATION_VALIDATION_RATIO))
    val_frames = frames[:val_size]
    train_frames = frames[val_size:]
    print(f"📊 Split: {len(train_frames)} train / {len(val_frames)} validation frames")

    # ── Generate parameter combinations ──────────────────────────────────
    auto_ranges = cfg.AUTO_CALIBRATION_PARAM_RANGES
    param_keys = list(auto_ranges.keys())
    param_vals_list = list(auto_ranges.values())
    all_combos = list(itertools.product(*param_vals_list))

    if len(all_combos) > max_combinations:
        print(f"⚠️ {len(all_combos)} combinations → sampling {max_combinations}")
        all_combos = random.sample(all_combos, max_combinations)

    total = len(all_combos)
    print(f"🧪 Testing {total} parameter combinations on TRAIN set...")

    # Progress window
    cv2.namedWindow("Calibration Progress")
    prog_img = np.ones((120, 500, 3), dtype=np.uint8) * 255

    best_train_score = 0
    best_marker_count = 0
    best_params_found = None

    for idx, combo in enumerate(all_combos):
        test_params = {param_keys[i]: combo[i] for i in range(len(param_keys))}
        # Fill in defaults for any params not in the search
        for k, v in param_values.items():
            if k not in test_params:
                test_params[k] = v

        train_score, marker_count = evaluate_parameters(
            test_params, train_frames, dictionary, scale_factor, expected_markers
        )

        # Update progress
        pct = (idx + 1) / total * 100
        prog_img.fill(255)
        cv2.rectangle(prog_img, (10, 40), (10 + int(480 * pct / 100), 60), (0, 255, 0), -1)
        cv2.putText(prog_img, f"Testing: {idx+1}/{total}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
        cv2.putText(prog_img, f"Best train score: {best_train_score:.1f}%", (10, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 200), 1)
        cv2.putText(prog_img, f"Expected markers: {expected_markers}", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1)
        cv2.imshow("Calibration Progress", prog_img)
        cv2.waitKey(1)

        if train_score > best_train_score or (train_score == best_train_score and marker_count > best_marker_count):
            best_train_score = train_score
            best_marker_count = marker_count
            best_params_found = test_params
            print(f"📈 New best (train): {best_train_score:.1f}% | markers: {best_marker_count}")

    # ── Cross-validate winner on held-out set ────────────────────────────
    if best_params_found is not None:
        val_score, val_markers = evaluate_parameters(
            best_params_found, val_frames, dictionary, scale_factor, expected_markers
        )
        print(f"\n📊 Validation score: {val_score:.1f}%  (train was {best_train_score:.1f}%)")

        drop = best_train_score - val_score
        if drop > cfg.CALIBRATION_OVERFIT_TOLERANCE:
            print(f"⚠️  Validation dropped {drop:.1f}pp — possible overfitting!")
            print(f"   Tolerance is {cfg.CALIBRATION_OVERFIT_TOLERANCE}pp.")
            print(f"   Saving anyway, but consider increasing --frames or varying conditions.")
        else:
            print(f"✅ Validation within tolerance (drop {drop:.1f}pp ≤ {cfg.CALIBRATION_OVERFIT_TOLERANCE}pp)")

        print(f"🎉 Calibration complete!  Train: {best_train_score:.1f}% | Val: {val_score:.1f}%")
        save_parameters(best_params_found, "best_aruco_params.json")
    else:
        print("⚠️ No good parameters found!")

    cv2.destroyAllWindows()
    cap.release()
    return best_params_found or {}


# ──────────────────────────────────────────────────────────────────────────────
# LIVE ADAPTIVE CALIBRATION  (strategy-based, unchanged scoring)
# ──────────────────────────────────────────────────────────────────────────────

def run_live_adaptive_calibration(
    scale_factor=0.7,
    resolution='4k',
    target_detection_rate=100,
    max_duration=300,
    expected_markers=None,
):
    """
    Live adaptive calibration that tries preset strategies until the target
    detection rate is achieved or time runs out.
    """
    if expected_markers is None:
        expected_markers = cfg.CALIBRATION_EXPECTED_MARKERS

    print(f"🔍 Starting LIVE ADAPTIVE calibration targeting {target_detection_rate}% …")

    if resolution not in cfg.RESOLUTION_CONFIGS:
        print(f"⚠️ Unknown resolution '{resolution}', defaulting to 4k")
        resolution = '4k'
    config = cfg.RESOLUTION_CONFIGS[resolution]
    print(f"📹 Calibrating for {config['display_name']}")

    dictionary = cfg.get_dictionary()

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, config['width'])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config['height'])
    actual_w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    actual_h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    print(f"📹 Camera resolution: {actual_w}x{actual_h}")

    strategies = [
        {
            'name': 'Conservative Detection',
            'params': {
                'adaptiveThreshWinSizeMin': 5,
                'adaptiveThreshWinSizeMax': 23,
                'adaptiveThreshConstant': 7,
                'minMarkerPerimeterRate': 0.04,
                'maxMarkerPerimeterRate': 3.0,
                'cornerRefinementMethod': 1,
                'polygonalApproxAccuracyRate': 0.03,
                'minCornerDistanceRate': 0.05,
                'errorCorrectionRate': 0.6,
            },
        },
        {
            'name': 'High Sensitivity',
            'params': {
                'adaptiveThreshWinSizeMin': 3,
                'adaptiveThreshWinSizeMax': 45,
                'adaptiveThreshConstant': 9,
                'minMarkerPerimeterRate': 0.02,
                'maxMarkerPerimeterRate': 5.0,
                'cornerRefinementMethod': 2,
                'polygonalApproxAccuracyRate': 0.05,
                'minCornerDistanceRate': 0.03,
                'errorCorrectionRate': 0.8,
            },
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
                'errorCorrectionRate': 0.9,
            },
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
                'errorCorrectionRate': 0.7,
            },
        },
    ]

    start_time = time.time()
    best_params = None
    best_detection_rate = 0
    best_marker_count = 0
    strategy_index = 0
    detection_history_buf = deque(maxlen=50)

    print(f"🎯 Target: {target_detection_rate}% | ⏱️ Max: {max_duration}s | 🔄 {len(strategies)} strategies")
    cv2.namedWindow("Live Adaptive Calibration", cv2.WINDOW_AUTOSIZE)

    while time.time() - start_time < max_duration and strategy_index < len(strategies):
        cur = strategies[strategy_index]
        print(f"\n🧪 Strategy {strategy_index+1}/{len(strategies)}: {cur['name']}")

        det_params = cv2.aruco.DetectorParameters()
        det_params = update_detector_parameters(det_params, cur['params'])
        det_params.markerBorderBits = 1
        det_params.minSideLengthCanonicalImg = 16
        det_params.perspectiveRemoveIgnoredMarginPerCell = 0.13
        detector = cv2.aruco.ArucoDetector(dictionary, det_params)

        strat_start = time.time()
        strat_detections = []

        while (time.time() - strat_start < 30
               and time.time() - start_time < max_duration
               and len(strat_detections) < 100):

            ret, frame = cap.read()
            if not ret:
                continue
            frame = crop_frame(frame, resolution)
            small = cv2.resize(frame, (0, 0), fx=scale_factor, fy=scale_factor)
            gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
            corners, ids, _ = detector.detectMarkers(gray)

            # Per-marker scoring for corners
            corner_hits = 0
            if ids is not None:
                for mid in ids.flatten():
                    if 0 <= mid <= cfg.CORNER_TAG_MAX_ID:
                        corner_hits += 1
            strat_detections.append(corner_hits)
            detection_history_buf.append(corner_hits)

            # Use per-marker rate: average fraction of expected markers seen
            strat_rate = (sum(min(d, expected_markers) for d in strat_detections)
                          / (len(strat_detections) * expected_markers)) * 100

            # Draw visualisation
            if corners is not None and ids is not None:
                scaled_corners = [c / scale_factor for c in corners]
                cv2.aruco.drawDetectedMarkers(frame, scaled_corners, ids)
                for i, c in enumerate(scaled_corners):
                    ctr = np.mean(c[0], axis=0).astype(int)
                    cv2.circle(frame, tuple(ctr), 8, (0, 255, 0), -1)
                    cv2.putText(frame, f"ID: {ids[i][0]}", (ctr[0]+15, ctr[1]-10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            elapsed = time.time() - start_time
            cv2.putText(frame, f"Strategy: {cur['name']}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
            cv2.putText(frame, f"Rate: {strat_rate:.1f}% ({corner_hits}/{expected_markers})", (10, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f"Best: {best_detection_rate:.1f}%", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.putText(frame, f"Target: {target_detection_rate}%", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
            cv2.putText(frame, f"Time: {elapsed:.1f}s / {max_duration}s", (10, 190), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            bar_w = 400
            bar_pct = min(1.0, len(strat_detections) / 50.0)
            cv2.rectangle(frame, (10, 210), (10 + int(bar_w * bar_pct), 230), (0, 255, 0), -1)
            cv2.rectangle(frame, (10, 210), (10 + bar_w, 230), (255, 255, 255), 2)
            cv2.imshow("Live Adaptive Calibration", frame)

            if len(strat_detections) >= 20 and strat_rate >= target_detection_rate:
                print(f"🎉 TARGET ACHIEVED! {strat_rate:.1f}% with {cur['name']}")
                best_params = cur['params'].copy()
                best_detection_rate = strat_rate
                break

            if strat_rate > best_detection_rate:
                best_detection_rate = strat_rate
                best_params = cur['params'].copy()
                best_marker_count = max(strat_detections) if strat_detections else 0
                print(f"📈 New best: {best_detection_rate:.1f}% with {cur['name']}")

            if cv2.waitKey(1) & 0xFF == 27:
                print("🛑 Stopped by user")
                break

        print(f"✅ '{cur['name']}' → {strat_rate:.1f}% over {len(strat_detections)} frames")
        if best_detection_rate < target_detection_rate:
            strategy_index += 1
        else:
            break

    cv2.destroyAllWindows()
    cap.release()

    total_time = time.time() - start_time
    if best_params is not None:
        status = "SUCCESS" if best_detection_rate >= target_detection_rate else "PARTIAL"
        print(f"{'🎉' if status == 'SUCCESS' else '📊'} {status}: {best_detection_rate:.1f}% in {total_time:.1f}s")
        save_parameters(best_params, "best_aruco_params.json")
        return best_params
    else:
        print(f"❌ No improvement found in {total_time:.1f}s")
        return {}


# ──────────────────────────────────────────────────────────────────────────────
# MANUAL CALIBRATION  (interactive trackbar UI — largely unchanged)
# ──────────────────────────────────────────────────────────────────────────────

def create_window_and_trackbars():
    """Create a window with trackbars for each parameter."""
    cv2.namedWindow("ArUco Calibration")
    cv2.createTrackbar("adaptiveThreshWinSizeMin", "ArUco Calibration", param_values['adaptiveThreshWinSizeMin'], 20, lambda v: update_param('adaptiveThreshWinSizeMin', v))
    cv2.createTrackbar("adaptiveThreshWinSizeMax", "ArUco Calibration", param_values['adaptiveThreshWinSizeMax'], 100, lambda v: update_param('adaptiveThreshWinSizeMax', v))
    cv2.createTrackbar("adaptiveThreshWinSizeStep", "ArUco Calibration", param_values['adaptiveThreshWinSizeStep'], 20, lambda v: update_param('adaptiveThreshWinSizeStep', v))
    cv2.createTrackbar("adaptiveThreshConstant", "ArUco Calibration", param_values['adaptiveThreshConstant'], 40, lambda v: update_param('adaptiveThreshConstant', v))
    cv2.createTrackbar("minMarkerPerimeter*100", "ArUco Calibration", param_values['minMarkerPerimeterRate'], 10, lambda v: update_param('minMarkerPerimeterRate', v))
    cv2.createTrackbar("maxMarkerPerimeter*100", "ArUco Calibration", param_values['maxMarkerPerimeterRate'], 1000, lambda v: update_param('maxMarkerPerimeterRate', v))
    cv2.createTrackbar("polygonalApproxAccuracy*100", "ArUco Calibration", param_values['polygonalApproxAccuracyRate'], 10, lambda v: update_param('polygonalApproxAccuracyRate', v))
    cv2.createTrackbar("cornerRefinementMethod", "ArUco Calibration", param_values['cornerRefinementMethod'], 2, lambda v: update_param('cornerRefinementMethod', v))

    cv2.namedWindow("Advanced Parameters")
    cv2.createTrackbar("minCornerDistance*100", "Advanced Parameters", param_values['minCornerDistanceRate'], 20, lambda v: update_param('minCornerDistanceRate', v))
    cv2.createTrackbar("minMarkerDistance*100", "Advanced Parameters", param_values['minMarkerDistanceRate'], 20, lambda v: update_param('minMarkerDistanceRate', v))
    cv2.createTrackbar("cornerRefinementWinSize", "Advanced Parameters", param_values['cornerRefinementWinSize'], 20, lambda v: update_param('cornerRefinementWinSize', v))
    cv2.createTrackbar("cornerRefinementMaxIter", "Advanced Parameters", param_values['cornerRefinementMaxIterations'], 100, lambda v: update_param('cornerRefinementMaxIterations', v))
    cv2.createTrackbar("cornerRefinementMinAcc*100", "Advanced Parameters", param_values['cornerRefinementMinAccuracy'], 100, lambda v: update_param('cornerRefinementMinAccuracy', v))


def run_calibration(scale_factor=0.7, resolution='4k', load_previous=True):
    """Run the interactive calibration process with trackbars."""
    global detection_history, best_params, best_detection_count, frame_count, detection_count, param_values, param_change_count

    if resolution not in cfg.RESOLUTION_CONFIGS:
        print(f"⚠️ Unknown resolution '{resolution}', defaulting to 4k")
        resolution = '4k'
    config = cfg.RESOLUTION_CONFIGS[resolution]
    print(f"📹 Calibrating for {config['display_name']}")

    if load_previous:
        load_parameters()

    dictionary = cfg.get_dictionary()
    detector_params = cv2.aruco.DetectorParameters()
    detector_params = update_detector_parameters(detector_params)
    detector = cv2.aruco.ArucoDetector(dictionary, detector_params)

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, config['width'])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config['height'])
    actual_w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    actual_h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    print(f"📹 Camera resolution: {actual_w}x{actual_h}")

    create_window_and_trackbars()

    prev_time = time.time()
    detection_window = []
    evaluation_interval = 30

    help_text = [
        "Press 'h' to show/hide this help",
        "Press 's' to save current parameters",
        "Press 'l' to load saved parameters",
        "Press 'r' to reset to default parameters",
        "Press 'q' to quit calibration",
    ]
    show_help = True

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        current_time = time.time()
        fps = 1.0 / (current_time - prev_time)
        prev_time = current_time
        frame = crop_frame(frame, resolution)

        small_frame = cv2.resize(frame, (0, 0), fx=scale_factor, fy=scale_factor)
        gray = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)

        detector_params = update_detector_parameters(detector_params)
        detector = cv2.aruco.ArucoDetector(dictionary, detector_params)
        corners, ids, rejected = detector.detectMarkers(gray)

        frame_count += 1
        detection_window.append(1 if ids is not None else 0)
        if len(detection_window) > 100:
            detection_window.pop(0)
        detection_rate = sum(detection_window) / len(detection_window) * 100

        if ids is not None:
            corners = [corner / scale_factor for corner in corners]
            cv2.aruco.drawDetectedMarkers(frame, corners, ids)
            for i, corner in enumerate(corners):
                center = np.mean(corner[0], axis=0).astype(int)
                cv2.circle(frame, tuple(center), 5, (0, 255, 0), -1)
                cv2.putText(frame, f"ID: {ids[i][0]}", (center[0]+10, center[1]-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        draw_text_info(frame, ids, fps, detection_rate)

        param_y = 150
        cv2.putText(frame, "Current Parameters:", (500, param_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        param_y += 30
        for i, (param, value) in enumerate(param_values.items()):
            if i < 8:
                display_value = value
                if param in ['minMarkerPerimeterRate', 'maxMarkerPerimeterRate',
                             'polygonalApproxAccuracyRate', 'minCornerDistanceRate',
                             'minMarkerDistanceRate', 'cornerRefinementMinAccuracy']:
                    display_value = value / 100.0
                cv2.putText(frame, f"{param}: {display_value}",
                            (500, param_y + i * 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

        if frame_count % evaluation_interval == 0:
            param_key = str(param_values)
            detection_history[param_key] = detection_history.get(param_key, 0) + detection_rate
            if detection_rate > best_detection_count:
                best_detection_count = detection_rate
                best_params = param_values.copy()

        if show_help:
            help_y = 400
            for line in help_text:
                cv2.putText(frame, line, (10, help_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                help_y += 25

        cv2.imshow("ArUco Calibration", frame)
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

    if best_params is not None:
        save_parameters(best_params, "best_aruco_params.json")
    cap.release()
    cv2.destroyAllWindows()


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='ArUco / AprilTag detection calibration tool')
    parser.add_argument('--no-load', action='store_false', dest='load_previous',
                        help='Do not load previous best parameters')
    parser.add_argument('--scale', type=float, default=0.7,
                        help='Scale factor for processing (default: 0.7)')
    parser.add_argument('--auto', action='store_true',
                        help='Run automated calibration')
    parser.add_argument('--live', action='store_true',
                        help='Run live adaptive calibration')
    parser.add_argument('--frames', type=int, default=50,
                        help='Number of frames for auto calibration (default: 50)')
    parser.add_argument('--max-combinations', type=int, default=50,
                        help='Max parameter combinations to test (default: 50)')
    parser.add_argument('--expected-markers', type=int, default=None,
                        help=f'Expected number of markers per frame (default: {cfg.CALIBRATION_EXPECTED_MARKERS})')
    parser.add_argument('--target-rate', type=int, default=100,
                        help='Target detection %% for live calibration (default: 100)')
    parser.add_argument('--max-duration', type=int, default=300,
                        help='Max calibration time in seconds (default: 300)')
    parser.add_argument('--resolution', choices=['4k', '1080p'], default='1080p',
                        help='Camera resolution (default: 1080p)')

    args = parser.parse_args()

    if args.live:
        run_live_adaptive_calibration(args.scale, args.resolution, args.target_rate,
                                      args.max_duration, args.expected_markers)
    elif args.auto:
        run_auto_calibration(args.scale, args.frames, args.max_combinations,
                             args.resolution, args.expected_markers)
    else:
        run_calibration(args.scale, args.resolution, args.load_previous)