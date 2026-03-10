# Mac-compatible version - removed Linux-specific CPU affinity commands
#
# ─── Calibration (run once) ──────────────────────────────────────
#   Step 1 - Camera lens calibration (ChArUco board):
#     python3 main.py --camera-calibrate --resolution 4k
#
#   Step 2 - ArUco detection parameters:
#     python3 main.py --auto-calibrate --resolution 1080p
#
# ─── Run Tracker ─────────────────────────────────────────────────
#   python3 main.py --resolution 1080p
#   python3 main.py --resolution 1080p --movement-threshold 2.0

import asyncio
import websockets
import json
import os
import subprocess
import platform
import sys
import argparse
#from aruco_tracker import track_aruco_tags
from aruco_tracker_2 import track_aruco_tags

connected_clients = set()
lock_state = False  # Global lock state
lock_queue = asyncio.Queue()  # Queue to send updates to aruco_tracker.py


async def broadcast_message(message):
    """Send a message to all connected clients and prune dead connections."""
    if not connected_clients:
        return

    clients = tuple(connected_clients)
    results = await asyncio.gather(
        *(client.send(message) for client in clients),
        return_exceptions=True,
    )

    disconnected_clients = set()
    payload_too_big = None

    for client, result in zip(clients, results):
        if isinstance(result, websockets.exceptions.PayloadTooBig):
            payload_too_big = result
        elif isinstance(result, (websockets.exceptions.ConnectionClosed, websockets.exceptions.ConnectionClosedOK)):
            disconnected_clients.add(client)
            print(f"Client disconnected: {result}")
        elif isinstance(result, Exception):
            disconnected_clients.add(client)
            print(f"Client send failed: {result}")

    for client in disconnected_clients:
        connected_clients.discard(client)

    if payload_too_big is not None:
        raise payload_too_big


async def track_and_broadcast(resolution='4k', scale_factor=0.7, movement_threshold=1.0):
    # No CPU affinity on macOS - instead use higher task priority if needed
    
    async for output_dict in track_aruco_tags(lock_queue, scale_factor, resolution, movement_threshold):  # Pass all parameters
        # Add auto-lock info to the message if available
        if "auto_lock_info" in output_dict:
            auto_lock_info = output_dict.pop("auto_lock_info")
            # Reduce image quality to decrease message size
            if "frame" in output_dict:
                # Keep the frame data for visualization (should be small enough now)
                # If WebSocket messages are still too large, this will be handled in the try/except below
                pass
                
            tracking_message = json.dumps({
                "type": "tracking_data",
                "data": output_dict,
                "auto_lock_info": auto_lock_info
            })
        else:
            # Reduce image quality to decrease message size
            if "frame" in output_dict:
                # Keep the frame data for visualization (should be small enough now with reduced resolution)
                # If WebSocket messages are still too large, this will be handled in the try/except below
                pass
                
            tracking_message = json.dumps({"type": "tracking_data", "data": output_dict})
            
        if connected_clients:
            try:
                await broadcast_message(tracking_message)
            except websockets.exceptions.PayloadTooBig:
                # If message is still too big, send without the frame data
                print("Warning: Message too large. Removing frame data.")
                if "frame" in output_dict:
                    output_dict.pop("frame")
                    print("Removed frame data from message due to size constraints")
                tracking_message = json.dumps({"type": "tracking_data", "data": output_dict})
                await broadcast_message(tracking_message)


async def handler(websocket):
    
    global lock_state

    connected_clients.add(websocket)
    try:
        async for message in websocket:
            data = json.loads(message)

            if data["type"] == "lock_state":
                lock_state = data["data"]
                print(f"Lock state updated: {lock_state}")
                # Send new lock state to queue
                await lock_queue.put(lock_state)
                
            if data["type"] == "auto_lock":
                # Handle auto-lock command
                print("Auto-lock command received: Starting auto detection of corners")
                # Add a quick response to the client to acknowledge the command
                await websocket.send(json.dumps({
                    "type": "command_ack",
                    "message": "Auto-lock command received"
                }))
                # Then queue the actual command
                await lock_queue.put({"auto_lock": True})

            if data["type"] == "match_dict":
                match_message = json.dumps({"type": "match_dict", "data": data["data"]})
                if connected_clients:
                    await broadcast_message(match_message)
    except (websockets.exceptions.ConnectionClosedError, websockets.exceptions.ConnectionClosedOK):
        pass
    finally:
        connected_clients.discard(websocket)


async def main(resolution='4k', scale_factor=0.7, movement_threshold=1.0):
    # Check if we're on macOS
    is_macos = platform.system() == 'Darwin'
    
    # Set process priority if possible
    try:
        if not is_macos:
            import psutil
            p = psutil.Process(os.getpid())
            p.nice(psutil.HIGH_PRIORITY_CLASS if os.name == 'nt' else -10)
    except (ImportError, PermissionError):
        print("Note: Could not set process priority")

    # Launch zigbee.py as a subprocess using virtual environment
    venv_python = "/Users/jhumechatronics/Desktop/mechatronics/.venv/bin/python"
    zigbee_process = subprocess.Popen([venv_python, "zigbee.py"])
    
    print(f"Running on: {platform.system()} {platform.machine()}")
    
    # On M1 Macs, we rely on the system's task scheduler instead of manual CPU affinity
    if is_macos and "arm" in platform.machine().lower():
        print("Detected Apple Silicon - using system scheduler for performance")
    
    # Start WebSocket server and tracking loop
    server = await websockets.serve(
        handler, 
        "localhost", 
        8765,
        max_size=10485760  # 10MB limit to be extra safe
    )
    print("WebSocket server started on ws://localhost:8765")
    print(f"zigbee.py started on PID {zigbee_process.pid}")

    try:
        await asyncio.gather(server.wait_closed(), track_and_broadcast(resolution, scale_factor, movement_threshold))
    finally:
        zigbee_process.terminate()
        print("zigbee.py terminated.")

def run_calibration_mode(auto_mode=False, frames=20, max_combinations=50, scale_factor=0.7, resolution='4k'):
    """Run the ArUco marker detection parameter calibration"""
    print(f"Starting ArUco detection calibration with {resolution} resolution...")
    try:
        import aruco_tracker_calibration
        if auto_mode:
            print("Running automated calibration process...")
            aruco_tracker_calibration.run_auto_calibration(scale_factor, frames, max_combinations, resolution)
        else:
            aruco_tracker_calibration.run_calibration(scale_factor, resolution)
    except ImportError:
        print("Error: Could not import aruco_tracker_calibration module")
    except Exception as e:
        print(f"Error in calibration mode: {e}")


def run_camera_calibration(resolution='4k', captures=20):
    """Run camera lens distortion calibration using ChArUco board"""
    print(f"Starting camera calibration with {resolution} resolution...")
    try:
        import calibrate_camera
        calibrate_camera.interactive_capture(
            resolution=resolution,
            use_charuco=True,
            board_size=(8, 11),        # 8x11 ChArUco board
            square_size_mm=25.0,       # 25mm squares
            marker_size_mm=18.0,       # 18mm markers
            num_captures=captures
        )
    except ImportError:
        print("Error: Could not import calibrate_camera module")
    except Exception as e:
        print(f"Error in camera calibration: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="ArUco tracking system for Mechatronics",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
─── Calibration Commands ───────────────────────────────────────

  Step 1: Camera lens calibration (run once per camera/resolution)
    python3 main.py --camera-calibrate --resolution 4k

  Step 2: ArUco detection parameter calibration
    python3 main.py --auto-calibrate --resolution 1080p

  Step 3: Manual ArUco calibration (interactive)
    python3 main.py --calibrate --resolution 1080p

─── Run Tracker ────────────────────────────────────────────────

    python3 main.py --resolution 1080p
    python3 main.py --resolution 1080p --movement-threshold 2.0
        """
    )

    # --- Calibration modes ---
    parser.add_argument("--camera-calibrate", action="store_true",
                      help="Calibrate camera lens distortion using ChArUco board (run first, once per camera)")
    parser.add_argument("--calibrate", "-c", action="store_true",
                      help="Run interactive ArUco detection parameter calibration")
    parser.add_argument("--auto-calibrate", "-a", action="store_true",
                      help="Run automatic ArUco detection parameter calibration")

    # --- Calibration options ---
    parser.add_argument("--frames", type=int, default=50,
                      help="Number of frames for auto calibration (default: 50)")
    parser.add_argument("--max-combinations", type=int, default=50,
                      help="Max parameter combinations to test (default: 50)")
    parser.add_argument("--captures", type=int, default=20,
                      help="Number of captures for camera calibration (default: 20)")

    # --- Runtime options ---
    parser.add_argument("--scale", type=float, default=0.7,
                      help="Scale factor for processing (default: 0.7)")
    parser.add_argument("--movement-threshold", type=float, default=1.0,
                      help="Min movement in cm to update position (default: 1.0)")
    parser.add_argument("--resolution", "-r", choices=['4k', '1080p'], default='1080p',
                      help="Camera resolution (default: 1080p)")
    
    args = parser.parse_args()
    
    if args.camera_calibrate:
        # Step 1: Camera lens distortion calibration
        run_camera_calibration(args.resolution, args.captures)
    elif args.calibrate:
        # Step 2a: Interactive ArUco parameter calibration
        run_calibration_mode(False, args.frames, args.max_combinations, args.scale, args.resolution)
    elif args.auto_calibrate:
        # Step 2b: Automatic ArUco parameter calibration
        run_calibration_mode(True, args.frames, args.max_combinations, args.scale, args.resolution)
    else:
        # Normal tracking mode
        if platform.system() == 'Darwin' and "arm" in platform.machine().lower():
            try:
                import uvloop
                asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
                print("Using uvloop for improved performance")
            except ImportError:
                print("Note: Install uvloop with 'pip install uvloop' for better performance")
        
        asyncio.run(main(args.resolution, args.scale, args.movement_threshold))
