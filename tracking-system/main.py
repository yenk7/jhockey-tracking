# Mac-compatible version - removed Linux-specific CPU affinity commands
# To launch the visualization: open Chrome and navigate to file:///Users/jhumechatronics/Desktop/mechatronics/visualize.html
#run calibration mode with:
# 4K mode (maximum accuracy)
#   python3 main.py

# 1080p mode (better performance) 
#   python3 main.py --resolution 1080p

# With movement stability (reduce jitter)
#   python3 main.py --resolution 1080p --movement-threshold 2.0
#python3 main.py --resolution 4k --movement-threshold 2.0

# Calibration with resolution
#   python3 main.py --auto-calibrate --resolution 1080p
#   python3 main.py --live-calibrate --resolution 1080p --target-rate 100

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
                await asyncio.gather(
                    *[client.send(tracking_message) for client in connected_clients]
                )
            except websockets.exceptions.PayloadTooBig:
                # If message is still too big, send without the frame data
                print("Warning: Message too large. Removing frame data.")
                if "frame" in output_dict:
                    output_dict.pop("frame")
                    print("Removed frame data from message due to size constraints")
                tracking_message = json.dumps({"type": "tracking_data", "data": output_dict})
                await asyncio.gather(
                    *[client.send(tracking_message) for client in connected_clients]
                )
            except (websockets.exceptions.ConnectionClosed, websockets.exceptions.ConnectionClosedOK) as e:
                # Client disconnected - clean up the client list
                print(f"Client disconnected: {e}")
                # Remove closed connections
                connected_clients_copy = set(client for client in connected_clients if not client.closed)
                connected_clients.clear()
                connected_clients.update(connected_clients_copy)


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
                    await asyncio.gather(
                        *[client.send(match_message) for client in connected_clients]
                    )
    except websockets.exceptions.ConnectionClosedError:
        pass
    finally:
        connected_clients.remove(websocket)


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

def run_calibration_mode(auto_mode=False, live_mode=False, frames=20, max_combinations=50, scale_factor=0.7, resolution='4k', target_rate=100, max_duration=300):
    """Run the ArUco marker calibration tool"""
    print(f"Starting ArUco calibration mode with {resolution} resolution...")
    try:
        import aruco_tracker_calibration
        if live_mode:
            print(f"Running LIVE ADAPTIVE calibration targeting {target_rate}% detection...")
            aruco_tracker_calibration.run_live_adaptive_calibration(scale_factor, resolution, target_rate, max_duration)
        elif auto_mode:
            print("Running automated calibration process...")
            aruco_tracker_calibration.run_auto_calibration(scale_factor, frames, max_combinations, resolution)
        else:
            aruco_tracker_calibration.run_calibration(scale_factor, resolution)
    except ImportError:
        print("Error: Could not import aruco_tracker_calibration module")
    except Exception as e:
        print(f"Error in calibration mode: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ArUco tracking system for Mechatronics")
    parser.add_argument("--calibrate", "-c", action="store_true",
                      help="Run in calibration mode to optimize ArUco detection parameters")
    parser.add_argument("--auto-calibrate", "-a", action="store_true",
                      help="Run automatic calibration to find optimal parameters without manual intervention")
    parser.add_argument("--live-calibrate", "-l", action="store_true",
                      help="Run LIVE adaptive calibration that adjusts parameters until 100% detection is achieved")
    parser.add_argument("--frames", type=int, default=20,
                      help="Number of frames to capture for automated calibration (default: 20)")
    parser.add_argument("--max-combinations", type=int, default=50,
                      help="Maximum parameter combinations to test in auto calibration (default: 50)")
    parser.add_argument("--target-rate", type=int, default=100,
                      help="Target detection rate percentage for live calibration (default: 100)")
    parser.add_argument("--max-duration", type=int, default=300,
                      help="Maximum calibration duration in seconds (default: 300)")
    parser.add_argument("--scale", type=float, default=0.7,
                      help="Scale factor for processing (default: 0.7)")
    parser.add_argument("--movement-threshold", type=float, default=1.0,
                      help="Minimum movement in cm to update position (reduces jitter, default: 1.0)")
    parser.add_argument("--resolution", "-r", choices=['4k', '1080p'], default='4k',
                      help="Camera resolution: 4k (3840x2160) or 1080p (1920x1080) (default: 4k)")
    
    args = parser.parse_args()
    
    if args.calibrate:
        # Manual calibration mode
        run_calibration_mode(False, False, args.frames, args.max_combinations, args.scale, args.resolution, args.target_rate, args.max_duration)
    elif args.auto_calibrate:
        # Automatic calibration mode
        run_calibration_mode(True, False, args.frames, args.max_combinations, args.scale, args.resolution, args.target_rate, args.max_duration)
    elif args.live_calibrate:
        # Live adaptive calibration mode
        run_calibration_mode(False, True, args.frames, args.max_combinations, args.scale, args.resolution, args.target_rate, args.max_duration)
    else:
        # For Mac M1, optimize event loop policy if available
        if platform.system() == 'Darwin' and "arm" in platform.machine().lower():
            try:
                import uvloop
                asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
                print("Using uvloop for improved performance")
            except ImportError:
                print("Note: Install uvloop with 'pip install uvloop' for better performance")
        
        asyncio.run(main(args.resolution, args.scale, args.movement_threshold))

