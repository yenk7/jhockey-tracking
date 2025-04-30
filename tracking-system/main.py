# # Open Chrome with 
# # Note: taskset command not available on macOS

import asyncio
import websockets
import json
import os
import subprocess
import platform
from aruco_tracker import track_aruco_tags

connected_clients = set()
lock_state = False  # Global lock state
lock_queue = asyncio.Queue()  # Queue to send updates to aruco_tracker.py


async def track_and_broadcast():
    # CPU affinity not supported on macOS
    if platform.system() != "Darwin":
        import psutil
        p = psutil.Process(os.getpid())
        try:
            p.cpu_affinity([0])  # Assign main.py to core 1
        except AttributeError:
            print("CPU affinity not supported on this platform")
    
    async for output_dict in track_aruco_tags(lock_queue):  # Pass queue here
        tracking_message = json.dumps({"type": "tracking_data", "data": output_dict})
        if connected_clients:
            await asyncio.gather(
                *[client.send(tracking_message) for client in connected_clients]
            )


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


async def main():
    # Launch zigbee.py as a subprocess
    zigbee_process = subprocess.Popen(["python3", "zigbee.py"])
    
    # Set CPU affinity if supported (not on macOS)
    if platform.system() != "Darwin":
        try:
            import psutil
            zigbee_psutil_process = psutil.Process(zigbee_process.pid)
            zigbee_psutil_process.cpu_affinity([1])
            core_info = "(core 2)"
        except (ImportError, AttributeError):
            core_info = "(CPU affinity not supported)"
    else:
        core_info = "(CPU affinity not available on macOS)"

    # Start WebSocket server and tracking loop
    server = await websockets.serve(handler, "localhost", 8765)
    print(f"WebSocket server started on ws://localhost:8765")
    print(f"zigbee.py started on PID {zigbee_process.pid} {core_info}")

    try:
        await asyncio.gather(server.wait_closed(), track_and_broadcast())
    finally:
        zigbee_process.terminate()
        print("zigbee.py terminated.")


if __name__ == "__main__":
    asyncio.run(main())

