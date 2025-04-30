import asyncio
import websockets
import json
import serial.tools.list_ports
import platform
import time

from digi.xbee.devices import XBeeDevice

BAUD_RATE = 115200
# Use a more dynamic approach to find the XBee device on Mac
TARGET_HWID_SUBSTRING = "FT" # Common substring for FTDI chips used with XBee


def find_serial_port_by_hwid(target_hwid_substring):
    """Find the serial port with a specific HWID substring, with Mac compatibility."""
    print("Searching for XBee device...")
    ports = serial.tools.list_ports.comports()
    
    # Print all available ports for debugging
    print("Available ports:")
    for port in ports:
        print(f"  {port.device}: {port.description} [{port.hwid}]")
    
    # First try to find by HWID or description that contains target substring
    for port in ports:
        if (target_hwid_substring in port.hwid) or (target_hwid_substring in port.description):
            print(f"Found XBee device on port: {port.device} (matched by HWID or description)")
            return port.device
    
    # On Mac, also check by common USB-serial adapter descriptors
    if platform.system() == 'Darwin':
        for port in ports:
            # Common descriptors for XBee on Mac
            if any(name in port.description.lower() for name in ['xbee', 'digi', 'ftdi', 'usb to uart', 'usb serial', 'uart']):
                print(f"Found XBee device on port: {port.device} (matched by description)")
                return port.device
    
    # If only one serial port with real hardware info is available, use that
    real_ports = [port for port in ports if port.hwid != 'n/a']
    if len(real_ports) == 1:
        print(f"Only one real port available, using: {real_ports[0].device}")
        return real_ports[0].device
        
    raise Exception(f"No serial port with HWID containing '{target_hwid_substring}' found.")


def construct_payload(robot_tags, match_dict):
    """
    Constructs the payload string with a start byte, match information,
    robot positions, a computed checksum, and a trailing semicolon.
    """
    payload = '>'
    # Add match bit (1 digit) and match time (4 digits)
    payload += f"{match_dict.get('match_bit', 0):01d}"
    payload += f"{match_dict.get('match_time', 0):04d}"

    # Process robot tags (limit to 15 robots)
    parsed_robots = []
    for tag_id, position in robot_tags.items():
        x_cm = max(0, int(position[0]))
        y_cm = max(0, int(position[1]))
        parsed_robots.append((int(tag_id), x_cm, y_cm))
    parsed_robots = parsed_robots[:15]

    # Append each robot's data: letter (starting from 'A'), x and y coordinates
    for tag_id, x, y in parsed_robots:
        payload += chr(61 + tag_id)  # 'A' corresponds to 0, 'B' for 1, etc.
        payload += f"{x:03d}"        # X coordinate padded to 3 digits
        payload += f"{y:03d}"        # Y coordinate padded to 3 digits

    # Compute checksum: sum of ASCII values plus the semicolon, modulo 64
    checksum = (sum(ord(c) for c in payload) + ord(';')) % 64
    payload += f"{checksum:02d}"    # Checksum padded to 2 digits

    payload += ';'  # End byte

    return payload


async def receive_data(device):
    uri = "ws://localhost:8765"
    retry_interval = 1.0  # seconds between reconnection attempts
    
    while True:
        try:
            async with websockets.connect(uri) as websocket:
                print("Connected to WebSocket server.")
                robot_tags = {}
                match_dict = {"match_bit": 0, "match_time": 0}
                
                last_send_time = time.time()
                rate_limit = 0.05  # Limit to sending at most every 50ms

                while True:
                    message = await websocket.recv()
                    message = json.loads(message)

                    # Update robot tags or match info based on message type.
                    if message["type"] == "tracking_data":
                        robot_tags = message["data"].get("robot_tags", {})
                    elif message["type"] == "match_dict":
                        match_dict = message["data"]

                    # Rate-limit the XBee transmissions to avoid overwhelming it
                    current_time = time.time()
                    if current_time - last_send_time >= rate_limit:
                        # Construct the payload from the received data.
                        payload = construct_payload(robot_tags, match_dict)
                        
                        try:
                            # Send the payload using the XBee device.
                            device.send_data_broadcast(payload)
                            print(f"Sent: {payload}", end="\r")
                            last_send_time = current_time
                        except Exception as e:
                            print(f"\nError sending data via XBee: {e}")

        except (websockets.exceptions.ConnectionClosed, 
                websockets.exceptions.WebSocketException,
                ConnectionRefusedError) as e:
            print(f"WebSocket connection error: {e}. Retrying in {retry_interval} seconds...")
            await asyncio.sleep(retry_interval)
        except Exception as e:
            print(f"Unexpected error: {e}. Retrying in {retry_interval} seconds...")
            await asyncio.sleep(retry_interval)


async def main():
    # Keep trying to find and connect to the XBee device
    while True:
        try:
            # Find the serial port for the XBee device.
            serial_port = find_serial_port_by_hwid(TARGET_HWID_SUBSTRING)
            device = XBeeDevice(serial_port, BAUD_RATE)
            
            print(f"Connecting to XBee on {serial_port}...")
            device.open()
            print("XBee device opened.")
            
            # Optimize for M1 Mac
            if platform.system() == 'Darwin' and 'arm' in platform.machine().lower():
                print("Using optimized settings for Apple Silicon")
                
            try:
                await receive_data(device)
            except Exception as e:
                print(f"Error in receive_data: {e}")
            finally:
                if device is not None and device.is_open():
                    device.close()
                    print("XBee device closed.")
                    
        except Exception as e:
            print(f"Connection error: {e}")
            print("Retrying in 5 seconds...")
            await asyncio.sleep(5)


if __name__ == "__main__":
    # For Mac M1, optimize event loop if uvloop available
    if platform.system() == 'Darwin' and 'arm' in platform.machine().lower():
        try:
            import uvloop
            asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
            print("Using uvloop for improved performance")
        except ImportError:
            pass
    
    asyncio.run(main())
