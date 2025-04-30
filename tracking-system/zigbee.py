import asyncio
import websockets
import json
import serial.tools.list_ports
import platform

from digi.xbee.devices import XBeeDevice

BAUD_RATE = 115200
# Use a more generic approach for port detection
TARGET_HWID_SUBSTRING = "0"


def find_serial_port():
    """Find the XBee serial port with platform-specific approaches."""
    ports = serial.tools.list_ports.comports()
    
    # Print available ports for debugging
    print("Available serial ports:")
    for port in ports:
        print(f"  {port.device}: {port.description} [hwid: {port.hwid}]")
    
    # On macOS, look for ports with specific patterns
    if platform.system() == "Darwin":
        # First try to find XBee devices specifically
        for port in ports:
            if "XBee" in port.description or "FTDI" in port.description:
                print(f"Found likely XBee device on port: {port.device}")
                return port.device
                
        # Fall back to typical macOS USB-Serial patterns
        for port in ports:
            if "usbserial" in port.device or "usbmodem" in port.device:
                print(f"Found USB serial device on port: {port.device}")
                return port.device
    
    # General approach - use first available port if no specific device found
    if ports:
        print(f"Using first available port: {ports[0].device}")
        return ports[0].device
        
    raise Exception("No serial ports found. Check if XBee device is connected.")


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

    async with websockets.connect(uri) as websocket:
        print("Connected to WebSocket server.")
        robot_tags = {}
        match_dict = {"match_bit": 0, "match_time": 0}

        while True:
            message = await websocket.recv()
            message = json.loads(message)

            # Update robot tags or match info based on message type.
            if message["type"] == "tracking_data":
                robot_tags = message["data"].get("robot_tags", {})
            elif message["type"] == "match_dict":
                match_dict = message["data"]

            # Construct the payload from the received data.
            payload = construct_payload(robot_tags, match_dict)
            # print("Sending payload:", payload)

            try:
                # Send the payload using the XBee device.
                device.send_data_broadcast(payload)
            except Exception as e:
                print(f"Error sending data via XBee: {e}")


async def main():
    # Find the serial port for the XBee device with improved detection
    try:
        serial_port = find_serial_port()
        device = XBeeDevice(serial_port, BAUD_RATE)
        device.open()
        print(f"XBee device opened on port {serial_port}.")
        await receive_data(device)
    except Exception as e:
        print("An error occurred:", e)
    finally:
        if 'device' in locals() and device is not None and device.is_open():
            device.close()
            print("XBee device closed.")


if __name__ == "__main__":
    asyncio.run(main())
