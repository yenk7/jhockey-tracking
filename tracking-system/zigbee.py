# import asyncio
# import websockets
# import json
# import struct
# import serial
# import serial.tools.list_ports

# from digi.xbee.devices import XBeeDevice

# BAUD_RATE = 115200
# TARGET_HWID_SUBSTRING = "D30DP79H"


# def find_serial_port_by_hwid(target_hwid_substring):
#     """Find the serial port with a specific HWID substring."""
#     ports = serial.tools.list_ports.comports()
#     for port in ports:
#         if target_hwid_substring in port.hwid:
#             print(f"Found target device on port: {port.device}")
#             return port.device
#     raise Exception(f"No serial port with HWID containing '{target_hwid_substring}' found.")


# async def receive_data():
#     uri = "ws://localhost:8765"

#     # Automatically select serial port based on HWID substring
#     serial_port = find_serial_port_by_hwid(TARGET_HWID_SUBSTRING)

#     # Open the serial port
#     ser = serial.Serial(serial_port, BAUD_RATE)

#     async with websockets.connect(uri) as websocket:
#         print("Connected to WebSocket server.")

#         robot_tags = {}
#         match_dict = {"match_bit": 0, "match_time": 0}

#         while True:
#             message = await websocket.recv()
#             message = json.loads(message)

#             if message["type"] == "tracking_data":
#                 robot_tags = message["data"].get("robot_tags", {})

#             elif message["type"] == "match_dict":
#                 match_dict = message["data"]

#             # Construct and send payload after receiving updated data
#             payload = construct_payload(robot_tags, match_dict)

#             print(payload)
#             ser.write(payload.encode())


# def construct_payload(robot_tags, match_dict):
#     """
#     Constructs the binary payload according to the match and robot tracking info.
#     """

#     payload = ''
#     payload += '>'
#     payload += f"{match_dict['match_bit']:01d}"
#     payload += f"{match_dict['match_time']:04d}"

#     # Parse robot positions and limit to 15 robots
#     parsed_robots = []
#     for tag_id, position in robot_tags.items():
#         x_cm = max(0, int(position[0]))
#         y_cm = max(0, int(position[1]))
#         parsed_robots.append((int(tag_id), x_cm, y_cm))

#     # Limit to 15 robots
#     parsed_robots = parsed_robots[:15]

#     # Add robot information to the payload
#     for tag_id, x, y in parsed_robots:
#         payload += chr(61 + tag_id)  # 'A' for 0, 'B' for 1, etc.
#         payload += f"{x:03d}"  # X coordinate padded to 3 digits
#         payload += f"{y:03d}"  # Y coordinate padded to 3 digits

#     # Compute checksum as sum of ASCII values mod 100 (2 digits)
#     checksum = sum([ord(c) for c in payload] + [ord(';')]) % 64
#     payload += f"{checksum:02d}"  # Add checksum padded to 2 digits

#     payload += ';'  # End byte

#     return payload


# if __name__ == "__main__":
#     asyncio.run(receive_data())


import asyncio
import websockets
import json
import serial.tools.list_ports

from digi.xbee.devices import XBeeDevice

BAUD_RATE = 115200
# Use the HWID substring from the working function.
TARGET_HWID_SUBSTRING = "0"


def find_serial_port_by_hwid(target_hwid_substring):
    """Find the serial port with a specific HWID substring."""
    ports = serial.tools.list_ports.comports()
    for port in ports:
        if target_hwid_substring in port.hwid:
            print(f"Found XBee device on port: {port.device}")
            return port.device
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
    # Find the serial port for the XBee device.
    serial_port = find_serial_port_by_hwid(TARGET_HWID_SUBSTRING)
    device = XBeeDevice(serial_port, BAUD_RATE)
    try:
        device.open()
        print("XBee device opened.")
        await receive_data(device)
    except Exception as e:
        print("An error occurred:", e)
    finally:
        if device is not None and device.is_open():
            device.close()
            print("XBee device closed.")


if __name__ == "__main__":
    asyncio.run(main())
