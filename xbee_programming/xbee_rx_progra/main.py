"""
MicroPython Script for XBee 3 Modules

Description:
    This script is designed to receive data from the Xbee Transmitter (Tx) and parse it for further processing. It is specifically tailored for use with XBee 3 Modules running the 802.15.4 firmware.

Author: Anway Pimpalkar
Date: 03/05/2026
"""

import xbee
import utime
from parse_string import parse_string
from sys import stdin, stdout

# Unique ID for each robot
# Tags 0-3 are field corner markers (reserved). Robot tags start at ID 4.
# Mapping: chr(61 + tag_id) — tag_id 4 → 'A', 5 → 'B', 6 → 'C', 7 → 'D', ...
ROBOT_ID = "I"  # Change to 'B', 'C', etc. for other robots

# Parsing parameters — must match emitter payload format exactly
startLen   = 1  # '>' start byte
timeLen    = 4  # match time (4 digits)
robotIDLen = 1  # single-char robot letter emitted by zigbee.py
coordLen   = 3  # x or y coordinate (3 digits each)
# Note: emitter does NOT send an angle field

# Store the parameters in a list (angleLen slot kept for parse_string.py compat but unused)
parsingParameters = [startLen, timeLen, robotIDLen, coordLen, 0]

lastDataTime = None  # None means no packet received yet
timeout = 3000       # 3 seconds — clear last_payload if no RF for this long

# Variable to store the last payload received
last_payload = None

while True:
    # Check if there is any data to be received in a non-blocking way
    payload = xbee.receive()

    # If there is data, store it in last_payload
    if payload:
        last_payload = payload

    # Read data from stdin
    data = stdin.buffer.read()

    # If data is received, start processing it
    if data:
        nowTime = utime.ticks_ms()

        # Only apply timeout if we've received at least one packet
        if lastDataTime is not None and utime.ticks_diff(nowTime, lastDataTime) > timeout:
            last_payload = None

        if b"?" in data:
            if last_payload is not None:
                # Decode the payload
                receivedMsg = last_payload["payload"].decode("utf-8")

                # If the payload is not empty, parse it
                if receivedMsg:
                    # Find the start and end of the payload
                    start = receivedMsg.find(">")
                    end = receivedMsg.find(";") + 1

                    # If the start and end are found, parse the payload
                    if start != -1 and end != 0:
                        # Extract the string from the payload
                        string = receivedMsg[start:end]

                        # Parse the string
                        parsedDict = parse_string(string, parsingParameters)

                        # Get match time if available, otherwise default
                        if "time" in parsedDict:
                            matchTime = parsedDict["time"]
                        else:
                            matchTime = "9" * timeLen

                        # Get match bit if available, otherwise default
                        if "matchbit" in parsedDict:
                            matchBit = parsedDict["matchbit"]
                        else:
                            matchBit = "9"

                        # Get robot coordinates if available, otherwise default
                        if ROBOT_ID in parsedDict:
                            robotCoords = parsedDict[ROBOT_ID]
                        else:
                            robotCoords = "9" * coordLen + "," + "9" * coordLen

                        # Create output string for stdout (Arduino/UART interface)
                        out = matchTime + "," + matchBit + "," + robotCoords + "\n"

                        # Write the output string to stdout
                        stdout.buffer.write(out.encode())

                        lastDataTime = nowTime
                    else:
                        out = "?,????,999999999\n"
                        stdout.buffer.write(out.encode())
            else:
                out = "no active tx found\n"
                stdout.buffer.write(out.encode())