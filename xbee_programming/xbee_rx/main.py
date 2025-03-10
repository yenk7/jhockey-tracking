"""
MicroPython Script for XBee 3 Modules

Description:
    This script is designed to receive data from the Xbee Transmitter (Tx) and parse it for further processing. It is specifically tailored for use with XBee 3 Modules running the 802.15.4 firmware.

Author: Anway Pimpalkar
Date: 01/23/2024

"""

import xbee
import utime
from parse_string import parse_string
from sys import stdin, stdout

# Unique ID for each robot
<<<<<<< Updated upstream
ROBOT_ID = "BA"
=======
ROBOT_ID = "L"
>>>>>>> Stashed changes

# Parsing parameters
startLen = 1
timeLen = 4
robotIDLen = 2
coordLen = 3
angleLen = 3

# Store the parameters (Start Length, Time Length, Robot ID Length, Coordinate Length, Angle Length) in a list
parsingParameters = [startLen, timeLen, robotIDLen, coordLen, angleLen]

lastDataTime = 0
nowTime = 0
timeout = 1000

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
<<<<<<< Updated upstream
    if data and data.decode() == "?":
=======
    if data:
        nowTime = utime.ticks_ms()

        if (utime.ticks_diff(lastDataTime, nowTime)) > timeout:
            last_payload = None

        if "?" in data.decode():
>>>>>>> Stashed changes

        if last_payload is not None:
            # Decode the payload
            receivedMsg = last_payload["payload"].decode("utf-8")

<<<<<<< Updated upstream
            # If the payload is not empty, parse it
            if receivedMsg:
                # Find the start and end of the payload
                start = receivedMsg.find(">")
                end = receivedMsg.find(";") + 1
=======
                # If the payload is not empty, parse it
                if receivedMsg:
                    # print(receivedMsg)
                    # Find the start and end of the payload
                    start = receivedMsg.find(">")
                    end = receivedMsg.find(";") + 1
>>>>>>> Stashed changes

                # If the start and end are found, parse the payload
                if start != -1 and end != -1:
                    # Extract the string from the payload
                    string = receivedMsg[start:end]

                    # Parse the string
                    parsedDict = parse_string(string, parsingParameters)

                    # Check if the robot ID is a key in the dictionary
                    if ROBOT_ID in parsedDict:
                        # If the robot ID is a key in the dictionary, set the match time, match bit, and robot coordinates
                        matchTime = parsedDict["time"]
                        matchBit = parsedDict["matchbit"]
                        robotCoords = parsedDict[ROBOT_ID]

<<<<<<< Updated upstream
                    else:
                        # If the robot ID is not a key in the dictionary, set everything to 9s
                        matchTime = "9" * timeLen
                        matchBit = "9"
                        robotCoords = "9" * (coordLen * 2) + "9" * angleLen
=======
                            if not checksumValid:
                                out = "/,////,---,---"
                                stdout.buffer.write(out.encode())
                                continue
>>>>>>> Stashed changes

                    # Create output string for stdout (Arduino/UART interface)
                    out = matchTime + "," + matchBit + "," + robotCoords + "\n"

                    # Write the output string to stdout
                    stdout.buffer.write(out.encode())

<<<<<<< Updated upstream
        else:
            out = "no active tx found\n"
            stdout.buffer.write(out.encode())
=======
                                if "matchbit" in parsedDict:
                                    matchBit = parsedDict["matchbit"]
                                else:
                                    matchBit = "-"

                                if ROBOT_ID in parsedDict:
                                    robotCoords = parsedDict[ROBOT_ID]
                                else:
                                    robotCoords = "---,---"

                                # Create output string for stdout (Arduino/UART interface)
                                out = matchBit + "," + matchTime + "," + robotCoords

                                # Write the output string to stdout
                                stdout.buffer.write(out.encode())
                                lastDataTime = utime.ticks_ms()
            else:
                out = "?,????,---,---"
                stdout.buffer.write(out.encode())
>>>>>>> Stashed changes
