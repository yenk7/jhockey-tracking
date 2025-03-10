""""
Function to parse the string of data received from the robot.
"""


def parse_string(data, parsingParameters):
    # Get the parsing parameters from the list
    startLen = parsingParameters[0]
    timeLen = parsingParameters[1]
    robotIDLen = parsingParameters[2]
    coordLen = parsingParameters[3]
    angleLen = parsingParameters[4]

    # Create a dictionary to store the parsed data
    parsedData = {}

    # Parse the data
    parsedData["start"] = data[0:startLen]
    parsedData["time"] = data[startLen : startLen + timeLen]
    parsedData["matchbit"] = data[startLen + timeLen]

    # Get the robot data
    i = startLen + timeLen + 1

    # Check if there is no robot data
    if data[i] == ";":
        return parsedData

    while i < len(data):
        toCheck = data[i:]

        robotName = toCheck[0:robotIDLen]
        parsedData[robotName] = (
            toCheck[robotIDLen : robotIDLen + coordLen]
            + ","
            + toCheck[robotIDLen + coordLen : robotIDLen + (coordLen * 2)]
            + ","
            + toCheck[
                robotIDLen + (coordLen * 2) : robotIDLen + (coordLen * 2) + angleLen
            ]
        )

        i = i + 11

        if data[i] == ";":
            break

    return parsedData
