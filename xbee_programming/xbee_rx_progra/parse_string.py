"""
Function to parse the string of data received from the XBee emitter.

Emitter payload format:
    > [matchbit:1] [time:4] [robot_letter:1] [x:3] [y:3] ... [checksum:2] ;

Example:
    >01200E120085XX;
     ^ matchbit (1 digit)
      ^^^^ time (4 digits)
          ^ robot letter (1 char, e.g. 'E' for tag_id=4)
           ^^^ x (3 digits)
              ^^^ y (3 digits)
                ^^ checksum (2 digits, ignored)
                  ^ end byte
"""


def parse_string(data, parsingParameters):
    # Get the parsing parameters from the list
    startLen   = parsingParameters[0]  # 1  — the '>' start byte
    timeLen    = parsingParameters[1]  # 4  — match time digits
    robotIDLen = parsingParameters[2]  # 1  — single-char robot letter from emitter
    coordLen   = parsingParameters[3]  # 3  — x or y digits
    # parsingParameters[4] (angleLen) is unused — emitter does not send angle

    # Create a dictionary to store the parsed data
    parsedData = {}

    # Parse header fields — emitter order: start, matchbit, time
    parsedData["start"]    = data[0:startLen]                          # '>'
    parsedData["matchbit"] = data[startLen]                            # 1-digit match bit
    parsedData["time"]     = data[startLen + 1 : startLen + 1 + timeLen]  # 4-digit time

    # Start of robot data section
    i = startLen + 1 + timeLen  # = 6

    # Strip the 2-digit checksum before ';' from consideration
    # Valid robot data occupies data[6 : len(data)-3]
    end_of_robots = len(data) - 3  # -2 for checksum digits, -1 for ';'

    # Return early if no robot data present
    if i >= end_of_robots or data[i] == ";":
        return parsedData

    while i < end_of_robots:
        toCheck = data[i:]

        # Read 1-char robot letter + x (3) + y (3) = 7 bytes per robot
        robotName = toCheck[0:robotIDLen]
        parsedData[robotName] = (
            toCheck[robotIDLen : robotIDLen + coordLen]              # x
            + ","
            + toCheck[robotIDLen + coordLen : robotIDLen + coordLen * 2]  # y
        )

        i += robotIDLen + coordLen * 2  # advance by 7

    return parsedData
