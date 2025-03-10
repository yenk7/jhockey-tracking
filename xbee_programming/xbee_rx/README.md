# Receiving Data: Expected Format

The current implementation of the MicroPython expects the message payload to be organized as the following:

`>ttttmAAxxxyyyaaaABxxxyyyaaaBAxxxyyyaaa` ...

- `>` = start (1)
- `tttt` = time bytes (4)
- `m` = match byte (1)
- `AA` = ROBOT_ID (2)
- `xxx` = x coordinates of robot (3)
- `yyy` = y coordinates of robot (3)
- `aaa` = angle of robot (3)
- `;` = end (1)

> The maximum length of the payload can be 114 bytes.
> Therefore, `9 robots` can be supported within the payload.