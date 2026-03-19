//--------------------------------------------------------------------------
// Code to setup basic ZigBee functionality (jhockey tracking)
// Updated by Anway Pimpalkar 03/20/24
// Modified: added parsing, debug stats, speed calculation,
//           and BROADCAST_FORMAT support
//--------------------------------------------------------------------------

// ===== CONFIGURATION =====
// Comment out the next line to only print X and Y
#define DEBUG

// In DEBUG mode, comment out the next line to hide invalid messages
// (they are still counted in stats, just not printed)
#define SHOW_INVALID

// --- FORMAT FLAG ---
// Define BROADCAST_FORMAT to parse the coordinator's broadcast payload:
//   >MTTTTRXXXYYY...CC;
// Comment it out to use the old comma-separated format:
//   matchByte,gameTime,X,Y
#define BROADCAST_FORMAT

// Set your robot ID here (the letter assigned to this bot's XBee module)
// In broadcast mode this must match the letter the coordinator assigns
// (chr(61 + tag_id) in zigbee.py, so tag_id 4 → 'A', 5 → 'B', etc.)
#define ROBOT_ID  'H'

// --- FILTER FLAG (broadcast mode only) ---
// Define FILTER_MY_ROBOT to only process messages containing ROBOT_ID.
// Comment it out to accept all valid broadcasts and display every robot.
//#define FILTER_MY_ROBOT

// Timeout (ms) – if no new byte arrives within this window, treat
// whatever is in the buffer as a complete message.
#define RX_TIMEOUT_MS  5

// ===== END CONFIGURATION =====

// Parsed fields
int   matchByte  = 0;
long  gameTime   = 0;
int   xPos       = 0;
int   yPos       = 0;

#ifdef BROADCAST_FORMAT
// Store all robots from the broadcast message
struct RobotEntry {
  char letter;
  int  x;
  int  y;
};
#define MAX_ROBOTS 15
RobotEntry robots[MAX_ROBOTS];
int numRobots = 0;
#endif

// Previous position & time for speed calculation
int   prevX      = 0;
int   prevY      = 0;
unsigned long prevTimeMicros = 0;
float speed      = 0.0;  // units per second

// Stats (used in DEBUG mode)
unsigned long totalResponses   = 0;
unsigned long validCoords      = 0;
unsigned long invalidResponses = 0;

#ifndef BROADCAST_FORMAT
// Only used in legacy query mode
unsigned long totalQueries     = 0;
#endif

// Sampling rate measurement
unsigned long hzCounter       = 0;
unsigned long lastHzTime       = 0;
float         samplingRateHz   = 0.0;

// Buffer for incoming XBee data
char rxBuffer[128];
int  rxIndex = 0;

// Timestamp of last received byte (for timeout detection)
unsigned long lastRxTime = 0;

// ---------------------------------------------------------------
// Helper: extract N decimal digits from buf starting at pos
// Returns the integer value, advances pos by N.
// Returns -1 if any character is not a digit.
// ---------------------------------------------------------------
int extractDigits(const char* buf, int len, int &pos, int numDigits) {
  int value = 0;
  for (int i = 0; i < numDigits; i++) {
    if (pos >= len) return -1;
    char c = buf[pos++];
    if (c < '0' || c > '9') return -1;
    value = value * 10 + (c - '0');
  }
  return value;
}

#ifdef BROADCAST_FORMAT
// ---------------------------------------------------------------
// Parse broadcast format:  >MTTTTRXXXYYY...CC;
//   '>'       start byte
//   M         match bit   (1 digit)
//   TTTT      match time  (4 digits)
//   RXXXYYY   robot entry (letter + 3-digit X + 3-digit Y), repeated
//   CC        checksum    (2 digits)
//   ';'       end byte
//
// Finds the entry whose letter == ROBOT_ID and populates globals.
// Returns true on success.
// ---------------------------------------------------------------
bool parseBroadcast(const char* buf) {
  int len = strlen(buf);

  // Minimum valid message: >M TTTT R XXX YYY CC ;  = 13 chars
  if (len < 13) return false;

  // Check start and end bytes
  if (buf[0] != '>')       return false;
  if (buf[len - 1] != ';') return false;

  // --- Verify checksum ---
  // Checksum in payload = (sum of ASCII from '>' through last robot entry + ord(';')) % 64
  // The two digits just before ';' are the transmitted checksum
  int txChk = (buf[len - 3] - '0') * 10 + (buf[len - 2] - '0');
  if (buf[len - 3] < '0' || buf[len - 3] > '9') return false;
  if (buf[len - 2] < '0' || buf[len - 2] > '9') return false;

  // Compute expected checksum over everything before the checksum digits, plus ';'
  int calcChk = 0;
  for (int i = 0; i < len - 3; i++) {   // sum '>' through last robot byte
    calcChk += (unsigned char)buf[i];
  }
  calcChk += ';';   // the Python code includes ord(';') in the sum
  calcChk %= 64;

  if (calcChk != txChk) return false;

  // --- Parse header ---
  int pos = 1;  // skip '>'

  int mBit = extractDigits(buf, len, pos, 1);
  if (mBit < 0) return false;

  int mTime = extractDigits(buf, len, pos, 4);
  if (mTime < 0) return false;

  // --- Parse header values into globals ---
  matchByte = mBit;
  gameTime  = mTime;

  // --- Scan robot entries ---
  // Each entry is 7 chars: 1 letter + 3-digit X + 3-digit Y
  // Entries end when we hit the 2-digit checksum + ';' (3 chars from end)
  int dataEnd = len - 3;  // index of first checksum digit
  bool foundSelf = false;
  numRobots = 0;

  while (pos + 7 <= dataEnd && numRobots < MAX_ROBOTS) {
    char robotLetter = buf[pos++];

    int rx = extractDigits(buf, len, pos, 3);
    if (rx < 0) return false;

    int ry = extractDigits(buf, len, pos, 3);
    if (ry < 0) return false;

    // Store every robot in the array
    robots[numRobots].letter = robotLetter;
    robots[numRobots].x      = rx;
    robots[numRobots].y      = ry;
    numRobots++;

    if (robotLetter == ROBOT_ID) {
      xPos      = rx;
      yPos      = ry;
      foundSelf = true;
    }
  }

#ifdef FILTER_MY_ROBOT
  return foundSelf;           // only succeed if our robot is present
#else
  return (numRobots > 0);     // succeed if any robots were parsed
#endif
}
#endif

// ---------------------------------------------------------------
// Parse old comma-separated response: "matchByte,gameTime,X,Y"
// Returns true if parsing succeeded
// ---------------------------------------------------------------
bool parseCsv(const char* buf) {
  int   f1 = 0;
  long  f2 = 0;
  int   f3 = 0;
  int   f4 = 0;

  int matched = sscanf(buf, "%d,%ld,%d,%d", &f1, &f2, &f3, &f4);
  if (matched == 4) {
    matchByte = f1;
    gameTime  = f2;
    xPos      = f3;
    yPos      = f4;
    return true;
  }
  return false;
}

// ---------------------------------------------------------------
// Top-level parser — picks the right format based on the flag
// ---------------------------------------------------------------
bool parseResponse(const char* buf) {
#ifdef BROADCAST_FORMAT
  return parseBroadcast(buf);
#else
  return parseCsv(buf);
#endif
}

// ---------------------------------------------------------------
// Process a completed message sitting in rxBuffer
// ---------------------------------------------------------------
void processMessage() {
  if (rxIndex == 0) return;

  rxBuffer[rxIndex] = '\0';  // null-terminate
  totalResponses++;

  if (parseResponse(rxBuffer)) {
    validCoords++;
    hzCounter++;

    // ---------- OUTPUT ----------
#ifdef DEBUG
    float successPct = (totalResponses > 0)
                         ? (validCoords * 100.0 / totalResponses)
                         : 0.0;

#ifdef BROADCAST_FORMAT
    // Print robots from the broadcast
    for (int i = 0; i < numRobots; i++) {
      bool isSelf = (robots[i].letter == ROBOT_ID);

#ifdef FILTER_MY_ROBOT
      if (!isSelf) continue;  // skip other robots when filtering
#endif

      // Speed calculation for our robot only
      if (isSelf) {
        unsigned long nowMicros = micros();
        float dtSec = (nowMicros - prevTimeMicros) / 1000000.0;
        if (dtSec > 0.001 && prevTimeMicros > 0) {
          float dx = xPos - prevX;
          float dy = yPos - prevY;
          speed = sqrt(dx * dx + dy * dy) / dtSec;
        }
        prevX = xPos;
        prevY = yPos;
        prevTimeMicros = nowMicros;
      }

      Serial.print(isSelf ? '*' : ' ');
      Serial.print(robots[i].letter);  Serial.print(" | ");
      Serial.print(matchByte);         Serial.print(" | ");
      Serial.print(robots[i].x);       Serial.print(" | ");
      Serial.print(robots[i].y);       Serial.print(" | ");
      if (isSelf) {
        Serial.print(speed, 1);
      } else {
        Serial.print('-');
      }
      Serial.print(" | ");
      Serial.print(samplingRateHz, 1); Serial.print(F(" Hz | "));
      Serial.print(gameTime);          Serial.print(" | ");
      Serial.print(totalResponses);    Serial.print(" | ");
      Serial.print(validCoords);       Serial.print(" | ");
      Serial.print(invalidResponses);  Serial.print(" | ");
      Serial.print(successPct, 1);     Serial.println('%');
    }
#else
    // Legacy CSV mode — single robot
    unsigned long nowMicros = micros();
    float dtSec = (nowMicros - prevTimeMicros) / 1000000.0;
    if (dtSec > 0.001 && prevTimeMicros > 0) {
      float dx = xPos - prevX;
      float dy = yPos - prevY;
      speed = sqrt(dx * dx + dy * dy) / dtSec;
    }
    prevX = xPos;
    prevY = yPos;
    prevTimeMicros = nowMicros;

    Serial.print((char)ROBOT_ID);    Serial.print(" | ");
    Serial.print(matchByte);         Serial.print(" | ");
    Serial.print(xPos);              Serial.print(" | ");
    Serial.print(yPos);              Serial.print(" | ");
    Serial.print(speed, 1);          Serial.print(" | ");
    Serial.print(samplingRateHz, 1); Serial.print(F(" Hz | "));
    Serial.print(gameTime);          Serial.print(" | ");
    Serial.print(totalResponses);    Serial.print(" | ");
    Serial.print(validCoords);       Serial.print(" | ");
    Serial.print(invalidResponses);  Serial.print(" | ");
    Serial.print(successPct, 1);     Serial.println('%');
#endif
#else
    // Minimal output: just X and Y
    Serial.print(xPos);
    Serial.print(" , ");
    Serial.println(yPos);
#endif

  } else {
    invalidResponses++;

#if defined(DEBUG) && defined(SHOW_INVALID)
    Serial.print(F("[INVALID] raw: \""));
    Serial.print(rxBuffer);
    Serial.println(F("\""));
#endif
  }

  rxIndex = 0;  // reset buffer
}

// ---------------------------------------------------------------
void setup() {
  Serial.begin(115200);   // UART to computer
  Serial1.begin(115200);  // UART to XBee module (Pins 18 & 19 on Arduino Mega)
  delay(500);

  Serial.println(F("=== XBee Tracking Started ==="));
  Serial.print(F("Robot ID: "));
  Serial.println((char)ROBOT_ID);

#ifdef BROADCAST_FORMAT
  Serial.println(F("FORMAT: BROADCAST (>MTTTTRXXXYYY...CC;)"));
#else
  Serial.println(F("FORMAT: CSV (matchByte,gameTime,X,Y)"));
#endif

#ifdef DEBUG
  Serial.println(F("MODE: DEBUG (full stats)"));
  Serial.println(F("RobotID | MatchByte | X | Y | Speed | Hz | GameTime | Responses | ValidCoords | Invalid | CoordSuccess%"));
#else
  Serial.println(F("MODE: MINIMAL (X, Y only)"));
  Serial.println(F("X , Y"));
#endif

  Serial.println(F("----------------------------"));

  prevTimeMicros = micros();
  lastRxTime     = millis();
  lastHzTime     = millis();
}

// ---------------------------------------------------------------
void loop() {

#ifndef BROADCAST_FORMAT
  // Legacy mode: send '?' query every loop iteration
  Serial1.print('?');
  totalQueries++;
#endif

  // --- Update Hz measurement every second ---
  unsigned long now = millis();
  if (now - lastHzTime >= 1000) {
    samplingRateHz = (float)hzCounter;
    hzCounter = 0;
    lastHzTime = now;
  }

  // --- Forward anything typed in the Serial Monitor to XBee ---
  if (Serial.available()) {
    char outgoing = Serial.read();
    Serial1.print(outgoing);
  }

  // --- Read incoming bytes from XBee ---
  while (Serial1.available()) {
    char c = Serial1.read();
    lastRxTime = millis();

#ifdef BROADCAST_FORMAT
    // In broadcast mode, use ';' as the message delimiter
    if (c == ';') {
      // Include the ';' in the buffer so the parser can verify it
      if (rxIndex < (int)(sizeof(rxBuffer) - 1)) {
        rxBuffer[rxIndex++] = c;
      }
      processMessage();
    } else if (c == '>') {
      // Start of a new message — reset buffer
      rxIndex = 0;
      rxBuffer[rxIndex++] = c;
    } else {
      if (rxIndex < (int)(sizeof(rxBuffer) - 1)) {
        rxBuffer[rxIndex++] = c;
      } else {
        rxIndex = 0;  // overflow — discard
      }
    }
#else
    // Legacy mode: newline-delimited messages
    if (c == '\n' || c == '\r') {
      processMessage();
    } else {
      if (rxIndex < (int)(sizeof(rxBuffer) - 1)) {
        rxBuffer[rxIndex++] = c;
      } else {
        processMessage();
      }
    }
#endif
  }

  // --- Timeout: if we have data in the buffer and no new byte
  //     arrived for RX_TIMEOUT_MS, treat it as a complete message ---
  if (rxIndex > 0 && (millis() - lastRxTime >= RX_TIMEOUT_MS)) {
    processMessage();
  }
}
