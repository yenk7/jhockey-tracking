#include <SoftwareSerial.h>
SoftwareSerial XBee(A0, A1);

void setup() {
  XBee.begin(115200);
  Serial.begin(115200);

}

void loop() {

  if (Serial.available()) {
    XBee.write(Serial.read());
  }

  if (XBee.available()) {
    Serial.write(XBee.read());
  }
  
}