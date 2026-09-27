#include <Arduino.h>
#include <camera.h>
#include <himax.h> 
#include "hm0360.h"
// Required for the Portenta Vision Shield camera sensor

// Instantiate the Himax image sensor (Vision Shield Rev.1)
HM0360 himax;

// Pass the sensor instance into the Camera constructor
Camera camera(himax);

void setup() {
  Serial.begin(115200);
  while (!Serial);

  if (!camera.begin(CAMERA_R320x240, CAMERA_GRAYSCALE, 30)) {
    Serial.println("Camera initialization failed");
    while (true) {
    }
  }

  Serial.println("Vision Shield camera initialized successfully");
}

void loop() {
  // Capture/process image using the Portenta camera API.
}