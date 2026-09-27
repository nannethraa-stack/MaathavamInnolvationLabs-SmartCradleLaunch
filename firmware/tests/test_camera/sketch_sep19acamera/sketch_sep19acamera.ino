/*
  Portenta H7 Lite Connected + Portenta Vision Shield Rev.2
  HM0360 camera frame-grab diagnostic

  Arduino IDE board:
  Tools -> Board -> Arduino Mbed OS Portenta Boards -> Portenta H7 (M7 core)

  Serial Monitor:
  115200 baud
*/

#include <Arduino.h>
#include <mbed.h>       // Must be before hm0360.h
//#include <camera.h>
//#include <hm0360.h>

#include "arducam_dvp.h"

#define ARDUCAM_CAMERA_HM0360

#ifdef ARDUCAM_CAMERA_HM01B0
    #include "Himax_HM01B0/himax.h"
    HM01B0 himax;
    Camera cam(himax);
    #define IMAGE_MODE CAMERA_GRAYSCALE
#elif defined(ARDUCAM_CAMERA_HM0360)
    #include "Himax_HM0360/hm0360.h"
    HM0360 himax;
    Camera cam(himax);
    #define IMAGE_MODE CAMERA_GRAYSCALE
#elif defined(ARDUCAM_CAMERA_OV767X)
    #include "OV7670/ov767x.h"
    // OV7670 ov767x;
    OV7675 ov767x;
    Camera cam(ov767x);
    #define IMAGE_MODE CAMERA_RGB565
#elif defined(ARDUCAM_CAMERA_GC2145)
    #include "GC2145/gc2145.h"
    GC2145 galaxyCore;
    Camera cam(galaxyCore);
    #define IMAGE_MODE CAMERA_RGB565
#endif

//HM0360 himax;
//Camera cam(himax);

const int FRAME_WIDTH = 320;
const int FRAME_HEIGHT = 240;

FrameBuffer fb(FRAME_WIDTH, FRAME_HEIGHT, 2);

unsigned long frameNumber = 0;

void blinkRedForever() {
  while (true) {
    digitalWrite(LEDR, LOW);
    delay(250);
    digitalWrite(LEDR, HIGH);
    delay(250);
  }
}

void setup() {
  pinMode(LEDR, OUTPUT);
  pinMode(LEDG, OUTPUT);
  pinMode(LEDB, OUTPUT);

  digitalWrite(LEDR, HIGH);
  digitalWrite(LEDG, HIGH);
  digitalWrite(LEDB, HIGH);

  Serial.begin(921600);

 unsigned long startTime = millis();
 while (!Serial && millis() - startTime < 5000) {
    delay(10);
  }

  Serial.println();
  Serial.println("========================================");
  Serial.println("Portenta H7 + Vision Shield Rev.2 test");
  Serial.println("HM0360 camera initialization starting");
  himax.debug(Serial);
  cam.debug(Serial);
  int cameraStatus = cam.begin(CAMERA_R320x240, CAMERA_GRAYSCALE, 30);
  //int cameraStatus = cam.begin();

  /*if (cameraStatus != 0) {
    Serial.print("CAMERA INIT FAILED. Error code: ");
    Serial.println(cameraStatus);
    blinkRedForever();
  }*/

  Serial.println("CAMERA INIT SUCCESS");
  Serial.println("Expected grayscale frame size: 76800 bytes");

  digitalWrite(LEDG, LOW);
}

void loop() {
  int frameStatus = cam.grabFrame(fb, 3000);

  if (frameStatus == 0) {
    frameNumber++;

    uint8_t *image = fb.getBuffer();

    Serial.print("FRAME OK #");
    Serial.print(frameNumber);
    Serial.print(" | Size: ");
    Serial.print(cam.frameSize());
    Serial.print(" bytes | First 8 pixels: ");

    for (int i = 0; i < 8; i++) {
      Serial.print(image[i]);

      if (i < 7) {
        Serial.print(", ");
      }
    }

    Serial.println();

    digitalWrite(LEDB, LOW);
    delay(50);
    digitalWrite(LEDB, HIGH);

  } else {
    Serial.print("FRAME GRAB FAILED. Error code: ");
    Serial.println(frameStatus);

    digitalWrite(LEDR, LOW);
    delay(150);
    digitalWrite(LEDR, HIGH);
  }

  delay(1000);
}