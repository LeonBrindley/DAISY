#pragma once

// Include
#include "Camera.h"
#include "SDCard.h"
#include "GPS.h"
#include "Accelerometer.h"
#include <Arduino.h>

// Define
// SPI pins
#define MCU_sck_ D8
#define MCU_miso_ D9
#define MCU_mosi_ D10
#define MCU_cs_ 21

class Daisy
{
public:
    // Public variables.

    // Public functions.
    Daisy();
    void Initialize();
    void MainLoop();

private:
    // Private variables.
    Camera Camera_;
    SDCard SDCard_;
    GPS GPS_;
    long int Sleep_Time; // Sleep period in seconds.

    // Private functions.
    void HandleErrors();
    void InitialBlink();
    float MonitorBattery();
};