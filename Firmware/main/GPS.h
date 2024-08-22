#pragma once

// Include
#include <Arduino.h>
#include <SparkFun_u-blox_GNSS_v3.h> // https://github.com/sparkfun/SparkFun_u-blox_GNSS_v3

class GPS {
public:

    // Public variables.

    // Public functions.
    GPS();
    bool Initialize();
    bool DeInitialize();
    bool IsInitialized();
    const char* ReadData(); // This function calls both ReadTime and ReadLocation.
    const char* ReadTime();
    const char* ReadLocation();

private:

    // Private variables.
    bool Initialized;
    bool TimeValid, DateValid, LLHValid; // LLH = Longitude, Latitude and Height.
    SFE_UBLOX_GNSS GNSS;

    // Private functions.

};