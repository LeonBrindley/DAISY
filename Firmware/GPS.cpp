#include "GPS.h"

// Default constructor to initialise member variables to zero.
GPS::GPS() : 
  Initialized(false)
{}

// Function to connect to the SAM-M10Q GPS module using u-blox library.
bool GPS::Initialize() {
  pinMode(D0,OUTPUT);
  digitalWrite(D0,LOW); // Turn on the MOSFET to set V_GATED to VCC.

  int max_attempts = 300; // Only attempt to initialise the GPS module up to 120 times.

  for (int num_attempts = 0; (num_attempts < max_attempts) && (!Initialized); num_attempts++) { // Connect to the SAM-M10Q using I2C.
    Initialized = GNSS.begin(); // Initialized is set high if this is successful.
    delay(50); // Wait for half a second before trying again.
  }

  // GNSS.enableDebugging(); // Uncomment for useful debug messages.
  GNSS.setI2COutput(COM_TYPE_NMEA); // Set the I2C port to receive NMEA (as opposed to UBX) messages.
  GNSS.saveConfiguration(); // Save the current settings to flash and BBR.
    
  return Initialized;
}

bool GPS::IsInitialized() { // Getter for the private member variable 'Initialized'.
  return Initialized;
}

bool GPS::DeInitialize() {
  digitalWrite(D0,HIGH);  // Turn off the MOSFET to set V_GATED to floating.
  Initialized = false;

  return Initialized;
}

const char* GPS::ReadData() {
    static char data[110];
    unsigned long startMillis = millis();
    unsigned long currentMillis;

    while (true) {
        currentMillis = millis();
        if (currentMillis - startMillis > 60000) { // Check if 60 seconds have elapsed.
            sprintf(data, "-1,-1,-1,-1");
            break;
        }

        const char* locationString = ReadLocation();
        // Serial.println(locationString);
        if (strstr(locationString, "-17000") == NULL) { // Check if the altitude does not contain "-17000"
            const char* timeString = ReadTime();
            sprintf(data, "%s,%s", timeString, locationString);
            break;
        }
    }

    Serial.println(data);
    return data;
}



const char* GPS::ReadTime() {
  static char isoTime[25]; // Buffer to hold the time as an ISO-formatted string.

  int day = -1; // Set day to sentinel value of -1 (invalid data).
  int month = -1; // Set month to sentinel value of -1 (invalid data).
  int year = -1; // Set year to sentinel value of -1 (invalid data).
  int hour = -1; // Set hour to sentinel value of -1 (invalid data).
  int minute = -1; // Set minute to sentinel value of -1 (invalid data).
  int second = -1; // Set second to sentinel value of -1 (invalid data).
  int num_attempts = 0; // Counter for the number of attempts.
  int max_attempts = 60; // Only attempt to read the time up to 60 times.

  while ((day <= 0) && (num_attempts < max_attempts)) { // Day cannot be negative, so try again.
    if (GNSS.getPVT() == true) { // Only if valid position/velocity/time is available.
      day = GNSS.getDay();
      month = GNSS.getMonth();
      year = GNSS.getYear();
      hour = GNSS.getHour();
      minute = GNSS.getMinute();
      second = GNSS.getSecond();
      TimeValid = GNSS.getTimeValid();
      DateValid = GNSS.getDateValid();

      // Format the date and time in the ISO 8601 format: https://www.iso.org/iso-8601-date-and-time-format.html
      sprintf(isoTime, "%04d-%02d-%02dT%02d:%02d:%02dZ", year, month, day, hour, minute, second);
      Serial.print(F("TimeValid: "));
      Serial.print(TimeValid);
      Serial.print(F(", DateValid: "));
      Serial.print(DateValid);    
      Serial.println(isoTime);
    } 
    else {
      Serial.println(F("Failed to acquire valid position/velocity/time (PVT)."));
      sprintf(isoTime, "-1"); // If isoTime equals -1, this variable is invalid.
    }

    delay(1000); // Wait for one second before trying again.
    num_attempts += 1; // Increment the number of attempts so far.

  }
  return isoTime;
}

const char* GPS::ReadLocation() {
    static char location[80];
    double latitude, longitude, altitude;
    int pdop;
    int num_attempts = 0;
    int max_attempts = 120;

    while (num_attempts < max_attempts) {
        if (GNSS.getPVT() == true) {
            latitude = GNSS.getLatitude();
            longitude = GNSS.getLongitude();
            altitude = GNSS.getAltitudeMSL();
            pdop = GNSS.getPDOP();
            LLHValid = !GNSS.getInvalidLlh();

            sprintf(location, "%.6f,%.6f,%.2f,%d", latitude, longitude, altitude, pdop);
            if (pdop < 1000 || !LLHValid) {
                break; // Valid data or invalid but not specific wrong altitude.
            }
            // Serial.println(location);
        } else {
            sprintf(location, "-1,-1,-1,-1");
            Serial.println(F("Failed to acquire valid PVT data."));
        }

        delay(500);
        num_attempts++;
    }
    return location;
}