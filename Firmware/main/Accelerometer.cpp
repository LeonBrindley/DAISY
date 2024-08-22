#include "Accelerometer.h"

#pragma region AccelerometerAddresses
#define LIS2DH12_ADDRESS 18 // Default LIS2DH12 I2C Address: https://www.st.com/resource/en/datasheet/lis2dh12.pdf
#define LIS3DH_ADDRESS 18 // Default LIS3DH I2C Address: https://www.st.com/resource/en/datasheet/lis3dh.pdf
#pragma endregion

// Default constructor to initialise the member variables to zero.
Accelerometer::Accelerometer() :
  XAcceleration(0),
  YAcceleration(0),
  ZAcceleration(0),
  temperature(0)
{}

bool Accelerometer::connectLIS3DH() {
  Serial.println(F("Initialising LIS3DH accelerometer."));
  Serial.println(F("I2C address of LIS3DH accelerometer = 0x18 = 0d24."));
  DaisyLIS3DH.settings.tempEnabled = 0; // Set tempEnabled to 1 if you wish to record the temperature.
  DaisyLIS3DH.settings.accelRange = 2; // Set the full-scale range to be +/- 2g.
  while (DaisyLIS3DH.begin() == false) { // Connect to the LIS3DH using I2C.
    Serial.println(F("Searching for LIS3DH accelerometer at I2C address 0x18."));
    delay (2000);
  }
  return true;
}

bool Accelerometer::readLIS3DH() {
  XAcceleration = DaisyLIS3DH.readFloatAccelX();
  YAcceleration = DaisyLIS3DH.readFloatAccelY();
  ZAcceleration = DaisyLIS3DH.readFloatAccelZ();
  return true;
}

bool Accelerometer::connectLIS2DH12() { // See Example5_LowestPower.ino for the most efficient settings.
  Serial.println(F("Initialising LIS2DH12 accelerometer."));
  Serial.println(F("I2C address of LIS2DH12 accelerometer = 0x18 = 0d24."));
  // DaisyLIS2DH12.disableTemperature(); // Call this function if you wish to disable temperature readings.
  DaisyLIS2DH12.setScale(LIS2DH12_2g); // Set the full-scale range to be +/- 2g.
  DaisyLIS2DH12.setMode(LIS2DH12_HR_12bit); // Set the resolution to be 12 bits.
  DaisyLIS2DH12.setDataRate(LIS2DH12_ODR_1Hz); // Set the output data rate (ODR) to 1 Hz.
  while (DaisyLIS2DH12.begin() == false) { // Connect to the LIS2DH12 using I2C.
    Serial.println(F("Searching for LIS2DH12 accelerometer at I2C address 0x18."));
    delay (2000);
  }
  return true;
}

bool Accelerometer::readLIS2DH12() {
  if(DaisyLIS2DH12.available()) { // Only read the accelerometer if new data is available.
    XAcceleration = DaisyLIS2DH12.getX();
    YAcceleration = DaisyLIS2DH12.getY();
    ZAcceleration = DaisyLIS2DH12.getZ();
    temperature = DaisyLIS2DH12.getTemperature();
  }
  return true;
}

// Function to print the X, Y and Z acceleration to the serial monitor.
bool Accelerometer::displayAccelerometerData() {
  Serial.print(F("[Acceleration] X: "));
  Serial.print(XAcceleration); // Print the x-axis acceleration.
  Serial.print(F(", Y: "));
  Serial.print(YAcceleration); // Print the y-axis acceleration.
  Serial.print(F(", Z: "));
  Serial.print(ZAcceleration); // Print the z-axis acceleration.
  Serial.print(F(" [Temperature] "));
  Serial.print(temperature); // Print the temperature in degrees Celsius.
  Serial.print(F("°C"));
  Serial.println();
  return true;
}

bool Accelerometer::saveAccelerometerData(char path[32], const char *fileName) {
  Serial.println(F("Saving accelerometer data to the SD card."));

  sprintf(path, "/accelerometer/%s.csv", fileName);

  File CSVFile = SD.open(path, FILE_WRITE); // Open the correct CSV file in WRITE mode.
  if (!CSVFile) {
    Serial.println("Failed to open accelerometer CSV file for writing.");
    return false; // Return false if the CSV file cannot be opened.
  }

  // Write the accelerometer data to the CSV file.
  CSVFile.print(XAcceleration);
  CSVFile.print(",");
  CSVFile.print(YAcceleration);
  CSVFile.print(",");
  CSVFile.print(ZAcceleration);
  CSVFile.print(",");
  CSVFile.print(temperature);
  
  // Finally, close the CSV file.
  CSVFile.close();
  return true;
}