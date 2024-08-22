#pragma once

#include "SparkFunLIS3DH.h" // https://github.com/sparkfun/SparkFun_LIS3DH_Arduino_Library
#include "SparkFun_LIS2DH12.h" // https://github.com/sparkfun/SparkFun_LIS2DH12_Arduino_Library
#include "SD.h"
#include "SPI.h"

class Accelerometer {
	
  public:

  // Default constructor.
  Accelerometer();

  // Accelerometer functions.
  bool connectLIS3DH();
  bool readLIS3DH();
  bool connectLIS2DH12();
  bool readLIS2DH12(); 
  bool displayAccelerometerData();
  bool saveAccelerometerData(char path[32], const char *fileName);

  private:

  // Accelerometer variables.
  LIS3DH DaisyLIS3DH; // https://www.st.com/resource/en/datasheet/lis3dh.pdf
  SPARKFUN_LIS2DH12 DaisyLIS2DH12; // https://www.st.com/resource/en/datasheet/lis2dh12.pdf
  
  float XAcceleration, YAcceleration, ZAcceleration, temperature;

};