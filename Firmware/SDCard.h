#pragma once

// Include
#include <SD.h>
#include <SPI.h>
#include <esp_camera.h>
#include <Arduino.h>

// Define
#define CSV_FILENAME "/data.txt"

class SDCard {
public:

    // Public variables.

    // Public functions.
    SDCard();
    bool Initialize(int MCU_cs_);
    bool IsInitialized();
	bool SaveImage(camera_fb_t* Image_Data, const char* ImageFileName);
    bool SaveData(camera_fb_t* Image_Data, const char* GPSData, float BatteryVoltage);

private:

    // Private variables.
    uint64_t cardSize;
    const char* dirname;
    bool Initialized;

    // Private functions.

    const char* ChooseName();

    #pragma region base_SDCard_functions
    void listDir(uint8_t levels);
    void createDir(const char *path);
    void removeDir(const char *path);
    void readFile(const char *path);
    void writeFile(const char *path, const char *message);
    void appendFile(const char *path, const char *message);
    void renameFile(const char *path1, const char *path2);
    void deleteFile(const char *path);
    void testFileIO(const char *path);
    #pragma endregion

};