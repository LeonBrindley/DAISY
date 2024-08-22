#pragma once

// Include
#include <esp_camera.h>
#include "camera_index.h"
#include <Arduino.h>
// #include "ESP32_OV5640_AF.h" // https://github.com/0015/ESP32-OV5640-AF

// Define
#define MAX_FB_SIZE 5800000
#pragma region cameraPins
#define PWDN_GPIO_NUM  -1
#define RESET_GPIO_NUM -1
#define XCLK_GPIO_NUM  10
#define SIOD_GPIO_NUM  40
#define SIOC_GPIO_NUM  39
#define Y9_GPIO_NUM    48
#define Y8_GPIO_NUM    11
#define Y7_GPIO_NUM    12
#define Y6_GPIO_NUM    14
#define Y5_GPIO_NUM    16
#define Y4_GPIO_NUM    18
#define Y3_GPIO_NUM    17
#define Y2_GPIO_NUM    15
#define VSYNC_GPIO_NUM 38
#define HREF_GPIO_NUM  47
#define PCLK_GPIO_NUM  13
#pragma endregion

class Camera {
public:

    // Public variables.
    
    // Public functions.
    Camera();
    bool Initialize();
    bool DeInitialize();
    bool IsInitialized();
    camera_fb_t* ReadImage();
    void ChangeSettings();

private:

    // Private variables.
    camera_config_t Config;
    sensor_t* Camera_Sensors;
    camera_fb_t* Image_Data; // Data structure of camera frame buffer including:
                    // uint8_t * buf;              /*!< Pointer to the pixel data */
                    // size_t len;                 /*!< Length of the buffer in bytes */
                    // size_t width;               /*!< Width of the buffer in pixels */
                    // size_t height;              /*!< Height of the buffer in pixels */
                    // pixformat_t format;         /*!< Format of the pixel data */
                    // struct timeval timestamp;   /*!< Timestamp since boot of the first DMA buffer of the frame */
    bool Initialized;

    // Private functions.

};