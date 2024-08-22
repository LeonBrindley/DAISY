#include "Camera.h"

Camera::Camera()
    : Initialized(false), Image_Data(nullptr), Camera_Sensors(nullptr) {
    Image_Data = new camera_fb_t; 
    Image_Data->buf = new uint8_t[MAX_FB_SIZE]; 
    Image_Data->len = 0;
}

bool Camera::Initialize()
{
    // Activate PWDN pin.
    pinMode(D1,OUTPUT);
    digitalWrite(D1, LOW);
    delay(1000);
    
    // Configure the camera.
    #pragma region CameraConfig
    // Configure fixed attributes for OV-type cameras.
    Config.pin_d0 = Y2_GPIO_NUM;
    Config.pin_d1 = Y3_GPIO_NUM;
    Config.pin_d2 = Y4_GPIO_NUM;
    Config.pin_d3 = Y5_GPIO_NUM;
    Config.pin_d4 = Y6_GPIO_NUM;
    Config.pin_d5 = Y7_GPIO_NUM;
    Config.pin_d6 = Y8_GPIO_NUM;
    Config.pin_d7 = Y9_GPIO_NUM;
    Config.pin_xclk = XCLK_GPIO_NUM;
    Config.pin_pclk = PCLK_GPIO_NUM;
    Config.pin_vsync = VSYNC_GPIO_NUM;
    Config.pin_href = HREF_GPIO_NUM;
    Config.pin_pclk = PCLK_GPIO_NUM;
    Config.pin_sccb_sda = SIOD_GPIO_NUM;
    Config.pin_sccb_scl = SIOC_GPIO_NUM;
    // Config.sccb_i2c_port = -1; // If pin_sccb_sda is -1, use the already configured I2C bus by number.
    // Configure attributes that can be changed for OV-type cameras.
    Config.xclk_freq_hz = 20000000; // Frequency of XCLK signal, in Hz. EXPERIMENTAL: Set to 16MHz on ESP32-S2 or ESP32-S3 to enable EDMA mode.
    Config.ledc_channel = LEDC_CHANNEL_0; // LEDC channel to be used for generating XCLK.
    Config.ledc_timer = LEDC_TIMER_0; // LEDC timer to be used for generating XCLK.
    Config.pixel_format = PIXFORMAT_JPEG;  // The pixel format of the image: PIXFORMAT_ + YUV422|GRAYSCALE|RGB565|JPEG
    Config.frame_size = FRAMESIZE_QSXGA; // The resolution size of the image: FRAMESIZE_ + QVGA|CIF|VGA|SVGA|XGA|SXGA|UXGA
    Config.jpeg_quality = 10; // The quality of the JPEG image, ranging from 0 to 63 (where lower number means higher quality for OV cameras).
    Config.fb_count = 1; // The number of frame buffers to use. If greater than one, then each frame will be acquired (at double speed). When fb_count is greater than one, the driver will work in continuous mode, and camera grab needs to be changed too!
    Config.fb_location = CAMERA_FB_IN_PSRAM; // Set the frame buffer storage location.
    Config.grab_mode = CAMERA_GRAB_WHEN_EMPTY; // CAMERA_GRAB_LATEST; //  The image capture mode.
    
    #pragma endregion
    
    // Make sure that configuration is achieved.
    esp_err_t err = esp_camera_init(&Config);
    if (err != ESP_OK) {
        Serial.print("Camera initialisation failed with error 0x");
        Serial.println(err, HEX);
        return false;
    } 
    
    // Get pointer to the image sensor control structure.
    Camera_Sensors = esp_camera_sensor_get();
    if (Camera_Sensors == nullptr)
    {
        Serial.println("Failed to get sensor control structure");
        return false;
    }

    // Use sensor control sensor structure to decide image settings.
    #pragma region ControlSettings
    // Light (influenced by resolution, acutance, and noise).
    Camera_Sensors->set_brightness(Camera_Sensors, 0);                  // -2 to 2
    Camera_Sensors->set_contrast(Camera_Sensors, 0);                    // -2 to 2
    Camera_Sensors->set_sharpness(Camera_Sensors, 0);                   // -2 to 2
    Camera_Sensors->set_raw_gma(Camera_Sensors, 1);                     // 0 = disable , 1 = enable, when ON there is more light and detail 
    // Exposure
    Camera_Sensors->set_exposure_ctrl(Camera_Sensors, 1);               // 0 = disable , 1 = enable
    Camera_Sensors->set_aec2(Camera_Sensors, 0);                        // 0 = disable , 1 = enable
    Camera_Sensors->set_aec_value(Camera_Sensors, 300);                 // 0 to 1200
    Camera_Sensors->set_ae_level(Camera_Sensors, 0);                    // -2 to 2
    // Gain
    Camera_Sensors->set_gain_ctrl(Camera_Sensors, 1);                   // 0 = disable , 1 = enable
    Camera_Sensors->set_agc_gain(Camera_Sensors, 0);                    // 0 to 30
    Camera_Sensors->set_gainceiling(Camera_Sensors, (gainceiling_t)0);  // 0 to 6
    // Saturation and white balance
    Camera_Sensors->set_saturation(Camera_Sensors, 0);                  // -2 to 2
    Camera_Sensors->set_colorbar(Camera_Sensors, 0);                    // 0 = disable , 1 = enable
    Camera_Sensors->set_whitebal(Camera_Sensors, 1);                    // 0 = disable , 1 = enable
    Camera_Sensors->set_awb_gain(Camera_Sensors, 1);                    // 0 = disable , 1 = enable
    Camera_Sensors->set_wb_mode(Camera_Sensors, 0);                     // 0 to 4, the modes are listed in the sensor.h file
    // Optics
    Camera_Sensors->set_lenc(Camera_Sensors, 0);                        // 0 = disable , 1 = enable, lents correction
    Camera_Sensors->set_hmirror(Camera_Sensors, 0);                     // 0 = disable , 1 = enable, horizontal mirror
    Camera_Sensors->set_vflip(Camera_Sensors, 0);                       // 0 = disable , 1 = enable, vertical mirror
    // Others
    Camera_Sensors->set_dcw(Camera_Sensors, 1);                         // 0 = disable , 1 = enable, Downsize EN
                                                                        //When DCW is on, the image that you receive will be the size that you requested (VGA, QQVGA, etc)
                                                                        //When DCW is off, the image that you receive will be one of UXGA, SVGA, or CIF
    Camera_Sensors->set_special_effect(Camera_Sensors, 0);              // 0 to 6
    Camera_Sensors->set_bpc(Camera_Sensors, 0);                         // 0 = disable , 1 = enable, Black-Point Compensation
    Camera_Sensors->set_wpc(Camera_Sensors, 0);                         // 0 = disable , 1 = enable, White-Point Compensation 
    #pragma endregion

    Initialized = true;

    return Initialized;
}

bool Camera::DeInitialize() {
    
    esp_camera_deinit(); // Deinitialise the camera driver.
    digitalWrite(D1, HIGH); // PWDN pin active.
    Initialized = false;

    return Initialized;
}

bool Camera::IsInitialized() {
    return Initialized;
}

camera_fb_t* Camera::ReadImage() {

    // Capture the image.
    camera_fb_t* fb = esp_camera_fb_get();
    if (!fb) {
        Serial.println("Camera capture failed");
        return nullptr; // Indicate an error by returning nullptr.
    }

    // Copy data from fb to Image_Data.
    Image_Data->len = fb->len;
    Image_Data->width = fb->width;
    Image_Data->height = fb->height;
    Image_Data->format = fb->format;

    // Ensure the buffer is large enough.
    if (fb->len <= MAX_FB_SIZE) {
        memcpy(Image_Data->buf, fb->buf, fb->len);
    } else {
        Serial.println("Buffer overflow while storing image");
        esp_camera_fb_return(fb);
        return nullptr;
    }

    // Return the frame buffer back to the camera driver.
    esp_camera_fb_return(fb);

    return Image_Data;
}

void Camera::ChangeSettings() {
    if (Serial.available() > 0) {
        char input[128]; // Increased buffer size to handle multiple commands.
        Serial.readBytesUntil('\n', input, sizeof(input) - 1);
        input[127] = '\0'; // Ensure the string is null-terminated.

        char* command = strtok(input, ";"); // Split input by semicolon.

        while (command != nullptr) {
            // Trim leading whitespace.
            while (*command == ' ') command++;

            // Process each command.
            if (strncmp(command, "brightness=", 11) == 0) {
                int value = atoi(command + 11);
                if (value >= -2 && value <= 2) {
                    Camera_Sensors->set_brightness(Camera_Sensors, value);
                    Serial.print("Brightness set to ");
                    Serial.println(value);
                } else {
                    Serial.println("Brightness value out of range (-2 to 2)");
                }
            } else if (strncmp(command, "contrast=", 9) == 0) {
                int value = atoi(command + 9);
                if (value >= -2 && value <= 2) {
                    Camera_Sensors->set_contrast(Camera_Sensors, value);
                    Serial.print("Contrast set to ");
                    Serial.println(value);
                } else {
                    Serial.println("Contrast value out of range (-2 to 2)");
                }
            } else if (strncmp(command, "sharpness=", 10) == 0) {
                int value = atoi(command + 10);
                if (value >= -2 && value <= 2) {
                    Camera_Sensors->set_sharpness(Camera_Sensors, value);
                    Serial.print("Sharpness set to ");
                    Serial.println(value);
                } else {
                    Serial.println("Sharpness value out of range (-2 to 2)");
                }
            } else if (strncmp(command, "exposure_ctrl=", 14) == 0) {
                int value = atoi(command + 14);
                if (value == 0 || value == 1) {
                    Camera_Sensors->set_exposure_ctrl(Camera_Sensors, value);
                    Serial.print("Exposure Control set to ");
                    Serial.println(value);
                } else {
                    Serial.println("Exposure Control value out of range (0 or 1)");
                }
            } else if (strncmp(command, "aec2=", 5) == 0) {
                int value = atoi(command + 5);
                if (value == 0 || value == 1) {
                    Camera_Sensors->set_aec2(Camera_Sensors, value);
                    Serial.print("AEC2 set to ");
                    Serial.println(value);
                } else {
                    Serial.println("AEC2 value out of range (0 or 1)");
                }
            } else if (strncmp(command, "aec_value=", 10) == 0) {
                int value = atoi(command + 10);
                if (value >= 0 && value <= 1200) {
                    Camera_Sensors->set_aec_value(Camera_Sensors, value);
                    Serial.print("AEC Value set to ");
                    Serial.println(value);
                } else {
                    Serial.println("AEC Value out of range (0 to 1200)");
                }
            } else if (strncmp(command, "ae_level=", 9) == 0) {
                int value = atoi(command + 9);
                if (value >= -2 && value <= 2) {
                    Camera_Sensors->set_ae_level(Camera_Sensors, value);
                    Serial.print("AE Level set to ");
                    Serial.println(value);
                } else {
                    Serial.println("AE Level value out of range (-2 to 2)");
                }
            } else if (strncmp(command, "gain_ctrl=", 10) == 0) {
                int value = atoi(command + 10);
                if (value == 0 || value == 1) {
                    Camera_Sensors->set_gain_ctrl(Camera_Sensors, value);
                    Serial.print("Gain Control set to ");
                    Serial.println(value);
                } else {
                    Serial.println("Gain Control value out of range (0 or 1)");
                }
            } else if (strncmp(command, "agc_gain=", 9) == 0) {
                int value = atoi(command + 9);
                if (value >= 0 && value <= 30) {
                    Camera_Sensors->set_agc_gain(Camera_Sensors, value);
                    Serial.print("AGC Gain set to ");
                    Serial.println(value);
                } else {
                    Serial.println("AGC Gain value out of range (0 to 30)");
                }
            } else if (strncmp(command, "raw_gma=", 8) == 0) {
                int value = atoi(command + 8);
                if (value == 0 || value == 1) {
                    Camera_Sensors->set_raw_gma(Camera_Sensors, value);
                    Serial.print("RAW GMA set to ");
                    Serial.println(value);
                } else {
                    Serial.println("RAW GMA value out of range (0 or 1)");
                }
            } else if (strncmp(command, "whitebal=", 9) == 0) {
                int value = atoi(command + 9);
                if (value == 0 || value == 1) {
                    Camera_Sensors->set_whitebal(Camera_Sensors, value);
                    Serial.print("White Balance set to ");
                    Serial.println(value);
                } else {
                    Serial.println("White Balance value out of range (0 or 1)");
                }
            } else if (strncmp(command, "awb_gain=", 9) == 0) {
                int value = atoi(command + 9);
                if (value == 0 || value == 1) {
                    Camera_Sensors->set_awb_gain(Camera_Sensors, value);
                    Serial.print("AWB Gain set to ");
                    Serial.println(value);
                } else {
                    Serial.println("AWB Gain value out of range (0 or 1)");
                }
            } else {
                Serial.println("Invalid command");
            }

            command = strtok(nullptr, ";"); // Get the next command.
        }
    }
}