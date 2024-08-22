#include "SDCard.h"

// Constructor
SDCard::SDCard()
    : dirname("/images"),
    Initialized(false) 
{}

bool SDCard::Initialize(int MCU_cs_) {

    // Start I2C bus.
    if (!SD.begin(MCU_cs_)) {
        return false;
    } 

    // Initialise SD card type.
    uint8_t cardType = SD.cardType();
    if (cardType == CARD_NONE) {
        return false;
    } 

    // Store the variable cardSize.
    cardSize = SD.cardSize() / (1024 * 1024);

    Initialized = true;
    return true;
}

bool SDCard::IsInitialized() { // Getter for the variable Initialized.
    return Initialized;
}

const char* SDCard::ChooseName() {
    if (!SD.exists(dirname)) {
        SD.mkdir(dirname);  // Try to create the directory if it doesn't yet exist.
    }

    File root = SD.open(dirname);
    if (!root) {
        Serial.println("Failed to open directory");
        return nullptr;  // Return nullptr if the directory cannot be opened.
    }

    File file = root.openNextFile();
    int highestNumber = -1;

    while (file) {
        if (!file.isDirectory()) {
            const char *fileName = file.name();
            int number = atoi(fileName); // Convert file name to number.
            if (number > highestNumber) {
                highestNumber = number;
            }
        }
        file = root.openNextFile();
    }

    static char newFileName[32];  // Increased size for safety.
    sprintf(newFileName, "%d", highestNumber + 1);

    return newFileName;
}

bool SDCard::SaveImage(camera_fb_t* Image_Data, const char* ImageFileName){


    char path[32];
    sprintf(path, "%s/%s.jpg", dirname, ImageFileName);
    
    // Save the image to the SD card.
    File file = SD.open(path, FILE_WRITE);
    if (!file) {
        Serial.println("[ERROR] Failed to open image file for writing");
        return false; // Return false to indicate failure.
    }

    // Write image data to the file.
    file.write(Image_Data->buf, Image_Data->len);
    file.close();

    Serial.printf("Saved file: %s\n", path);

    return true;
}

bool SDCard::SaveData(camera_fb_t* Image_Data, const char* GPSData, float BatteryVoltage) {

    // Choose a filename (a number higher than previous files).
    const char* ImageFileName = ChooseName();
    if (ImageFileName == nullptr) {
        Serial.println("[ERROR] Failed to get a file name");
        return false; // Return false to indicate failure
    }

    // Save the captured image.
    SaveImage(Image_Data, ImageFileName);

    // Save the accompanying data.
    File file = SD.open(CSV_FILENAME);
    if (!file) {
        file = SD.open(CSV_FILENAME, FILE_WRITE);
        file.println("Image name, ISO Time, Latitude, Longitude, Altitude, PDOP, Battery voltage");
    } else {
        // File exists, so close it and reopen in append mode.
        file.close();
        file = SD.open(CSV_FILENAME, FILE_APPEND);
    }

    // Check if the file has been created.
    if (file) {
        // Write data to the file.
        char buffer[100]; 
        sprintf(buffer, "%s.jpg,%s,%.2f", ImageFileName, GPSData, BatteryVoltage);
        file.println(buffer);
        file.close();
        Serial.println("Data written to CSV.");
    } else {
        Serial.println("[ERROR] Issue opening CSV file.");
        return false;
    }
    return true;
}

 #pragma region base_SDCard_functions
    void SDCard::listDir(uint8_t levels) {
        Serial.printf("Listing directory: %s\n", dirname);

        File root = SD.open(dirname);
        if (!root) {
            Serial.println("Failed to open directory");
            return;
        }
        if (!root.isDirectory()) {
            Serial.println("Not a directory");
            return;
        }

        File file = root.openNextFile();
        while (file) {
            if (file.isDirectory()) {
            Serial.print("  DIR : ");
            Serial.println(file.name());
            if (levels) {
                listDir(levels - 1);
            }
            } else {
            Serial.print("  FILE: ");
            Serial.print(file.name());
            Serial.print("  SIZE: ");
            Serial.println(file.size());
            }
            file = root.openNextFile();
        }
    }

    void SDCard::createDir(const char *path) {
        Serial.printf("Creating Dir: %s\n", path);
        if (SD.mkdir(path)) {
            Serial.println("Dir created");
        } else {
            Serial.println("mkdir failed");
        }
        }

    void SDCard::removeDir(const char *path) {
        Serial.printf("Removing Dir: %s\n", path);
        if (SD.rmdir(path)) {
            Serial.println("Dir removed");
        } else {
            Serial.println("rmdir failed");
        }
        }

    void SDCard::readFile(const char *path) {
        Serial.printf("Reading file: %s\n", path);

        File file = SD.open(path);
        if (!file) {
            Serial.println("Failed to open file for reading");
            return;
        }

        Serial.print("Read from file: ");
        while (file.available()) {
            Serial.write(file.read());
        }
        file.close();
        }

    void SDCard::writeFile(const char *path, const char *message) {
        Serial.printf("Writing file: %s\n", path);

        File file = SD.open(path, FILE_WRITE);
        if (!file) {
            Serial.println("Failed to open file for writing");
            return;
        }
        if (file.print(message)) {
            Serial.println("File written");
        } else {
            Serial.println("Write failed");
        }
        file.close();
        }

    void SDCard::appendFile(const char *path, const char *message) {
        Serial.printf("Appending to file: %s\n", path);

        File file = SD.open(path, FILE_APPEND);
        if (!file) {
            Serial.println("Failed to open file for appending");
            return;
        }
        if (file.print(message)) {
            Serial.println("Message appended");
        } else {
            Serial.println("Append failed");
        }
        file.close();
        }

    void SDCard::renameFile(const char *path1, const char *path2) {
        Serial.printf("Renaming file %s to %s\n", path1, path2);
        if (SD.rename(path1, path2)) {
            Serial.println("File renamed");
        } else {
            Serial.println("Rename failed");
        }
        }

    void SDCard::deleteFile(const char *path) {
        Serial.printf("Deleting file: %s\n", path);
        if (SD.remove(path)) {
            Serial.println("File deleted");
        } else {
            Serial.println("Delete failed");
        }
        }

    void SDCard::testFileIO(const char *path) {
        File file = SD.open(path);
        static uint8_t buf[512];
        size_t len = 0;
        uint32_t start = millis();
        uint32_t end = start;
        if (file) {
            len = file.size();
            size_t flen = len;
            start = millis();
            while (len) {
            size_t toRead = len;
            if (toRead > 512) {
                toRead = 512;
            }
            file.read(buf, toRead);
            len -= toRead;
            }
            end = millis() - start;
            Serial.printf("%u bytes read for %lu ms\n", flen, end);
            file.close();
        } else {
            Serial.println("Failed to open file for reading");
        }

        file = SD.open(path, FILE_WRITE);
        if (!file) {
            Serial.println("Failed to open file for writing");
            return;
        }

        size_t i;
        start = millis();
        for (i = 0; i < 2048; i++) {
            file.write(buf, 512);
        }
        end = millis() - start;
        Serial.printf("%u bytes written for %lu ms\n", 2048 * 512, end);
        file.close();
        }
    #pragma endregion