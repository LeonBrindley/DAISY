#include "Daisy.h"

// Constructor.
Daisy::Daisy()
    : Camera_(),
      SDCard_(),
      GPS_(),
      Sleep_Time(300) 
{}

void Daisy::Initialize()
{
    // Serial initialisation.
    Serial.begin(115200);
    if(Serial)
        delay(5000);

    Serial.println(" ______       ____   .-./`)   .-'''-.   ____     __ ");        
    Serial.println("|    _ `''. .'  __ `\\. .-.') / _     \\  \\   \\   /  /");        
    Serial.println("| _ | ) _  /   '  \\  / `-' \\(`' )/`--'   \\  _. /  ' ");        
    Serial.println("|( ''_'  ) |___|  /  |`-'`\"(_ o _).       _( )_ .'  ");        
    Serial.println("| . (_) `. |  _.-`   |.---. (_,_). '. ___(_ o _)'   ");        
    Serial.println("|(_    ._) .'   _    ||   |.---.  \\  |   |(_,_)'    ");        
    Serial.println("|  (_.\\.' /|  _( )_  ||   |\\    `-'  |   `-'  /     ");        
    Serial.println("|       .' \\ (_ o _) /|   | \\       / \\      /      ");        
    Serial.println("'-----'`    '.(_,_).' '---'  `-...-'   `-..-'       "); 

    // Pin initialisation.
    pinMode(GPIO_NUM_21,OUTPUT);  
    pinMode(A2, INPUT);
    InitialBlink();

    // Initialise buses.
    SPI.begin(MCU_sck_, MCU_miso_, MCU_mosi_, MCU_cs_);
    Wire.begin();

    // Error detection
    // HandleErrors();
                                                      
}

void Daisy::MainLoop()
{
    digitalWrite(GPIO_NUM_21, LOW);

    // Module initialisation.
    const char* GPSData;

    Serial.println("[INFO] Initialising GPS:");
    if (GPS_.Initialize()) //
    {
        Serial.println(" [INFO] Recording GPS data...");
        GPSData = GPS_.ReadData();
    } else {
        Serial.println(" [ERROR] GPS module not detected...");
        GPSData = "-1,-1,-1,-1,-1";
    }
        
    GPS_.DeInitialize();

    if (!Camera_.Initialize())
    {
        Serial.println("[ERROR] Camera module not detected...");
        delay(1000);
        esp_deep_sleep(5 * 60 * 1000000); // Sleep for five minutes and then retry.
    }

    if (!SDCard_.Initialize(MCU_cs_))
    {
        Serial.println("[ERROR] SD card module not detected...");
        delay(1000);
        esp_deep_sleep(5 * 60 * 1000000); // Sleep for five minutes and then retry.
    }

    // Extract the hour from the GPS data.
    int hour = -1;
    sscanf(GPSData, "%*d-%*d-%*dT%d", &hour); // This ignores the date part and reads the hour.

    // Take and record data.
    SDCard_.SaveData(Camera_.ReadImage(), GPSData, MonitorBattery());

    // Deinitialise devices.
    delay(1000);
    Camera_.DeInitialize();
    digitalWrite(GPIO_NUM_21, HIGH);

    // If current hour is after 20:00 (8 PM), sleep until 08:00 AM.
    if (hour >= 20 || hour < 8) {
        // unsigned long sleepTime = 0.5; // Hours until 8 AM SAFETY FEATURE. JUST RECHECK EACH HALF an hour (32 - hour) % 24
        // esp_deep_sleep(sleepTime * 3600 * 1000000); // Convert hours to microseconds.
        Serial.println("[DEBUG] Sleeping at night");
        delay(1000);
        // esp_deep_sleep(Sleep_Time * 1000000); // Regular sleep time for debug.
    } else {
        Serial.println("[DEBUG] Day loop");
        esp_deep_sleep(Sleep_Time * 1000000); // Regular sleep time.
    }
}

void Daisy::HandleErrors()
{

}

float Daisy::MonitorBattery()
{
    uint32_t Vbatt = 0;
    for(int i = 0; i < 16; i++){
        Vbatt = Vbatt + analogReadMilliVolts(A2); // ADC with correction.
    }
    float Vbattf = 2 * Vbatt / 16 / 1000.0;     // Attenuation ratio 1/2, mV --> V.
    return Vbattf;
}

void Daisy::InitialBlink()
{
    digitalWrite(GPIO_NUM_21, LOW);
    delay(500);
    digitalWrite(GPIO_NUM_21, HIGH);
}