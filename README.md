# DAISY

Welcome to the Git repository for the Sensor CDT's 2024 Team Challenge: Automated Vegetation Monitoring Using Animal-Mounted Sensors.

## Hardware

You can find out about each component used in D**AI**SY via the links below.

- [Seeed Studio XIAO ESP32S3 Sense + Expansion Board](https://wiki.seeedstudio.com/xiao_esp32s3_getting_started)
- [OmniVision OV5640 CMOS Image Sensor](https://cdn.sparkfun.com/datasheets/Sensors/LightImaging/OV5640_datasheet.pdf)
- [SanDisk 32GB High-Endurance microSD Card](https://documents.westerndigital.com/content/dam/doc-library/en_us/assets/public/sandisk/product/memory-cards/high-endurance-uhs-i-microsd/data-sheet-high-endurance-uhs-i-microsd.pdf)
- [JLCPCB Flexible Printed Circuit Board](https://jlcpcb.com/capabilities/flex-pcb-capabilities)
- [RS PRO 18650 Lithium-Ion Rechargeable Batteries](https://docs.rs-online.com/9c01/A700000008874176.pdf)
- [Kvikk Durable Collar for Cattle](https://www.shearwell.co.uk/collars-for-cattle)

## Arduino Libraries

To compile our code, please include the following libraries through the Arduino IDE's Library Manager. Furthermore, the board should be set to **XIAO_ESP32S3**, the correct COM port must be selected and [Espressif's ESP32 board package](https://raw.githubusercontent.com/espressif/arduino-esp32/gh-pages/package_esp32_index.json) should be installed under File->Preferences.

- [SD](https://www.arduino.cc/reference/en/libraries/sd)
- [OV5640 Auto Focus for ESP32 Camera](https://www.arduino.cc/reference/en/libraries/ov5640-auto-focus-for-esp32-camera)
- [SparkFun LIS2DH12 Arduino Library](https://www.arduino.cc/reference/en/libraries/sparkfun-lis2dh12-arduino-library)
- [SparkFun LIS3DH Arduino Library](https://www.arduino.cc/reference/en/libraries/sparkfun-lis3dh-arduino-library)
- [SparkFun u-blox GNSS v3](https://www.arduino.cc/reference/en/libraries/sparkfun-u-blox-gnss-v3)

## Dataset

All dataset-related scripts and data can be found in the `Dataset` directory.

The raw JPEG files of the image fragments used in the dataset can be downloaded [here](https://sensor-cdt-group-project.s3.eu-north-1.amazonaws.com/data.zip).

## ML Models

The ML model implementation, experiment scripts, notebooks and HPC configurations can be found in the `Model` directory.

You can download the weights (`model.pth` file) for our best ML models using the links below.

* Best individual models: [iNaturalist](https://sensor-cdt-group-project.s3.eu-north-1.amazonaws.com/best_models/best_indiv/f1/inaturalist_uf5_pFalse_dp0.1_wd1e-05_lr0.0001/model.pth), [DenseNet121](https://sensor-cdt-group-project.s3.eu-north-1.amazonaws.com/best_models/best_indiv/keras_acc/densenet121_uf7_pTrue_dp0.3_wd1e-05_lr0.0001/model.pth)
* [Best ensemble model](https://sensor-cdt-group-project.s3.eu-north-1.amazonaws.com/best_models/best_ensemble/f1/ensemble_nm5_s2_dp0.3_wd0.01_lr0.001/model.pth)

## More Information

To find out more about this project, please visit our website ([daisysensing.com](https://daisysensing.com)) or Instagram profile ([instagram.com/daisysensing](https://www.instagram.com/daisysensing)).
