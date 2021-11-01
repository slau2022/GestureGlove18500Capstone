#include <Adafruit_ICM20X.h>
#include <Adafruit_ICM20948.h>
#include <Adafruit_Sensor.h>
#include <Wire.h>

Adafruit_ICM20948 icm;
uint16_t measurement_delay_us = 65535; // Delay between measurements for testing
// For SPI mode, we need a CS pin
#define ICM_CS 10
// For software-SPI mode we need SCK/MOSI/MISO pins
#define ICM_SCK 13
#define ICM_MISO 12
#define ICM_MOSI 11

const int flexPin_Thumb = A0; //pin A0 to read analog input
const int flexPin_Index = A1;
const int flexPin_Middle = A2;
const int flexPin_Ring = A3;
const int flexPin_Pinky = A6;

int value_Thumb;
int value_Index;
int value_Middle;
int value_Ring;
int value_Pinky;

const float VCC = 4.98; // Measured voltage of Ardunio 5V line
const float R_DIV = 47500.0; // Measured resistance of 3.3k resistor

// Upload the code, then try to adjust these values to more
// accurately calculate bend degree.
const float STRAIGHT_RESISTANCE = 37300.0; // resistance when straight
const float BEND_RESISTANCE = 90000.0; // resistance at 90 deg


void setup() {
  Serial.begin(115200);
  while (!Serial)
    delay(10); // will pause Zero, Leonardo, etc until serial console opens

  Serial.println("Adafruit ICM20948 test!");

  // Try to initialize!
  if (!icm.begin_I2C()) {
    // if (!icm.begin_SPI(ICM_CS)) {
    // if (!icm.begin_SPI(ICM_CS, ICM_SCK, ICM_MISO, ICM_MOSI)) {

    Serial.println("Failed to find ICM20948 chip");
    while (1) {
      delay(10);
    }
  }
  Serial.println("ICM20948 Found!");
  // icm.setAccelRange(ICM20948_ACCEL_RANGE_16_G);
  Serial.print("Accelerometer range set to: ");
  switch (icm.getAccelRange()) {
  case ICM20948_ACCEL_RANGE_2_G:
    Serial.println("+-2G");
    break;
  case ICM20948_ACCEL_RANGE_4_G:
    Serial.println("+-4G");
    break;
  case ICM20948_ACCEL_RANGE_8_G:
    Serial.println("+-8G");
    break;
  case ICM20948_ACCEL_RANGE_16_G:
    Serial.println("+-16G");
    break;
  }
  Serial.println("OK");

  // icm.setGyroRange(ICM20948_GYRO_RANGE_2000_DPS);
  Serial.print("Gyro range set to: ");
  switch (icm.getGyroRange()) {
  case ICM20948_GYRO_RANGE_250_DPS:
    Serial.println("250 degrees/s");
    break;
  case ICM20948_GYRO_RANGE_500_DPS:
    Serial.println("500 degrees/s");
    break;
  case ICM20948_GYRO_RANGE_1000_DPS:
    Serial.println("1000 degrees/s");
    break;
  case ICM20948_GYRO_RANGE_2000_DPS:
    Serial.println("2000 degrees/s");
    break;
  }

  //  icm.setAccelRateDivisor(4095);
  uint16_t accel_divisor = icm.getAccelRateDivisor();
  float accel_rate = 1125 / (1.0 + accel_divisor);

  Serial.print("Accelerometer data rate divisor set to: ");
  Serial.println(accel_divisor);
  Serial.print("Accelerometer data rate (Hz) is approximately: ");
  Serial.println(accel_rate);

  //  icm.setGyroRateDivisor(255);
  uint8_t gyro_divisor = icm.getGyroRateDivisor();
  float gyro_rate = 1100 / (1.0 + gyro_divisor);

  Serial.print("Gyro data rate divisor set to: ");
  Serial.println(gyro_divisor);
  Serial.print("Gyro data rate (Hz) is approximately: ");
  Serial.println(gyro_rate);

  // icm.setMagDataRate(AK09916_MAG_DATARATE_10_HZ);
  Serial.print("Magnetometer data rate set to: ");
  switch (icm.getMagDataRate()) {
  case AK09916_MAG_DATARATE_SHUTDOWN:
    Serial.println("Shutdown");
    break;
  case AK09916_MAG_DATARATE_SINGLE:
    Serial.println("Single/One shot");
    break;
  case AK09916_MAG_DATARATE_10_HZ:
    Serial.println("10 Hz");
    break;
  case AK09916_MAG_DATARATE_20_HZ:
    Serial.println("20 Hz");
    break;
  case AK09916_MAG_DATARATE_50_HZ:
    Serial.println("50 Hz");
    break;
  case AK09916_MAG_DATARATE_100_HZ:
    Serial.println("100 Hz");
    break;
  }
  Serial.println();
}

void loop() {
  sensors_event_t accel;
  sensors_event_t gyro;
  sensors_event_t mag;
  sensors_event_t temp;
  
  icm.getEvent(&accel, &gyro, &temp, &mag);
  
  value_Thumb = analogRead(flexPin_Thumb);
  value_Index = analogRead(flexPin_Index);
  value_Middle = analogRead(flexPin_Middle);
  value_Ring = analogRead(flexPin_Ring);
  value_Pinky = analogRead(flexPin_Pinky);

  float flexV_Thumb = value_Thumb * VCC / 1023.0;
  float flexR_Thumb = R_DIV * (VCC / flexV_Thumb - 1.0);
  float angle_Thumb = map(flexR_Thumb, STRAIGHT_RESISTANCE, BEND_RESISTANCE,
                   0, 90.0);
  float flexV_Index = value_Index * VCC / 1023.0;
  float flexR_Index = R_DIV * (VCC / flexV_Index - 1.0);
  float angle_Index = map(flexR_Index, STRAIGHT_RESISTANCE, BEND_RESISTANCE,
                   0, 90.0);
  float flexV_Middle = value_Middle * VCC / 1023.0;
  float flexR_Middle = R_DIV * (VCC / flexV_Middle - 1.0);
  float angle_Middle = map(flexR_Middle, STRAIGHT_RESISTANCE, BEND_RESISTANCE,
                   0, 90.0);
  float flexV_Ring = value_Ring * VCC / 1023.0;
  float flexR_Ring = R_DIV * (VCC / flexV_Index - 1.0);
  float angle_Ring = map(flexR_Ring, STRAIGHT_RESISTANCE, BEND_RESISTANCE,
                   0, 90.0);
  float flexV_Pinky = value_Pinky * VCC / 1023.0;
  float flexR_Pinky = R_DIV * (VCC / flexV_Pinky - 1.0);
  float angle_Pinky = map(flexR_Pinky, STRAIGHT_RESISTANCE, BEND_RESISTANCE,
                   0, 90.0);

  float to_send[] = {angle_Thumb, angle_Index, angle_Middle, angle_Ring, angle_Pinky, 
                     accel.acceleration.x, accel.acceleration.y, accel.acceleration.z,
                     mag.magnetic.x, mag.magnetic.y, mag.magnetic.z,
                     gyro.gyro.x, gyro.gyro.y, gyro.gyro.z};

  Serial.println("\nbegin");
  Serial.print("\n");
  String sendstr = "";
  for (int i = 0; i < 14; i++) {
    if (i < 13) {
      sendstr = String(sendstr + String(to_send[i]) + ", ");
    } else {
      sendstr = String(sendstr + String(to_send[i]));
    }
  }
      
  Serial.print(sendstr);
  Serial.print("\nend");
  
  /*Serial.print("\t\tThumb angle: ");          
  Serial.print(angle_Thumb);
  Serial.print("\tIndex angle: ");  
  Serial.print(angle_Index); 
  Serial.print("\tMiddle angle: ");  
  Serial.print(angle_Middle); 
  Serial.print("\tRing angle: ");  
  Serial.print(angle_Ring); 
  Serial.print("\tPinky angle: ");  
  Serial.println(angle_Pinky); 
  Serial.println();

  Serial.print("\t\tAccel X: ");
  Serial.print(accel.acceleration.x);
  Serial.print(" \tY: ");
  Serial.print(accel.acceleration.y);
  Serial.print(" \tZ: ");
  Serial.print(accel.acceleration.z);
  Serial.println(" m/s^2 ");

  Serial.print("\t\tMag X: ");
  Serial.print(mag.magnetic.x);
  Serial.print(" \tY: ");
  Serial.print(mag.magnetic.y);
  Serial.print(" \tZ: ");
  Serial.print(mag.magnetic.z);
  Serial.println(" uT"); */

  /* Display the results (acceleration is measured in m/s^2) */
  /*Serial.print("\t\tGyro X: ");
  Serial.print(gyro.gyro.x);
  Serial.print(" \tY: ");
  Serial.print(gyro.gyro.y);
  Serial.print(" \tZ: ");
  Serial.print(gyro.gyro.z);
  Serial.println(" radians/s ");
  Serial.println(); */

  delay(100);
  // put your main code here, to run repeatedly:

}
