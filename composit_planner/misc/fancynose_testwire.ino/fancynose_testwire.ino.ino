/////////////////////
// I2C Scanner
// Written by Nick Gammon
// Date: 20th April 2011

#include <Wire.h>

void setup() {
  Serial.begin (115200);
  uint8_t address = 0x52;

  // Leonardo: wait for serial port to connect
  while (!Serial) {}
  Wire.begin();
  //stop_furnace_and_pump();
  start_furnace_and_pump();
  set_sensor_tension();

  Wire.end();
  delay(1500);

}
void loop() {
  // read the system status
  read_status();
  read_sensor_resistance();
  delay(250);
}

unsigned char read_status()
{
  uint8_t address;
  uint8_t nBytes;
  uint8_t ReadData[4];

  address = 0x52; // 52*2 = A4
  nBytes = 4;
  // errMsg = I2C_Read( address, nBytes, ReadData, SendStop );
  Wire.requestFrom(address, nBytes);

  int i = 0;
  while (Wire.available()) {
    ReadData[i] = Wire.read(); // receive a byte as character
    ++i;
  }
  Serial.print("Status Code: "); Serial.println(int(ReadData[0]));
  Serial.print("Temperature: "); Serial.println(int(ReadData[1]) / 2.0);
  Serial.print("Pump Voltage: "); Serial.println(int(ReadData[2]) / 20.0);
  Serial.print("Hydrometry: "); Serial.println(int(ReadData[3]) / 2.0);
}

unsigned char read_sensor_resistance()
{
  uint8_t address;
  uint8_t nBytes;
  uint8_t WriteData[1];
  uint8_t ReadData[6];
  int SensorReadings[3];

  address = 0x57; // 57*2 = AE
  nBytes = 1;
  WriteData[0] = 0x00;
  Wire.beginTransmission (address);
  Wire.write(WriteData, nBytes);
  if (Wire.endTransmission () == 0) {} // end of good response
  else {
    Serial.print("Read sensor resistance (write) exited with error message");
  }

  nBytes = 6;
  Wire.requestFrom(address, nBytes);
  int i = 0;
  while (Wire.available()) {
    ReadData[i] = Wire.read(); // receive a byte as character
    ++i;
  }
  if (i != nBytes) {
    Serial.print("Read less then 6 bytes:");
    Serial.println(i);
  }

  SensorReadings[0] = (ReadData[0] << 8) | ReadData[1];
  SensorReadings[1] = (ReadData[2] << 8) | ReadData[3];
  SensorReadings[2] = (ReadData[4] << 8) | ReadData[5];
  Serial.print("Sensor 1 resistance in Ohms: "); Serial.println(SensorReadings[0]);
  Serial.print("Sensor 2 resistance in Ohms: "); Serial.println(SensorReadings[1]);
  Serial.print("Sensor 3 resistance in Ohms: "); Serial.println(SensorReadings[2]);
//  Serial.print(SensorReadings[0]);
//  Serial.print("\t ");
//  Serial.print(SensorReadings[1]);
//  Serial.print("\t ");
//  Serial.println(SensorReadings[2]);  
  return 0;
}

unsigned char start_furnace_and_pump()
{
  uint8_t address;
  uint8_t nBytes;
  uint8_t WriteData[2];

  address = 0x52; // 52*2 = A4
  nBytes = 2;
  WriteData[0] = 80;
  Serial.println("Setting furance temperature setpoint to 80 (40 C)...");
  WriteData[1] = 100;
  Serial.println("Setting pump voltage to 100 (5 V)...");
  Wire.beginTransmission (address);
  Wire.write(WriteData, nBytes);
  if (Wire.endTransmission () == 0) {} // end of good response
  return 0;
}

unsigned char set_sensor_tension()
{
  uint8_t address;
  uint8_t nBytes;
  uint8_t WriteData[4];

  address = 0x57; // 57*2 = AE
  nBytes = 4;
  WriteData[0] = 0x06;
  WriteData[1] = 86;
  Serial.println("Setting sensor 1 tension to 86 (2.39 V)...");
  WriteData[2] = 244;
  Serial.println("Setting sensor 2 tension to 244 (5.1 V)...");
  WriteData[3] = 244;
  Serial.println("Setting sensor 3 tension to 244 (5.1 V)...");

  Wire.beginTransmission (address);
  Wire.write(WriteData, nBytes);
  if (Wire.endTransmission () == 0) {} // end of good response
  return 0;
}

unsigned char stop_furnace_and_pump()
{
  uint8_t address;
  uint8_t nBytes;
  uint8_t WriteData[2];

  address = 0x52; // 52*2 = A4
  nBytes = 2;
  WriteData[0] = 00;
  Serial.println("Setting furance temperature setpoint to 0 (0 C)...");
  WriteData[1] = 0;
  Serial.println("Setting pump voltage to 0 (0 V)...");
  Wire.beginTransmission (address);
  Wire.write(WriteData, nBytes);
  if (Wire.endTransmission () == 0) {} // end of good response
  return 0;
}
