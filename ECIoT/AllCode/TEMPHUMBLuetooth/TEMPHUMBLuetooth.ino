/*
  Code originates from "LED" example of ArduinoBLE library.

  Modified by Tobias Erbacher, 2023
  Modified by Jonas Fischer, 2023
*/

#include <ArduinoBLE.h>
#include <Wire.h>
#include "Adafruit_HTU21DF.h"

BLEService ledService("19b10001-e8f2-537e-4f6c-d104768a1214"); // Bluetooth® Low Energy LED Service

// Bluetooth® Low Energy Characteristics
BLEIntCharacteristic tempCharacteristic("19b10002-e8f2-537e-4f6c-d104768a1211", BLERead | BLENotify);
BLEIntCharacteristic humidityCharacteristic("19b10002-e8f2-537e-4f6c-d104768a1212", BLERead | BLENotify);
BLEBoolCharacteristic co2Characteristic("19b10002-e8f2-537e-4f6c-d104768a1213", BLERead | BLENotify);
BLEIntCharacteristic isConnectedCharacteristic("19b10002-e8f2-537e-4f6c-d104768a1215", BLERead | BLENotify);

const int ledPin = LED_BUILTIN; // pin to use for the LED
Adafruit_HTU21DF htu = Adafruit_HTU21DF();


/************************Hardware Related Macros************************************/
#define MG_PIN                       (A0)     // define which analog input channel you are going to use
#define READ_SAMPLE_INTERVAL         (50)    // define how many samples you are going to take in normal operation
#define READ_SAMPLE_TIMES            (5)     // define the time interval(in milliseconds) between each sample in normal operation
#define DC_GAIN                      (8.5)   // define the DC gain of the amplifier

/**********************Application Related Macros**********************************/
#define ZERO_POINT_VOLTAGE           (0.220) // define the output of the sensor in volts when the concentration of CO2 is 400PPM
#define REACTION_VOLTAGE             (0.030) // define the voltage drop of the sensor when moving the sensor from air into 1000ppm CO2
#define CO2_THRESHOLD 800 // Adjust this value based on your requirements


/*****************************Globals***********************************************/
float CO2Curve[3] = {2.602, ZERO_POINT_VOLTAGE, (REACTION_VOLTAGE / (2.602 - 3))};

long previousMillis; 
float temp, rel_hum;
int co2Percentage; 
bool highCO2;

void setup() {
  Serial.begin(9600);
  while (!Serial);

  // set LED pin to output mode
  pinMode(ledPin, OUTPUT);

  // begin initialization
  if (!BLE.begin()) {
    Serial.println("Starting Bluetooth® Low Energy module failed!");
    while (1);
  }

  BLE.setLocalName("environment");
  BLE.setAdvertisedService(ledService);

  // Add characteristics to the service
  ledService.addCharacteristic(tempCharacteristic);
  ledService.addCharacteristic(humidityCharacteristic);
  ledService.addCharacteristic(co2Characteristic);
  ledService.addCharacteristic(isConnectedCharacteristic);

  BLE.addService(ledService);

previousMillis = millis(); 
temp = 0.0;
rel_hum = 0.0;
co2Percentage = 0;
bool highCO2;

  // Set the initial values for the characteristics
  tempCharacteristic.writeValue(0);
  humidityCharacteristic.writeValue(0);
  co2Characteristic.writeValue(false);
  isConnectedCharacteristic.writeValue(0);


  BLE.advertise();

  Serial.println("BLE Peripheral");

  Serial.println("Temp and CO2 sensor connected");

  if (!htu.begin()) {
    Serial.println("Couldn't find sensor!");
    while (1);
  }
}

void loop() {
  // Listen for Bluetooth® Low Energy peripherals to connect:
  BLEDevice central = BLE.central();
  Serial.println("Listen for central...");
  //BLE.setLocalName("environment");

  //Serial.println("My Name: " + BLE.LocalName() );
  


  // If a central is connected to the peripheral:
  if (central) {

    Serial.print("Connected to central: ");
    Serial.println(central.address());
    Serial.println(central.localName());

    // While the central is still connected to the peripheral:
    while (central.connected()) {

      long currentMillis = millis();
      if (currentMillis - previousMillis > 10000){
        previousMillis = currentMillis;
        getData();

        tempCharacteristic.writeValue((int)temp);
        humidityCharacteristic.writeValue((int)rel_hum);
        co2Characteristic.writeValue(highCO2);

      }


      // Print values to Serial for debugging

      Serial.print("\n");
      bool connected = BLE.connected();
      Serial.println("Connected: " + String(connected));
      uint8_t readConnected = 4;
      isConnectedCharacteristic.readValue(readConnected);
      Serial.println("IsConnected: " + String(readConnected));

      //if (readConnected == 0){
        //BLE.disconnect();
        //isConnectedCharacteristic.writeValue(0);
        //Serial.print(F("Disconnected from central: "));
        //Serial.println(central.address());
      //}

      delay(500);
    }

    // When the central disconnects, print it out:
    Serial.print(F("Disconnected from central: "));
    Serial.println(central.address());

  }
    delay(500);
}

float MGRead(int mg_pin) {
  int i;
  float v = 0;

  for (i = 0; i < READ_SAMPLE_TIMES; i++) {
    v += analogRead(mg_pin);
    delay(READ_SAMPLE_INTERVAL);
  }
  v = (v / READ_SAMPLE_TIMES) * 5 / 1024;
  return v;
}

void getData(){
  temp = htu.readTemperature();
  rel_hum = htu.readHumidity();

  co2Percentage = MGGetPercentage(MGRead(MG_PIN), CO2Curve);
  highCO2 = (co2Percentage != -1) && (co2Percentage > CO2_THRESHOLD);


  Serial.print("Temp: "); Serial.print((int)temp); Serial.print(" C");
  Serial.print("\t\t");
  Serial.print("Humidity: "); Serial.print((int)rel_hum); Serial.println(" %");

  if (highCO2) {
     Serial.print("High");
  } else {
    Serial.println("Normal");
  }
}

int MGGetPercentage(float volts, float *pcurve) {
  if ((volts / DC_GAIN) >= ZERO_POINT_VOLTAGE) {
    return -1;
  } else {
    return pow(10, ((volts / DC_GAIN) - pcurve[1]) / pcurve[2] + pcurve[0]);
  }

}