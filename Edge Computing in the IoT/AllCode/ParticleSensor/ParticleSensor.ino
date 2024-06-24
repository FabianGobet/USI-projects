#include <Arduino.h>
#include <SDS011.h>
#include <ArduinoBLE.h>
#define FREQ_FETCH 5000
                      
#define SERVICE_UUID "19b10001-e8f2-537e-4f6c-d104768a1214"
#define CHAR_UUID "19b10002-e8f2-537e-4f6c-d104768a1210"


BLEService particleservice(SERVICE_UUID);
BLEByteCharacteristic particlecharacteristic(CHAR_UUID, BLERead | BLEWrite);

GuL::SDS011 sds(Serial1);
int analogPin = 13;
int val = 0;
unsigned long fetch = 0;
unsigned long prev_ms = 0;
unsigned long curr_ms = 0;

std::string outputFormat = "PM2.5 (STD) \t= % 6.2f µg/µ3 \n"
                           "PM10 (STD) \t= % 6.2f µg/µ3 \n"
                           "\n";

void setup()
{
  
  Serial.begin(9600);
  fetch = millis();
  Serial1.begin(9600);//, SERIAL_8N1, D13, D14);
  //pinMode(analogPin, INPUT);
  sds.setToPassiveReporting();
  get_data();

  if(!BLE.begin()){
    Serial.println("Failed to begin");
    while(1);
  } 

  BLE.setLocalName("particles");
  particleservice.addCharacteristic(particlecharacteristic);
  BLE.addService(particleservice);
  BLE.advertise();
  Serial.println("Initialized."); 
}



void get_data(){
  val = analogRead(analogPin); 
  Serial.println((byte)map(val,0,999,0,255)); 
}

void loop()
{
  //get_data();
  BLEDevice central = BLE.central();

  if (central){
    Serial.println("Found one central."); 
    while(central.connected()){
      curr_ms = millis();
      if(curr_ms - prev_ms > 1000){
        prev_ms = curr_ms;
        get_data();

        particlecharacteristic.writeValue(byte(map(val, 0, 999, 0, 255)));
      }
    }
  }
  if(millis()-fetch > FREQ_FETCH){
    get_data();
    fetch = millis();
  }
  delay(50);
}