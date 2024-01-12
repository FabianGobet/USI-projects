#include "TileSetup.h"
#include <ArduinoBLE.h>
#include <String>

#define CS D6
#define DC D5
#define RST D4

#define BACK D0
#define SET D3
#define LEFT D1
#define RIGHT D2


#define DUST_THRESHOLD 240
#define DEBOUNCE_DELAY 50
#define TIME_BETWEEN_ACTION 15000
#define REFRESH_TIME 2500

#define P1C1_UUID "19b10002-e8f2-537e-4f6c-d104768a1214" //read windows status
#define P1C2_UUID "19b10002-e8f2-537e-4f6c-d104768a1215" //write order windows
#define PER1_LOCALNAME "windows"

#define P2C1_UUID "19b10002-e8f2-537e-4f6c-d104768a1211" //read temp
#define P2C2_UUID "19b10002-e8f2-537e-4f6c-d104768a1212" //read humidity
#define P2C3_UUID "19b10002-e8f2-537e-4f6c-d104768a1213" //read toxic
#define PER2_LOCALNAME "environment"

#define P3C1_UUID "19b10002-e8f2-537e-4f6c-d104768a1210" //read particles
#define PER3_LOCALNAME "particles"

BLECharacteristic p1c1,p1c2,p2c1,p2c2,p2c3,p3c1;
BLEDevice per1,per2,per3;
U8G2_SSD1327_WS_128X128_F_4W_HW_SPI u8g2(U8G2_R0, CS, DC, RST);

volatile bool draw_flag = true, searching = false;
bool isOpen = false, isClosed = false, dev1=false, dev2=false, dev3=true;
Tile* current_tile;

volatile unsigned long lastRightPress = 0;
volatile unsigned long lastLeftPress = 0;
volatile unsigned long lastSetPress = 0;
volatile unsigned long lastBackPress = 0;
unsigned long lastaction = 0;
unsigned long fetchTime = 0;


void setup() {
  Serial.begin(115200);
  Serial.println("Serial started.");
  u8g2.begin();
  u8g2.setFont(u8g2_font_7x14_mr);
  pinMode(RIGHT,INPUT);
  pinMode(BACK,INPUT);
  pinMode(SET,INPUT);
  pinMode(LEFT,INPUT);
  attachInterrupt(RIGHT,intRIGHT,RISING);
  attachInterrupt(LEFT,intLEFT,RISING);
  attachInterrupt(SET,intSET,RISING);
  attachInterrupt(BACK,intBACK,RISING);
  setupTilesAndLeaves();
  current_tile = &status;
  BLE.setLocalName("central");
  BLE.begin();
  lastaction = millis();
  fetchTime = millis();
}

void intRIGHT() {
  lastRightPress = millis();
}

void intLEFT() {
  lastLeftPress = millis();
}

void intSET() {
  lastSetPress = millis();
}

void intBACK() {
  lastBackPress = millis();
}

bool allSet(){
  return per1.connected() && per2.connected() && per3.connected();
}

void printInfo(BLEDevice peripheral){
  Serial.print("Address: ");
  Serial.println(peripheral.address());
  if (peripheral.hasLocalName()) {
    Serial.print("Local Name: ");
    Serial.println(peripheral.localName());
  }
}

void connect(){
  BLEDevice peripheral = BLE.available();
  if(peripheral){
    printInfo(peripheral);
    if(!per1.connected()){
      if(peripheral.localName().equals(String(PER1_LOCALNAME))){
        if(!tryConnect(peripheral)) return;
        peripheral.discoverAttributes();
        per1 = peripheral;
        p1c1 = per1.characteristic(P1C1_UUID);
        p1c2 = per1.characteristic(P1C2_UUID);
        Serial.println("Connected to 1.");
      }
    }

    if(!per2.connected()){
      if(peripheral.localName().equals(String(PER2_LOCALNAME))){
        if(!tryConnect(peripheral)) return;
        peripheral.discoverAttributes();
        per2 = peripheral;
        p2c1 = per2.characteristic(P2C1_UUID);
        p2c2 = per2.characteristic(P2C2_UUID);
        p2c3 = per2.characteristic(P2C3_UUID);
        Serial.println("Connected to 2.");
      }
    }

    if(!per3.connected()){
      if(peripheral.localName().equals(String(PER3_LOCALNAME))){
        if(!tryConnect(peripheral)) return;
        peripheral.discoverAttributes();
        per3 = peripheral;
        p3c1 = per3.characteristic(P3C1_UUID);
        Serial.println("Connected to 3.");
      }
    }
  }

  if(allSet() && searching){
    BLE.stopScan();
    searching = false;
    Serial.println("All devices connected, turning off scan.");
  }
}

void getData(){
  switchdevice();
  if(per1.connected() && dev1){
    uint8_t windowsState = 0;
    p1c1.readValue(windowsState);
    Serial.println("Read values for 1 -> windows:"+String(windowsState));
    switch (byte(windowsState)){
      case 0xA:
        windowsValues[0] = 0;
        windowsValues[1] = 0;
        break;
      case 0xB:
        windowsValues[0] = 1;
        windowsValues[1] = 1;
        break;
      case 0xC:
        windowsValues[0] = 1;
        windowsValues[1] = 0;
        break;
      case 0xD:
        windowsValues[0] = 1;
        windowsValues[1] = 0;
        break;
    }
  }
  if(per3.connected() && dev2){
    byte particles = byte(outsideValues[2]);
    p3c1.readValue(particles);
    outsideValues[2] = particles;
    Serial.println("Read values for 3 -> particles:"+String(particles));
  }
  if(per2.connected() && dev3){
    uint8_t temperature=insideValues[0], humidity=insidealues[1], toxic=insideValues[2];
    p2c1.readValue(temperature);
    p2c2.readValue(humidity);
    p2c3.readValue(toxic);
    insideValues[0]=temperature;
    insideValues[1]=humidity;
    insideValues[2]=toxic;
    Serial.println("Read values for 2 -> temp:"+String(temperature)+", hum:"+String(humidity)+", toxic:"+String(toxic));
  }
}

void decisions(){
  if(auto_wind==1){
    if(insideValues[2]==1 && toxic_vent==1){
      isOpen=true;
    }
    else if(outsideValues[2]<DUST_THRESHOLD){
      if((insideValues[0]<min_temp && outsideValues[0]>min_temp) || (insideValues[0]>max_temp && outsideValues[0]<max_temp)){
        isOpen=true;
      } else if(insideValues[0]>=min_temp && insideValues[0]<=max_temp){
        if((insideValues[1]<min_hum && outsideValues[1]>min_hum) || (insideValues[1]>max_hum && outsideValues[1]<max_hum)){
          isOpen=true;
        } else if(insideValues[1]>=min_hum && insideValues[1]<=max_hum){
          return;
        } else {
          isClosed=true;
        }
      } else
        isClosed=true;
    }
  }
}

void openWindows(){
  if((millis() - lastaction > TIME_BETWEEN_ACTION) && per1.connected()){
    //uint8_t buff = 0;
    //p1c2.readValue(buff);
    //Serial.println(buff);
    p1c2.writeValue(byte(0x06));
    //p1c2.readValue(buff);
    //Serial.println(buff);
    Serial.println("Ordered open windows.");
    lastaction = millis();
    isOpen=false;
  } 
}

bool tryConnect(BLEDevice device){
  int i=0;
  while(i<15){
    if(device.connect())
      return true;
    i++;
    delay(10);
  }
  return false;
}

void closeWindows(){
  if((millis() - lastaction > TIME_BETWEEN_ACTION) && per1.connected()){
    p1c2.writeValue(byte(0x07));
    Serial.println("Ordered close windows.");
    lastaction = millis();
    isClosed=false;
  }
}


void handleButtonPresses() {
  unsigned long currentMillis = millis();

  if (lastRightPress != 0 && (currentMillis - lastRightPress) > DEBOUNCE_DELAY) {
    current_tile = current_tile->right();
    lastRightPress = 0; 
  }

  if (lastLeftPress != 0 && (currentMillis - lastLeftPress) > DEBOUNCE_DELAY) {
    current_tile = current_tile->left();
    lastLeftPress = 0;
  }

  if (lastSetPress != 0 && (currentMillis - lastSetPress) > DEBOUNCE_DELAY) {
    current_tile = current_tile->set();
    lastSetPress = 0; 
  }

  if (lastBackPress != 0 && (currentMillis - lastBackPress) > DEBOUNCE_DELAY) {
    current_tile = current_tile->back();
    lastBackPress = 0;
  }

  draw_flag = true;
}

void fnDraw(){
  u8g2.firstPage();
  do {
    current_tile->drawStuff();
  } while (u8g2.nextPage() );
  draw_flag = false;
}

void switchdevice(){
  if(dev1){
    dev1 = false;
    dev2 = true;
  } else if(dev2){
    dev2 = false;
    dev3 = true;
  } else if(dev3){
    dev3 = false;
    dev1 = true;
  }
}

void loop(void) {
  handleButtonPresses();
  if(draw_flag) fnDraw();
  if(!allSet()){ 
    if(!searching){
      Serial.println("At least one device not connected, initializing search.");
      BLE.scan();
      searching=true;
    }
    //Serial.println("connecting");
    connect();
  }
  if(millis() - fetchTime > REFRESH_TIME){
    getData();
    fetchTime = millis();
  }
  decisions();
  if(isOpen) openWindows();
  else if(isClosed) closeWindows();
  delay(50);
}