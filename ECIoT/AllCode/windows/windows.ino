/*
  Code originates from "LED" example of ArduinoBLE library.

  Modified by Tobias Erbacher, 2023
*/

#include <ArduinoBLE.h>
#include <AccelStepper.h>

// Sensors (both use same code): https://wiki.dfrobot.com/Digital_magnetic_sensor_SKU__DFR0033
// Note: Sensor 2 reqiures the magnet to be very close and only works with one orientation (south pole apparently)
// https://wiki.seeedstudio.com/Grove-Hall_Sensor/

BLEService ledService("19B10000-E8F2-537E-4F6C-D104768A1214"); // Bluetooth® Low Energy LED Service

// Bluetooth® Low Energy LED Switch Characteristic - custom 128-bit UUID, read and writable by central
BLEByteCharacteristic switchCharacteristic("19B10002-E8F2-537E-4F6C-D104768A1215", BLERead | BLEWrite); // Listen for instructions from central
BLEByteCharacteristic switchCharacteristic2("19B10002-E8F2-537E-4F6C-D104768A1214", BLERead | BLEWrite); // Stores the Window State to be read by central

const int ledPin = LED_BUILTIN; // pin to use for the LED

// Initialize with all Windows closed (i.e. all magnets present)!
int inputPin1 = 14;
int inputPin2 = 13;
int WindowValue1 = 0; // Open = 1, Closed = 0
int WindowValue2 = 0; // Open = 1, Closed = 0
int isWindowOpen1 = 0; // True = 1, False = 0
int isWindowOpen2 = 0; // True = 1, False = 0
int turn1 = 512;
int turn2 = 0;
int checkerVal1 = 0;
int checkerVal2 = 0;
int checkConnect;

#define FULLSTEP 4
AccelStepper stepper1(FULLSTEP, 6, 8, 7, 9);
AccelStepper stepper2(FULLSTEP, 3, 5, 2, 4);

void setup() {
  Serial.begin(9600);
  while (!Serial);

  // set LED pin to output mode
  pinMode(ledPin, OUTPUT);

  // begin initialization
  if (!BLE.begin()) {
    Serial.println("starting Bluetooth® Low Energy module failed!");

    //while (1);
  }

  // set advertised local name and service UUID:
  //BLE.setLocalName("Arduino Window Opener");
  BLE.setLocalName("windows");
  BLE.setAdvertisedService(ledService);

  // add the characteristic to the service
  ledService.addCharacteristic(switchCharacteristic);
  ledService.addCharacteristic(switchCharacteristic2);

  // add service
  BLE.addService(ledService);

  // set the initial value for the characeristic:
  switchCharacteristic.writeValue(0);
  switchCharacteristic2.writeValue(0);

  Serial.println("Initiating BLE");

  // start advertising
  BLE.advertise();

  //Serial.println("BLE Peripheral");

  pinMode(inputPin1, INPUT);
  pinMode(inputPin2, INPUT);

  stepper1.setMaxSpeed(400.0);
	stepper1.setAcceleration(50.0);

  stepper2.setMaxSpeed(400.0);
	stepper2.setAcceleration(50.0);
}

void loop() { // Initialize this in both windows closed state
  // listen for Bluetooth® Low Energy peripherals to connect:
  BLEDevice central = BLE.central();

  // if a central is connected to peripheral:
  if (central) {
    Serial.print("Connected to central: ");
    Serial.print(central.localName());
    Serial.print(" (");
    Serial.print(central.address());
    Serial.print(")\n");
    

    // while the central is still connected to peripheral:
    while (central.connected()) {
        // The definition here is inverted between the two windows because sensor 1 gives 1 if magnet is present and sensor 2 gives 0 if magnet is present.
        isWindowOpen1 = 1 - digitalRead(inputPin1);
        isWindowOpen2 = digitalRead(inputPin2);
        checkerfunction(isWindowOpen1, isWindowOpen2);

        if (switchCharacteristic.value() == 0) {        // Deactivate LED
          digitalWrite(ledPin, HIGH);

        } else if (switchCharacteristic.value() == 1) { // Activate LED / listen for the disconnected flag from central, disconnect if 1.
          digitalWrite(ledPin, LOW);
          //BLE.disconnect();

        } else if (switchCharacteristic.value() == 2) { // Open Window 1
          if (checkerVal1 == 0) {
            if (isWindowOpen1 == 0) {
              WindowValue1 = 0;
            } else {
              WindowValue1 = 1;
            }
            checkerVal1 = 1;
          } else if (checkerVal1 == 1) {
            if (WindowValue1 == 0) {
              stepper1.moveTo(turn1);
              stepper1.run();
            } else if (stepper1.distanceToGo() == 0) {
              Serial.println("Window 1 Has Been Opened.");
              WindowValue1 = 1;
              checkerVal1 = 0;
            } else {
              Serial.println("Window 1 Is Already Open: No Action.");
              WindowValue1 = 1;
              checkerVal1 = 0;
            }
          }
        } else if (switchCharacteristic.value() == 3) { // Close Window 1
          if (checkerVal1 == 0) {
            if (isWindowOpen1 == 1) {
              WindowValue1 = 1;
            } else {
              WindowValue1 = 0;
            }
            checkerVal1 = 1;
          } else if (checkerVal1 == 1) {
            if (WindowValue1 == 1) {
              stepper1.moveTo(turn2);
              stepper1.run();
            } else if (stepper1.distanceToGo() == 0) {
              Serial.println("Window 1 Has Been Closed.");
              WindowValue1 = 0;
              checkerVal1 = 0;
            } else {
              Serial.println("Window 1 Is Already Closed: No Action.");
              WindowValue1 = 0;
              checkerVal1 = 0;
            }
          }
        } else if (switchCharacteristic.value() == 4) { // Open Window 2
          if (checkerVal2 == 0) {
            if (isWindowOpen2 == 0) {
              WindowValue2 = 0;
            } else {
              WindowValue2 = 1;
            }
            checkerVal2 = 1;
          } else if (checkerVal2 == 1) {
            if (WindowValue2 == 0) {
              stepper2.moveTo(turn1);
              stepper2.run();
            } else if (stepper2.distanceToGo() == 0) {
              Serial.println("Window 2 Has Been Opened.");
              WindowValue2 = 1;
              checkerVal2 = 0;
            } else {
              Serial.println("Window 2 Is Already Open: No Action.");
              WindowValue2 = 1;
              checkerVal2 = 0;
            }
          }
        } else if (switchCharacteristic.value() == 5) { // Close Window 2
          if (checkerVal2 == 0) {
            if (isWindowOpen2 == 1) {
              WindowValue2 = 1;
            } else {
              WindowValue2 = 0;
            }
            checkerVal2 = 1;
          } else if (checkerVal2 == 1) {
            if (WindowValue2 == 1) {
              stepper2.moveTo(turn2);
              stepper2.run();
            } else if (stepper2.distanceToGo() == 0) {
              Serial.println("Window 2 Has Been Closed.");
              WindowValue2 = 0;
              checkerVal2 = 0;
            } else {
              Serial.println("Window 2 Is Already Closed: No Action.");
              WindowValue2 = 0;
              checkerVal2 = 0;
            }
          }
        } else if (switchCharacteristic.value() == 6) { // Open Windows 1 & 2
          if (checkerVal1 == 0) {
            if (isWindowOpen1 == 0) {
              WindowValue1 = 0;
            } else {
              WindowValue1 = 1;
            }
            checkerVal1 = 1;
          } else if (checkerVal1 == 1) {
            if (WindowValue1 == 0) {
              stepper1.moveTo(turn1);
              stepper1.run();
            } else if (stepper1.distanceToGo() == 0) {
              Serial.println("Window 1 Has Been Opened.");
              WindowValue1 = 1;
              checkerVal1 = 0;
            } else {
              Serial.println("Window 1 Is Already Open: No Action.");
              WindowValue1 = 1;
              checkerVal1 = 0;
            }
          }
          if (checkerVal2 == 0) {
            if (isWindowOpen2 == 0) {
              WindowValue2 = 0;
            } else {
              WindowValue2 = 1;
            }
            checkerVal2 = 1;
          } else if (checkerVal2 == 1) {
            if (WindowValue2 == 0) {
              stepper2.moveTo(turn1);
              stepper2.run();
            } else if (stepper2.distanceToGo() == 0) {
              Serial.println("Window 2 Has Been Opened.");
              WindowValue2 = 1;
              checkerVal2 = 0;
            } else {
              Serial.println("Window 2 Is Already Open: No Action.");
              WindowValue2 = 1;
              checkerVal2 = 0;
            }
          }
        } else if (switchCharacteristic.value() == 7) { // Close Windows 1 & 2
          if (checkerVal1 == 0) {
            if (isWindowOpen1 == 1) {
              WindowValue1 = 1;
            } else {
              WindowValue1 = 0;
            }
            checkerVal1 = 1;
          } else if (checkerVal1 == 1) {
            if (WindowValue1 == 1) {
              stepper1.moveTo(turn2);
              stepper1.run();
            } else if (stepper1.distanceToGo() == 0) {
              Serial.println("Window 1 Has Been Closed.");
              WindowValue1 = 0;
              checkerVal1 = 0;
            } else {
              Serial.println("Window 1 Is Already Closed: No Action.");
              WindowValue1 = 0;
              checkerVal1 = 0;
            }
          }
          if (checkerVal2 == 0) {
            if (isWindowOpen2 == 1) {
              WindowValue2 = 1;
            } else {
              WindowValue2 = 0;
            }
            checkerVal2 = 1;
          } else if (checkerVal2 == 1) {
            if (WindowValue2 == 1) {
              stepper2.moveTo(turn2);
              stepper2.run();
            } else if (stepper2.distanceToGo() == 0) {
              Serial.println("Window 2 Has Been Closed.");
              WindowValue2 = 0;
              checkerVal2 = 0;
            } else {
              Serial.println("Window 2 Is Already Closed: No Action");
              WindowValue2 = 0;
              checkerVal2 = 0;
            }
          }
        }  
    }
    // when the central disconnects, print it out:
    Serial.print(F("Disconnected from central: "));
    Serial.println(central.address());
  }
}

void checkerfunction(int val1, int val2) {
  // WindowStates:
  // 10 / 0A: Both closed
  // 11 / 0B: Both open
  // 12 / 0C: Window 1 open, Window 2 closed
  // 13 / 0D: Window 1 closed, Window 2 open
  if (val1 == 0 && val2 == 0) {
    switchCharacteristic2.writeValue(10);
  } else if (val1 == 1 && val2 == 1) {
    switchCharacteristic2.writeValue(11);
  } else if (val1 == 1 && val2 == 0) {
    switchCharacteristic2.writeValue(12);
  } else if (val1 == 0 && val2 == 1) {
    switchCharacteristic2.writeValue(13);
  }
}