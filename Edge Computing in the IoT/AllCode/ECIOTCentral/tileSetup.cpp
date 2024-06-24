// TileSetup.cpp

#include "TileSetup.h"
#define INSIDE 3
#define OUTSIDE 3
#define WINDOWS 2



byte min_temp = 15;
byte max_temp = 20;
byte min_hum = 15;
byte max_hum = 30;
byte toxic_vent = true;
byte push_not = true;
byte auto_wind = true;
int insideValues[INSIDE] = {0,0,0};
int outsideValues[OUTSIDE] = {27,10,220};
int windowsValues[WINDOWS] = {0};
String insideDescriptions[INSIDE] = {String("Temperature"),String("Humidity"),String("Toxic")};
String outsideDescriptions[OUTSIDE] = {String("Temperature"),String("Humidity"),String("Particles")};
String windowsDescriptions[WINDOWS] = {String("Window 1"),String("Window 2")};

Info insideInfo(&u8g2, String("Inside"), INSIDE, insideDescriptions, insideValues);
Info outsideInfo(&u8g2, String("Outside"), OUTSIDE, outsideDescriptions, outsideValues);
Info windowsInfo(&u8g2, String("Windows"), WINDOWS, windowsDescriptions, windowsValues);
Leaf minTempSet(&u8g2, String("Min. temp."), &min_temp);
Leaf maxTempSet(&u8g2, String("Max. temp."), &max_temp);
Leaf minHumSet(&u8g2, String("Min. hum."), &min_hum);
Leaf maxHumSet(&u8g2, String("Max. hum."), &max_hum);
Leaf2 toxicVentSet(&u8g2, String("Toxic. vent."), &toxic_vent);
Leaf2 pushNotifSet(&u8g2, String("Push notif."), &push_not);
Leaf2 autoWindSet(&u8g2, String("Auto wind."), &auto_wind);

Tile status(&u8g2, String("Status"));
Tile settings(&u8g2, String("Settings"));
Tile outside(&u8g2, String("Outside"));
Tile inside(&u8g2, String("Inside"));
Tile windows(&u8g2, String("Windows"));

Tile minTemp(&u8g2, String("Min. temp."));
Tile maxTemp(&u8g2, String("Max. temp."));
Tile minHum(&u8g2, String("Min. hum."));
Tile maxHum(&u8g2, String("Max. hum."));
Tile toxicVent(&u8g2, String("Toxic. vent."));
Tile pushNotif(&u8g2, String("Push notif."));
Tile autoWind(&u8g2, String("Auto wind."));

//Tile toDo(&u8g2, String("To Do"));

void setupTilesAndLeaves() {
  minTempSet.set_neighbours(&minTemp);
  maxTempSet.set_neighbours(&maxTemp);
  minHumSet.set_neighbours(&minHum);
  maxHumSet.set_neighbours(&maxHum);
  toxicVentSet.set_neighbours(&toxicVent);
  pushNotifSet.set_neighbours(&pushNotif);
  autoWindSet.set_neighbours(&autoWind);

  status.set_neighbours(&status,&settings,&settings,&inside);
  settings.set_neighbours(&settings,&status,&status,&minTemp);

  inside.set_neighbours(&status,&windows,&outside,&insideInfo);
  outside.set_neighbours(&status,&inside,&windows,&outsideInfo);
  windows.set_neighbours(&status,&outside,&inside,&windowsInfo);

  minTemp.set_neighbours(&status,&autoWind,&maxTemp,&minTempSet);
  maxTemp.set_neighbours(&status,&minTemp,&minHum,&maxTempSet);
  minHum.set_neighbours(&status,&maxTemp,&maxHum,&minHumSet);
  maxHum.set_neighbours(&status,&minHum,&toxicVent,&maxHumSet);
  toxicVent.set_neighbours(&status,&maxHum,&pushNotif,&toxicVentSet);
  pushNotif.set_neighbours(&status,&toxicVent,&autoWind,&pushNotifSet);
  autoWind.set_neighbours(&status,&pushNotif,&minTemp,&autoWindSet);

  insideInfo.set_neighbours(&inside);
  outsideInfo.set_neighbours(&outside);
  windowsInfo.set_neighbours(&windows);

}
