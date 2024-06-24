// TileSetup.h

#include <Arduino.h>
#include <U8g2lib.h>
#include <String>
#include "TileLeaf.h"

extern U8G2_SSD1327_WS_128X128_F_4W_HW_SPI u8g2;

extern byte min_temp;
extern byte max_temp;
extern byte min_hum;
extern byte max_hum;
extern byte toxic_vent;
extern byte push_not;
extern byte auto_wind;

extern Leaf minTempSet;
extern Leaf maxTempSet;
extern Leaf minHumSet;
extern Leaf maxHumSet;
extern Leaf2 toxicWindSet;
extern Leaf2 pushNotifSet;
extern Leaf2 autoWindSet;

extern Tile status;
extern Tile settings;
extern Tile outside;
extern Tile inside;
extern Tile windows;

extern Tile minTemp;
extern Tile maxTemp;
extern Tile minHum;
extern Tile maxHum;
extern Tile toxicVent;
extern Tile pushNotif;
extern Tile autoWind;

extern int insideValues[];
extern int outsideValues[];
extern int windowsValues[];

void setupTilesAndLeaves();
