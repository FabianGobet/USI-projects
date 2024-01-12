#include "TileLeaf.h"

// Tile implementation
Tile::Tile(U8G2_SSD1327_WS_128X128_F_4W_HW_SPI* lcd, String message) {
  this->lcd = lcd;
  this->message = message;
}

void Tile::set_neighbours(Tile *t_back, Tile *t_left, Tile *t_right, Tile *t_set){
  this->t_back = t_back;
  this->t_left = t_left;
  this->t_set = t_set;
  this->t_right = t_right;
}

Tile* Tile::right() {
  return t_right;
}

Tile* Tile::left() {
  return t_left;
}

Tile* Tile::set() {
  return t_set;
}

Tile* Tile::back() {
  return t_back;
}

void Tile::baseDraw() {
  (*lcd).drawTriangle(15,56, 15,72, 2,64);
  (*lcd).drawTriangle(113,56, 113,72, 126,64);
  byte width = (*lcd).getStrWidth(this->message.c_str());
  (*lcd).drawStr(15 + 2 + (94 - width) / 2, 69, this->message.c_str());
}

void Tile::drawStuff(){
  this->baseDraw();
}

Tile::~Tile() {
}



// Leaf implementation
Leaf::Leaf(U8G2_SSD1327_WS_128X128_F_4W_HW_SPI* lcd, String message, byte* parameter)
    : Tile(lcd,message) {
  this->parameter = parameter;
  this->my_value = *parameter;
}

void Leaf::set_neighbours(Tile* t){
  this->t_back = t;
  this->t_set = t;
}

Tile* Leaf::left() {
  this->my_value = this->my_value - 1;
  return this;
}

Tile* Leaf::right() {
  this->my_value = this->my_value + 1;
  return this;
}

Tile* Leaf::set() {
  *this->parameter = this->my_value;
  return this->t_set;
}

Tile* Leaf::back() {
  this->my_value = *this->parameter;
  return this->t_back;
}

void Leaf::drawStuff() {
  this->baseDraw();
  byte width = (*lcd).getStrWidth(std::to_string(this->my_value).c_str());
  (*lcd).drawStr(15 + 2 + (94 - width) / 2, 90, std::to_string(this->my_value).c_str());
}

Leaf::~Leaf() {
}


// Info implementation
Info::Info(U8G2_SSD1327_WS_128X128_F_4W_HW_SPI* lcd, String message, byte numElements, String* descriptions, int* values)
    : Tile(lcd,message) {
  this->numElements = numElements;
  this->descriptions = descriptions;
  this->values = values;
}

void Info::set_neighbours(Tile* t){
  this->t_back = t;
  this->t_set = t;
}

Tile* Info::left() {
  return this;
}

Tile* Info::right() {
  return this;
}

Tile* Info::set() {
  return this;
}

Tile* Info::back() {
  return this->t_back;
}

void Info::drawStuff() {
  byte calc = static_cast<byte>((128 - this->numElements * 12) / (this->numElements + 1));
  for(byte i = 0; i < this->numElements; i++){
    byte ypos = (i+1)*calc;
    String sentence = this->descriptions[i] + ": " + String(this->values[i]);
    (*lcd).drawStr(1, ypos, sentence.c_str());
  }
}

Info::~Info() {
}



// Leaf2 implementation
Leaf2::Leaf2(U8G2_SSD1327_WS_128X128_F_4W_HW_SPI* lcd, String message, byte* parameter)
    : Leaf(lcd,message,parameter) {
}


Tile* Leaf2::left() {
  this->my_value = (this->my_value - 1)%2;
  return this;
}

Tile* Leaf2::right() {
  this->my_value = (this->my_value + 1)%2;
  return this;
}

void Leaf2::drawStuff() {
  this->baseDraw();
  String val = (this->my_value==1 ? "True":"False");
  byte width = (*lcd).getStrWidth(val.c_str());
  (*lcd).drawStr(15 + 2 + (94 - width) / 2, 90, val.c_str());
}

Leaf2::~Leaf2() {
}
