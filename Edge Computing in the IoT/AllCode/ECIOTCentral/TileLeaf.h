#ifndef TILE_LEAF_H
#define TILE_LEAF_H
#include <U8g2lib.h>
#include <String>



class Tile {
  public:
    U8G2_SSD1327_WS_128X128_F_4W_HW_SPI* lcd;
    Tile* t_back;
    Tile* t_set;
    Tile* t_left;
    Tile* t_right;
    String message;

    Tile(U8G2_SSD1327_WS_128X128_F_4W_HW_SPI* lcd, String message);
    
    void set_neighbours(Tile* t_back, Tile* t_left, Tile* t_right, Tile* t_set);

    virtual Tile* right();
    virtual Tile* left();
    virtual Tile* set();
    virtual Tile* back();

    void baseDraw();
    virtual void drawStuff();

    virtual ~Tile();
};

class Info : public Tile{
  public:
    byte numElements;
    String* descriptions;
    int* values;
    Info(U8G2_SSD1327_WS_128X128_F_4W_HW_SPI* lcd, String message, byte numElements, String* descriptions, int* values);

    Tile* set() override;
    Tile* back() override;
    Tile* left() override;
    Tile* right() override;
    void set_neighbours(Tile* t);
    void drawStuff() override;

    ~Info();
};

class Leaf : public Tile {
  public:
    byte my_value;
    String message;
    byte* parameter;

    Leaf(U8G2_SSD1327_WS_128X128_F_4W_HW_SPI* lcd, String message, byte* parameter);

    virtual Tile* left() override;
    virtual Tile* right() override;
    Tile* set() override;
    Tile* back() override;

    void set_neighbours(Tile* t);
    virtual void drawStuff() override;

    ~Leaf();
};


class Leaf2 : public Leaf {
  public:
    Leaf2(U8G2_SSD1327_WS_128X128_F_4W_HW_SPI* lcd, String message, byte* parameter);

    Tile* left() override;
    Tile* right() override;

    void drawStuff() override;

    ~Leaf2();
};

#endif // TILE_LEAF_H
