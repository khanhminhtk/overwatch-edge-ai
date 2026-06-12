#include <string>
#include <stdexcept>

class BoudingBox {
    public:
        BoudingBox(float x, float y, float width, float height) 
        : x_(x), y_(y), width_(width), height_(height) {
            if (x<0 || y<0) {
                throw std::invalid_argument("Width and height must be non-negative");
            };
        };
        float get_x(){return x_;};
        float get_y(){return y_;};
        float get_width(){return width_;};
        float get_height(){return height_;};

    private:
        float x_;
        float y_;
        float width_;
        float height_;
};