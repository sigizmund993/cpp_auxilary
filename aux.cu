//Модуль вспомогательной математики и утилит
#include <iostream>
#include <cmath>
class Point{
    public:
    float x,y;
    Point(float x, float y): x(x),y(y){}
    Point operator+(const Point& other) const{
        return Point(x+other.x,y+other.y);
    }
    Point operator-(const Point& other) const{
        return Point(x-other.x,y-other.y);
    }
    double mag(){
        return sqrt(x*x+y*y);
    }
    void print(){
        std::cout << x << " " << y << std::endl;
    }

};
int main(){
    Point p1(10,10);
    Point p2(2,8);
    Point p3 = p1 - p2;
    
    std::cout << p3.mag() << std::endl;
    return 0;
}