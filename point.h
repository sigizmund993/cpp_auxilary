struct Point {
    float x, y;
    __host__ __device__ Point() : x(0), y(0) {}
    __host__ __device__ Point(float x_, float y_) : x(x_), y(y_) {}
    __host__ __device__ Point operator-(const Point& b) const { return Point(x - b.x, y - b.y); }
    __host__ __device__ float mag() const { return sqrtf(x * x + y * y); }
    __host__ __device__ Point operator+(Point b) {
        return Point(x + b.x, y + b.y);
    }
    __host__ __device__ Point operator-(Point b) {
        return Point(x - b.x, y - b.y);
    }
    __host__ __device__ Point operator*(float scalar) {
        return Point(x * scalar, y * scalar);
    }
    __host__ __device__ Point operator/(float scalar) {
        return Point(x / scalar, y / scalar);
    }
    __host__ __device__
    bool operator==(Point other) {
        return x == other.x && y == other.y;
    }
    __host__ __device__
    bool operator!=(Point other) {
        return !(*this == other);
    }
    __host__ __device__ float scalar(Point b) {
        return x * b.x + y * b.y;
    }

    __host__ __device__ float vector(Point b) {
        return x * b.y - y * b.x;
    }

    __host__ __device__ float mag() {
        return sqrtf(x * x + y * y);
    }

    __host__ __device__ Point unity() {
        float len = mag();
        return len > 0.0f ? Point(x / len, y / len) : Point(0.0f, 0.0f);
    }

    __host__ __device__ float arg() {
        return atan2f(y, x);
    }
};