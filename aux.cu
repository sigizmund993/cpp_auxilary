struct Vec2 {
    float x, y;

    __host__ __device__ Vec2() : x(0), y(0) {}
    __host__ __device__ Vec2(float x_, float y_) : x(x_), y(y_) {}

    __host__ __device__ Vec2 operator+(const Vec2& b) const {
        return Vec2(x + b.x, y + b.y);
    }

    __host__ __device__ Vec2 operator-(const Vec2& b) const {
        return Vec2(x - b.x, y - b.y);
    }

    __host__ __device__ Vec2 operator*(float scalar) const {
        return Vec2(x * scalar, y * scalar);
    }

    __host__ __device__ float dot(const Vec2& b) const {
        return x * b.x + y * b.y;
    }

    __host__ __device__ float mag() const {
        return sqrtf(x * x + y * y);
    }

    __host__ __device__ Vec2 unity() const {
        float len = length();
        return len > 0.0f ? Vec2(x / len, y / len) : Vec2(0.0f, 0.0f);
    }

    __host__ __device__ float arg() const {
        return atan2f(y, x);
    }
};