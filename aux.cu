#include <stdio.h>
#include <math.h>
#define GRID_SIZE 2110
#define PI 3.14159265358979323846f
#define ROBOT_R 100.0f
#define OBSTACLE_ANGLE (PI / 5)
#define GOAL_VIEW_ANGLE (PI / 4)
#define DIST_TO_ENEMY 1500
#define SHOOT_ANGLE (PI / 8)
#define DANGER_ZONE_DIST 400
#define MAX_ENEMIES_COUNT 6
#define GOAL_DX 4500
#define GOAL_DY 1000
#define FIELD_DX 4500
#define FIELD_DY 3000
#define POLARITY 1
#define ZONE_DX 1000
#define ZONE_DY 2000
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
struct Field {
    Point hull[4];
    Point enemy_hull[4], ally_hull[4];
    Point enemy_goal[2], ally_goal[2];
    __host__ __device__ Field() {
        hull[0] = Point(FIELD_DX, FIELD_DY);
        hull[1] = Point(FIELD_DX, -FIELD_DY);
        hull[2] = Point(-FIELD_DX, -FIELD_DY);
        hull[3] = Point(-FIELD_DX, FIELD_DY);
        enemy_hull[0] = Point(FIELD_DX * POLARITY, ZONE_DY / 2);
        enemy_hull[1] = Point(FIELD_DX * POLARITY, -ZONE_DY / 2);
        enemy_hull[2] = Point((FIELD_DX - ZONE_DX) * POLARITY, -ZONE_DY / 2);
        enemy_hull[3] = Point((FIELD_DX - ZONE_DX) * POLARITY, ZONE_DY / 2);
        ally_hull[0] = Point(FIELD_DX * -POLARITY, ZONE_DY / 2);
        ally_hull[1] = Point(FIELD_DX * -POLARITY, -ZONE_DY / 2);
        ally_hull[2] = Point((FIELD_DX - ZONE_DX) * -POLARITY, -ZONE_DY / 2);
        ally_hull[3] = Point((FIELD_DX - ZONE_DX) * -POLARITY, ZONE_DY / 2);
        enemy_goal[0] = Point(GOAL_DX * POLARITY, GOAL_DY / 2);
        enemy_goal[1] = Point(GOAL_DX * POLARITY, -GOAL_DY / 2);
        ally_goal[0] = Point(GOAL_DX * -POLARITY, GOAL_DY / 2);
        ally_goal[1] = Point(GOAL_DX * -POLARITY, -GOAL_DY / 2);
    }
};

__host__ __device__ int sign(float a) {
    if (a > 0) return 1;
    if (a < 0) return -1;
    return 0;
}

__host__ __device__ float wind_down_angle(float angle) {
    if (fabsf(angle) > 2 * PI) {
        angle = fmodf(angle, 2 * PI);
    }
    if (fabsf(angle) > PI) {
        angle -= 2 * PI * sign(angle);
    }
    return angle;
}

__host__ __device__ float get_angle_between_points(Point a, Point b, Point c) {
    return wind_down_angle((a - b).arg() - (c - b).arg());
}

__host__ __device__ void circles_inter(Point p0, Point p1, float r0, float r1, Point* out) {
    float d = (p0 - p1).mag();
    float a = (r0 * r0 - r1 * r1 + d * d) / (2 * d);
    float h = sqrtf(r0 * r0 - a * a);
    float x2 = p0.x + a * (p1.x - p0.x) / d;
    float y2 = p0.y + a * (p1.y - p0.y) / d;
    out[0].x = x2 + h * (p1.y - p0.y) / d;
    out[0].y = y2 - h * (p1.x - p0.x) / d;
    out[1].x = x2 - h * (p1.y - p0.y) / d;
    out[1].y = y2 + h * (p1.x - p0.x) / d;
}

__host__ __device__ int get_tangent_points(Point point0, Point point1, float r, Point* out) {
    float d = (point1 - point0).mag();
    if (d < r) {
        return 0;
    }   

    if (d == r) {
        out[0] = point1;
        return 1;
    }
    circles_inter(point0, Point((point0.x + point1.x) / 2, (point0.y + point1.y) / 2), r, d / 2, out);
    return 2;
}

__host__ __device__ Point closest_point_on_line(Point point1, Point point2, Point point, char type = 'S') {
    float line_len = (point1 - point2).mag();
    if (line_len == 0) {
        return point1;
    }
    Point line_dir = (point2 - point1).unity();
    Point point_vec = point - point1;
    float dot_product = point_vec.scalar(line_dir);
    // if (type == 'S') {
    //     return 0;
    // }
    if (dot_product <= 0 && type != 'L') {
        return point1;
    }
    if (dot_product >= line_len && type == 'S') {
        return point2;
    }
    return line_dir * dot_product + point1;
}

__host__ __device__ Point nearest_point_on_poly(Point p, Point *poly, int ed_n) {
    float min_ = -1, d;
    Point ans(0, 0), pnt(0, 0);
    for (int i = 0; i < ed_n; i++) {
        pnt = closest_point_on_line(poly[i], poly[i > 0 ? i - 1 : ed_n - 1], p);
        d = (pnt - p).mag();
        if (d < min_ || min_ < 0) {
            min_ = d;
            ans = pnt;
        }
    }
    return ans;
}

__host__ __device__ bool is_point_inside_poly(Point p, Point *points, int ed_n) {
    float old_sign = sign((p - points[ed_n - 1]).vector(points[0] - points[ed_n - 1]));
    for (int i = 0; i < ed_n - 1; i++) {
        if (old_sign != sign((p - points[i]).vector(points[i + 1] - points[i]))) {
            return false;
        }  
    }
    return true;
}

__host__ __device__ Point find_nearest_robot(Point point, Point *team, int te_n) {
    Point ans = Point(0, 0);
    float min_dist = -1, dist;

    if (te_n == 0) {
        return Point(0, 0);
    }
    for (int i = 0; i < te_n; i++) {
        dist = (team[i] - point).mag();
        if (dist < min_dist || min_dist < 0) {
            ans = team[i];
            min_dist = dist;
        }
    }
    return ans;
}

__host__ __device__ float estimate_pass_point(Point *enemies, int en_n, Point frm, Point to) {
    float lerp = 0.0f;
    float ang, ang1, ang2;
    for (int i = 0; i < en_n; i++) {
        float frm_enemy = (enemies[i] - frm).mag();
        if (frm_enemy > ROBOT_R) {
            if(frm_enemy <= (frm - to).mag()) {
                Point tgs[2];
                get_tangent_points(enemies[i], frm, ROBOT_R, tgs);
                ang1 = get_angle_between_points(to, frm, tgs[0]);
                ang2 = get_angle_between_points(to, frm, tgs[1]);
                ang = fminf(fabsf(ang1), fabsf(ang2));
                if (ang1 * ang2 < 0 && fabsf(ang1) < PI / 2 && fabsf(ang2) < PI / 2) {
                    ang *= -1;
                }
            }
            else {
                ang = 2 * asinf(((enemies[i] - to).mag() / 2) / frm_enemy) - asinf(ROBOT_R / frm_enemy);
            }
        }

        if (ang < OBSTACLE_ANGLE) {
            lerp += powf(fabsf((OBSTACLE_ANGLE - ang) / OBSTACLE_ANGLE), 1.5);
        }
            
    }
    return lerp;
}

__host__ __device__ float estimate_goal_view(Point point, Field fld) {
    return fminf(fabsf(get_angle_between_points(fld.enemy_goal[0], point, fld.enemy_goal[1])/GOAL_VIEW_ANGLE), 1);
}

__host__ __device__ float estimate_dist_to_boarder(Point point, Field fld) {
    float dist_to_goal_zone = (point - nearest_point_on_poly(point, fld.enemy_hull, 4)).mag();
    if (is_point_inside_poly(point, fld.enemy_hull, 4)) {
        dist_to_goal_zone *= -1;
    }
    
    float dist_to_field_boarder = (point - nearest_point_on_poly(point, fld.hull, 4)).mag();

    float dist_to_danger_zone = fminf(dist_to_goal_zone, dist_to_field_boarder);

    return fmaxf(1 - dist_to_danger_zone / DANGER_ZONE_DIST, 0);
}

__host__ __device__ float estimate_dist_to_enemy(Point point, Point *active_enemies, int en_n) {

    if (en_n == 0) {
        return 0;
    }
    return fmaxf(1 - (find_nearest_robot(point, active_enemies, en_n) - point).mag() / DIST_TO_ENEMY, 0);
}

__host__ __device__ float estimate_shoot(Point point, Field fld, Point *enemies, int en_n) {
    float lerp = 0.0f;
    float ang, ang1, ang2;
    float frm_enemy;
    for (int i = 0; i < en_n; i++) {
        frm_enemy = (point - enemies[i]).mag();
        if (frm_enemy > ROBOT_R) {
            ang1 = get_angle_between_points(fld.enemy_goal[0], point, enemies[i]);
            ang2 = get_angle_between_points(fld.enemy_goal[1], point, enemies[i]);

            ang = fminf(fabsf(ang1), fabsf(ang2));

            if (ang < SHOOT_ANGLE) {
                lerp += powf(fabsf((SHOOT_ANGLE - ang) / SHOOT_ANGLE), 1.5);
            }
        }
    }
    return lerp;
}

__host__ __device__ float estimate_point(Field fld, Point point, Point kick_point, Point *enemies, int en_n) {
    return estimate_goal_view(point, fld) - estimate_pass_point(enemies, en_n, kick_point, point) - estimate_dist_to_boarder(point, fld) - 
    estimate_shoot(point, fld, enemies, en_n) - estimate_dist_to_enemy(point, enemies, en_n);

}
__host__ __device__ float estimate_point_by_id(Field fld, Point kick_point, Point *enemies,int grid_dens, int en_n,int idx,int N)
{

    if(idx < N)
    {
        Point cur_pos(
            grid_dens * (idx % int(FIELD_DX*2 / grid_dens))-FIELD_DX,
            grid_dens * int(idx / int(FIELD_DX*2 / grid_dens))-FIELD_DY
        );
        return -estimate_point(fld,cur_pos,kick_point,enemies,en_n);
    }
    return 1e10f;
}
extern "C" __global__ void find_best_pass_point(Point *field_poses,int en_count, int grid_dens, float *out, int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    Field fld = Field();
    Point enemies[MAX_ENEMIES_COUNT];
    for(int i = 0;i<MAX_ENEMIES_COUNT;i++)
        enemies[i] = field_poses[i+1];
    out[idx] = estimate_point_by_id(fld,field_poses[0],enemies,grid_dens,en_count,idx,N);

    __syncthreads();
    int tid = threadIdx.x;
    __shared__ float sharedVals[256];
    float val = (idx < N) ? out[idx] : 1e10f;
    sharedVals[tid] = val;
    __syncthreads();
    for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
        if (tid < offset) {
            sharedVals[tid] = fminf(sharedVals[tid], sharedVals[tid + offset]);
        }
        __syncthreads();
    }
    if (tid == 0) {
        out[blockIdx.x] = sharedVals[0];
        out[blockIdx.x*2] = idx;
    }

    // Поток 0 блока 0 ищет глобальный минимум среди всех блоков
    //TODO
    /*
    if (blockIdx.x == 0 && tid == 0) {
        float globalMin = 1e10f;
        for (int i = 0; i < 256; i++) {
            globalMin = fminf(globalMin, out[i]);
        }
        out[0] = globalMin;
    }
    */

}