#include <stdio.h>
#include <iostream>
#include <math.h>
#include <chrono>
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
    Point() : x(0), y(0) {}
    Point(float x_, float y_) : x(x_), y(y_) {}
    Point operator-(const Point& b) const { return Point(x - b.x, y - b.y); }
    float mag() const { return sqrtf(x * x + y * y); }
    Point operator+(Point b) {
        return Point(x + b.x, y + b.y);
    }
    Point operator-(Point b) {
        return Point(x - b.x, y - b.y);
    }
    Point operator*(float scalar) {
        return Point(x * scalar, y * scalar);
    }
    Point operator/(float scalar) {
        return Point(x / scalar, y / scalar);
    }
     
    bool operator==(Point other) {
        return x == other.x && y == other.y;
    }
     
    bool operator!=(Point other) {
        return !(*this == other);
    }
    float scalar(Point b) {
        return x * b.x + y * b.y;
    }

    float vector(Point b) {
        return x * b.y - y * b.x;
    }

    float mag() {
        return sqrtf(x * x + y * y);
    }

    Point unity() {
        float len = mag();
        return len > 0.0f ? Point(x / len, y / len) : Point(0.0f, 0.0f);
    }

    float arg() {
        return atan2f(y, x);
    }
};
struct Field {
    // float GOAL_DX, GOAL_DY, ZONE_DX, ZONE_DY, FIELD_DX, FIELD_DY;
    // int POLARITY;
    Point hull[4];
    Point enemy_hull[4], ally_hull[4];
    Point enemy_goal[2], ally_goal[2];
    // Field(float gx, float gy, float zx, float zy, float fy, int pol) {
    Field() {
        // GOAL_DX = gx;
        // GOAL_DY = gy;
        // ZONE_DX = zx;
        // ZONE_DY = zy;
        // FIELD_DX = gx;
        // FIELD_DY = fy;
        // POLARITY = pol;
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

int sign(float a) {
    if (a > 0) return 1;
    if (a < 0) return -1;
    return 0;
}

float wind_down_angle(float angle) {
    if (fabsf(angle) > 2 * PI) {
        angle = fmodf(angle, 2 * PI);
    }
    if (fabsf(angle) > PI) {
        angle -= 2 * PI * sign(angle);
    }
    return angle;
}

float get_angle_between_points(Point a, Point b, Point c) {
    return wind_down_angle((a - b).arg() - (c - b).arg());
}

void circles_inter(Point p0, Point p1, float r0, float r1, Point* out) {
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

int get_tangent_points(Point point0, Point point1, float r, Point* out) {
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

Point closest_point_on_line(Point point1, Point point2, Point point, char type = 'S') {
    float line_len = (point1 - point2).mag();
    if (line_len == 0) {
        return point1;
    }
    Point line_dir = (point1 - point2).unity();
    Point point_vec = point - point1;
    float dot_product = point_vec.scalar(line_dir);
    if (dot_product <= 0 && type != 'L') {
        return point1;
    }
    if (dot_product >= line_len && type == 'S') {
        return point2;
    }
    return line_dir * dot_product + point1;
}

Point nearest_point_on_poly(Point p, Point *poly, int ed_n) {
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

bool is_point_inside_poly(Point p, Point *points, int ed_n) {
    float old_sign = sign((p - points[ed_n - 1]).vector(points[0] - points[ed_n - 1]));
    for (int i = 0; i < ed_n - 1; i++) {
        if (old_sign != sign((p - points[i]).vector(points[i + 1] - points[i]))) {
            return false;
        }  
    }
    return true;
}

Point find_nearest_robot(Point point, Point *team, int te_n) {
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

float estimate_pass_point(Point *enemies, int en_n, Point frm, Point to) {
    float lerp = 0.0f;
    float ang, ang1, ang2;
    for (int i = 0; i < en_n; i++) {
        float frm_enemy = (enemies[i] - frm).mag();
        if (frm_enemy > ROBOT_R) {
            Point tgs[2];
            get_tangent_points(enemies[i], frm, ROBOT_R, tgs);
            ang1 = get_angle_between_points(to, frm, tgs[0]);
            ang2 = get_angle_between_points(to, frm, tgs[1]);
            ang = fminf(fabsf(ang1), fabsf(ang2));
            if (ang1 * ang2 < 0 && fabsf(ang1) < PI / 2 && fabsf(ang2) < PI / 2)
                    ang *= -1;
        }
        else {
            ang = 2 * asinf(((enemies[i] - to).mag() / 2) / frm_enemy) - asinf(ROBOT_R / frm_enemy);
        }

        if (ang < OBSTACLE_ANGLE) {
            lerp += powf(fabsf((OBSTACLE_ANGLE - ang) / OBSTACLE_ANGLE), 1.5);
        }
            
    }
    return lerp;
}

float estimate_goal_view(Point point, Field fld) {
    return fminf(fabsf(get_angle_between_points(fld.enemy_goal[0], point, fld.enemy_goal[1])), 1);
}

float estimate_dist_to_boarder(Point point, Field fld) {
    float dist_to_goal_zone = (point - nearest_point_on_poly(point, fld.enemy_hull, 4)).mag();
    if (is_point_inside_poly(point, fld.enemy_goal, 4)) {
        dist_to_goal_zone *= -1;
    }
    
    float dist_to_field_boarder = (point - nearest_point_on_poly(point, fld.hull, 4)).mag();

    float dist_to_danger_zone = fminf(dist_to_goal_zone, dist_to_field_boarder);

    return fmaxf(1 - dist_to_danger_zone / DANGER_ZONE_DIST, 0);
}

float estimate_dist_to_enemy(Point point, Point *active_enemies, int en_n) {

    if (en_n == 0) {
        return 0;
    }
    return fmaxf(1 - (find_nearest_robot(point, active_enemies, en_n) - point).mag() / DIST_TO_ENEMY, 0);
}

float estimate_shoot(Point point, Field fld, Point *enemies, int en_n) {
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

float estimate_point(Field fld, Point point, Point kick_point, Point *enemies, int en_n) {
    return -(estimate_goal_view(point, fld) - estimate_pass_point(enemies, en_n, kick_point, point) - estimate_dist_to_boarder(point, fld) - 
    estimate_shoot(point, fld, enemies, en_n) - estimate_dist_to_enemy(point, enemies, en_n));

}
//extern "C" __global__ void find_best_pass_point(float *field_info,Point *enemies, int en_count, Point kick_point,int grid_dens, float *out, int N)
// extern "C" __global__ void find_best_pass_point(float *field_info, Point *field_poses,int en_count, int grid_dens, float *out, int N)
int main()
{
    auto start = std::chrono::system_clock::now();   
    time_t end_time = 0;
    Point field_poses[7];
    int en_count = 6;
    int grid_dens = 5;
    float minVal = 1e10f;
    int minX = -1, minY = -1;
    float curVal = 0;
    Field fld = Field(
        // field_info[0],
        // field_info[1],
        // field_info[2],
        // field_info[3],
        // field_info[4],
        // field_info[5]
    );
    field_poses[0] = Point(10,10);
    field_poses[1] = Point(1120,1230);
    field_poses[2] = Point(1210,1440);
    field_poses[3] = Point(20,-1210);
    field_poses[4] = Point(-3810,120);
    field_poses[5] = Point(-310,1120);
    field_poses[6] = Point(-1290,0);

    Point enemies[MAX_ENEMIES_COUNT];
    for(int i = 0;i<MAX_ENEMIES_COUNT;i++)
        enemies[i] = field_poses[i+1];

    minVal = 1e10f;
    for(int i = 0;i<FIELD_DX*2;i+=grid_dens)
    {
        for(int j = 0;j<FIELD_DY*2;j+=grid_dens)
        {
            Point cur_pos(i,j);
            curVal = estimate_point(fld,cur_pos,field_poses[0],enemies,en_count);
            // printf("%f at %f, %f, min: %f\n",curVal,cur_pos.x,cur_pos.y,minVal);
            
            if(curVal<minVal)
            {
                minVal = curVal;
                minX = cur_pos.x;
                minY = cur_pos.y;
            }
        }
    }
    // for(int i = 0;i<FIELD_DX*2/grid_dens*FIELD_DY*2/grid_dens;i++)
    // {
    //     Point cur_pos(
    //         grid_dens * (i % int(FIELD_DX*2 / grid_dens))-FIELD_DX,
    //         grid_dens * int(i / int(FIELD_DX*2 / grid_dens))-FIELD_DY//field_info[0]*2 - размер поля по x
    //     );
    //     curVal = estimate_point(fld,cur_pos,field_poses[0],enemies,en_count);
    //     printf("%f, %f\n",cur_pos.x,cur_pos.y);
    //     if(curVal<minVal)
    //         minVal = curVal;
    //         minX = cur_pos.x;
    //         minY = cur_pos.y;
    // }
    printf("%f at %i,%i\n",minVal,minX,minY);
    auto end = std::chrono::system_clock::now();
    std::cout <<
        std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    printf("\n");
    // printf("%f\n",estimate_point(fld,Point(3600,0),field_poses[0],enemies,en_count));
    return 0;
}
