// #include "point.h"

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
__host__ __device__ int gauss_sovle(double *a, double *b, int n, double *x) {
    static double maxV, val, k;
    static int maxJ, i1, i2, i, j, p;
    // a[0] = -1;
    // a[1] = 1;
    // a[2] = -2;
    // a[3] = 2;
    // b[0] = 1;
    // b[1] = 1;
    maxV = 0;
    maxJ = 0;
    for(i = 0; i < n; i++) {
        maxJ = 0;
        maxV = 0;
        for(j = 0; j < n - i; j++) {
            val = abs(a[(i + j) * n + i]);
            if(val > maxV) {
                maxV = val;
                maxJ = j;
            }
        }
        // cout << "jjjjjjj " << maxJ << endl;
        if(maxV == 0) {
            return 0;
        }
        if(maxJ != 0) {
            i1 = (maxJ + i) * n + i;
            i2 = i * n + i;
            for(j = 0; j < n - i; j++) {
                a[i1 + j] += a[i2 + j];
                a[i2 + j] = a[i1 + j] - a[i2 + j];
                a[i1 + j] -= a[i2 + j];
            }
            b[maxJ + i] += b[i];
            b[i] = b[maxJ + i] - b[i];
            b[maxJ + i] -= b[i];
        }
        // cout << "semen lobanov " << a[0] << ", " << a[1] << ", " << a[2] << ", " << a[3] << ", " << b[0] << ", " << b[1] << endl;
        for(j = 1; j < n - i; j++) {
            k = a[(i + j) * n + i] / a[i * n + i];
            for(p = 1; p < n - i; p++) {
                a[(i + j) * n + i + p] -= a[i * n + i + p] * k;
            }
            b[i + j] -= b[i] * k;
        }
    }
    // cout << "pupupu " << a[0] << ", " << a[1] << ", " << a[2] << ", " << a[3] << ", " << b[0] << ", " << b[1] << endl;
    for(i = n - 1; i >= 0; i--) {
        for(j = n - 1; j > i; j--) {
            b[i] -= a[i * n + j] * x[j];
        }
        if(a[i * n + i] == 0) {
            // cout << "нашел пидора" << endl;
            x[i] = 0;
        }
        else {
            x[i] = b[i] / a[i * n + i];
        }
        
    }
    // cout << "gauss " << a[0] << ", " << b[0] << ", " << x[0] << endl;
    return 1;
}
__host__ __device__ int newton(void (*jac)(void (*)(double*, double*, double*), double*, double*, double*, double*, int, double), 
    void (*f)(double*, double*, double*), double *args, double *x, int n, double tol = 1e-7, int max_iter = 100, double d = 1e-7) {
    static int i, j;
    static bool flag;
    double jacobian[n * n], fx[n], dx[n];
    // cout << "start" << endl;
    // static high_resolution_clock::time_point start, end;
    // cout << "start " << x[0] << endl;
    for(i = 0; i < max_iter; i++) {
        // start = high_resolution_clock::now();
        jac(f, x, jacobian, args, fx, n, d);
        // if (n == 1) {
        //     cout << x[0] << ", " << fx[0] << ", " << jacobian[0] << endl;
        // }
        flag = true;
        for(j = 0; j < n; j++) {
            if(abs(fx[j]) > tol) {
                flag = false;
            }
        }
        if(flag) {
            return 2;
        }

        if(gauss_sovle(jacobian, fx, n, dx) == 0) {
            // cout << "err with gauss solve (maybe something with zero jacobian)" << endl;
            return 1;
        }
        // cout << "gauss " << dx[0] << endl;
        for(j = 0; j < n; j++) {
            x[j] -= dx[j]; 
        }
        // end = high_resolution_clock::now();
        // duration<double, micro> duration_us = duration_cast<duration<double, micro>>(end - start);
        // cout << "time " << duration_us.count() << endl;
        // cout << x[0] << endl;
    }
    return 0;
}
__host__ __device__ void jac1(void (*f)(double*, double*, double*), double *Vm, double *jac, double *args, double *fx, int n, double d) {
    static double ma, mb;
    ma = sqrt((args[0] - Vm[0]) * (args[0] - Vm[0]) + (args[1] - Vm[1]) * (args[1] - Vm[1]));
    mb = sqrt((args[2] - Vm[0]) * (args[2] - Vm[0]) + (args[3] - Vm[1]) * (args[3] - Vm[1]));
    fx[0] = 2 * args[6] * args[4] - (Vm[0] + args[0]) * ma - (Vm[0] + args[2]) * mb;
    fx[1] = 2 * args[6] * args[5] - (Vm[1] + args[1]) * ma - (Vm[1] + args[3]) * mb;
    jac[0] = -ma - (Vm[0] * Vm[0] - args[0] * args[0]) / ma - mb - (Vm[0] * Vm[0] - args[2] * args[2]) / mb;
    jac[1] = -(Vm[0] + args[0]) * (Vm[1] - args[1]) / ma - (Vm[0] + args[2]) * (Vm[1] - args[3]) / mb;
    jac[2] = -(Vm[1] + args[1]) * (Vm[0] - args[0]) / ma - (Vm[1] + args[3]) * (Vm[0] - args[2]) / mb;
    jac[3] = -ma - (Vm[1] * Vm[1] - args[1] * args[1]) / ma - mb - (Vm[1] * Vm[1] - args[3] * args[3]) / mb;
}
__host__ __device__ void func1(double *Vm, double *args, double *fx) {
    static double ma, mb;
    ma = sqrt((args[0] - Vm[0]) * (args[0] - Vm[0]) + (args[1] - Vm[1]) * (args[1] - Vm[1]));
    mb = sqrt((args[2] - Vm[0]) * (args[2] - Vm[0]) + (args[3] - Vm[1]) * (args[3] - Vm[1]));
    // cout << "mamb " << ma << ", " << mb << ", " << (args[0] - Vm[0]) << ", " << (args[1] - Vm[1]) << endl;
    fx[0] = 2 * args[6] * args[4] - (Vm[0] + args[0]) * ma - (Vm[0] + args[2]) * mb;
    fx[1] = 2 * args[6] * args[5] - (Vm[1] + args[1]) * ma - (Vm[1] + args[3]) * mb;
}
__host__ __device__ void jac2(void (*f)(double*, double*, double*), double *ang, double *jac, double *args, double *fx, int n, double d) {
    static double ma, mb, dma, dmb, lhs[2], dl[2], Vm[2], ln, c2, s2; //, tc2, ts2, tVm[2], tma, tmb, tlhs[2];
    c2 = cos(ang[0]);
    s2 = sin(ang[0]);
    Vm[0] = c2 * args[7];
    Vm[1] = s2 * args[7];
    ma = sqrt((args[0] - Vm[0]) * (args[0] - Vm[0]) + (args[1] - Vm[1]) * (args[1] - Vm[1]));
    mb = sqrt((args[2] - Vm[0]) * (args[2] - Vm[0]) + (args[3] - Vm[1]) * (args[3] - Vm[1]));
    lhs[0] = 2 * args[6] * args[4] - (Vm[0] + args[0]) * ma - (Vm[0] + args[2]) * mb;
    lhs[1] = 2 * args[6] * args[5] - (Vm[1] + args[1]) * ma - (Vm[1] + args[3]) * mb;
    ln = sqrt(lhs[0] * lhs[0] + lhs[1] * lhs[1]);
    fx[0] = (lhs[0] * c2 + lhs[1] * s2) / ln;
    dma = (Vm[1] * args[0] - Vm[0] * args[1]) / ma;
    dmb = (Vm[1] * args[2] - Vm[0] * args[3]) / mb;
    dl[0] = Vm[1] * ma - (Vm[0] + args[0]) * dma + Vm[1] * mb - (Vm[0] + args[2]) * dmb;
    dl[1] = -Vm[0] * ma - (Vm[1] + args[1]) * dma - Vm[0] * mb - (Vm[1] + args[3]) * dmb;
    jac[0] = -((-s2 * lhs[0] + c2 * dl[0] + c2 * lhs[1] + s2 * dl[1]) / ln - (lhs[0] * dl[0] + lhs[1] * dl[1]) * fx[0] / (ln * ln)); // / sqrt(1 - (fx[0]  * fx[0]));
    fx[0] = -fx[0] + 1;
}
__host__ __device__ void func2(double *ang, double *args, double *fx) {
    static double Vm[2], lhs[2], ma, mb, ln;
    // static high_resolution_clock::time_point start, end;
    // start = high_resolution_clock::now();
    Vm[0] = cos(ang[0]) * args[7];
    Vm[1] = sin(ang[0]) * args[7];
    ma = sqrt((args[0] - Vm[0]) * (args[0] - Vm[0]) + (args[1] - Vm[1]) * (args[1] - Vm[1]));
    mb = sqrt((args[2] - Vm[0]) * (args[2] - Vm[0]) + (args[3] - Vm[1]) * (args[3] - Vm[1]));
    lhs[0] = 2 * args[6] * args[4] - (Vm[0] + args[0]) * ma - (Vm[0] + args[2]) * mb;
    lhs[1] = 2 * args[6] * args[5] - (Vm[1] + args[1]) * ma - (Vm[1] + args[3]) * mb;
    // cout << lhs[0] << ", " << lhs[1] << ", " << Vm[0] << ", " << Vm[1] << ", " << ma << ", " << mb << endl;
    ln = sqrt(lhs[0] * lhs[0] + lhs[1] * lhs[1]);
    fx[0] = -(lhs[0] * Vm[0] + lhs[1] * Vm[1]) / ln / args[7] + 1;
}

__host__ __device__ Point bangbang(Point start, Point end, Point dr, double Amax, double Vmax, int nshort = 10, int nst = 10, int mult = 2, int barrier = 10000) { // gang-bang
    static double args[8], angle[1], rn, Vm[2], zero, imin, vmin, vnow[1];
    static int i, n;
    args[0] = start.x;
    args[1] = start.y;
    args[2] = end.x;
    args[3] = end.y;
    args[4] = dr.x;
    args[5] = dr.y;
    args[6] = Amax;
    args[7] = Vmax;
    rn = sqrt(args[4] * args[4] + args[5] * args[5]);Vm[0] = args[4] / rn * Vmax;
    Vm[1] = args[5] / rn * Vmax;
    static Point res = {0, 0};
    int g = 0;
    zero = atan2(args[5], args[4]);
    for(i = 0; g != 2 && i < nshort; i++) {
        Vm[0] = cos(zero + 2 * M_PI * i / nshort) * Vmax;
        Vm[1] = sin(zero + 2 * M_PI * i / nshort) * Vmax;
        g = newton(jac1, func1, args, Vm, 2);
    }
    // if(g != 2) {
    //     printf("g %i, ",g,);
    //     cout << "g " << g << ", " << start[0] << ", " <<  start[1] << ", " << end[0] << ", " << end[1] << ", " << args[4] << ", " << args[5] << endl;
    //     testScore ++;
    // }
    // cout << "stage 2 " << sqrt(Vm[0] * Vm[0] + Vm[1] * Vm[1]) << endl;
    if(sqrt(Vm[0] * Vm[0] + Vm[1] * Vm[1]) > Vmax * 1.001) {
        g = 0;
        zero = atan2(Vm[1], Vm[0]);
        angle[0] = zero;
        g = newton(jac2, func2, args, angle, 1);
        for(n = nst; g != 2 && n <= barrier; n *= mult) {
            vmin = 2;
            for(i = 0; i < n; i++) {
                if(n == nst || i % mult != 0) {
                    angle[0] = zero + 2 * M_PI * i / n;
                    func2(angle, args, vnow);
                    if(vnow[0] < vmin) {
                        vmin = vnow[0];
                        imin = i;
                    }
                }
            }
            angle[0] = zero + 2 * M_PI * imin / n;
            g = newton(jac2, func2, args, angle, 1);
        }
        // cout << g << endl;
        // if(g != 2) {
        //     cout << "g2 " << g << ", " << start[0] << ", " << start[1] << ", " << end[0] << ", " << end[1] << ", " << r[0] << ", " << r[1] << endl;
        //     testScore ++;
        // }
        
        Vm[0] = cos(angle[0]) * Vmax;
        Vm[1] = sin(angle[0]) * Vmax;
    }
    res.x = Vm[0];
    res.y = Vm[1];
    // cout << "res" << Vm[0] << ", " << Vm[1] << endl;
    return res;
}
__host__ __device__ float estimate_time_for_speed(float speed,Point start_r,Point start_v,Point mid_r,Point mid_v,Point tgt_r,Point tgt_v,float max_acc,float max_speed)
{
    Point v_m1 = bangbang(start_v,mid_v,mid_r-start_r,max_acc,max_speed);
    Point v_m2 = bangbang(mid_v,tgt_v,tgt_r-mid_r,max_acc,max_speed);

    Point r1 = (v_m1 + start_v) * (v_m1 - start_v).mag() / (2 * max_acc);
    Point r3 = (v_m1 + mid_v) * (v_m1 - mid_v).mag() / (2 * max_acc);
    Point r2 = mid_r - start_r - r1 - r3;
    double t1 = (v_m1 -start_v).mag() / max_acc + r2.mag() / max_speed + (mid_v - v_m1).mag() / max_acc;
    r1 = (v_m2 + mid_v) * (v_m2 - mid_v).mag() / (2 * max_acc);
    r3 = (v_m2 + tgt_v) * (v_m2 - tgt_v).mag() / (2 * max_acc);
    r2 = tgt_r - mid_r - r1 - r3;
    double t2 = (v_m2 -mid_v).mag() / max_acc + r2.mag() / max_speed + (tgt_v - v_m2).mag() / max_acc;
    return t1+t2;
}
extern "C" __global__ void find_best_bb_speed(Point *speeds_n_poses,float *out,int N,float max_acc,float max_speed)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    Point start_r = speeds_n_poses[0];
    Point start_v = speeds_n_poses[1];
    Point mid_r = speeds_n_poses[2];
    Point mid_v = speeds_n_poses[3];
    Point tgt_r = speeds_n_poses[4];
    Point tgt_v = speeds_n_poses[5];
    out[0] = 10;
    
}