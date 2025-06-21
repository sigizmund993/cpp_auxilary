#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <vector>
#include <iostream>
#include <cmath>
#include <chrono>
using namespace std;
using namespace std::chrono;

vector<double> Va = {650, -890};
vector<double> Vb = {0, -1350};
vector<double> r = {-2600, -1460};
vector<double> it = {0, 0};
double Amax = 1000, Vmax = 1500;
long testScore = 0, testNum = 0;

int sgn(double a) {
    if(a > 0) return 1;
    if(a < 0) return -1;
    return 0;
}

double deltares(double *f, int n) {
    static double summ;
    summ = 0;
    for(int i = 0; i < n; i++) {
        summ += f[i] * f[i];
    }
    return sqrt(summ);
}

int gauss_sovle(double *a, double *b, int n, double *x) {
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

void num_jac(void (*f)(double*, double*, double*), double *x, double *jac, double *args, double *f0, int n, double d) {
    static int i, j;
    double f1[n];
    f(x, args, f0);
    for(i = 0; i < n; i++) {
        x[i] += d;
        f(x, args, f1);
        x[i] -= d;
        for(j = 0; j < n; j++) {
            jac[j * n + i] = (f1[j] - f0[j]) / d;
        }
    }
}

void jac1(void (*f)(double*, double*, double*), double *Vm, double *jac, double *args, double *fx, int n, double d) {
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

void jac2(void (*f)(double*, double*, double*), double *ang, double *jac, double *args, double *fx, int n, double d) {
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
    // fx[0] = acos(fx[0]);

    // static int i, j;
    // double f1[n];
    // f(ang, args, fx);
    // for(i = 0; i < n; i++) {
    //     ang[i] += d;
    //     f(ang, args, f1);
    //     ang[i] -= d;
    //     for(j = 0; j < n; j++) {
    //         jac[j * n + i] = (f1[j] - fx[j]) / d;
    //     }
    // }
    // cout << "num " << ang[0] << ", " << fx[0] << ", " << jac[0] << endl;
}

int newton(void (*jac)(void (*)(double*, double*, double*), double*, double*, double*, double*, int, double), 
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

void func1(double *Vm, double *args, double *fx) {
    static double ma, mb;
    ma = sqrt((args[0] - Vm[0]) * (args[0] - Vm[0]) + (args[1] - Vm[1]) * (args[1] - Vm[1]));
    mb = sqrt((args[2] - Vm[0]) * (args[2] - Vm[0]) + (args[3] - Vm[1]) * (args[3] - Vm[1]));
    // cout << "mamb " << ma << ", " << mb << ", " << (args[0] - Vm[0]) << ", " << (args[1] - Vm[1]) << endl;
    fx[0] = 2 * args[6] * args[4] - (Vm[0] + args[0]) * ma - (Vm[0] + args[2]) * mb;
    fx[1] = 2 * args[6] * args[5] - (Vm[1] + args[1]) * ma - (Vm[1] + args[3]) * mb;
}

void func2(double *ang, double *args, double *fx) {
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
    // end = high_resolution_clock::now();
    // duration<double, micro> duration_us = duration_cast<duration<double, micro>>(end - start);
    // cout << "time " << duration_us.count() << endl;
}

vector<double> bangbang(vector<double> start, vector<double> end, vector<double> dr, double Amax, double Vmax, int nshort = 10, int nst = 10, int mult = 2, int barrier = 10000) { // gang-bang
    static double args[8], angle[1], rn, Vm[2], zero, imin, vmin, vnow[1];
    static int i, n;
    args[0] = start[0];
    args[1] = start[1];
    args[2] = end[0];
    args[3] = end[1];
    args[4] = dr[0];
    args[5] = dr[1];
    args[6] = Amax;
    args[7] = Vmax;
    rn = sqrt(args[4] * args[4] + args[5] * args[5]);Vm[0] = args[4] / rn * Vmax;
    Vm[1] = args[5] / rn * Vmax;
    static vector<double> res = {0, 0};
    int g = 0;
    zero = atan2(args[5], args[4]);
    for(i = 0; g != 2 && i < nshort; i++) {
        Vm[0] = cos(zero + 2 * M_PI * i / nshort) * Vmax;
        Vm[1] = sin(zero + 2 * M_PI * i / nshort) * Vmax;
        g = newton(jac1, func1, args, Vm, 2);
    }
    if(g != 2) {
        cout << "g " << g << ", " << start[0] << ", " <<  start[1] << ", " << end[0] << ", " << end[1] << ", " << args[4] << ", " << args[5] << endl;
        testScore ++;
    }
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
        if(g != 2) {
            cout << "g2 " << g << ", " << start[0] << ", " << start[1] << ", " << end[0] << ", " << end[1] << ", " << r[0] << ", " << r[1] << endl;
            testScore ++;
        }
        
        Vm[0] = cos(angle[0]) * Vmax;
        Vm[1] = sin(angle[0]) * Vmax;
    }
    res[0] = Vm[0];
    res[1] = Vm[1];
    // cout << "res" << Vm[0] << ", " << Vm[1] << endl;
    return res;
}

PYBIND11_MODULE(solver, m) {
    m.def("bangbang", &bangbang, "гэнгбэнг, хуле");
}

void test() {
    static high_resolution_clock::time_point start, end;
    static int i, j, p1, p2;
    static float a1, a2;
    start = high_resolution_clock::now();
    for(i = -3000; i <= 3000; i += 300) {
        for(j = -3000; j <= 3000; j += 300) {
            for(a1 = 0; a1 < 2 * M_PI - 0.01; a1 += M_PI / 10) {
                for(p1 = 0; p1 <= Vmax; p1 += 100) {
                    for(a2 = 0; a2 < 2 * M_PI - 0.01; a2 += M_PI / 10) {
                        for(p2 = 0; p2 <= Vmax; p2 += 100) {
                            testNum ++;
                            Va = {p1 * cos(a1), p1 * sin(a1)};
                            Vb = {p2 * cos(a2), p2 * sin(a2)};
                            r = {double(i), double(j)};
                            // cout << "hihi" << endl;
                            bangbang(Va, Vb, r, Amax, Vmax);
                            // if(duration_us.count() > 1000) {
                            //     cout << Va[0] << ", " << Va[1] << ", " << Vb[0] << ", " << Vb[1] << ", " << r[0] << ", " << r[1] << ", " << duration_us.count() << endl;
                            // }
                        }
                    }
                }
            }
            cout << "lol " << i << ", " << j << endl;
        }
    }
    end = high_resolution_clock::now();
    duration<double, micro> duration_us = duration_cast<duration<double, micro>>(end - start);
    cout << "summ time in " << testNum / 1000000 << " " << testNum % 1000000 / 1000 << " " << testNum % 1000 << " tests (s) " << duration_us.count() / 1000000.0 << endl;
    cout << "average time (mcs) " << duration_us.count() / testNum << endl;
    cout << "test score " << testScore << endl;
}

int main() {
    // test();
    // static high_resolution_clock::time_point start, end;
    // vector<double> Vm = {0, 0};
    // Vm = bangbang(Va, Vb, r, Amax, Vmax);
    // start = high_resolution_clock::now();
    // Vm = bangbang(Va, Vb, r, Amax, Vmax);
    // end = high_resolution_clock::now();
    // duration<double, micro> duration_us = duration_cast<duration<double, micro>>(end - start);
    // cout << Vm[0] << ", " << Vm[1] << " time " << duration_us.count() << endl;
    // double args[8] = {-809, 587.8, -1331.5, 432.6, -3000, -3000, 1000, 1500};
    // double e[1], j[1], x[1] = {1.2};
    // func2(x, args, e);
    // cout << e[0] << endl;
    // jac2(func2, x, j, args, e, 1, 1e-7);
    // cout << e[0] << endl;
    return 0;
}