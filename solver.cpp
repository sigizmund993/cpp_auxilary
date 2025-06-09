#include <iostream>
#include <cmath>
#include <chrono>
using namespace std;
using namespace std::chrono;

double Va[2] = {5, 5};
double Vb[2] = {1, -12};
double r[2] = {180, 80};
double Amax = 1, Vmax = 13;

double deltares(double *f, int n) {
    static double summ;
    summ = 0;
    for(int i = 0; i < n; i++) {
        summ += f[i] * f[i];
    }
    return sqrt(summ);
}

int gauss_sovle(double *a, double *b, int n, double *x) {
    static double max_row, maxV, val, k;
    static int maxJ, i1, i2, i, j, p;
    maxV = 0;
    maxJ = 0;
    for(i = 0; i < n; i++) {
        for(j = 0; j < n - i; j++) {
            val = abs(a[(i + j) * n + i]);
            if(val > maxV) {
                maxV = val;
                maxJ = 0;
            }
        }
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
        for(j = 1; j < n - i; j++) {
            k = a[(i + j) * n + i] / a[i * n + i];
            for(p = 1; p < n - 1; p++) {
                a[(i + j) * n + i + p] -= a[i * n + i + p] * k;
            }
            b[i + j] -= b[i] * k;
        }
    }
    for(i = n - 1; i >= 0; i--) {
        for(j = n - 1; j > i; j--) {
            b[i] -= a[i * n + j] * x[j];
        }
        x[i] = b[i] / a[i * n + i];
    }
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

int newton(void (*jac)(void (*)(double*, double*, double*), double*, double*, double*, double*, int, double), 
    void (*f)(double*, double*, double*), double *args, double *x, int n, double tol = 1e-10, int max_iter = 1000, double d = 1e-8) {
    static int i, j;
    static bool flag;
    double jacobian[n * n], fx[n], dx[n];
    // static high_resolution_clock::time_point start, end;
    // cout << "start " << x[0] << endl;
    for(i = 0; i < max_iter; i++) {
        // start = high_resolution_clock::now();
        jac(f, x, jacobian, args, fx, n, d);
        // cout << "jac " << jacobian[0] << endl;
        flag = false;
        for(j = 0; j < n * n; j++) {
            if(jacobian[j] == 0) {
                flag = true;
                cout << "flag" << endl;
                break;
            }
        }
        if(flag) {
            jac(f, x, jacobian, args, fx, n, -d);
            // cout << "jac " << jacobian[0] << endl;
        }
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
            return 1;
        }
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
    // cout << lhs[0] << " " << lhs[1] << " " << Vm[0] << " " << Vm[1] << " " << ma << " " << mb << endl;
    ln = sqrt(lhs[0] * lhs[0] + lhs[1] * lhs[1]);
    fx[0] = acos((lhs[0] * Vm[0] + lhs[1] * Vm[1]) / ln / args[7]);
    // end = high_resolution_clock::now();
    // duration<double, micro> duration_us = duration_cast<duration<double, micro>>(end - start);
    // cout << "time " << duration_us.count() << endl;
}

void bangbang(double *start, double *end, double *r, double Amax, double Vmax, double *Vm) { // gang-bang
    static double args[8] = {start[0], start[1], end[0], end[1], r[0], r[1], Amax, Vmax}, fx[2], angle[1], rn = sqrt(r[0] * r[0] + r[1] * r[1]);
    static int g;
    Vm[0] = r[0] / rn * Vmax;
    Vm[1] = r[1] / rn * Vmax;
    g = newton(num_jac, func1, args, Vm, 2, 1e-7);
    // cout << "Res 1 stage " << Vm[0] << " " << Vm[1] << " status (2 - ok, 1 - jac err, 0 - iter err) " << g << endl;
    if(sqrt(Vm[0] * Vm[0] + Vm[1] * Vm[1]) > Vmax) {
        angle[0] = atan2(r[1], r[0]);
        // cout << "Vm > Vmax" << endl;
        g = newton(num_jac, func2, args, angle, 1);
        Vm[0] = cos(angle[0]) * Vmax;
        Vm[1] = sin(angle[0]) * Vmax;
        // cout << "Res 2 stage " << angle[0] << " " << Vm[0] <<  " " << Vm[1] << " status (2 - ok, 1 - jac err, 0 - iter err) " << g << endl;
    }
}

int main() {
    double Vm[2], Va[2] = {0, 0}, Vb[2] = {0, 0}, r[2] = {200, 200}, Amax = 1, Vmax = 13;
    high_resolution_clock::time_point start, end;
    bangbang(Va, Vb, r, Amax, Vmax, Vm);
    start = high_resolution_clock::now();
    bangbang(Va, Vb, r, Amax, Vmax, Vm);
    end = high_resolution_clock::now();
    duration<double, micro> duration_us = duration_cast<duration<double, micro>>(end - start);
    cout << "time " << duration_us.count() << " val " << Vm[0] << " " << Vm[1] << endl;
    return 0;
}