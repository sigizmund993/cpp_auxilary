#include <iostream>
#include <cmath>
#include <chrono>
using namespace std;
using namespace std::chrono;

double quadr(double *f, int n) {
    static double summ;
    summ = 0;
    for(int i = 0; i < n; i++) {
        summ += f[i] * f[i];
    }
    return summ;
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

void num_jac(void (*f)(double*, int, double*), double *x, int n, double *jac, double *f0, double d) {
    static int i, j;
    double f1[n];
    f(x, n, f0);
    for(i = 0; i < n; i++) {
        x[i] += d;
        f(x, n, f1);
        x[i] -= d;
        for(j = 0; j < n; j++) {
            jac[j * n + i] = (f1[j] - f0[j]) / d;
        }
    }
}

int newton(void (*jac)(void (*)(double*, int, double*), double*, int, double*, double*, double), void (*f)(double*, int, double*), 
    double *x, int n, double tol = 1e-9, int max_iter = 100, double d = 1e-6) {
    static int i, j;
    static float flag;
    double jacobian[n * n], fx[n], dx[n];
    for(i = 0; i < max_iter; i++) {
        jac(f, x, n, jacobian, fx, d);
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
    }
    return 0;
}

void func(double *x, int n, double *out) {
    out[0] = x[0] * x[0] * x[0] - x[0] * x[0] + 1;
    out[1] = x[1] * x[1] * x[1] - x[1] * x[1] + 1;
}

int main() {
    double x[2];
    int g;
    x[0] = 0.5;
    x[1] = 0.5;
    high_resolution_clock::time_point start = high_resolution_clock::now();
    g = newton(num_jac, func, x, 2);
    high_resolution_clock::time_point end = high_resolution_clock::now();
    duration<double, micro> duration_us = duration_cast<duration<double, micro>>(end - start);
    cout << "Привет, Code Runner! " << x[0] << " " << x[1] << " ошибка " << g << " время " << duration_us.count() << endl;
    return 0;
}