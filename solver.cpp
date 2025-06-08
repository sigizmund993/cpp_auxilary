#include <iostream>
#include <cmath>
using namespace std;

int main() {
    cout << "Привет, Code Runner!" << endl;
    return 0;
}

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
        for(j = 0; j < n; n++) {
            if(abs(fx[i]) > tol) {
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
            x[i] -= dx[i];
        }
    }
    return 0;
}

void func(double *x, int n, double *out) {

}