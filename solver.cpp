#include <iostream>
#include <cmath>
#include <chrono>
using namespace std;
using namespace std::chrono;

double Va[2] = {5, 5};
double Vb[2] = {1, -12};
double r[2] = {90, 40};
double Am = 1, Vmax = 13;

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
    double *x, int n, double tol = 1e-10, int max_iter = 1000, double d = 1e-8) {
    static int i, j;
    static bool flag;
    double jacobian[n * n], fx[n], dx[n];
    // static high_resolution_clock::time_point start, end;
    // cout << "start " << x[0] << endl;
    for(i = 0; i < max_iter; i++) {
        // start = high_resolution_clock::now();
        jac(f, x, n, jacobian, fx, d);
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
            jac(f, x, n, jacobian, fx, -d);
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

void func1(double *Vm, int n, double *fx) {
    static double ma, mb;
    ma = sqrt((Va[0] - Vm[0]) * (Va[0] - Vm[0]) + (Va[1] - Vm[1]) * (Va[1] - Vm[1]));
    mb = sqrt((Vb[0] - Vm[0]) * (Vb[0] - Vm[0]) + (Vb[1] - Vm[1]) * (Vb[1] - Vm[1]));
    fx[0] = 2 * Am * r[0] - (Vm[0] + Va[0]) * ma - (Vm[0] + Vb[0]) * mb;
    fx[1] = 2 * Am * r[1] - (Vm[1] + Va[1]) * ma - (Vm[1] + Vb[1]) * mb;
}

void func2(double *ang, int n, double *fx) {
    static double Vm[2], lhs[2], ma, mb, ln;
    // static high_resolution_clock::time_point start, end;
    // start = high_resolution_clock::now();
    Vm[0] = cos(ang[0]) * Vmax;
    Vm[1] = sin(ang[0]) * Vmax;
    ma = sqrt((Va[0] - Vm[0]) * (Va[0] - Vm[0]) + (Va[1] - Vm[1]) * (Va[1] - Vm[1]));
    mb = sqrt((Vb[0] - Vm[0]) * (Vb[0] - Vm[0]) + (Vb[1] - Vm[1]) * (Vb[1] - Vm[1]));
    lhs[0] = 2 * Am * r[0] - (Vm[0] + Va[0]) * ma - (Vm[0] + Vb[0]) * mb;
    lhs[1] = 2 * Am * r[1] - (Vm[1] + Va[1]) * ma - (Vm[1] + Vb[1]) * mb;
    // cout << lhs[0] << " " << lhs[1] << " " << Vm[0] << " " << Vm[1] << " " << ma << " " << mb << endl;
    ln = sqrt(lhs[0] * lhs[0] + lhs[1] * lhs[1]);
    fx[0] = acos((lhs[0] * Vm[0] + lhs[1] * Vm[1]) / ln / Vmax);
    // end = high_resolution_clock::now();
    // duration<double, micro> duration_us = duration_cast<duration<double, micro>>(end - start);
    // cout << "time " << duration_us.count() << endl;
}

int main() {
    double x[2], fx[2];
    int g;
    x[0] = 0.5;
    x[1] = 0.5;
    //func2(x, 1, fx);
    func1(x, 2, fx);
    high_resolution_clock::time_point start, end;
    start = high_resolution_clock::now();
    g = newton(num_jac, func1, x, 2);
    end = high_resolution_clock::now();
    // x[0] = 0.9301568129232883;
    func2(x, 1, fx);
    duration<double, micro> duration_us = duration_cast<duration<double, micro>>(end - start);
    cout << "Result " << x[0] << " " << x[1] << " error type (2 - ok, 1 - error with 0 jac / numjac, 0 - steps max count) " << g << " время " << duration_us.count() << endl;
    return 0;
}