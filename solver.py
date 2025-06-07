import numpy as np
import time 
ti = time.time() 
def jacobian(f, x):
    h = 1.0e-4
    n = len(x)
    Jac = np.zeros([n,n])
    f0 = f(x)
    for i in np.arange(0,n,1):
        tt = x[i].copy()
        x[i] = tt + h
        f1= f(x)
        x[i] = tt
        Jac [:,i] = (f1 - f0)/h
    return Jac, f0

def quadr(f):
    summ = 0
    for i in f:
        summ += i ** 2
    return summ

def gauss_solve(A, b):
    A = A.astype(float)  # чтобы избежать ошибок с целыми
    b = b.astype(float)
    n = len(b)

    # Прямой ход (приведение к верхнетреугольному виду)
    for i in range(n):
        # Находим ведущий элемент
        max_row = i + np.argmax(np.abs(A[i:, i]))
        if A[max_row, i] == 0:
            raise ValueError("Система не имеет единственного решения")

        # Меняем строки местами
        A[[i, max_row]] = A[[max_row, i]]
        b[[i, max_row]] = b[[max_row, i]]

        # Обнуляем элементы ниже ведущего
        for j in range(i + 1, n):
            factor = A[j, i] / A[i, i]
            A[j, i:] -= factor * A[i, i:]
            b[j] -= factor * b[i]

    # Обратная подстановка
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x[i] = (b[i] - np.dot(A[i, i + 1:], x[i + 1:])) / A[i, i]
    return x


def newton(f, x, tol=1.0e-9):
    iterMax = 50
    for i in range(iterMax):
        Jac, fO = jacobian(f, x)
        if np.sqrt(quadr(fO) / len(x)) < tol:
                return x, i                 
        dx = gauss_solve(Jac, fO)
        x = x - dx
    print ("Too many iterations for the Newton method")
n=1
def f(x):
    f = np.zeros([n])
    f[0] = x[0] ** 3 - x[0] ** 2 + 1
    # f[1] = np.sin(np.pi / 2 - x[1])
    # f = np.zeros([n])
    # for i in np.arange(0,n-1,1):
    #         f[i] = (3 + 2*x[i])*x[i] - x[i-1] - 2*x[i+1] - 2
    # f [0] = (3 + 2*x[0] )*x[0] - 2*x[1] - 3
    # f[n-1] = (3 + 2*x[n-1] )*x[n-1] - x[n-2] - 4
    return f
x0 = np.zeros([n])
x0[0] = 0.66666666666667
x, iter = newton(f, x0)
print ('Solution:\n', x)
print ('Newton iteration = ', iter)
print('Newton method time', round(time.time()-ti,7), 'seconds')