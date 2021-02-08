import numpy as np
import random as r

RANG_SP = 1.7
VelicinaProblema = 2


def RASTRIGIN(x):
    sum = 10 * VelicinaProblema
    for i in range(0, VelicinaProblema):
        sum += x[i] ** 2 - 10 * np.cos(2 * np.pi * x[i])
    return sum


OPSEG_RASTRIGIN = [-5.12, 5.12]

SCHAFFER2 = lambda x: 0.5 + ((np.sin(x[0] ** 2 - x[1] ** 2)) ** 2 - 0.5) / (1 + 0.001 * (x[0] ** 2 + x[1] ** 2)) ** 2

OPSEG_SCHAFFER2 = [-100, 100]


def ACKLEY(x, a=20, b=0.2, c=2 * np.pi):
    x = np.asarray_chkfinite(x)  # ValueError if any NaN or Inf
    n = len(x)
    s1 = sum(x ** 2)
    s2 = sum(np.cos(c * x))
    return -a * np.exp(-b * np.sqrt(s1 / n)) - np.exp(s2 / n) + a + np.exp(1)


OPSEG_ACKLEY = [-32.768, 32.768]


def DIXONPRICE(x):  # dp.m
    x = np.asarray_chkfinite(x)
    n = len(x)
    j = np.arange(2, n + 1)
    x2 = 2 * x ** 2
    return sum(j * (x2[1:] - x[:-1]) ** 2) + (x[0] - 1) ** 2


OPSEG_DIXONPRICE = [-10, 10]


def LEVY(x):
    x = np.asarray_chkfinite(x)
    n = len(x)
    z = 1 + (x - 1) / 4
    return (np.sin(np.pi * z[0]) ** 2
            + sum((z[:-1] - 1) ** 2 * (1 + 10 * np.sin(np.pi * z[:-1] + 1) ** 2))
            + (z[-1] - 1) ** 2 * (1 + np.sin(2 * np.pi * z[-1]) ** 2))


OPSEG_LEVY = [-10, 10]


def GRIEWANK(x, fr=4000):
    x = np.asarray_chkfinite(x)
    n = len(x)
    j = np.arange(1., n + 1)
    s = sum(x ** 2)
    p = np.prod(np.cos(x / np.sqrt(j)))
    return s / fr - p + 1


OPSEG_GRIEWANK = [-600, 600]


def ROSENBROCK(x):
    x = np.asarray_chkfinite(x)
    x0 = x[:-1]
    x1 = x[1:]
    return sum((1 - x0) ** 2) + 100 * sum((x1 - x0 ** 2) ** 2)


OPSEG_ROSENBROCK = [-2.048, 2.048]


def ZAKHAROV(x):  # zakh.m
    x = np.asarray_chkfinite(x)
    n = len(x)
    j = np.arange(1., n + 1)
    s2 = sum(j * x) / 2
    return sum(x ** 2) + s2 ** 2 + s2 ** 4


OPSEG_ZAKHAROV = [-5, 10]

HOLDER_TABLE = lambda x: -np.fabs(
    np.sin(x[0]) * np.cos(x[1]) * np.exp(np.fabs(1 - np.sqrt(x[0] ** 2 + x[1] ** 2) / np.pi)))

OPSEG_HOLDER_TABLE = [-10, 10]

DROP_WAVE = lambda x: -1 * (1 + np.cos(12 * np.sqrt(x[0] ** 2 + x[1] ** 2))) / (0.5 * (x[0] ** 2 + x[1] ** 2) + 2)

OPSEG_DROP_WAVE = [-5.12, 5.12]

PARABOLOID = lambda x: (x[0] - 3) ** 2 + (x[1] + 1) ** 2
OPSEG_PARABOLOID = [-5, 5]


def SPHERE(x):
    x = np.asarray_chkfinite(x)
    return sum(x ** 2)


OPSEG_SPHERE = [-5.12, 5.12]


def SUM_SQUARES(x):
    x = np.asarray_chkfinite(x)
    n = len(x)
    j = np.arange(1., n + 1)
    return sum(j * x ** 2)


OPSEG_SUM2 = [-5.12, 5.12]


def QUARTIC(x):
    n = len(x)
    sum = 0
    for i in range(0, n):
        sum += (i + 1) * x[i] ** 4
    return sum + r.random()


OPSEG_QUARTIC = [-1.28, 1.28]


def SCHWEFEL(x):  # schw.m
    x = np.asarray_chkfinite(x)
    n = len(x)
    return 418.9829 * n - sum(x * np.sin(np.sqrt(abs(x))))


OPSEG_SCHWEFEL = [-500, 500]

def MICHALEWICZ(x):
    x = np.asarray_chkfinite(x)
    n = len(x)
    j = np.arange(1., n+1)
    return - sum(np.sin(x) * np.sin(j * x**2 / np.pi ) ** (2 * 10))

OPSEG_MICHALEWICZ = [0, np.pi]

MATYAS = lambda x: 0.26 * (x[0] ** 2 + x[1] ** 2) - 0.48 * x[0] * x[1]
OPSEG_MATYAS = [-10, 10]
