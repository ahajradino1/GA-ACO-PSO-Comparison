from dataclasses import dataclass
import random
import numpy as np
import matplotlib.pyplot as plt
from plots import plot_scatter
from plots import plot_function_values


def levy(x1, x2):
    w1 = 1 + (x1 - 1) / 4
    w2 = 1 + (x2 - 1) / 4
    return np.sin(np.pi * w1) ** 2 + (x1 - 1) ** 2 * (1 + 10 * (np.sin(np.pi * x1 + 1)) ** 2) + (x2 - 1) ** 2 * (
            1 + 10 * (np.sin(np.pi * x2 + 1)) ** 2) + (w2 - 1) ** 2 * (1 + np.sin(2 * np.pi * w2) ** 2)


LEVY = lambda x: levy(x[0], x[1])
OPSEG_LEVY = [-10, 10]


def plot(f, interval, title, points, z_label="$f(x_1,x_2)$", x_label="$x_1$", y_label="$x_2$"):
    x1 = np.linspace(interval[0], interval[1], 100)
    x2 = np.linspace(interval[0], interval[1], 100)
    X1, X2 = np.meshgrid(x1, x2)
    fun = np.vectorize(f)
    Y = f([X1, X2])
    plt.figure(figsize=(5, 4))
    ax = plt.axes(projection="3d")
    ax.contour(X1, X2, Y, 50, cmap="binary", alpha=0.5)

    for point in points:
        ax.scatter(point[0][0], point[0][1], f(point[0]), color=point[1], marker=point[2], s=100)

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_zlabel(z_label)
    ax.set_title(title)
    plt.show()


@dataclass
class Particle:
    x: np.array  # pozicija
    v: np.array  # vektor tekuce brzine
    x_best: np.array  # najbolja pozicija


# N – maksimalan broj iteracija,
# M – maksimalan broj čestica: 20 - 40,
# eps – minimalna promjena po vrijednosti funkcije između dvije sukcesivne iteracije,
# P – vektor koji sadrži parametre W, C1 i C2.
def PSO(f, opseg, N, M, eps, P, n, plot):
    W, C1, C2, for_decreasing = P[0], P[1], P[2], 0.5 / N
    r1, r2 = random.uniform(0, 1), random.uniform(0, 1)
    x_min, x_max = (opseg[0] - opseg[1]) / 2, (opseg[1] - opseg[0]) / 2

    population = []

    for i in range(0, M):
        x, velocity = [], []
        for j in range(0, n):
            x.append(random.uniform(opseg[0], opseg[1]))
            velocity.append(random.uniform(x_min, x_max))
        population.append(Particle(np.array(x), np.array(velocity), np.array(x)))

    best_patricle = min(population, key=lambda par: f(par.x))
    x_best = best_patricle.x
    y_best = f(x_best)

    no_iterations = 0
    function_values = [y_best]

    while no_iterations < N:
        for i in range(0, M):
            for j in range(0, n):
                population[i].v[j] = W[j] * population[i].v[j] + C1[j] * r1 * (population[i].x_best[j] -
                                                                               population[i].x[j]) + C2[j] * r2 * (
                                             x_best[j] - population[i].x[j])

            population[i].x = population[i].x + population[i].v

            if not all(opseg[0] <= el <= opseg[1] for el in population[i].x):
                population[i].x = population[i].x_best

            if f(population[i].x) < f(population[i].x_best):
                population[i].x_best = population[i].x

            if abs(y_best - f(population[i].x)) < eps:
                return x_best, y_best

            if f(population[i].x) < y_best:
                x_best = population[i].x
                y_best = f(population[i].x)

        function_values.append(f(x_best))

        if plot and (no_iterations == 0 or no_iterations == int(N / 3) or no_iterations == int(
                2 * N / 3) or no_iterations == N - 1):
            x_draw = [el.x[0] for el in population]
            y_draw = [el.x[1] for el in population]
            result = [x_draw, y_draw]
            best = x_best
            print(no_iterations, " ", x_best, " ", f(x_best))
            plot_scatter(result, best, opseg)

        no_iterations += 1
        W = [w - for_decreasing for w in W]
    if plot:
        plot_function_values(function_values, 'Optimizacija rojem cestica')

    return x_best, y_best
