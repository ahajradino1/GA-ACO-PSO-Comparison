import random
import time
from comparative_statistics import find_statistics

from ga import Populacija
from pso import PSO
from aco import ACO

from definitions import RASTRIGIN
from definitions import OPSEG_RASTRIGIN
from definitions import DROP_WAVE
from definitions import OPSEG_DROP_WAVE
from definitions import HOLDER_TABLE
from definitions import OPSEG_HOLDER_TABLE
from definitions import LEVY
from definitions import OPSEG_LEVY
from definitions import ZAKHAROV
from definitions import OPSEG_ZAKHAROV
from definitions import ROSENBROCK
from definitions import OPSEG_ROSENBROCK
from definitions import SCHAFFER2
from definitions import OPSEG_SCHAFFER2
from definitions import ACKLEY
from definitions import OPSEG_ACKLEY
from definitions import DIXONPRICE
from definitions import OPSEG_DIXONPRICE
from definitions import SPHERE
from definitions import OPSEG_SPHERE
from definitions import SUM_SQUARES
from definitions import OPSEG_SUM2
from definitions import QUARTIC
from definitions import OPSEG_QUARTIC
from definitions import GRIEWANK
from definitions import OPSEG_GRIEWANK
from definitions import MATYAS
from definitions import OPSEG_MATYAS
from definitions import MICHALEWICZ
from definitions import OPSEG_MICHALEWICZ


def particles_prevelance():
    P = generate_p(2)
    x, f_x = PSO(RASTRIGIN, OPSEG_RASTRIGIN, 20, 30, 0, P, 2, True)
    print(x, f_x)

    p = Populacija(RASTRIGIN, OPSEG_RASTRIGIN, 2, 1.7, 30, 0.75, 0.15, 20, 1, 20, True)
    x, f_x = p.GenerisiGeneracije()
    print(x, f_x)

    x_best, f_best = ACO(RASTRIGIN, OPSEG_RASTRIGIN, 20, 30, 50, 0.0001, 0.5, 2, True)
    print(x_best, f_best)


def unimodal_separable(n, chromosome_length, no_iterations, population_size, no_executions, plot=False):
    P = generate_p(n)

    sphere_pso, sphere_ga, sphere_aco, sum2_pso = [], [], [], []
    sum2_ga, sum2_aco, quartic_pso, quartic_ga, quartic_aco = [], [], [], [], []

    for i in range(0, no_executions):
        print(i)
        x, f_x = PSO(SPHERE, OPSEG_SPHERE, no_iterations, population_size, 0, P, n, plot)
        sphere_pso.append(f_x)

        p = Populacija(SPHERE, OPSEG_SPHERE, n, 1.7, population_size, 0.75, 0.15, no_iterations, 1, chromosome_length,
                       plot)
        x, f_x = p.GenerisiGeneracije()
        sphere_ga.append(f_x)

        x, f_x = ACO(SPHERE, OPSEG_SPHERE, no_iterations, population_size, 50, 0.0001, 0.5, n, plot)
        sphere_aco.append(f_x)

        x, f_x = PSO(SUM_SQUARES, OPSEG_SUM2, no_iterations, population_size, 0, P, n, plot)
        sum2_pso.append(f_x)

        p = Populacija(SUM_SQUARES, OPSEG_SUM2, n, 1.7, population_size, 0.75, 0.15, no_iterations, 1,
                       chromosome_length, plot)
        x, f_x = p.GenerisiGeneracije()
        sum2_ga.append(f_x)

        x, f_x = ACO(SUM_SQUARES, OPSEG_SUM2, no_iterations, population_size, 50, 0.0001, 0.5, n, plot)
        sum2_aco.append(f_x)

        x, f_x = PSO(QUARTIC, OPSEG_QUARTIC, no_iterations, population_size, 0, P, n, plot)
        quartic_pso.append(f_x)

        p = Populacija(QUARTIC, OPSEG_QUARTIC, n, 1.7, population_size, 0.75, 0.15, no_iterations, 1, chromosome_length,
                       plot)
        x, f_x = p.GenerisiGeneracije()
        quartic_ga.append(f_x)

        x, f_x = ACO(QUARTIC, OPSEG_QUARTIC, no_iterations, population_size, 50, 0.0001, 0.5, n, plot)
        quartic_aco.append(f_x)

    print('PSO')
    print('Sphere:')
    find_statistics(sphere_pso)
    print('Sum Squares:')
    find_statistics(sum2_pso)
    print('Quartic')
    find_statistics(quartic_pso)

    print('------------------------------------------------------------------------')

    print('GA')
    print('Sphere:')
    find_statistics(sphere_ga)
    print('Sum Squares:')
    find_statistics(sum2_ga)
    print('Quartic')
    find_statistics(quartic_ga)

    print('------------------------------------------------------------------------')

    print('ACO')
    print('Sphere:')
    find_statistics(sphere_aco)
    print('Sum Squares:')
    find_statistics(sum2_aco)
    print('Quartic')
    find_statistics(quartic_aco)

    print('------------------------------------------------------------------------')


def multimodal_separable(n, chromosome_length, no_iterations, population_size, no_executions, plot=False):
    P = generate_p(n)
    rastrigin_pso, rastrigin_ga, rastrigin_aco, ht_pso, ht_ga, ht_aco, micha_pso, micha_ga, micha_aco = [], [], [], [], [], [], [], [], []

    for i in range(0, no_executions):
        print(i)
        x, f_x = PSO(RASTRIGIN, OPSEG_RASTRIGIN, no_iterations, population_size, 0, P, n, plot)
        rastrigin_pso.append(f_x)

        p = Populacija(RASTRIGIN, OPSEG_RASTRIGIN, n, 1.7, population_size, 0.75, 0.15, no_iterations, 1,
                       chromosome_length, plot)
        x, f_x = p.GenerisiGeneracije()
        rastrigin_ga.append(f_x)

        x, f_x = ACO(RASTRIGIN, OPSEG_RASTRIGIN, no_iterations, population_size, 50, 0.0001, 0.5, n, plot)
        rastrigin_aco.append(f_x)

        x, f_x = PSO(HOLDER_TABLE, OPSEG_HOLDER_TABLE, no_iterations, population_size, 0, P, 2, plot)
        ht_pso.append(f_x)

        p = Populacija(HOLDER_TABLE, OPSEG_HOLDER_TABLE, 2, 1.7, population_size, 0.75, 0.15, no_iterations, 1,
                       chromosome_length, plot)
        x, f_x = p.GenerisiGeneracije()
        ht_ga.append(f_x)

        x, f_x = ACO(HOLDER_TABLE, OPSEG_HOLDER_TABLE, no_iterations, population_size, 50, 0.0001, 0.5, 2, plot)
        ht_aco.append(f_x)

        x, f_x = PSO(MICHALEWICZ, OPSEG_MICHALEWICZ, no_iterations, population_size, 0, P, n, plot)
        micha_pso.append(f_x)

        p = Populacija(MICHALEWICZ, OPSEG_MICHALEWICZ, n, 1.7, population_size, 0.75, 0.15, no_iterations, 1,
                       chromosome_length, plot)
        x, f_x = p.GenerisiGeneracije()
        micha_ga.append(f_x)

        x, f_x = ACO(MICHALEWICZ, OPSEG_MICHALEWICZ, no_iterations, population_size, 50, 0.0001, 0.5, n, plot)
        micha_aco.append(f_x)

    print('PSO')
    print('Rastrigin:')
    find_statistics(rastrigin_pso)
    print('Holder-Table:')
    find_statistics(ht_pso)
    print('Michalewicz:')
    find_statistics(micha_pso)

    print('------------------------------------------------------------------------')

    print('GA')
    print('Rastrigin:')
    find_statistics(rastrigin_ga)
    print('Holder-Table:')
    find_statistics(ht_ga)
    print('Michalewicz:')
    find_statistics(micha_ga)

    print('------------------------------------------------------------------------')

    print('ACO')
    print('Rastrigin:')
    find_statistics(rastrigin_aco)
    print('Holder-Table:')
    find_statistics(ht_aco)
    print('Michalewicz:')
    find_statistics(micha_aco)

    print('------------------------------------------------------------------------')


def multimodal_non_separable(n, chromosome_length, no_iterations, no_executions, population_size, plot=False):
    P = generate_p(n)
    schaffer_pso, schaffer_ga, schaffer_aco = [], [], []
    dw_pso, dw_ga, dw_aco = [], [], []
    rosenbrock_pso, rosenbrock_ga, rosenbrock_aco = [], [], []
    ackley_pso, ackley_ga, ackley_aco = [], [], []
    levy_pso, levy_ga, levy_aco = [], [], []
    griewank_pso, griewank_ga, griewank_aco = [], [], []

    for i in range(0, no_executions):
        x, f_x = PSO(SCHAFFER2, OPSEG_SCHAFFER2, no_iterations, population_size, 0, P, 2, plot)
        schaffer_pso.append(f_x)

        p = Populacija(SCHAFFER2, OPSEG_SCHAFFER2, 2, 1.7, population_size, 0.75, 0.15, no_iterations, 1,
                       chromosome_length, plot)
        x, f_x = p.GenerisiGeneracije()
        schaffer_ga.append(f_x)

        x, f_x = ACO(SCHAFFER2, OPSEG_SCHAFFER2, no_iterations, population_size, 50, 0.0001, 0.5, 2, plot)
        schaffer_aco.append(f_x)

        x, f_x = PSO(DROP_WAVE, OPSEG_DROP_WAVE, no_iterations, population_size, 0, P, 2, plot)
        dw_pso.append(f_x)

        p = Populacija(DROP_WAVE, OPSEG_DROP_WAVE, 2, 1.7, population_size, 0.75, 0.15, no_iterations, 1,
                       chromosome_length, plot)
        p.GenerisiGeneracije()
        dw_ga.append(f_x)

        x, f_x = ACO(DROP_WAVE, OPSEG_DROP_WAVE, no_iterations, population_size, 50, 0.0001, 0.5, 2, plot)
        dw_aco.append(f_x)

        x, f_x = PSO(ROSENBROCK, OPSEG_ROSENBROCK, no_iterations, population_size, 0, P, n, plot)
        rosenbrock_pso.append(f_x)

        p = Populacija(ROSENBROCK, OPSEG_ROSENBROCK, n, 1.7, population_size, 0.75, 0.15, no_iterations, 1,
                       chromosome_length, plot)
        x, f_x = p.GenerisiGeneracije()
        rosenbrock_ga.append(f_x)

        x, f_x = ACO(ROSENBROCK, OPSEG_ROSENBROCK, no_iterations, population_size, 50, 0.0001, 0.5, n, plot)
        rosenbrock_aco.append(f_x)

        x, f_x = PSO(ACKLEY, OPSEG_ACKLEY, no_iterations, population_size, 0, P, n, plot)
        ackley_pso.append(f_x)

        p = Populacija(ACKLEY, OPSEG_ACKLEY, n, 1.7, population_size, 0.75, 0.15, no_iterations, 1, chromosome_length,
                       plot)
        x, f_x = p.GenerisiGeneracije()
        ackley_ga.append(f_x)

        x, f_x = ACO(ACKLEY, OPSEG_ACKLEY, no_iterations, population_size, 50, 0.0001, 0.5, n, plot)
        ackley_aco.append(f_x)

        x, f_x = PSO(LEVY, OPSEG_LEVY, no_iterations, population_size, 0, P, n, plot)
        levy_pso.append(f_x)

        p = Populacija(LEVY, OPSEG_LEVY, n, 1.7, population_size, 0.75, 0.15, no_iterations, 1, chromosome_length, plot)
        x, f_x = p.GenerisiGeneracije()
        levy_ga.append(f_x)

        x, f_x = ACO(LEVY, OPSEG_LEVY, no_iterations, population_size, 50, 0.0001, 0.5, n, plot)
        levy_aco.append(f_x)

        x, f_x = PSO(GRIEWANK, OPSEG_GRIEWANK, no_iterations, population_size, 0, P, n, plot)
        griewank_pso.append(f_x)

        p = Populacija(GRIEWANK, OPSEG_GRIEWANK, n, 1.7, population_size, 0.75, 0.15, no_iterations, 1,
                       chromosome_length, plot)
        x, f_x = p.GenerisiGeneracije()
        griewank_ga.append(f_x)

        x, f_x = ACO(GRIEWANK, OPSEG_GRIEWANK, no_iterations, population_size, 50, 0.0001, 0.5, n, plot)
        griewank_aco.append(f_x)

    print('PSO')
    print('Schaffer-2:')
    find_statistics(schaffer_pso)
    print('Drop-wave:')
    find_statistics(dw_pso)
    print('Rosenbrock:')
    find_statistics(rosenbrock_pso)
    print('Ackley')
    find_statistics(ackley_pso)
    print('Levy')
    find_statistics(levy_pso)
    print('Griewank')
    find_statistics(griewank_pso)

    print('------------------------------------------------------------------------')

    print('GA')
    print('Schaffer-2:')
    find_statistics(schaffer_ga)
    print('Drop-wave:')
    find_statistics(dw_ga)
    print('Rosenbrock:')
    find_statistics(rosenbrock_ga)
    print('Ackley')
    find_statistics(ackley_ga)
    print('Levy')
    find_statistics(levy_ga)
    print('Griewank')
    find_statistics(griewank_ga)

    print('------------------------------------------------------------------------')

    print('ACO')
    print('Schaffer-2:')
    find_statistics(schaffer_aco)
    print('Drop-wave:')
    find_statistics(dw_aco)
    print('Rosenbrock:')
    find_statistics(rosenbrock_aco)
    print('Ackley')
    find_statistics(ackley_aco)
    print('Levy')
    find_statistics(levy_aco)
    print('Griewank')
    find_statistics(griewank_aco)

    print('------------------------------------------------------------------------')


def unimodal_non_separable(n, chromosome_length, no_iterations, population_size, no_executions, plot=False):
    P = generate_p(n)

    dixon_price_pso, dixon_price_ga, dixon_price_aco = [], [], []

    zakharov_pso = []
    zakharov_ga = []
    zakharov_aco = []

    for i in range(0, no_executions):
        print(i)

    zakharov_pso, zakharov_ga, zakharov_aco = [], [], []

    matyas_pso, matyas_ga, matyas_aco = [], [], []

    for i in range(0, no_executions):
        x, f_x = PSO(DIXONPRICE, OPSEG_DIXONPRICE, no_iterations, population_size, 0, P, n, plot)
        dixon_price_pso.append(f_x)

        p = Populacija(DIXONPRICE, OPSEG_DIXONPRICE, n, 1.7, population_size, 0.75, 0.15, no_iterations, 2,
                       chromosome_length, plot)
        x, f_x = p.GenerisiGeneracije()
        dixon_price_ga.append(f_x)

        x, f_x = ACO(DIXONPRICE, OPSEG_DIXONPRICE, no_iterations, population_size, 50, 0.0001, 0.5, n, plot)
        dixon_price_aco.append(f_x)

        x, f_x = PSO(ZAKHAROV, OPSEG_ZAKHAROV, no_iterations, population_size, 0, P, n, plot)
        zakharov_pso.append(f_x)

        p = Populacija(ZAKHAROV, OPSEG_ZAKHAROV, n, 1.7, population_size, 0.75, 0.15, no_iterations, 1,
                       chromosome_length, plot)
        x, f_x = p.GenerisiGeneracije()
        zakharov_ga.append(f_x)

        x, f_x = ACO(ZAKHAROV, OPSEG_ZAKHAROV, no_iterations, population_size, 50, 0.0001, 0.5, n, plot)
        zakharov_aco.append(f_x)

        x, f_x = PSO(MATYAS, OPSEG_MATYAS, no_iterations, population_size, 0, P, 2, plot)
        matyas_pso.append(f_x)

        p = Populacija(MATYAS, OPSEG_MATYAS, 2, 1.7, population_size, 0.75, 0.15, no_iterations, 1,
                       chromosome_length, plot)
        x, f_x = p.GenerisiGeneracije()
        matyas_ga.append(f_x)

        x, f_x = ACO(MATYAS, OPSEG_MATYAS, no_iterations, population_size, 50, 0.0001, 0.5, 2, plot)
        matyas_aco.append(f_x)

    print('PSO')
    print('Dixon-Price:')
    find_statistics(dixon_price_pso)
    print('Zakharov:')
    find_statistics(zakharov_pso)

    print('Matyas:')
    find_statistics(matyas_pso)

    print('------------------------------------------------------------------------')

    print('GA')
    print('Dixon-Price:')
    find_statistics(dixon_price_ga)
    print('Zakharov:')
    find_statistics(zakharov_ga)

    print('Matyas:')
    find_statistics(matyas_ga)

    print('------------------------------------------------------------------------')

    print('ACO')
    print('Dixon-Price:')
    find_statistics(dixon_price_aco)
    print('Zakharov:')
    find_statistics(zakharov_aco)

    print('Matyas:')
    find_statistics(matyas_aco)

    print('------------------------------------------------------------------------')


def runtime_rastrigin():
    P = generate_p(2)

    start = time.time()
    for i in range(0, 30):
        PSO(RASTRIGIN, OPSEG_RASTRIGIN, 20, 30, 0, P, 2, False)
    end = time.time()
    print('PSO:', (end - start) / 30)

    start = time.time()
    for i in range(0, 30):
        p = Populacija(RASTRIGIN, OPSEG_RASTRIGIN, 2, 1.7, 30, 0.75, 0.15, 20, 1, 20, False)
        p.GenerisiGeneracije()
    end = time.time()
    print('GA', (end - start) / 30)

    start = time.time()
    for i in range(0, 30):
        ACO(RASTRIGIN, OPSEG_RASTRIGIN, 20, 30, 50, 0.0001, 0.5, 2, False)
    end = time.time()
    print('ACO', (end - start) / 30)


def generate_p(n):
    W = [0.8] * n
    C1 = []
    for i in range(0, n):
        C1.append(random.uniform(0, 1.47))

    C2 = []
    for i in range(0, n):
        C2.append(random.uniform(0, 1.47))

    return [W, C1, C2]


if __name__ == '__main__':
    # pozivi algoritama za prikaz rasprostanjenost tacaka u prostoru pretrazivanja (kroz iteracije)
    particles_prevelance()

    # pozivi algoritama nad benchmark funkcijama
    unimodal_separable(2, 20, 100, 50, 3)
    unimodal_separable(5, 20, 100, 50, 3)
    unimodal_separable(10, 20, 300, 20, 30)

    unimodal_non_separable(2, 16, 20, 40, 30)
    unimodal_non_separable(5, 15, 40, 40, 30)
    unimodal_non_separable(10, 20, 300, 20, 30)

    multimodal_separable(2, 20, 50, 20, 30)
    multimodal_separable(5, 20, 50, 20, 30)
    multimodal_separable(10, 20, 300, 20, 30)

    multimodal_non_separable(2, 16, 50, 30, 30)
    multimodal_non_separable(5, 20, 50, 30, 30)
    multimodal_non_separable(10, 50, 300, 40, 30)

    # racunanje prosječnog vremena izvršenja algoritama
    runtime_rastrigin()
