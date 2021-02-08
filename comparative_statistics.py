import numpy as np


def find_statistics(functionValues):
    print("Best", min(functionValues))
    print("Worst", max(functionValues))
    print("Mean", np.mean(functionValues))
    print("Std", np.std(functionValues))
