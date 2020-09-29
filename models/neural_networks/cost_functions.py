import numpy as np


def quadratic(target, output):
    return np.square(target - output) / (2 * len(target))

def quadratic_derivative(target, output):
    return output - target


def logloss(target, output):
    return np.square(target - output) / (2 * len(target))

def logloss_derivative(target, output):
    return (1 / (output * (1 - output))) * (output - target)




cost = {
    'quadratic': quadratic,
    'logloss': logloss
}
cost_derivative = {
    'quadratic': quadratic_derivative,
    'logloss': logloss_derivative
}
