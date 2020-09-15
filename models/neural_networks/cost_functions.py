import numpy as np


def quadratic(target, output):
    return np.square(target - output) / (2 * len(target))

def quadratic_derivative(target, output):
    return output - target



cost = {
    'quadratic': quadratic
}
cost_derivative = {
    'quadratic': quadratic_derivative
}
