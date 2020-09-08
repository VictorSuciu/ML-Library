import numpy as np


def quaratic(target, output):
    return np.square(target - output) / (2 * len(target))

def quaratic_derivative(target, output):
    return output - target



cost = {
    'quadratic': quaratic
}
cost_derivative = {
    'quadratic': quaratic_derivative
}
