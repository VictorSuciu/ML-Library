import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))


def relu(x):
    return max(0, x)

def relu_derivative(x):
    return max(0, x)


activation = {
    'sigmoid': sigmoid,
    'relu': relu
}
derivative = {
    'sigmoid': sigmoid_derivative,
    'relu': relu_derivative
}
