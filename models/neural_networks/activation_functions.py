import numpy as np
import copy

def linear(x):
    return x

def linear_derivative(x):
    return np.ones(x.shape)


def sigmoid(x):
    # print('sigmoid')
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    # print('sigmoid-prime')
    return sigmoid(x) * (1 - sigmoid(x))


def relu(x):
    y = copy.deepcopy(x)
    y[y<0] = 0
    return y

def relu_derivative(x):
    y = copy.deepcopy(x)
    y[y>=0] = 1
    y[y<0] = 0
    return y


def elu(x, alpha=0.2):
    return np.piecewise(x, [x < 0, x >= 0], [lambda x: ((np.exp(x) - 1) * alpha), lambda x: x])

def elu_derivative(x, alpha=0.2):
    # print('elu-prime')
    return (elu(x, alpha) + alpha)
 

def tanh(x):
    # print('tanh')
    return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))

def tanh_derivative(x):
    # print('tanh-prime')
    return 1 - (tanh(x) * tanh(x))


activation = {
    'linear': linear,
    'sigmoid': sigmoid,
    'relu': relu,
    'elu': elu,
    'tanh':tanh
}
derivative = {
    'linear': linear_derivative,
    'sigmoid': sigmoid_derivative,
    'relu': relu_derivative,
    'elu': elu_derivative,
    'tanh':tanh_derivative
}
