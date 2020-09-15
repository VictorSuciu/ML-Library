import numpy as np
import copy

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


def tanh(x):
    # print('tanh')
    return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))

def tanh_derivative(x):
    # print('tanh-prime')
    return 1 - (tanh(x) * tanh(x))


activation = {
    'sigmoid': sigmoid,
    'relu': relu,
    'tanh':tanh
}
derivative = {
    'sigmoid': sigmoid_derivative,
    'relu': relu_derivative,
    'tanh':tanh_derivative
}
