import numpy as np


def sigmoid(x):
    # print('sigmoid')
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    # print('sigmoid-prime')
    return sigmoid(x) * (1 - sigmoid(x))


def relu(x):
    # print('relu')
    x[x<0] = 0
    return x

def relu_derivative(x):
    # print('relu-prime')
    x[x<0] = 0
    x[x>=0] = 1
    return x


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
