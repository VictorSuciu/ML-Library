import numpy as np

class SGD():

    def __init__(self, a=0.01):
        self.layers = None
        self.a = a
    

    def set_layers(self, layers):
        self.layers = layers


    def step(self, batch_size, epoch_count):
        for layer in self.layers:
            weight_grad = (self.a / batch_size) * layer.weight_error
            bias_grad = (self.a / batch_size) * layer.bias_error
            layer.weights -= weight_grad
            layer.biases -= bias_grad
            
        



