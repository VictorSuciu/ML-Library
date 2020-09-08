import numpy as np
from .initializers import *
from .activation_functions import *

class Neuron():

    def __init__(self, input_size=1, activation_name='sigmoid', weight_init_name='uniform', bias=0):
        self.input_size = input_size
        self.weight_init_name = weight_init_name
        self.bias_init = bias_init
        self.next_layer = next_layer
        self.inputs = np.zeros(input_size)
        self.weights = None
        self.bias = bias
        self.weighted_input
        self.error = 0
        self.activation_func = activation[activation_name]
        self.activation_dir = derivative[activation_name]

    
    def size(self):
        return self.input_size


    def init_weights(self):
        self.weights = initialize_weights[self.weight_init_name](input_size, -0.1, 0.1)


    def set_input(self, input_vector):
        if(len(input_vector) != self.input_size):
            raise Exception("Error neuron.set_input(): input vector size is " + str(len(input_vector)) + " but neuron size is " + \
            str(self.input_size) + ". These two sizes must be equal.")
        
        self.inputs = input_vector


    def output(self):
        self.weighted_input = np.matmul(np.transpose(self.inputs), self.weights) + self.bias
        return activation_func(self.weighted_input)
        

    def curr_error(self):
        pass
