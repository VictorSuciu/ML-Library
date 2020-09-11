import numpy as np
from anotherml.models.neural_networks.activation_functions import *
from anotherml.models.neural_networks.cost_functions import *
from anotherml.models.neural_networks.backpropagation import *

class Layer():

    def __init__(self, num_neurons, input_size, weight_init='uniform', bias_init='constant', activation_name='sigmoid', cost_name='quadratic', weight_init_name='uniform', bias=0, next_layer=None):
        self.num_neurons = num_neurons
        self.input_size = input_size
        self.weight_init = weight_init
        self.bias_init = bias_init
        self.is_last = False

        self.activation_name = activation_name
        self.cost_name = cost_name
        self.activation_func = activation[activation_name]
        self.activation_der = derivative[activation_name]
        self.cost_func = cost[cost_name]
        self.cost_der = cost_derivative[cost_name]
        self.error_func = None

        self.next_layer = next_layer
        self.inputs = None
        self.weights = None
        self.biases = None
        self.weighted_inputs = None
        self.output = None

        self.curr_error = None
        self.weight_error = None
        self.bias_error = np.zeros((num_neurons, 1))

    
    def get_num_neurons(self):
        return self.num_neurons


    def get_input_size(self):
        return self.input_size


    def input_size(self):
        return self.input_size

    
    def set_next_layer(self, next_layer):
        self.next_layer = next_layer


    def init_weights(self):
        if self.weight_init == 'uniform':
            radius = 0.1
            self.weights = np.random.rand(self.num_neurons, self.input_size) * radius * 2 - radius
        elif self.weight_init == 'glorot':
            pass # TODO: implement glorot

        self.weight_error = np.zeros(self.weights.shape)


    def init_biases(self):
        if self.bias_init == 'constant':
            self.biases = np.zeros((self.num_neurons, 1))


    def set_input(self, input_vector):
        if input_vector.shape[0] != self.input_size:
            raise Exception("Error Layer.set_input(): This layer's input_size is " + str(self.input_size) + \
                " but you tried to give it a vector of size " + str(input_vector.shape[0]) + \
                ". These two sizes must be equal.")
        self.inputs = input_vector


    def compute_layer_error(self, target):
        if self.is_last:
            self.curr_error = last_layer_error(target, self.output, self.weighted_inputs, self.activation_der, self.cost_der)
        else:
            self.curr_error = layer_error(self.next_layer.weights, self.next_layer.curr_error, self.weighted_inputs, self.activation_der)

        # print("weight_error Multiplying:\n", self.curr_error, "\nby\n", self.inputs.T)
        # print("Result:\n", (self.curr_error @ self.inputs.T))
        # print("weight_error:\n", self.weight_error)
        self.weight_error += (self.curr_error @ self.inputs.T)
        self.bias_error += self.curr_error
        # print("weight_error:\n", self.weight_error, "\nbias_error\n", self.bias_error)
        

    def reset_error(self):
        self.weight_error = np.zeros(self.weights.shape)
        self.bias_error = np.zeros((self.num_neurons, 1))


    def set_as_last(self):
        self.is_last = True


    def feed_forward(self):
        pass
