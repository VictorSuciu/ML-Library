import numpy as np
from .layer import Layer

class OutputLayer(Layer):

    def __init__(self, num_neurons, input_size, weight_init='uniform', bias_init='constant', activation_name='sigmoid', weight_init_name='uniform', bias=0, next_layer=None):
        super().__init__(num_neurons, input_size=input_size, weight_init=weight_init, bias_init=bias_init, activation_name=activation_name, weight_init_name=weight_init_name, bias=bias, next_layer=next_layer)
    
