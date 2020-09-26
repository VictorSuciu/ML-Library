import numpy as np
from .layer import Layer

class DenseLayer(Layer):

    def __init__(self, num_neurons, input_size, weight_init='uniform', bias_init='constant', activation_name='sigmoid', weight_init_name='uniform', bias=0, next_layer=None):
        super().__init__(num_neurons, input_size=input_size, weight_init=weight_init, bias_init=bias_init, activation_name=activation_name, weight_init_name=weight_init_name, bias=bias, next_layer=next_layer)
    
    def set_next_layer(self, next_layer):
        if self.num_neurons != next_layer.input_size:
            raise Exception("Error DenseLayer.set_next_layer(): This layer's size is " + str(self.num_neurons) + \
                " but next_layer's input_size is " + str(next_layer.input_size) + \
                ". These two sizes must be equal.")
        
        self.next_layer = next_layer


    def feed_forward(self):
        self.weighted_inputs = (self.weights @ self.inputs) + self.biases
        # print('weighted_inputs = weights @ inputs\n', self.weighted_inputs , '\n=\n', self.weights, '\n@\n', self.inputs, ('\n'+'-'*30+'\n'))
        self.output = self.activation_func(self.weighted_inputs)
        
        if self.next_layer != None:
            self.next_layer.set_input(self.output)
