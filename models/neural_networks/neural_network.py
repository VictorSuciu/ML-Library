import numpy as np
from anotherml.models.model import Model
from anotherml.util.dataset import Dataset
from anotherml.models.neural_networks.layer import Layer
from anotherml.models.neural_networks.dense_layer import DenseLayer
from anotherml.models.neural_networks.output_layer import OutputLayer
import anotherml.models.neural_networks.backpropagation


class NeuralNetwork(Model):

    def __init__(self, learning_rate=0.05):
        super().__init__()
        self.learning_rate = learning_rate
        self.layers = []


    def add_layer(self, layer):
        if len(self.layers) != 0:
            self.layers[-1].set_next_layer(layer)
        
        self.layers.append(layer)


    def finalize(self):
        if len(self.layers) == 0: 
            raise Exception("Error NeuralNetwork.fit(): this neural network has 0 layers. " \
                "Needs at least 1 layer.")

        self.layers[-1].set_as_last()

        for layer in self.layers:
            layer.init_weights()
            layer.init_biases()
        


    def fit(self, dataset, epochs):
        self.finalize()
        # print(dataset.data)
        # print(dataset.labels)

        for e in range(epochs):         
            for vector, label in zip(dataset.data, dataset.labels):
                # give input to first layer and feed forward throughout the network
                # print('\tFEED FORWARD\n')
                self.predict(vector)

                # backpropagate error through network. This performs gradient descent
                # print('\tBACKPROP\n')
                for layer in reversed(self.layers):
                    layer.compute_layer_error(label)      
                
                # exit(0)

            # update weights and biases from results of gradient descent
            for layer in self.layers:
                layer.weights -= (self.learning_rate / dataset.size()) * layer.weight_error
                layer.biases -= (self.learning_rate / dataset.size()) * layer.bias_error
                
                layer.reset_error()

            # print("new weights")
            # print(self.layers[1].weights)
            # print("new biases")
            # print(self.layers[1].biases)
        
        print("\n----------\n")
        print(self.layers[0].weights)
        print(self.layers[1].weights)
        print(self.layers[2].weights)
        
    def predict(self, datapoint):
        self.layers[0].set_input(datapoint / np.linalg.norm(datapoint))
        for layer in self.layers:
            layer.feed_forward()
        return self.layers[-1].output


