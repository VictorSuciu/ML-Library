import numpy as np
from anotherml.models.model import Model
from anotherml.util.dataset import Dataset
from anotherml.models.neural_networks.layer import Layer
from anotherml.models.neural_networks.dense_layer import DenseLayer
from anotherml.models.neural_networks.output_layer import OutputLayer
import anotherml.models.neural_networks.backpropagation
from anotherml.models.neural_networks.cost_functions import quadratic


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
        


    def fit(self, dataset, batch_size, epochs):
        self.finalize()
        # print(dataset.data)
        # print(dataset.labels)
        loss_list = []
        
        for e in range(epochs):
            loss = np.zeros((1, 1))

            for i in range(dataset.size() // batch_size):
                batch_start = i * batch_size
                batch_end = min((i + 1) * batch_size - 1, dataset.size())
            
                for vector, label in zip( \
                    dataset.data[batch_start : batch_end], \
                    dataset.labels[batch_start : batch_end]):
                    
                    # give input to first layer and feed forward throughout the network
                    # print('\tFEED FORWARD\n')
                    tmp = self.predict(vector)
                    loss += quadratic(label, tmp)
                    # backpropagate error through network. This performs gradient descent
                    # print('\tBACKPROP\n')
                    for layer in reversed(self.layers):
                        layer.compute_layer_error(label)      
                    
                    # exit(0)

                # update weights and biases from results of gradient descent
                for layer in self.layers:
                    layer.weights -= (self.learning_rate / dataset.size()) * layer.weight_error
                    layer.biases -= (self.learning_rate / dataset.size()) * layer.bias_error
                    # for layer in self.layers:
                    #     print("\n+++++++++++++++\n")
                    #     print(layer.weights, '\n\n')
                    #     print(layer.biases)
                    layer.reset_error()

                loss /= dataset.size()
                loss_list.append(loss[0][0])
                # print("new weights")
                # print(self.layers[1].weights)
                # print("new biases")
                # print(self.layers[1].biases)

            dataset.shuffle()

        # for layer in self.layers:
        #     print("\n+++++++++++++++\n")
        #     print(layer.weights, '\n\n')
        #     print(layer.biases)
        # print(self.layers[2].weights)
        return loss_list
        
    def predict(self, datapoint, log=False):
        # self.layers[0].set_input(datapoint / np.linalg.norm(datapoint))
        self.layers[0].set_input(datapoint)
        for layer in self.layers:
            if log:
                print('.',end='')
            layer.feed_forward()
        if log:
            print()
        return self.layers[-1].output


