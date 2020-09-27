import numpy as np
from anotherml.models.model import Model
from anotherml.util.dataset import Dataset
from anotherml.models.neural_networks.layer import Layer
from anotherml.models.neural_networks.dense_layer import DenseLayer
from anotherml.models.neural_networks.output_layer import OutputLayer
import anotherml.models.neural_networks.backpropagation
from anotherml.models.neural_networks.cost_functions import quadratic
from anotherml.models.neural_networks.optimizers.adam import Adam


class NeuralNetwork(Model):

    def __init__(self):
        super().__init__()
        self.layers = []
        self.optimizer = None


    def add_layer(self, layer):
        if len(self.layers) != 0:
            self.layers[-1].set_next_layer(layer)
        
        self.layers.append(layer)

    
    def set_optimizer(self, optimizer):
        self.finalize()
        self.optimizer = optimizer
        self.optimizer.set_layers(self.layers)


    def finalize(self):
        if len(self.layers) == 0: 
            raise Exception("Error NeuralNetwork.fit(): this neural network has 0 layers. " \
                "Needs at least 1 layer.")

        self.layers[-1].set_as_last()

        for layer in self.layers:
            layer.init_weights()
            layer.init_biases()
        
        self.adam = Adam(self.layers)


    # def SGD(self, batch_size):
    #     for layer in self.layers:
    #         weight_grad = (self.learning_rate / batch_size) * layer.weight_error
    #         bias_grad = (self.learning_rate / batch_size) * layer.bias_error
    #         layer.weights -= weight_grad
    #         layer.biases -= bias_grad


        
    def fit(self, dataset, batch_size, epochs):
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
                    tmp = self.predict(vector)
                    loss += quadratic(label, tmp)

                    # backpropagate error through network. This performs gradient descent
                    for layer in reversed(self.layers):
                        layer.compute_layer_error(label)

                # update weights and biases from results of gradient descent
                # self.SGD(batch_size)
                self.optimizer.step(batch_size, e + 1)
                layer.reset_error()

                loss /= dataset.size()
                loss_list.append(loss[0][0])

            dataset.shuffle()
            
        return loss_list
        
    def predict(self, datapoint, log=False):
        self.layers[0].set_input(datapoint)
        for layer in self.layers:
            if log:
                print('.',end='')
            layer.feed_forward()
        if log:
            print()
        return self.layers[-1].output


