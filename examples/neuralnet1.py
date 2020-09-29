import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.datasets import make_moons
from sklearn.datasets import make_circles
from sklearn.datasets import make_classification

from anotherml.util.dataset import Dataset
from anotherml.models.neural_networks.neural_network import NeuralNetwork
from anotherml.models.neural_networks.dense_layer import DenseLayer
from anotherml.models.neural_networks.optimizers.sgd import SGD
from anotherml.models.neural_networks.optimizers.adam import Adam
from anotherml.util.visualizer import Visualizer

import sys


num_centers = 2
num_points = 300

# X, y = make_classification(n_features=2, n_redundant=0, n_informative=2,
# random_state=1, n_clusters_per_class=1)

X, y = make_moons(num_points, noise=0.2)
# X, y = make_circles(n_samples=num_points, noise=0.1, factor = 0.4)

# dataset = Dataset(np.array([[1, 1], [2, 4], [3, 9], [4, 16]]), np.array([1, 0, 1, 0]))
dataset = Dataset(X, y, colors=[0x0072bf, 0xc70000])

net = NeuralNetwork()
# net.add_layer(DenseLayer(input_size=2, num_neurons=128, activation_name='relu'))
# net.add_layer(DenseLayer(input_size=128, num_neurons=128, activation_name='relu'))
# net.add_layer(DenseLayer(input_size=128, num_neurons=1, activation_name='sigmoid'))

net.add_layer(DenseLayer(input_size=2, num_neurons=6, activation_name='tanh'))
net.add_layer(DenseLayer(input_size=6, num_neurons=4, activation_name='elu'))
net.add_layer(DenseLayer(input_size=4, num_neurons=1, activation_name='sigmoid'))

net.set_optimizer(Adam(a=0.01))
visualize = Visualizer()


# loss_list = net.fit(dataset, batch_size=(num_points), epochs=1)
# visualize.plot_binary_classification(net, dataset, 0, 1, resolution=150)

visualize.animate_binary_classification(net, dataset, dataset.size(), 1000, 0, 1, resolution=120, epoch_interval=10, ms_between_frames=90, ms_end_pause=2000)

