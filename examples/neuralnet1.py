import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs

from tinyml.util.dataset import Dataset
from tinyml.models.neural_networks.neural_network import NeuralNetwork
from tinyml.models.neural_networks.dense_layer import DenseLayer

num_centers = 2
X, y = make_blobs(n_samples=300, n_features=2, centers=num_centers, center_box=(-5.0, 5.0))

# dataset = Dataset([[1, 1], [2, 4], [3, 9], [4, 16]], [1, 0, 1, 0])
dataset = Dataset(X, y)

net = NeuralNetwork()
net.add_layer(DenseLayer(num_neurons=10, input_size=2))
net.add_layer(DenseLayer(num_neurons=1, input_size=10))
net.fit(dataset, 500)

xrange = np.arange(np.min(X[:,0]), np.max(X[:,0]), 0.05)
yrange = np.arange(np.min(X[:,1]), np.max(X[:,1]), 0.05)

prev = 0
curr = 0
xpoints = []
ypoints = []
color = []

colors = {
    0: '#99d6ff',
    1: '#ffb3b3'
}

print('Plotting results...')

for y in yrange:
    for x in xrange:
        curr = round(net.predict(np.array([x, y]))[0][0])
        xpoints.append(x)
        ypoints.append(y)
        color.append(colors[curr])
        # print(round(net.predict(np.array([x, y]))[0][0]))
        
        

plt.scatter(xpoints, ypoints, s=10, c=color)
dataset.plot()
