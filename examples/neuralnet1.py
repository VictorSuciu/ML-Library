import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.datasets import make_moons
from sklearn.datasets import make_classification

from anotherml.util.dataset import Dataset
from anotherml.models.neural_networks.neural_network import NeuralNetwork
from anotherml.models.neural_networks.dense_layer import DenseLayer

num_centers = 2
colors = {
    0: '#0072bf',
    1: '#c70000',
    2: '#99d6ff',
    3: '#ffb3b3'
}
# X, y = make_blobs(n_samples=300, n_features=2, centers=num_centers, center_box=(-5.0, 5.0))
# X, y = X, y = make_classification(n_features=2, n_redundant=0, n_informative=2,
#                           random_state=1, n_clusters_per_class=1)
X, y = make_moons(noise=0.3, random_state=0)

# dataset = Dataset([[1, 1], [2, 4], [3, 9], [4, 16]], [1, 0, 1, 0])
dataset = Dataset(X, y, colors=[colors[n] for n in y])

net = NeuralNetwork()
net.add_layer(DenseLayer(num_neurons=4, input_size=2, activation_name='tanh'))
net.add_layer(DenseLayer(num_neurons=6, input_size=4, activation_name='relu'))
net.add_layer(DenseLayer(num_neurons=1, input_size=6, activation_name='sigmoid'))
net.fit(dataset, 5000)

xrange = np.arange(np.min(X[:,0]), np.max(X[:,0]), 0.01)
yrange = np.arange(np.min(X[:,1]), np.max(X[:,1]), 0.01)

prev = 0
curr = 0
xpoints = []
ypoints = []
color = []



print('Plotting results...')

for y in yrange:
    for x in xrange:
        curr = round(net.predict(np.array([x, y]))[0][0])
        xpoints.append(x)
        ypoints.append(y)
        color.append(colors[curr + 2])
        # print(round(net.predict(np.array([x, y]))[0][0]))
        
        

plt.scatter(xpoints, ypoints, s=10, c=color)
dataset.plot()
