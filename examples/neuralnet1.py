import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.datasets import make_moons
from sklearn.datasets import make_classification

from anotherml.util.dataset import Dataset
from anotherml.models.neural_networks.neural_network import NeuralNetwork
from anotherml.models.neural_networks.dense_layer import DenseLayer
import sys

def hex_to_str(hexnum):
        zeros = '000000'
        hexstr = str(hex(hexnum))[2:]
        return '#' + (zeros[:6 - len(hexstr)]) + hexstr

# np.random.seed(10)
num_centers = 2
colors = {
    0: 0x0072bf,
    1: 0xc70000,
    2: 0x99d6ff,
    3: 0xffb3b3
}
# X, y = make_blobs(n_samples=300, n_features=2, centers=num_centers, center_box=(-5.0, 5.0))

# X, y = make_classification(n_features=2, n_redundant=0, n_informative=2,
# random_state=1, n_clusters_per_class=1)

X, y = make_moons(500, noise=0.2)

# dataset = Dataset(np.array([[1, 1], [2, 4], [3, 9], [4, 16]]), np.array([1, 0, 1, 0]))
dataset = Dataset(X, y, colors=[colors[n] for n in y])

net = NeuralNetwork(learning_rate=0.01)
# net.add_layer(DenseLayer(num_neurons=4, input_size=2, activation_name='tanh'))
# net.add_layer(DenseLayer(num_neurons=6, input_size=4, activation_name='tanh'))
# net.add_layer(DenseLayer(num_neurons=1, input_size=6, activation_name='sigmoid'))

# net.add_layer(DenseLayer(num_neurons=2, input_size=2, activation_name='tanh'))
net.add_layer(DenseLayer(num_neurons=100, input_size=2, activation_name='elu'))
net.add_layer(DenseLayer(num_neurons=1, input_size=100, activation_name='sigmoid'))

loss_list = net.fit(dataset, batch_size=500, epochs=2000)

xran = np.arange(np.min(X[:,0]), np.max(X[:,0]), 0.02)
yran = np.arange(np.min(X[:,1]), np.max(X[:,1]), 0.02)

prev = 0
curr = 0
xpoints = []
ypoints = []
color = []


np.set_printoptions(threshold=sys.maxsize)
print('Plotting results...')
colornums = []
for y in yran:
    for x in xran:
        temp = net.predict(np.array([[x], [y]]))[0][0]       
        curr = round(temp)
        xpoints.append(x)
        ypoints.append(y)
        color.append(hex_to_str(colors[curr + 2]))
        colornums.append(temp)
        # print(round(net.predict(np.array([x, y]))[0][0]))
        
print('PREDICTIONS\n', np.array(colornums))   
print('LOSS\n', np.array(loss_list))

plt.scatter(xpoints, ypoints, s=10, c=color)
dataset.plot()



