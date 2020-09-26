import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_blobs
from sklearn.datasets import make_moons
from sklearn.datasets import make_classification
import torch
import torch.nn as nn

X, y = make_moons(500, noise=0.2)

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=73)

# Define network dimensions
n_input_dim = x_train.shape[1]
# Layer size
n_hidden = 50 # Number of hidden nodes
n_output = 1 # Number of output nodes = for binary classifier

# Build your network
net = nn.Sequential(
    nn.Linear(n_input_dim, n_hidden),
    nn.ELU(),
    nn.Linear(n_hidden, n_output),
    nn.Sigmoid())

loss_func = nn.MSELoss()
learning_rate = 0.01
# optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
optimizer = torch.optim.ASGD(net.parameters(), lr=learning_rate)

train_loss = []
train_accuracy = []
iters = 5000
Y_train_t = torch.FloatTensor(y_train).reshape(-1, 1)
for i in range(iters):
    X_train_t = torch.FloatTensor(x_train)
    y_hat = net(X_train_t)
    loss = loss_func(y_hat, Y_train_t)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    y_hat_class = np.where(y_hat.detach().numpy()<0.5, 0, 1)
    accuracy = np.sum(y_train.reshape(-1,1)==y_hat_class) / len(y_train)
    train_accuracy.append(accuracy)
    train_loss.append(loss.item())



X_test_t = torch.FloatTensor(x_test)
y_hat_test = net(X_test_t)
y_hat_test_class = np.where(y_hat_test.detach().numpy()<0.5, 0, 1)
test_accuracy = np.sum(y_test.reshape(-1,1)==y_hat_test_class) / len(y_test)
print("Test Accuracy {:.2f}".format(test_accuracy))

# Plot the decision boundary
# Determine grid range in x and y directions
x_min, x_max = X[:, 0].min()-0.1, X[:, 0].max()+0.1
y_min, y_max = X[:, 1].min()-0.1, X[:, 1].max()+0.1

# Set grid spacing parameter
spacing = min(x_max - x_min, y_max - y_min) / 100

# Create grid
XX, YY = np.meshgrid(np.arange(x_min, x_max, spacing),
               np.arange(y_min, y_max, spacing))

# Concatenate data to match input
data = np.hstack((XX.ravel().reshape(-1,1), 
                  YY.ravel().reshape(-1,1)))

# Pass data to predict method
data_t = torch.FloatTensor(data)
db_prob = net(data_t)

clf = np.where(db_prob<0.5,0,1)

Z = clf.reshape(XX.shape)

plt.figure(figsize=(12,8))
plt.contourf(XX, YY, Z, cmap=plt.cm.Accent, alpha=0.5)
plt.scatter(x_test[:,0], x_test[:,1], c=y_test, 
            cmap=plt.cm.Accent)
plt.show()
