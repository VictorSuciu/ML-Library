import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from models.linear_regression import LinearRegressionModel
from util.dataset import Dataset

num_centers = 2
lineres = 100

X, y = make_blobs(n_samples=30, n_features=2, centers=num_centers, center_box=(-10.0, 10.0))


print(X)
print(y)
dataset = Dataset(X, y)
xdata = dataset.get_feature(0)
ydata = dataset.get_feature(1)
xmin = np.min(xdata) - 3
xmax = np.max(xdata) + 3
ymin = np.min(ydata) - 3
ymax = np.max(ydata) + 3

regression_model = LinearRegressionModel(2)
regression_model.fit(dataset)

xline = [xmin, xmax]
yline = [regression_model.predict([xmin]), regression_model.predict([xmax])]



plt.figure()
plt.scatter(xdata, ydata, c=y)
plt.plot(xline, yline)

plt.xlim(xmin, xmax)
plt.ylim(ymin, ymax)
plt.show()