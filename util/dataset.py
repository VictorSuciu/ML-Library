import numpy as np
import matplotlib.pyplot as plt

class Dataset():

    def __init__(self, data, labels):
        if(len(data) != len(labels)):
            raise Exception("Error: There are " + str(len(data)) + " data points and " + str(len(labels)) + \
            " labels. The number of data points must equal the number of labels.")
        self.data = self.set_data(np.array(data))
        self.labels = self.set_labels(np.array(labels))

    def set_data(self, data):
        formatted_data = np.zeros((data.shape[0], data.shape[1], 1))
        for i in range(data.shape[0]):
            formatted_data[i] = np.expand_dims(data[i].T, axis=0).T
        return formatted_data


    def set_labels(self, labels):
        if len(labels.shape) == 1:
            labels = np.expand_dims(labels, axis=0).T
        # print("labels:", labels)
        formatted_labels = np.zeros((labels.shape[0], labels.shape[1], 1))
        for i in range(labels.shape[0]):
            formatted_labels[i] = np.expand_dims(labels[i].T, axis=0).T
        # print("formatted_labels:", formatted_labels)
        return formatted_labels

    
    def size(self):
        return len(self.data)

    
    def plot(self, xfeat=0, yfeat=1):
        plt.scatter(self.data[:,xfeat].T[0], self.data[:,yfeat].T[0], s=20, c=self.labels.T)
        plt.show()
