import numpy as np
import matplotlib.pyplot as plt

class Dataset():

    def __init__(self, data, labels, colors=None):
        if(len(data) != len(labels)):
            raise Exception("Error: There are " + str(len(data)) + " data points and " + str(len(labels)) + \
            " labels. The number of data points must equal the number of labels.")
        
        self.data = self.set_data(np.array(data))
        self.labels = self.set_labels(np.array(labels))
        self.colormap = {}

        for i in range(len(colors)):
            self.colormap[i] = colors[i]

        if colors == 0:
            self.colors = labels
        else:
            self.colors = np.array([colors[l] for l in labels])


    def set_data(self, data):
        formatted_data = np.zeros((data.shape[0], data.shape[1], 1))
        for i in range(data.shape[0]):
            formatted_data[i] = np.expand_dims(data[i].T, axis=0).T
        return formatted_data


    def set_labels(self, labels):
        if len(labels.shape) == 1:
            labels = np.expand_dims(labels, axis=0).T
        formatted_labels = np.zeros((labels.shape[0], labels.shape[1], 1))
        for i in range(labels.shape[0]):
            formatted_labels[i] = np.expand_dims(labels[i].T, axis=0).T
        # print("formatted_labels:", formatted_labels)
        return formatted_labels

    
    def shuffle(self):
        permute = np.random.permutation(self.size())

        self.data = self.data[permute]
        self.labels = self.labels[permute]
        self.colors = self.colors[permute]


    def get_feature(self, feature_index):
        return self.data[:, feature_index, 0]

    def get_max(self, feature_index):
        return np.max(self.get_feature(feature_index))


    def get_min(self, feature_index):
        return np.min(self.get_feature(feature_index))


    def size(self):
        return len(self.data)

    
    def plot(self, xfeat=0, yfeat=1, file_name='', title='Dataset'):
        clr = [self.hex_to_str(n) for n in self.colors]
        plt.scatter(self.data[:,xfeat].T[0], self.data[:,yfeat].T[0], s=20, c=clr)
        plt.title(title, loc='left')

        if file_name == '':
            plt.show()
        else:
            plt.savefig(file_name, bbox_inches='tight')

    def hex_to_str(self, hexnum):
        zeros = '000000'
        hexstr = str(hex(hexnum))[2:]
        return '#' + (zeros[:6 - len(hexstr)]) + hexstr
