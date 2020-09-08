import numpy as np
from .model import Model
from util.dataset import Dataset

class LinearRegressionModel(Model):

    def __init__(self, dimentions):
        super().__init__()
        self.dimensions = dimentions
        self.coefficients = np.zeros((dimentions))


    def fit(self, data):
        """
        data: dataset objct of (x, y) pairs
        """
        if type(data) != Dataset:
            raise Exception("Error: data must be of type util.Dataset")

        a = 0
        b = 0
        xvals = data.get_feature(0)
        yvals = data.get_feature(1)
        xmean = np.mean(xvals)
        ymean = np.mean(yvals)

        bnumer = 0
        bdenom = 0
        for i in range(data.size()):
            bnumer += xvals[i] * (yvals[i] - ymean)
        
        for i in range(data.size()):
            bdenom += xvals[i] * (xvals[i] - xmean)

        b = bnumer / bdenom
        a = ymean - (b * xmean)
        self.coefficients[0] = a
        self.coefficients[1] = b
        print('y = ' + str(b) + 'x + ' + str(a))


    def predict(self, datapoint):
        """
        datapoint: numpy array of integer pairs
        returns: a prediction for datapoint
        """
        if(len(datapoint) != self.dimensions - 1):
            raise(Exception("Error: datapoint dimentionality must be one less than model dimentionality"))

        prediction = 0
        for i in range(len(self.coefficients) - 1):
            prediction += datapoint[i] * self.coefficients[i + 1]

        return prediction + self.coefficients[0]

