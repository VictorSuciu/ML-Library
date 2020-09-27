import numpy as np

class Adam():

    def __init__(self, a=0.01, b1=0.9, b2=0.999, e=0.001):
        self.layers = None

        self.a = a
        self.b1 = b1
        self.b2 = b2
        self.e = e

        self.m_weights = None
        self.m_biases = None

        self.v_weights = None
        self.v_biases = None
        
    
    def set_layers(self, layers):
        self.layers = layers
        self.m_weights = [np.zeros(layer.weights.shape) for layer in layers]
        self.m_biases = [np.zeros(layer.biases.shape) for layer in layers]

        self.v_weights = [np.zeros(layer.weights.shape) for layer in layers]
        self.v_biases = [np.zeros(layer.biases.shape) for layer in layers]
    
    def step(self, batch_size, epoch_count):
        for i in range(len(self.layers)):
            weight_grad = (self.a / batch_size) * self.layers[i].weight_error
            bias_grad = (self.a / batch_size) * self.layers[i].bias_error

            self.m_weights[i] = (self.b1 * self.m_weights[i]) + ((1 - self.b1) * weight_grad)
            self.m_biases[i] = (self.b1 * self.m_biases[i]) + ((1 - self.b1) * bias_grad)
            self.v_weights[i] = (self.b2 * self.v_weights[i]) + ((1 - self.b2) * (weight_grad * weight_grad))
            self.v_biases[i] = (self.b2 * self.v_biases[i]) + ((1 - self.b2) * (bias_grad * bias_grad))

            m_weights_hat = self.m_weights[i] / (1 - np.power(self.b1, epoch_count))
            m_biases_hat = self.m_biases[i] / (1 - np.power(self.b1, epoch_count))
            v_weights_hat = self.v_weights[i] / (1 - np.power(self.b2, epoch_count))
            v_biases_hat = self.v_biases[i] / (1 - np.power(self.b2, epoch_count))

            self.layers[i].weights -= (self.a * m_weights_hat) / (np.sqrt(v_weights_hat) + self.e)
            self.layers[i].biases -= (self.a * m_biases_hat) / (np.sqrt(v_biases_hat) + self.e)
            
        



