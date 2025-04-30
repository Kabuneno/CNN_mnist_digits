import numpy as np
from layer import Layer

from activation import Activation

class Sigmoid(Activation):
    def __init__(self):
        def sigmoid(x):
            return 1/ (1+ np.exp(-x))
        def sigmoid_prime(x):
            S = sigmoid(x)
            return S * (1-S)
        super().__init__(sigmoid,sigmoid_prime)

class Softmax(Layer):
    def forward(self, input):
        tmp = np.exp(input)
        self.output = tmp / np.sum(tmp)
        return self.output

    def backward(self, output_gradient, learning_rate):
        n = np.size(self.output)
        jacobian_matrix = np.zeros((n, n))
    
        for i in range(n):
            for j in range(n):
                if i == j:
                    jacobian_matrix[i, j] = self.output[i] * (1 - self.output[i])
                else:
                    jacobian_matrix[i, j] = -self.output[i] * self.output[j]
    
        return np.dot(jacobian_matrix, output_gradient)
