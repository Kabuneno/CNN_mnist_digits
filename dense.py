import numpy as np

from layer import Layer

class Dense(Layer):
    def __init__(self,input_size,output_size):
        self.weights = np.random.randn(output_size,input_size)
        self.bias = np.random.randn(output_size,1)
        # super().__init__()
    def forward(self, input):
        self.input = input
        return np.dot(self.weights,self.input) + self.bias
        # return super().forward(input)
    def backward(self, output_gradent ,learning_rate):
        weights_gradients = np.dot(output_gradent,self.input.T)
        input_gradients = np.dot(self.weights.T,output_gradent)
        self.weights -= learning_rate * weights_gradients
        self.bias -= learning_rate * output_gradent
        return input_gradients
        # return super().backward(output)


