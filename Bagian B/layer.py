from actifunction import af, drv, softmax_grad
import numpy as np

class Layer:
    def __init__(self, bias, weights, activationFunction):
        self.bias = bias
        self.weights = weights
        self.activationFunction = activationFunction
    
    def calcForward(self, input):
        self.input = input
        self.output_unctv = np.dot(self.input, self.weights) + self.bias
        self.output = af(self.output_unctv, self.activationFunction)
        return self.output
    
    def calcBackward(self, error, lrn_rate):
        unact_error = drv(self.output_unctv, self.activationFunction) * error
        input_error = np.dot(unact_error, self.weights.T)
        weights_error = np.dot(self.input.T, unact_error)
        
        self.weights = self.weights - lrn_rate * weights_error
        self.bias = self.bias - lrn_rate * unact_error
        
        return input_error
        
        