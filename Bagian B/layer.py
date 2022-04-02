from actifunction import af, drv, softmax_grad
import numpy as np

class Layer:
    def __init__(self, bias, weights, activationFunction):
        self.bias = bias
        self.weights = weights
        self.activationFunction = activationFunction
        self.unact_error = 0
        self.input_error = 0
        self.weights_error = 0
    
    def calcForward(self, input):
        self.input = input
        self.output_unctv = np.dot(self.input, self.weights) + self.bias
        self.output = af(self.output_unctv, self.activationFunction)
        return self.output
    
    def calcBackward(self, error, lrn_rate, is_updated=False, batch_size=None):        
        self.unact_error = self.unact_error + drv(self.output_unctv, self.activationFunction) * error
        self.input_error = self.input_error + np.dot(self.unact_error, self.weights.T)
        self.weights_error = self.weights_error + np.dot(self.input.T, self.unact_error)
        
        batch_input_error = 0
        
        if is_updated:
            batch_input_error = self.input_error/batch_size
            self.weights = self.weights + lrn_rate * self.weights_error/batch_size
            self.bias = self.bias + lrn_rate * self.unact_error/batch_size
            self.unact_error = 0
            self.input_error = 0
            self.weights_error = 0
        
        return batch_input_error 