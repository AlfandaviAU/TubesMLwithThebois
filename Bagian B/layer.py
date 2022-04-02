from actifunction import af, drv, softmax_grad
import numpy as np

class Layer:
    def __init__(self, input, bias, weights, activationFunction):
        self.input = input
        self.bias = bias
        self.weights = weights
        self.activationFunction = activationFunction
    
    def calcForward(self):
        pass
    
    def calcBackward(self, error, lrn_rate):
        unact_error = drv(self.output_unctv, self.activationFunction) * error
        input_error = np.dot(unact_error, self.weights.T)
        weights_error = np.dot(self.input.T, unact_error)
        
        self.weights -=  lrn_rate * weights_error
        self.bias -= lrn_rate * unact_error
        
        return input_error
        
        