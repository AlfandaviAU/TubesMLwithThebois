from .actifunction import af, drv, softmax_grad

class Layer:
    def __init__(self, input, weights, activationFunction):
        self.input = input
        self.weights = weights
        self.activationFunction = activationFunction
    
    def calcForward(self):
        pass
    
    def calcBackward(self):
        pass
        