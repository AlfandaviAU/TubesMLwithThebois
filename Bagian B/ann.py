from layer import Layer
from lossfunction import mse, drv_mse

class ANN:
    def __init__(self, layers):
        self.layers = layers
        
    def predict(self, input):
        result = []
        for i in range(len(input)):
            output = input[i]
            for layer in self.layers:
                output = layer.calcForward(output)
            result.append(output)
        return result 
        
    def fit(self, x_train, y_train, epochs, learning_rate):
        for i in range(epochs):
            error = 0
            for j in range(len(x_train)):
                output = x_train[j]
                for layer in self.layers:
                    output = layer.calcForward(output)
                error += mse(y_train[j], output)
                
                backError = drv_mse(y_train[j], output)
                for layer in reversed(self.layers):
                    backError = layer.calcBackward(backError, learning_rate)
            error /= len(x_train)
            print("Error di epoch %d adalah %f" %(i+1, error))
    
    def forwardPass(self):
        pass
        
    def backwardPass(self):
        pass        