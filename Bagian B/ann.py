from layer import Layer
from lossfunction import mse, drv_mse
import numpy as np

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
        
    def fit(self, x_train, y_train, learning_rate, batch_size=1, max_iter=200, error_threshold=0.0001):
        error = 1
        i = 0
        while(i < max_iter and error > error_threshold):
            error = 0
            for j in range(len(x_train)):
                output = x_train[j]
                
                for layer in self.layers:
                    output = layer.calcForward(output)
                error += mse(y_train[j], output)
                
                backError = drv_mse(y_train[j], output)
                if((j + 1) % batch_size == 0):
                    for layer in reversed(self.layers):
                        backError = layer.calcBackward(backError, learning_rate, is_updated=True, batch_size=batch_size)
                elif((j + 1) == len(x_train)):
                    for layer in reversed(self.layers):
                        backError = layer.calcBackward(backError, learning_rate, is_updated=True, batch_size=((j + 1) % batch_size))
                else:                    
                    for layer in reversed(self.layers):
                        backError = layer.calcBackward(backError, learning_rate)
            error /= len(x_train)
            print("Error di epoch %d adalah %f" %(i+1, error))
            i+=1
    
    def forwardPass(self):
        pass
        
    def backwardPass(self):
        pass        