import numpy as np
from fungsiAktivasi import linear, sigmoid, relu, softmax

class Layer:
    def __init__(self, neuron_num, weights, biases, activation_function = "linear"):
        activation_functions = {
            'linear': linear,
            'sigmoid': sigmoid,
            'relu': relu,
            'softmax': softmax
        }
        
        self.neuron_num = neuron_num
        self.weights = weights
        self.biases = biases
        self.activation_function = activation_functions[activation_function]
    
    def calculate_forward(self, input):
        self.input = input
        return self.activation_function(input)