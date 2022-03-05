import numpy as np
def linear(num):
    return num
def sigmoid(num):
    return 1 / (1 + np.exp(-num))
def relu(num):
    return np.maximum(num,0)
def softmax(num):
    return np.exp(num) / np.sum(np.exp(num))