import numpy as np

# Fungsi aktivasi yang dapat dipakai
def linear(num):
    return num
def sigmoid(num):
    return 1 / (1 + np.exp(-num))
def relu(num):
    return np.maximum(num,0)
def softmax(num):
    return np.exp(num) / np.sum(np.exp(num))

def af(num, type):
    if type == 'sigmoid':
        return sigmoid(num)
    elif type == 'relu':
        return relu(num)
    elif type == 'sigmoid':
        return sigmoid(num)
    else:
        return linear(num)
    

# Turunan fungsi aktivasi, dipakai dalam backpropagation
def drv_linear(num):
    return np.ones(num.shape)
def drv_sigmoid(num):
    return np.multiply(sigmoid(num), (1 - sigmoid(num)))
def drv_relu(num):
    return np.where(num < 0, 0, 1)    
def softmax_grad(num):
    num = softmax(num) 
    jacob_matrix = np.diag(num)
    for i in range(len(jacob_matrix)):
        for j in range(len(jacob_matrix)):
            if i != j:
                jacob_matrix[i][j] = -num[i] * num[j]
            else: 
                jacob_matrix[i][j] = num[i] * (1 - num[i])
    return jacob_matrix

def drv(num, type):
    if type == 'linear':
        return drv_linear(num)
    elif type == 'sigmoid':
        return drv_sigmoid(num)
    elif type == 'relu':
        return drv_relu(num)
