from fileinput import filename
import numpy as np
import json
#import fungsiAktivasi as FA

def linear(num):
    return num
def sigmoid(num):
    return 1 / (1 + np.exp(-num))
def relu(num):
    return np.maximum(num,0)
def softmax(num):
    return np.exp(num) / np.sum(np.exp(num))

filename = "data.json"
data = json.load(open(filename))

data_layer = []
data_n = 0
prediction = None

for layer in data:
    # iterasi setiap layer pada json obj
    data_layer.append([
        int(layer["n"]), #idx = 0
        layer["fungsiAktivasi"], #idx = 1
        np.array(layer["data"]), #idx = 2
        np.array(layer["bias"]), #idx = 3
        0# activation value : idx = 4 (default 0)
    ])
    data_n += 1

# print(data_layer)
# print(data_layer[0])
# print(data_n)

target = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
for i in range (data_n):
    # untuk data pertama
    print(data_layer[i])
    if (i == 0):
        value = np.dot(target, data_layer[i][2]) + data_layer[i][3]
    else:
        value = np.dot(target, data_layer[i-1][4]) + data_layer[i][3]
    if (data_layer[i][1] == "sigmoid"):
        passed_val = sigmoid(value)
    elif (data_layer[i][1] == "linear"):
        passed_val = linear
    elif (data_layer[i][1] == "relu"):
        passed_val = relu(value)
    elif (data_layer[i][1] == "softmax"):
        passed_val = softmax(value)
    
    data_layer[i-1][4] = passed_val
print(data_layer)

