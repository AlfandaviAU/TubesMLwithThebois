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

def drv_linear(num):
    return np.ones(num.shape)
def drv_sigmoid(num):
    return sigmoid(num) * (1 - sigmoid(num))
def drv_relu(num):
    return np.where(num < 0, 0, 1)    
# TODO: Buat fungsi derivatif dari softmax    

# Ganti nama filenya di sini
filename = "xor_sigmoid.json"
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

target = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
for i in range (data_n):
    # untuk data pertama
    if (i == 0):

        value = np.dot(target, data_layer[i][2]) + data_layer[i][3]
    else:
        value = np.dot(data_layer[i-1][4], data_layer[i][2]) + data_layer[i][3]
    if (data_layer[i][1] == "sigmoid"):
        passed_val = sigmoid(value)
    elif (data_layer[i][1] == "linear"):
        passed_val = linear(value)
    elif (data_layer[i][1] == "relu"):
        passed_val = relu(value)
    elif (data_layer[i][1] == "softmax"):
        passed_val = softmax(value)

    else:
        print("activation function not valid at data layer " + i)
        break

    data_layer[i][4] = passed_val

prediction = np.copy(data_layer[-1][4])
prediction = prediction.reshape(prediction.shape[0], 1)

for i in range(len(prediction)):
    if(prediction[i] > 0.5):
        prediction[i] = 1
    else:
        prediction[i] = 0


print("banyak layer: ", data_n)

for i in range(data_n):
    print("Layer", i+1, ": ")
    print("Banyak neuron: ", data_layer[i][0])
    print("Fungsi aktivasi: ", data_layer[i][1])
    print("Weight akhir: ", data_layer[i][2])
    print("Bias: ", data_layer[i][3])
    print("Nilai aktifasi: ", data_layer[i][4])

print("prediction", prediction)

