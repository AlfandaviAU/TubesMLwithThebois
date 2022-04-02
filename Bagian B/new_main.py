from ann import ANN
from layer import Layer
from fileinput import filename
import numpy as np
import json

def loadJSONData(filename):
    data = json.load(open(filename))
    data_layer = []
    data_n = 0
    for layer in data:
        # iterasi setiap layer pada json obj
        data_layer.append(
            Layer(np.array(layer['bias']), np.array(layer['data']), layer['fungsiAktivasi'])
            # [int(layer["n"]), #idx = 0
            # layer["fungsiAktivasi"], #idx = 1
            # np.array(layer["data"]), #idx = 2
            # np.array(layer["bias"]), #idx = 3
            # 0# activation value : idx = 4 (default 0)]
            )
        data_n += 1
    return(data_layer)

def make_batches(train, batch_size):
    mini_batches = []
    np.random.shuffle(train)
    num_of_batches = int(np.ceil(train.shape[0] // batch_size))
    for i in range(0, num_of_batches):
        mini_batch = train[i*batch_size : np.minimum((i + 1)*batch_size, train.shape[0])]
        mini_batches.append(mini_batch)
    return mini_batches

# XOR
x_train = np.array([[[0,0]], [[0,1]], [[1,0]], [[1,1]]])
y_train = np.array([[[0]], [[1]], [[1]], [[0]]])

# ANN
# network = ANN(loadJSONData('xor_relu.json'))
# network.fit(x_train, y_train, 70, 0.05)
# out = network.predict(x_train)
# print(out)

print(make_batches(x_train, 2))