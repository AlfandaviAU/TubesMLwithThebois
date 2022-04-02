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

def shuffle(x, y):
    p = np.random.permutation(len(x))
    return x[p], y[p]

# # XOR
# x_train = np.array([[[0,0]], [[0,1]], [[1,0]], [[1,1]]])
# y_train = np.array([[[0]], [[1]], [[1]], [[0]]])
# XOR
from keras.utils import np_utils
from sklearn.datasets import load_iris
iris = load_iris()
x = iris.data
y = iris.target

# x_train = np.array([[[0,0]], [[0,1]], [[1,0]], [[1,1]]])
# y_train = np.array([[[0]], [[1]], [[1]], [[0]]])

x_train, y_train = shuffle(x, y)
x_train = x_train.reshape(x_train.shape[0], 1, x_train.shape[1])
#print(x_train)

# ANN
network = ANN(loadJSONData('xor_relu_copy.json'))
print(network.predict(x_train[:5]))
# print(x_train_shuffled)
y_train = np_utils.to_categorical(y_train)
y_train = y_train.reshape(y_train.shape[0], 1, y_train.shape[1])
# print(y_train_shuffled)
network.fit(x_train, y_train, learning_rate=0.01, batch_size=3, max_iter=100, error_threshold=0.0001)
weights = []
bias = []
af = []
for i in range(len(network.layers)):
    weights.append(network.layers[i].weights.tolist())
    bias.append(network.layers[i].bias.tolist())
    af.append(network.layers[i].activationFunction)

data = {"weights": weights, "bias":bias, "activation_function": af}
with open('weights.json', 'w') as f:
    json.dump(data, f)

out = network.predict(x_train[:5])
print("hasil = ", out)
print("seharusnya = ", y_train[:5])