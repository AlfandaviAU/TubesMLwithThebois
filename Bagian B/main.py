from fileinput import filename
import numpy as np
import json
#import fungsiAktivasi as FA        

# Fungsi aktivasi yang dapat dipakai
def linear(num):
    return num
def sigmoid(num):
    return 1 / (1 + np.exp(-num))
def relu(num):
    return np.maximum(num,0)
def softmax(num):
    return np.exp(num) / np.sum(np.exp(num))

# Turunan fungsi aktivasi, dipakai dalam backpropagation
def drv_linear(num):
    return np.ones(num.shape)
def drv_sigmoid(num):
    return np.multiply(sigmoid(num), (1 - sigmoid(num)))
def drv_relu(num):
    return np.where(num < 0, 0, 1)    
# TODO: Buat fungsi derivatif dari softmax    

def drv(num, type):
    if type == 'linear':
        return drv_linear(num)
    elif type == 'sigmoid':
        return drv_sigmoid(num)
    elif type == 'relu':
        return drv_relu(num)

def make_batches(train, batch_size):
    mini_batches = []
    np.random.shuffle(train)
    num_of_batches = int(np.ceil(train.shape[0] // batch_size))
    for i in range(0, num_of_batches):
        mini_batch = train[i*batch_size : np.minimum((i + 1)*batch_size, train.shape[0])]
        mini_batches.append(mini_batch)
    return mini_batches

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

def loadJSONData(filename):
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
    return(data_layer)

def forwardPropagation(mini_data, layers_weights, activation_functions):
    neurons=[]
    result =[]
    for data in mini_data:
        neuron_per_data = []
        # print(data)
        data1 = data
        for i in range(len(layers_weights)):
            new_data = []
            for neuron in layers_weights[i]:
                cok = np.dot(data1, neuron[:-1]) + neuron[-1]
                new_data.append(cok)

            new_data = np.array(new_data)
            if(activation_functions[i] == "relu"):
                new_data= relu(new_data)
            elif(activation_functions[i] == "linear"):
                new_data= linear(new_data)
            elif(activation_functions[i] == "sigmoid"):
                new_data= sigmoid(new_data)
            elif(activation_functions[i] == "softmax"):
                new_data= softmax(new_data)
            neuron_per_data.append(new_data)
            data1 = new_data
            print(data1)
        neurons.append(neuron_per_data)
    # print(neurons)

    return neurons, result

def backwardPropagation(filename, y_pred, y_actual, batch_size, learning_rate, error_threshold=0.0001, max_iter=200):
    # filename buat ambil struktur jaringan
    # y_pred: output hasil prediksi
    # y_actual: output asli
    # Batch size: ukuran batch, dipakai dengan cara update nilai w nya setiap selesai nghitung sekian data
    # Sisanya bisa dicek
    data_layer = loadJSONData(filename)
    data_n = len(data_layer)
    
    weights = [data_layer[i][2] for i in range(len(data_layer))]
    biases = [data_layer[i][3] for i in range(len(data_layer))]
    weights_and_biases = []
    for i, weight in enumerate(weights):
        weight_and_bias = []
        for j, weight_neuron in enumerate(weight):
            weight_and_bias.append(np.append(weight_neuron, biases[i][j]))
        weights_and_biases.append(np.array(weight_and_bias))
    weights_and_biases = np.array(weights_and_biases)
    print(weights_and_biases)
    
    activation = [data_layer[i][1] for i in range(len(data_layer))]
    contoh_array = [
    [        # layer 1
        [
            1,
            2,
            3,
        ],
        # Layer 2
        [
            4,
            5,
            7,
        ]
    ],
    [
        [
            1,
            3,
            5,
        ],
        [
            2,
            5,
            6
        ]
    ]
    ]
        
    # --- Lakukan sampai error totalnya kurang dari error_threshold atau iterasinya sudah sampai max_iter
    
    lastLayer = []
    # Untuk tiap neuron k di layer output, hitung error termnya, d(k)
    # Untuk tiap neuron h di layer hidden, hitung error termnya, d(h)
    for i, data_array in enumerate(contoh_array):
        lastLayer.insert(0, -np.multiply((y_pred[i] - y_actual[i]), drv(weights_and_biases[-1])))
        # for j in reversed(range(len(data_array) - 1)):
            # lastLayer.insert(0, -np.multiply(drv(weights_and_biases[j]), np.sum([])))
        
        
    
    # CATATAN: error term itu yang kalau di spek nomor f
    # BTW, itu turunan fungsi aktivasi (kecuali softmax) kepakenya di dOut/dNet (mungkin perlu disesuaikan)
    # Untuk yang softmax, kita langsung dE/dNet jadi tinggal kalikan ke dNet/dw(ji)
    
    # Ganti setiap weight w(ji) (yakni, yang kalau forward dari i ke j)
    # Caranya, dari w(ji) menjadi w(ji) + delta(w(ji))
    # dengan delta(w(ji)) = learning_rate * d(j) * x(ji)
    
    return

# forwardPropagation(np.array([[0, 0], [0, 1], [1, 0], [1, 1]]), "xor_sigmoid.json")
backwardPropagation("xor_sigmoid.json", [1, 1, 1, 1], [1, 0, 0, 1], 10, 0.01)