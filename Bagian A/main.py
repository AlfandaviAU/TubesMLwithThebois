from fileinput import filename
import numpy as np
import json

filename = "data.json"
data = json.load(open(filename))

data_layer = []
data_n = 0

for layer in data:
    # iterasi setiap layer pada json obj
    data_layer.append((
        int(layer["n"]),
        layer["fungsiAktivasi"],
        np.array(layer["data"]),
        np.array(layer["bias"])
    ))
    data_n += 1

# print(data_layer)
# print(data_layer[0])
# print(data_n)

target = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
for i in range (data_n):
    # untuk data pertama
    if (i == 0):
        