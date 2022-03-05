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

print(data_layer[0])