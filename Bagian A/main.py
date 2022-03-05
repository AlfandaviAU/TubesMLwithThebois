from fileinput import filename
import numpy as np
import json

filename = "data.json"
data = json.load(open(filename))
print(data)
