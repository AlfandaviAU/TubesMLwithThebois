import numpy as np

def mse(real, pred):
    return np.mean(np.square(real - pred))

def drv_mse(real, pred):
    return 2*(real - pred)/real.size