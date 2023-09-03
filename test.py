import numpy as np

def sigmoid(x): return 1 / (1 + np.exp(-x))

def linear(theta, x):
    return sigmoid(theta.T@x)

print(linear(np.ones(4), np.ones(4)))