import numpy as np

def sigmoid(x):
    return 1/(1+np.exp(-x))

class Neuron:
    def __init__(self, w, b):
        self.w = w
        self.b = b

    def y(self, x):
        s = np.dot(self.w, x) + self.b
        return sigmoid(s)

Xi = np.array([2, 3])
Wi = np.array([0, 1])
bias = 4
n = Neuron(Wi, bias)
print("Y = ", n.y(Xi))