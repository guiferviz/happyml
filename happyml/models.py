

import numpy as np


class Hypothesis(object):

    def h(self, x):
        return 0

    def predict(self, X):
        return np.zeros(X.shape[0])


class Perceptron(Hypothesis):

    def __init__(self, w=None, b=None):
        self.w = w if w is not None else np.zeros(2)
        self.b = b if b is not None else 0

    def h(self, x):
        return np.sign(np.dot(self.w.T, x) + self.b)

    def predict(self, X):
        return np.sign(np.dot(X, self.w) + self.b)


class LinearRegression(Hypothesis):

    def __init__(self, w=None, b=None):
        self.w = w if w is not None else np.zeros(2)
        self.b = b if b is not None else 0

    def h(self, x):
        return np.dot(self.w.T, x) + self.b

    def predict(self, X):
        return np.dot(X, self.w) + self.b

