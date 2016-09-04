

import numpy as np

from utils import count_equals


class Hypothesis(object):

    def h(self, x):
        return 0

    def predict(self, X):
        return np.zeros(X.shape[0])

    def accuracy(self, X, y):
        output = self.predict(X)
        correct = count_equals(output, y)
        return float(correct) / X.shape[0]


class Perceptron(Hypothesis):

    def __init__(self, w=None, b=None):
        self.w = w if w is not None else np.zeros(2)
        self.b = b if b is not None else 0

    def h(self, x):
        # FIXME: np.sign output is in {-1, 0, +1}
        return np.sign(np.dot(self.w.T, x) + self.b)

    def predict(self, X):
        return np.sign(np.dot(X, self.w) + self.b)

    def pla(self, X, y, iterations=10):
        """Perceptron Learning Algorithm."""
        for i in range(iterations):
            for j in range(X.shape[0]):
                x = X[j, :].T
                out = self.h(x)
                diff = (y[j] - out)
                self.w += diff * x
                self.b += diff


class LinearRegression(Hypothesis):

    def __init__(self, w=None, b=None):
        self.w = w if w is not None else np.zeros(2)
        self.b = b if b is not None else 0

    def h(self, x):
        return np.dot(self.w.T, x) + self.b

    def predict(self, X):
        return np.dot(X, self.w) + self.b

