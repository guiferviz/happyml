

import numpy as np

from model import Model


class LinearRegression(Model):

    _plot_type = "line"


    def __init__(self, w=None, b=None, d=1):
        self.w = np.asarray(w) if w is not None \
                               else np.zeros(d)
        self.b = b if b is not None else 0


    def h(self, x):
        return np.dot(self.w.T, x) + self.b

    def predict(self, X):
        return np.dot(X, self.w) + self.b

    def fit(self, X, y):
        # Add column of ones.
        X = np.c_[np.ones(X.shape[0]), X]
        # w = pinv(X) * y    (1d array)
        self.w = np.dot(np.linalg.pinv(X), y).ravel()
        self.b = self.w[0]
        self.w = self.w[1:]
