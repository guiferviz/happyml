

import numpy as np

from model import Model
from happyml.kernels import linear, gaussian, \
                            polynomial, rational_quadratic


class PerceptronKernel(Model):

    plot_type = "binary_ones"


    def __init__(self, X, y, kernel="gaussian"):
        self.n = X.shape[0]
        self.d = X.shape[1]

        self.X = np.hstack((np.ones((self.n, 1)), X))
        self.y = y

        self.a = np.zeros(self.n)

        if kernel == "linear":
            self.kernel = linear
        elif kernel == "gaussian":
            self.kernel = gaussian
        else:
            self.kernel = kernel

    def h(self, x):
        return np.sign(self.plot_h(x))

    def plot_h(self, x):
        if x.shape[0] == self.d:
            x = np.hstack((1, x))
        out = 0
        for i in range(self.n):
            x_i = self.X[i, :].T
            out += self.a[i] * self.y[i] * self.kernel(x_i, x)
        return out

    def pla(self, iterations=10):
        self.a.fill(0)
        for i in range(iterations):
            errors = False
            for j in range(self.n):
                x_j = self.X[j, :].T
                if self.h(x_j) != self.y[j]:
                    errors = True
                    self.a[j] += 1
            if not errors:
                return i
        return iterations
