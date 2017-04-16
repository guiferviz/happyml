

import numpy as np

from model import Model


def linear_kernel(x_1, x_2):
    return np.dot(x_1, x_2)

def gaussian_kernel(x_1, x_2, sigma=1):
    return np.exp(-np.linalg.norm(x_1 - x_2, 2) ** 2 / (2. * sigma ** 2))

def rational_quadratic_kernel(x_1, x_2, sigma=0.3):
    norm = np.linalg.norm(x_1 - x_2, 2) ** 2
    return 1 - norm / (norm + sigma)

def polynomial_kernel(x_1, x_2, c=0, d=1):
    return np.power(np.dot(x_1, x_2) + c, d)

def get_polynomial_kernel(c=0, d=1):
    return lambda x_1, x_2: polynomial_kernel(x_1, x_2, c=c, d=d)

def get_gaussian_kernel(sigma=1):
    return lambda x_1, x_2: gaussian_kernel(x_1, x_2, sigma=sigma)


class PerceptronKernel(Model):

    def __init__(self, X, y, kernel):
        self.n = X.shape[0]
        self.d = X.shape[1]

        self.X = np.hstack((np.ones((self.n, 1)), X))
        self.y = y

        self.a = np.zeros(self.n)

        if kernel == "linear":
            self.kernel = linear_kernel
        elif kernel == "gaussian":
            self.kernel = gaussian_kernel
        else:
            self.kernel = kernel

    def h(self, x):
        return np.sign(self._h(x))

    def _h(self, x):
        if x.shape[0] == self.d:
            x = np.hstack((1, x))
        out = 0
        for i in range(self.n):
            x_i = self.X[i, :].T
            out += self.a[i] * self.y[i] * self.kernel(x_i, x)
        return out

    def pla(self, iterations=10):
        for i in range(iterations):
            for j in range(self.n):
                x_j = self.X[j, :].T
                if self.h(x_j) != self.y[j]:
                    self.a[j] += 1
