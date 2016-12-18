

import numpy as np

from utils import count_equals, flatten_one_hot


class Model(object):

    def h(self, x):
        return 0

    def predict(self, X):
        # Predict first example to get the output dim.
        x = X[0, :].T
        h = self.h(x)
        if len(h.shape) == 0:
            h = h.reshape((1,))
        output = np.empty((X.shape[0], h.shape[0]))
        output[0, :] = h
        # Predict all
        for i in range(1, X.shape[0]):
            x = X[i, :].T
            output[i, :] = self.h(x)
        return output

    def accuracy(self, X, y):
        output = self.predict(X)
        if len(output.shape) > 1:
            output = flatten_one_hot(output)
            y = flatten_one_hot(y)
        correct = count_equals(output, y)
        return float(correct) / X.shape[0]


class Perceptron(Model):

    def __init__(self, w=None, b=None, d=2):
        self.w = w if w is not None else np.zeros(d)
        self.b = b if b is not None else 0

    def h(self, x):
        # FIXME: np.sign output is in {-1, 0, +1}
        return np.sign(np.dot(self.w.T, x) + self.b)

    def _h(self, x):
        return np.dot(self.w.T, x) + self.b

    def predict(self, X):
        return np.sign(np.dot(X, self.w) + self.b)

    def pla(self, X, y, iterations=10):
        """Perceptron Learning Algorithm."""
        y = y.flatten()
        idx = np.arange(X.shape[0])
        for i in range(iterations):
            np.random.shuffle(idx)
            error = False
            for j in idx:
                x = X[j, :].T
                if self.h(x) != y[j]:
                    error = True
                    self.w += y[j] * x
                    self.b += y[j]
                    break
            if not error: return i
        return iterations

    def pocket(self, X, y, iterations=10):
        max_accuracy = 0.0
        y = y.flatten()
        pocket_w = np.array(self.w)
        pocket_b = self.b
        idx = np.arange(X.shape[0])
        for iteration in range(iterations):
            np.random.shuffle(idx)
            for i in idx:
                x = X[i, :].T
                if self.h(x) != y[i]:
                    self.w += y[i] * x
                    self.b += y[i]
                    accuracy = self.accuracy(X, y)
                    if accuracy > max_accuracy:
                        max_accuracy = accuracy
                        pocket_w = np.array(self.w)
                        pocket_b = self.b
                    break
        self.w = pocket_w
        self.b = pocket_b


class LinearRegression(Model):

    def __init__(self, w=None, b=None, d=1):
        self.w = w if w is not None else np.zeros(d)
        self.b = b if b is not None else 0

    def h(self, x):
        return np.dot(self.w.T, x) + self.b

    def transform(self, X):
        return X

    def predict(self, X):
        X = self.transform(X)
        return np.dot(X, self.w) + self.b

    def fit(self, X, y):
        # Add column of ones.
        X = np.c_[np.ones(X.shape[0]), X]
        # w = pinv(X) * y    (1d array)
        self.w = np.dot(np.linalg.pinv(X), y).flatten()
        self.b = self.w[0]
        self.w = self.w[1:]


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
