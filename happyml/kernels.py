

import numpy as np


def linear(x_1, x_2):
    return np.dot(x_1, x_2)

def gaussian(x_1, x_2, sigma=1):
    return np.exp(-np.linalg.norm(x_1 - x_2, 2) ** 2 / (2. * sigma ** 2))

def rational_quadratic(x_1, x_2, sigma=0.3):
    norm = np.linalg.norm(x_1 - x_2, 2) ** 2
    return 1 - norm / (norm + sigma)

def polynomial(x_1, x_2, c=0, d=1):
    return np.power(np.dot(x_1, x_2) + c, d)


def create_polynomial(c=0, d=1):
    return lambda x_1, x_2: polynomial(x_1, x_2, c=c, d=d)

def create_gaussian(sigma=1):
    return lambda x_1, x_2: gaussian(x_1, x_2, sigma=sigma)
