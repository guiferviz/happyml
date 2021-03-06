

import math

import numpy as np


def count_equals(v1, v2):
    """Count the number of elements that are exactly equals.

    Args:
        v1 (numpy.ndarray): Vector to compare.
        v2 (numpy.ndarray): Vector to compare.

    Returns:
        Number of equal elements.

    """
    if v1.ndim != 1:
        v1 = v1.ravel()
    if v2.ndim != 1:
        v2 = v2.ravel()

    return np.count_nonzero(v1 == v2)


def get_f(expr):
    """Returns a 2-dimensional function that evaluates the expression
    passed by parameter.

    Args:
        expr (str): String with a mathematical expression that uses "x"
                    and "y" variables. Must be a valid Python expression
                    because it will be evaluated using "eval" function.
                    The expression can contain also calls to functions in
                    the Python "math" module (e.g. "sqrt(x)+log(y)"). Be
                    aware that using functions like "sqrt(x)" or "x**0.5"
                    in negative numbers will rise an exception. Change the
                    evaluation domain to positive numbers to avoid that
                    problem.

    Returns:
        Return a lambda function that receives 2 parameters: x and y.
        When the returned function is called the expression is evaluated
        using the given values of x and y.

    """
    return lambda x, y: eval(expr, math.__dict__, {"x": x, "y": y})


def central_difference(f, x, y, h=0.1):
    """Compute the derivative of the function f in x, y using the
    central difference method.

    Args:
        f (function): Function with two parameters (x and y).
        x (float): Coordinate x where the gradient will be computed.
        y (float): Coordinate y where the gradient will be computed.
        h (float): Step size.

    Returns:
        gradient (numpy.ndarray): Two dimensional gradient vector.

    """
    gradient = np.zeros(2)
    gradient[0] = (f(x + h, y) - f(x - h, y)) / (2 * h)
    gradient[1] = (f(x, y + h) - f(x, y - h)) / (2 * h)
    return gradient


def one_hot(vector):
    max_output = np.max(vector)
    min_output = np.min(vector)
    N = vector.shape[0]
    k = int(max_output - min_output + 1)
    indexes = np.add(vector, -min_output)
    indexes = indexes.astype(int).reshape(N)
    vector = np.zeros((N, k))
    vector[np.arange(0, N), indexes] = 1

    return vector


def flatten_one_hot(vector, axis=1):
    return np.argmax(vector, axis=axis)


def shuffle(x, y):
    assert len(x) == len(y)
    p = np.random.permutation(len(x))
    return x[p], y[p]


def function2predict(f):
    """Return a function that receives single inputs to a
    function that process several inputs one by one.

    This method uses a loop, so it is not optimized.

    """
    return lambda X: f_predict(f, X)


def f_predict(f, X):
    """Iterate through the first dimension of X and call f with each
    of those values. """
    X = np.atleast_2d(X)
    return np.apply_along_axis(f, 1, X)
