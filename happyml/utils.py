

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
    if v1.ndim != 1 or v2.ndim != 1:
    	raise ValueError("count_equals expect two vectors")
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
                    aware that using functions like "sqrt(x)" in negative
                    numbers will rise an exception. Use "x**0.5" instead
                    or change the evaluation domain to positive numbers
                    to avoid that problem.

    Returns:
        Return a lambda function that receives 2 parameters: x and y.
        When the returned function is called the expression is evaluated
        using the given values of x and y.

    """
    return lambda x, y: eval(expr, math.__dict__, {"x": x, "y": y})

