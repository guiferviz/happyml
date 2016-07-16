

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

