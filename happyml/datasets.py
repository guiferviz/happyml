

import sys

import numpy as np

from happyml.utils import one_hot


class DataSet(object):
    """Generic collection of inputs and outputs.

    """

    X = np.empty((0, 0))

    Y = np.empty((0, 0))


    def get_N(self):
        """Gets the number of samples in the dataset.

        """
        # The next two expressions are not necessarily equivalent:
        # self.X.shape[0] and self.Y.shape[0]
        # self.Y.shape[0] <-- Can be 0 if no output assigned.
        return self.X.shape[0]

    def get_d(self):
        """Gets the dimension of each sample in the dataset.

        """
        return self.X.shape[1]

    def get_k(self):
        """Gets the number of outputs of each sample.
        
        """
        return self.Y.shape[1]


def load(filename, delimiter="", **kwargs):
    # Set delimiters if filename has a know extension.
    if delimiter is "":
        delimiter = "," if filename.endswith(".csv") else None
    # Open file and load dataset from stream.
    return load_from_stream(open(filename), delimiter=delimiter, **kwargs)


def load_from_stream(stream, delimiter=",", n_outputs=1,
                     one_hot_output=False, header=False):
    # Check parameters.
    assert not (one_hot_output and abs(n_outputs) != 1), \
        "If one-hot output is selected the number of outputs must be 1."
    # Read stream.
    data = np.loadtxt(stream, delimiter=delimiter, skiprows=int(header))
    # Check feature dimensions.
    d = data.shape[1]
    assert d >= abs(n_outputs), \
        "Number of outputs greater than or equal to the number of data columns."
    # Set starts/ends of the submatrixes X and Y.
    if n_outputs <= 0:
        start_X = 0
        end_X = start_Y = d + n_outputs
        end_Y = d
    else:
        start_Y = 0
        end_Y = start_X = n_outputs
        end_X = d
    # Create DataSet object.
    dataset = DataSet()
    dataset.X = data[:, start_X:end_X]
    dataset.Y = data[:, start_Y:end_Y]
    if one_hot_output:
        dataset.Y = one_hot(dataset.Y)

    return dataset


def save(file, dataset, delimiter=",", header="", footer=""):
    data = np.column_stack((dataset.Y, dataset.X))
    np.savetxt(file, data, delimiter=delimiter, header=header, footer=footer)


def print(dataset, delimiter=","):
    save(sys.stdout, dataset, delimiter=delimiter)


def print_numpy(dataset):
    sys.stdout.write("X = np.%s\n" % repr(dataset.X))
    sys.stdout.write("Y = np.%s\n" % repr(dataset.Y))
