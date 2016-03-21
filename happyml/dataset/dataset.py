
import numpy as np


class DataSet():
    """
        Generic collection of inputs and outputs.
    """

    X = np.array([])

    Y = np.array([])

    def get_N(self):
        """
            Gets the number of samples in the dataset.
        """
        # Both expressions are equivalent.
        # return self.X.shape[0]
        return self.Y.shape[0]

    def get_d(self):
        """
            Gets the dimension of each sample in the dataset.
        """
        return self.X.shape[1]

    def get_k(self):
        """
            Gets the number of outputs of each sample.
        """
        return self.Y.shape[1]


def load(filename, delimiter="", n_outputs=1, one_shot_output=False, header=False):
    # Set delimiters if filename has a know extension.
    if delimiter is "":
        if filename.endswith(".csv"):
            delimiter = ","
        else:
            delimiter = None
    # Open file and load dataset from stream.
    return load_from_stream(open(filename), delimiter=delimiter, n_outputs=n_outputs,
                            one_shot_output=one_shot_output, header=header)


def load_from_stream(stream, delimiter=",", n_outputs=1,
                     one_shot_output=False, header=False):
    # Check parameters.
    assert not (one_shot_output and abs(n_outputs) != 1), \
        "If one-shot output is selected the number of outputs must be 1."
    # Read stream.
    data = np.loadtxt(stream, delimiter=delimiter, skiprows=int(header))
    # Check feature dimensions.
    d = data.shape[1]
    assert d >= abs(n_outputs), \
        "Number of outputs greater than number of data columns."
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
    if one_shot_output:
        max_output = dataset.Y.max()
        min_output = dataset.Y.min()
        N = dataset.get_N()
        k = max_output - min_output + 1
        indexes = np.add(dataset.Y, -min_output)
        indexes = indexes.astype(int).reshape(N)
        dataset.Y = np.zeros((N, k))
        dataset.Y[np.arange(0, N), indexes] = 1

    return dataset
