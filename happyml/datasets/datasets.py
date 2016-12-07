

from collections import Counter
import sys
from StringIO import StringIO

import numpy as np

from happyml.utils import one_hot, flatten_one_hot


DATASET_TYPES = [
    "unknown",
    "binary", "binary-one-hot",
    "multiclass", "multiclass-one-hot", "multiclass-multioutput",
    "continuous", "continuous-multioutput",
]


class DataSet(object):
    """Generic collection of inputs and outputs.

    """

    X = None

    Y = None

    _type = None

    _classes = None


    def __init__(self, X=None, Y=None):
        self.X = np.empty((0, 0)) if X is None else X
        self.Y = np.empty((0, 0)) if Y is None else Y
        if self.X.ndim == 1:
            self.X = self.X.reshape((-1, 1))
        if self.Y.ndim == 1:
            self.Y = self.Y.reshape((-1, 1))


    def get_N(self):
        """Get the number of samples in the dataset.

        """
        # The next two expressions are not necessarily equivalent:
        # self.X.shape[0] and self.Y.shape[0]
        # self.Y.shape[0] <-- Can be 0 if no output assigned.
        return self.X.shape[0]

    def get_d(self):
        """Get the dimension of each sample in the dataset.

        """
        return self.X.shape[1]

    def get_k(self):
        """Get the number of outputs of each sample.

        """
        return self.Y.shape[1]

    def get_flatten_classes(self):
        """Return a flattened array with the target values.

        It is not defined with continuous/multiclass multioutput
        datasets.
        """
        _type = self.get_type()
        if "one-hot" in _type:
            return flatten_one_hot(self.Y).flatten().astype(int)
        elif "binary" == _type or "multiclass" == _type:
            return self.Y.flatten().astype(int)
        elif "continuous" == _type:
            return self.Y.flatten()

        raise ValueError("No flattened defined for a %s dataset." %
                         _type)

    def get_classes(self, force=False):
        """Return an array with the unique target values.

        """
        if force or self._classes is None:
            self._classes = np.unique(self.get_flatten_classes())

        return self._classes

    def get_class(self, class_number):
        """Return the indexes of the samples of the given class.

        """
        flattened_classes = self.get_flatten_classes()
        return np.where(flattened_classes == class_number)[0]

    def get_type(self, force=False):
        if force or self._type is None:
            self._type = get_type(self.Y)

        return self._type

    def __getitem__(self, index):
        return self.X[index, :].T, self.Y[index, :].T

    def __str__(self):
        io = StringIO()
        io.write("DataSet. Type: %s.\n" % self.get_type())
        save(io, self)
        return io.getvalue()

    def __len__(self):
        return self.get_N()


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


def show(dataset, delimiter=","):
    save(sys.stdout, dataset, delimiter=delimiter)


def show_numpy(dataset):
    sys.stdout.write("X = np.%s\n" % repr(dataset.X))
    sys.stdout.write("Y = np.%s\n" % repr(dataset.Y))


def get_type(y):
    """Try to guess the type of the numpy array.

    The differents types are:
    * unknown
    * binary
    * binary-one-hot
    * multiclass
    * multiclass-one-hot
    * multiclass-multioutput
    * continuous
    * continuous-multioutput

    Arrays with strings are never guessed as *-one-hot.

    This method is inspired by *sklearn.utils.multiclass.type_of_target*.

    """
    y = np.asarray(y)

    if y.ndim > 2 or y.ndim < 1:
        return "unknown"

    if y.shape[0] == 0 or (y.ndim == 2 and y.shape[1] == 0):
        return "unknown"

    multioutput = False
    if y.ndim == 2 and y.shape[1] > 1:
        multioutput = True

    if y.dtype == float and np.any(y != y.astype(int)):
        if multioutput:
            return "continuous-multioutput"
        return "continuous"

    classes = np.unique(y)

    one_hot = False
    if multioutput and len(classes) == 2 and \
       y.dtype.kind in "biuf" and \
       0 in classes and 1 in classes and \
       np.array_equal(np.sum(y, axis=1), np.ones(y.shape[0])):
        one_hot = True

    if len(classes) > 2 or (multioutput and not (
                            one_hot and y.shape[1] <= 2)):
        if multioutput:
            if one_hot:
                return "multiclass-one-hot"
            return "multiclass-multioutput"
        return "multiclass"

    if one_hot:
        return "binary-one-hot"
    return "binary"


def equilibrate_dataset(dataset, n=None):
    dataset_type = dataset.get_type()
    if dataset_type == "binary" or dataset_type == "multiclass":
        classes = dataset.get_classes()
        y = dataset.Y.flatten()
        counter = Counter(y)
        count_classes = counter.most_common(len(classes))
        if n is None:
            max_class, n = count_classes[0]  # max class number
        count_classes = dict(count_classes)
        X = np.empty((0, dataset.get_d()))
        Y = np.empty((0, dataset.get_k()))
        for c in classes:
            idx = y == c
            to_add = n - count_classes[c]
            if to_add < 0:
                _x = dataset.X[idx, :][0:to_add, :]
                _y = dataset.Y[idx, :][0:to_add, :]
            else:
                _x = np.pad(dataset.X[idx, :], ((0, to_add), (0, 0)), "wrap")
                _y = np.pad(dataset.Y[idx, :], ((0, to_add), (0, 0)), "wrap")
            X = np.vstack((X, _x))
            Y = np.vstack((Y, _y))
        dataset.X, dataset.Y = X, Y
