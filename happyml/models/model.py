

import numpy as np

from happyml.utils import count_equals, \
                          flatten_one_hot
from happyml.plot import model as model_plot


class Model(object):
    """This class represents a Machine Learning model. """

    """Type of plot that will be used when 'plot' method is called."""
    plot_type = None

    """Special predict function used only for plotting.

    The reason for this function is that it improves the graphs in models
    that return non-continuous values. Avoids sharpened edges.
    """
    plot_predict = None


    def h(self, x):
        """Hypothesis. Infer the output of a given input tensor.

        By default, the hypothesis method h calls the predict method adding
        at least one dimension to the input tensor.

        Args:
            x (number or numpy.ndarray): Input tensor to infer.

        Returns:
            The output inferred by the model. The type depends on the model.

        """
        # Check if 'predict' method has been overwritten to avoid infinite
        # recursion between the two default methods 'predict' and 'h'.
        if self.__class__.predict == Model.predict:
            raise NotImplementedError("You must overwrite 'predict' or 'h'"
                                      " methods of the model")

        x = np.atleast_1d(x)
        X = x[np.newaxis, ...]

        return np.squeeze(self.predict(X))

    def predict(self, X):
        """Run the model on a collection of input tensors stored on the
        first dimension of X.

        By default, predict calls the hypothesis method on each of the elements
        of the first dimension of X.

        Args:
            X (number or numpy.ndarray): Input tensors to infer. You can use
                                         a singel number or an one-dimensional
                                         vector in which each element will be
                                         treat as an input to the model.

        Returns:
            Numpy array with the outputs inferred by the model. The type
            of each element depends on the model.

        """
        # Check if 'h' method has been overwritten to avoid infinite
        # recursion between the two default methods 'predict' and 'h'.
        if self.__class__.h == Model.h:
            raise NotImplementedError("You must overwrite 'predict' or 'h'"
                                      " methods of the model")

        X = np.atleast_2d(X)
        output = np.apply_along_axis(self.h, 1, X)
        return np.squeeze(output)

    def accuracy(self, X, Y):
        output = self.predict(X)
        if len(output.shape) > 1:
            output = flatten_one_hot(output)
            Y = flatten_one_hot(Y)
        correct = count_equals(output, Y)
        return float(correct) / X.shape[0]

    def error(self, X, Y):
        return 1 - self.accuracy(X, Y)

    def mse(self, X, Y):
        """Compute the Mean Square Error (MSE).

        Basic error measure for regression models.

        """
        output = self.predict(X)
        assert X.shape == Y.shape
        error = np.mean(np.square(output - Y), axis=0)

        return error

    def plot(self, plot_predict=None, plot_type=None, **kwargs):
        """Delegate the plot to the plot module. """
        # Read or infer params.
        plot_predict = plot_predict or self.plot_predict or self.predict
        plot_type = plot_type or self.get_plot_type()
        # Call plot method on plot module.
        model_plot(plot_predict, plot_type, **kwargs)

    def get_plot_type(self, force=False):
        # Infer plot type only the first time (unless force = T).
        if force or self.plot_type is None:
            self.plot_type = infer_plot_type(self)

        return self.plot_type


def infer_plot_type(model, **kwargs):
    """Infer the plot type. """
    try:
        model.h([0, 0])
        return "binary_ones"
    except:
        # Error with 2d input vector.
        pass

    try:
        model.h([0])
        return "line"
    except:
        # Error with 1d input vector.
        pass

    return "unknown"
