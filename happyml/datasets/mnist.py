"""
This file is almost a copy&paste of one of the examples
of the Lasagne library.
https://github.com/Lasagne/Lasagne/blob/master/examples/mnist.py
"""

import gzip
import os
import sys
if sys.version_info[0] == 2:
    from urllib import urlretrieve
else:
    from urllib.request import urlretrieve

import numpy as np

from happyml.config import happyml_data_dir
from .datasets import DataSet


YANN_LECUN_MNIST = "http://yann.lecun.com/exdb/mnist/"
MNIST_DATA_SUBDIR = "mnist"


def load(data_dir=None, normalize=True, invert=True):
    if data_dir is None:
        data_dir = os.path.join(happyml_data_dir(), MNIST_DATA_SUBDIR)
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)

    def download(filename, path, source=YANN_LECUN_MNIST):
        print("Downloading %s" % filename)
        urlretrieve(source + filename, path)

    def load_mnist_images(filename):
        # Download if needed.
        path = os.path.join(data_dir, filename)
        if not os.path.exists(path):
            download(filename, path)
        # Read downloaded gz file.
        with gzip.open(path, 'rb') as f:
            data = np.frombuffer(f.read(), np.uint8, offset=16)
        # Reshape, normalize and reverse inputs.
        # We reverse the data becouse we want white background
        # and black foreground.
        data = data.reshape(-1, 28, 28)
        if normalize:
            data = data / 255.
            if invert:
                return 1. - data
            return data
        if invert:
            return 255. - data
        return data

    def load_mnist_labels(filename):
        # Download if needed.
        path = os.path.join(data_dir, filename)
        if not os.path.exists(path):
            download(filename, path)
        # Read downloaded gz file.
        with gzip.open(path, 'rb') as f:
            data = np.frombuffer(f.read(), np.uint8, offset=8)
        # Labels are a vector of integers.
        return data

    # Download and read the training and test set images and labels.
    X_train = load_mnist_images('train-images-idx3-ubyte.gz')
    y_train = load_mnist_labels('train-labels-idx1-ubyte.gz')
    X_test = load_mnist_images('t10k-images-idx3-ubyte.gz')
    y_test = load_mnist_labels('t10k-labels-idx1-ubyte.gz')

    dataset_train = DataSet(X_train, y_train)
    dataset_test = DataSet(X_test, y_test)

    return dataset_train, dataset_test
