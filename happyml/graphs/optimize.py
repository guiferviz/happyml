

import numpy as np

from happyml.graphs.loss import LMS
from happyml.graphs.core import forward_all, backward_all


class Optimizer(object):
    
    def update(gradients):
        pass


class SGD(Optimizer):

    def __init__(self, learning_rate=0.1):
        self.learning_rate = learning_rate

    def update(self, gradients):
        for i in gradients:
            i.value -= self.learning_rate * gradients[i]


def fit(f, dataset, loss=None, optimizer=None, epochs=10,
        batch_size=1):
    optimizer = optimizer or SGD(0.1)
    loss = loss or LMS()

    inputs = [i for i in f.get_computation_path() if i.is_input]
    # FIXME: computational graphs accepts only one input.
    f_input = inputs[0]

    n = dataset.get_N()
    idx = np.arange(n)
    for i in range(epochs):
        np.random.shuffle(idx)
        total_loss = 0
        index = 0
        while index < n:
            for j in range(index, min(n, index + batch_size)):
                f_input.set_value(dataset.X[idx[j], :].T)
                y = dataset.Y[idx[j], :].T.reshape(f.shape)
                forward_all(f)
                total_loss += loss.loss(y, f.value)
                loss_gradient = loss.loss_gradient(y, f.value)
                gradients = backward_all(f, loss_gradient)
                optimizer.update(gradients)
            index += batch_size
        print "Epoch", i, "\tLoss:", total_loss
