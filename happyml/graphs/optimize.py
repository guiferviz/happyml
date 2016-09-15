

import numpy as np

from happyml.graphs.loss import LMS
from happyml.graphs.core import forward_all, backward_all


class Optimizer(object):
    
    def update(self, gradients):
        raise NotImplementedError()

    def fit(self, loss, dataset):
        pass


class SGD(Optimizer):

    def __init__(self, learning_rate=0.1):
        self.learning_rate = learning_rate

    def update(self, gradients):
        for i in gradients:
            i.value -= self.learning_rate * gradients[i]


def minimize(loss, dataset, optimizer=None, epochs=10, batch_size=1):
    optimizer = optimizer or SGD(0.1)

    inputs = [i for i in loss.get_computation_path() if i.is_input]

    n = dataset.get_N()
    idx = np.arange(n)
    for i in range(epochs):
        np.random.shuffle(idx)
        total_loss = 0
        index = 0
        while index < n:
            for j in range(index, min(n, index + batch_size)):
                x, y = dataset[idx[j]]
                inputs[0].set_value(x)
                if len(inputs) > 1: inputs[1].set_value(y)
                forward_all(loss)
                total_loss += loss.value
                gradients = backward_all(loss)
                optimizer.update(gradients)
            index += batch_size
        print "Epoch:", i, "\tLoss:", total_loss
