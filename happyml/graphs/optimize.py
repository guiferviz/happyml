

import numpy as np

from happyml.graphs.loss import LMS
from happyml.graphs.core import forward_all, backward_all


class Optimizer(object):
    
    def update(self, gradients):
        raise NotImplementedError()

    def fit(self, loss, dataset):
        pass

    def minimize(self, loss, dataset,
                 epochs=10, batch_size=1, offset_epoch=0, feed=None,
                 shuffle=True):
        if feed is None:
            inputs = [i for i in loss.get_computation_path() if i.is_input]
            feed = {"x": inputs[0], "y": inputs[1]}

        n = dataset.get_N()
        idx = np.arange(n)
        batches = n // batch_size
        print "-------------------------------------------------"
        for i in range(epochs):
            if shuffle: np.random.shuffle(idx)
            total_loss = 0
            index = 0
            while index < n:
                gradients = {}
                for j in range(index, min(n, index + batch_size)):
                    x, y = dataset[idx[j]]
                    feed["x"].set_value(x)
                    feed["y"].set_value(y)
                    forward_all(loss)
                    total_loss += loss.value
                    backward_all(loss, gradients)
                #for k in gradients:
                #    gradients[k] *= 1. / batch_size
                self.update(gradients)
                index += batch_size
            print "Epoch:", offset_epoch + i, "\tLoss:", total_loss


class SGD(Optimizer):

    def __init__(self, learning_rate=0.1):
        self.learning_rate = learning_rate

    def update(self, gradients):
        for i in gradients:
            i.value -= self.learning_rate * gradients[i]


class Momentum(Optimizer):

    def __init__(self, learning_rate=0.1, momentum=0.9):
        self.learning_rate = learning_rate
        self.momentum = 0.9
        self._v = {}

    def update(self, gradients):
        for i in gradients:
            self._v.setdefault(i, np.zeros)
            i.value -= self.learning_rate * gradients[i]

