
from happyml.datasets import mnist
from happyml import plot
from happyml.graphs import core

import numpy as np


train, test = mnist.load()

indices = np.arange(train.get_N())
np.random.shuffle(indices)

img = plot.imshow(train.X[indices[0]], title="MNIST")
def update_fig(iteration, *args):
    i = indices[iteration]
    print "Image index: %d\tLabel: %d" % (i, train.Y[i])
    img.set_data(train.X[i])
    return img,

plot.animation(update_fig, interval=500)  # interval in ms
