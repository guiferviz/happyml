

import numpy as np

from happyml import datasets
from happyml.graphs import core
from happyml.graphs.optimize import minimize, SGD
from happyml import plot


dataset = datasets.load("linearly_separable.csv")
dataset.plot()

x = core.Input(shape=(2,), name="x")
W = core.Parameter(shape=(10, 2), name="W")
b = core.Parameter(name="b")
a = W.dot(x) + b
a = core.Tanh(a)
"""
W2 = core.Parameter(shape=(10, 10), name="W2")
b2 = core.Parameter(name="b2")
a2 = W2.dot(a) + b2
a2 = core.Tanh(a2)
nn = core.ReduceSum(a2)
"""
nn = a
y = core.Input(name="y")
loss = (y - nn) ** 2
"""
minimize(loss, dataset, feed={"x": x, "y": y},
         optimizer=SGD(learning_rate=0.1),
         epochs=10,
         batch_size=1)#dataset.get_N())
"""
model = nn.to_model()
model.plot(plot_type="binary_ones")

plot.show()
