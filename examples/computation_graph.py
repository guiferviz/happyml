

import numpy as np

from happyml import datasets
from happyml.graphs.core import Input, Parameter, Add, Prod, Square, \
                                forward_all
from happyml.graphs.viz import graph2dot
from happyml.graphs.optimize import minimize, SGD
from happyml.graphs.loss import LMS
from happyml import plot


def plot_all(h, loss, dataset):
    # Visualize using graphviz.
    g = graph2dot(h, filename="graph", format="png")
    g.render()
    #g.view()
    plot.figure(figsize=(15, 3))
    plot.subplot(131)
    plot.imshow("graph.png",
                title="Computational Graph: %s" % str(h))

    # Visualize dataset and initial predictions.
    plot.subplot(132)
    dataset.plot()
    x, y = plot.predict_1d_area(h.to_model())
    plot.plot_line(x, y, title="Before training",
                   limits=[-1, 1, -1, 1])

    # Optimize. Provided optimizer and loss are
    # the default options. You can delete them if you prefer.
    minimize(loss, dataset,
             optimizer=SGD(learning_rate=0.1),
             epochs=50,
             batch_size=5)

    # Visualize dataset and final predictions.
    plot.subplot(133)
    dataset.plot()
    x, y = plot.predict_1d_area(h.to_model())
    plot.plot_line(x, y, title="After training",
                   limits=[-1, 1, -1, 1])

    plot.show()


# Dataset
dataset = datasets.load("parabola.csv")


# Linear computation graph.
x = Input(name="x")
y = Input(name="y")
w = Parameter(name="w")
b = Parameter(name="b")
h = w * x + b
loss = (h - y) ** 2

# Plot and fit.
plot_all(h, loss, dataset)


# Squared computation graph.
x = Input(name="x")
y = Input(name="y")
w1 = Parameter(name="w1")
w2 = Parameter(name="w2")
b = Parameter(name="b")
h = Add([b, w1 * x, Prod([w2, x, x])])
# The next line is the same but with more nodes.
#h = b + w1 * x + w2 * x * x
loss = (h - y) ** 2

# Plot and fit.
plot_all(h, loss, dataset)
