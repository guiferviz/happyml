

import numpy as np

from happyml import datasets
from happyml.graphs.core import Input, Parameter, Add, Prod, Square, \
                                forward_all, check_gradients
from happyml.graphs.viz import graph2dot
from happyml.graphs.optimize import minimize, SGD
from happyml.graphs.loss import LMS
from happyml import plot


def plot_all(x_input, y_input, h, loss, dataset):
    # Visualize graph using graphviz.
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

    # Provided SGD optimizer is the default option.
    # You can delete it if you prefer.
    print "-------------------------------------------------"
    minimize(loss, dataset, feed={"x": x_input, "y": y_input},
             optimizer=SGD(learning_rate=0.1),
             epochs=10,
             batch_size=1)

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

# Plot, fit and plot.
plot_all(x, y, h, loss, dataset)


# Squared computation graph.
x = Input(name="x")
y = Input(name="y")
w1 = Parameter(name="w1")
w2 = Parameter(name="w2")
b = Parameter(name="b")
h = Add([b, w1 * x, w2 * x ** 2])
# The next line is the same but with more nodes.
#h = b + w1 * x + w2 * x * x
loss = (h - y) ** 2

# Plot, fit and plot.
plot_all(x, y, h, loss, dataset)
