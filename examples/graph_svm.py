

import numpy as np

from happyml import datasets
from happyml.graphs.core import Input, Parameter, Add, Prod, Square, \
                                Dot, Max, ReduceSum, backward_all, \
                                forward_all, check_gradients
from happyml.graphs.viz import graph2dot
from happyml.graphs.optimize import SGD
from happyml import plot


def train_and_plot(x_input, y_input, h, loss, dataset):
    # Visualize graph using graphviz.
    g = graph2dot(h, filename="graph", format="png")
    g.render()
    plot.figure(figsize=(15, 3))
    plot.subplot(131)
    plot.imshow("graph.png",
                title="Computational Graph: %s" % str(h))

    # Visualize dataset and initial predictions.
    plot.subplot(132)
    dataset.plot()
    X, Y, Z = plot.predict_area(h.to_model())
    plot.binary_margins(X, Y, Z, title="Before training",
                        limits=[-1, 1, -1, 1])

    # Checking gradients before training.
    check_gradients(loss)

    # Using a SGD optimizer for minimizing loss.
    optimizer = SGD(learning_rate=0.1)
    optimizer.minimize(loss, dataset,
                       feed={"x": x_input, "y": y_input},
                       epochs=50,
                       batch_size=1)

    # Visualize dataset and final predictions.
    plot.subplot(133)
    dataset.plot()
    X, Y, Z = plot.predict_area(h.to_model())
    plot.binary_margins(X, Y, Z, title="After training",
                        limits=[-1, 1, -1, 1])

    plot.show()


############
# Dataset. #
############
dataset = datasets.load("linearly_separable.csv")


#############################
# Linear computation graph. #
#############################
x = Input(shape=(2,), name="x")
y = Input(name="y")
w = Parameter(shape=(2,), name="w")
b = Parameter(name="b")
h = Dot(w, x) + b
# Hinge loss + l2 loss
lambda_ = 0.001
l2_loss = lambda_ * ReduceSum(w ** 2)
loss = Max(0, 1 - y * h) + l2_loss

# Plot, fit and plot.
train_and_plot(x, y, h, loss, dataset)


##############################
# Squared computation graph. #
##############################
x = Input(shape=(2,), name="x")
y = Input(name="y")
w1 = Parameter(shape=(2,), name="w1")
b1 = Parameter(name="b1")
a1 = w1.dot(x) + b1
w2 = Parameter(shape=(2,), name="w2")
b2 = Parameter(name="b2")
a2 = w2.dot(x ** 2) + b2
h = a1 + a2
# Hinge loss + l2 loss
lambda_ = 0.001
l2_loss = ReduceSum(w1 ** 2) * lambda_ + ReduceSum(w2 ** 2) * lambda_
loss = Max(0, 1 - y * h) + l2_loss

# Plot, fit and plot.
train_and_plot(x, y, h, loss, dataset)
