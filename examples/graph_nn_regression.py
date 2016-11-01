

import numpy as np

from happyml import datasets
from happyml.graphs import core
from happyml.graphs.optimize import minimize, SGD
from happyml.graphs.viz import graph2dot
from happyml import plot


def add_layer(in_element, neurons, prefix="", activation=None):
    """Connect an in_element to a new layer and returns that layer."""
    in_dim = in_element.shape[0]
    W = core.Parameter(shape=(neurons, in_dim),
                       name="%sW" % prefix)
    b = core.Parameter(shape=(neurons,),
                       name="%sb" % prefix)
    s = W.dot(in_element) + b
    if activation is not None:
        return activation(s, name="%sReLU" % prefix)
    return s  # Linear neurons


# Load a linear regression dataset.
#dataset = datasets.load("roller_coaster.csv")
dataset = datasets.load("parabola.csv")
#dataset = datasets.load("cubic.csv")

# Build neural network computation graph.
neurons_layers = [1, 4, 1]
x = core.Input(shape=(neurons_layers[0],), name="x")
layers = [x]
n_layers = len(neurons_layers)
for i in range(1, n_layers):
    last = i == n_layers - 1
    layers += [add_layer(layers[i - 1], neurons_layers[i],
                         prefix="Layer%d/" % i,
                         activation=None if last else core.ReLU)]
nn = core.ReduceSum(layers[-1])

# Define loss function.
y = core.Input(name="y")
loss = (y - nn) ** 2

# Check the automatic gradients computation using central difference method.
core.check_gradients(nn)

# Visualize graph using graphviz.
g = graph2dot(nn, filename="graph", format="png")
g.render()
plot.figure(figsize=(15, 4))
plot.imshow("graph.png",
            title="Neural Network Architecture")
plot.show()

# Train and plot each 1000 epochs until the end of the universe.
model = nn.to_model()
while True:
    minimize(loss, dataset, feed={"x": x, "y": y},
             optimizer=SGD(learning_rate=0.01),
             epochs=100,
             batch_size=1)#dataset.get_N())
    model.plot(plot_type="line")
    dataset.plot()
    plot.show()
