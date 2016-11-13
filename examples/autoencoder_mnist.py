
from happyml.datasets import mnist
from happyml import plot
from happyml.graphs import core
from happyml.graphs.optimize import SGD

import numpy as np


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


# Load MNIST dataset.
train, test = mnist.load()
train.labels = train.Y.flatten()
train.Y = train.X
# subsample
ones = np.where(train.labels == 1)[0]
zeros = np.where(train.labels == 0)[0]
indices = np.hstack((ones[:100], zeros[:100]))
train.X = train.X[indices]
train.Y = train.Y[indices]

# Create network.
input_size = 28 * 28
neurons_layers = [input_size, 256, 128, 256, input_size]
print "Network architecture:", neurons_layers
x = core.Input(shape=(input_size,), name="x")
layers = [x]
n_layers = len(neurons_layers)
for i in range(1, n_layers):
    last = i == n_layers - 1
    layers += [add_layer(layers[i - 1], neurons_layers[i],
                         prefix="Layer%d/" % i,
                         activation=core.Sigmoid if last else core.Sigmoid)]
nn = layers[-1]

# Define loss function.
y = core.Input(shape=(input_size,), name="y")
loss = core.ReduceSum((y - nn) ** 2)

# Check the automatic gradients computation using central difference method.
#core.check_gradients(loss)


# Train and plot until the end of the universe.
model = nn.to_model(out_shape=y.shape)
optimizer = SGD(learning_rate=0.01)
fig = plot.figure()
plot.subplot(121)
input_img = plot.imshow(x.value, shape=(28, 28))
plot.subplot(122)
output_img = plot.imshow(y.value, shape=(28, 28))
def update_fig(iteration, *args):
    optimizer.minimize(loss, train,
                       feed={"x": x, "y": y},
                       epochs=1,
                       offset_epoch=iteration,
                       batch_size=1)
    input_img.set_data(x.value)
    output_img.set_data(nn.value)
    return input_img, output_img


plot.animation(update_fig, interval=100)
