

import numpy as np

import happyml
from happyml import datasets
from happyml import plot
from happyml.graphs import core
from happyml.graphs.optimize import SGD


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


# Create dataset.
img = plot.imread("rubik.jpg", (64, 64))
print "Image with shape %s loaded." % str(img.shape)

size = np.prod(img.shape[0:2])

dataset = datasets.DataSet()
dataset.X = np.zeros((size, 2))
dataset.Y = np.zeros((size, 3))

idx = 0
for i in np.ndindex(img.shape[0:2]):
    dataset.X[idx, :] = i
    dataset.Y[idx, :] = img[i[0], i[1], 0:3]
    idx += 1
# Normalizing the input.
dataset.X = (dataset.X - np.mean(dataset.X)) / np.std(dataset.X)
# Normalize the output if you use Tanh or Sigmoid activations.
#dataset.Y /= 255.
print "Dataset ready for use."
plot.imshow(img, title="Training Set :)")
plot.show()


# Create network.
hidden_size = 50
hidden_number = 6
neurons_layers = [2,] + ([hidden_size,] * hidden_number) + [3,]
print "Network architecture:", neurons_layers
x = core.Input(shape=(neurons_layers[0],), name="x")
layers = [x]
n_layers = len(neurons_layers)
for i in range(1, n_layers):
    last = i == n_layers - 1
    layers += [add_layer(layers[i - 1], neurons_layers[i],
                         prefix="Layer%d/" % i,
                         activation=None if last else core.ReLU)]
nn = layers[-1]

# Define loss function.
y = core.Input(shape=(3,), name="y")
loss = core.ReduceSum((y - nn) ** 2)

# Check the automatic gradients computation using central difference method.
core.check_gradients(loss)


# Train and plot each 1000 epochs until the end of the universe.
model = nn.to_model(out_shape=y.shape)
optimizer = SGD(learning_rate=0.000001)
fig = plot.figure()
plot_img = plot.imshow(dataset.Y, shape=img.shape)
iteration = 0
def update_fig(*args):
    global iteration, model, plot_img, optimizer, loss, dataset, x, y
    optimizer.minimize(loss, dataset,
                       feed={"x": x, "y": y},
                       epochs=1,
                       offset_epoch=iteration,
                       batch_size=1)
    iteration += 1
    outimg = model.predict(dataset.X)
    plot_img.set_data(outimg)
    return plot_img,


import matplotlib.animation as animation
ani = animation.FuncAnimation(fig, update_fig, interval=1000, blit=True)
plot.show()
