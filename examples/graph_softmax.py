
import happyml
from happyml import datasets
from happyml.datasets import mnist, equilibrate_dataset
from happyml import plot
from happyml.graphs import core
from happyml.graphs.optimize import SGD
from happyml.utils import one_hot


def add_layer(in_element, neurons, prefix="", activation=None):
    """Connect an in_element to a new layer and returns that layer."""
    in_dim = in_element.shape[0]
    W = core.Parameter(shape=(neurons, in_dim),
                       name="%sW" % prefix)
    b = core.Parameter(shape=(neurons,),
                       name="%sb" % prefix)
    s = W.dot(in_element) + b
    if activation is not None:
        return activation(s, name="%sSigmoid" % prefix)
    return s  # Linear neurons


# Load multiclass dataset.
dataset = datasets.load("multiclass.csv", one_hot_output=True)
equilibrate_dataset(dataset)
dataset, test = mnist.load()
dataset.X = dataset.X[0:5000, :]
dataset.Y = dataset.Y[0:5000, :]
dataset.Y = one_hot(dataset.Y)
test.Y = one_hot(test.Y)

# Create network.
d = 28*28#dataset.get_d()
k = dataset.get_k()
neurons_layers = [d, k]
print "Network architecture:", neurons_layers
x = core.Input(shape=(d,), name="x")
layers = [x]
n_layers = len(neurons_layers)
for i in range(1, n_layers):
    last = i == n_layers - 1
    layers += [add_layer(layers[i - 1], neurons_layers[i],
                         prefix="Layer%d/" % i,
                         activation=core.ReLU if not last else None)]
#nn = layers[-1]
nn = core.Softmax(layers[-1])

# Define loss function.
y = core.Input(shape=(k,), name="y")
loss = core.ReduceSum(-y * core.Log(nn))
#loss = core.ReduceSum((y - nn) ** 2)

# Check the automatic gradients computation.
core.check_gradients(loss)

# Train and plot each 100 epochs until the end of the universe.
model = nn.to_model()
optimizer = SGD(learning_rate=0.01)
i = 0
while True:
    print "Train Accuracy:", model.accuracy(dataset.X, dataset.Y)
    print "Test Accuracy:", model.accuracy(test.X, test.Y)
    epochs = 1
    optimizer.minimize(loss, dataset,
                       feed={"x": x, "y": y},
                       epochs=epochs,
                       offset_epoch=i * epochs,
                       batch_size=1)
    if d == 2:
        model.plot(plot_type="multiclass", samples=100)
        dataset.plot()
        plot.show()
    i += 1
