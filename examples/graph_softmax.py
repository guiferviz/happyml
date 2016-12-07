
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
        return activation(s, name="%sSigmoid" % prefix)
    return s  # Linear neurons


# Load multiclass dataset.
dataset = datasets.load("multiclass.csv", one_hot_output=True)

# Create network.
d = dataset.get_d()
k = dataset.get_k()
neurons_layers = [d, 10, k]
print "Network architecture:", neurons_layers
x = core.Input(shape=(d,), name="x")
layers = [x]
n_layers = len(neurons_layers)
for i in range(1, n_layers):
    last = i == n_layers - 1
    layers += [add_layer(layers[i - 1], neurons_layers[i],
                         prefix="Layer%d/" % i,
                         activation=core.Sigmoid)]
nn = layers[-1]

# Define loss function.
y = core.Input(shape=(k,), name="y")
loss = core.ReduceSum((y - nn) ** 2)

# Train and plot each 100 epochs until the end of the universe.
model = nn.to_model()
optimizer = SGD(learning_rate=0.01)
while True:
    optimizer.minimize(loss, dataset,
                       feed={"x": x, "y": y},
                       epochs=100,
                       batch_size=1)
    model.plot(plot_type="multiclass")
    dataset.plot()
    plot.show()
