

from happyml.datasets import load
from happyml import plot
from happyml.models import Perceptron


dataset = load("linearly_separable.csv")

plot.set_theme("minus_plus")
plot.figure(figsize=(24, 6))
plot.suptitle('Perceptron', fontsize=24)

plot.subplot(131)
dataset.plot(limits=[-1, 1, -1, 1], grid=True,
             xlabel="$x_1$", ylabel="$x_2$", label_size=20,
             title="Dataset")


plot.subplot(132)
p = Perceptron()
p.pla(dataset.X, dataset.Y)
title = r'$\mathbf{w} = [%0.2f, %0.2f]$  $b=%0.2f$ (Accuracy: $%0.1f$%%)' % (p.w[0], p.w[1], p.b, p.accuracy(dataset.X, dataset.Y) * 100)
p.plot(xlabel="$x_1$", ylabel="$x_2$", label_size=20,
       title=title)
dataset.plot()


plot.subplot(133)
def feasible_w(w1, w2):
    p = Perceptron()
    p.w[0] = w1
    p.w[1] = w2
    return -1 if p.accuracy(dataset.X, dataset.Y) < 1 else 1
X, Y, Z = plot.grid_function_slow(feasible_w)
plot.binary_ones(X, Y, Z, title="Feasible region",
                 xlabel="$w_1$", ylabel="$w_2$", label_size=20)


plot.show()
