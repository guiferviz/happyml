

import numpy as np

from happyml import datasets
from happyml import plot
from happyml.models import LinearRegression


# 1D Dataset.
dataset = datasets.load("parabola.csv")

################################
# Linear Regression: x * w + b #
################################
h = LinearRegression()
h.fit(dataset.X, dataset.Y)

# Plotting.
dataset.plot()
h.plot(title="b=%.3f, w=%.3f" % (h.b, h.w))
plot.show()

############################################
# Linear Regression: x^2 * w2 + x * w1 + b #
############################################
h = LinearRegression(d=2)
def transform(X):
	return np.c_[X, np.square(X[:, 0])]
h.transform = transform
X = transform(dataset.X)
h.fit(X, dataset.Y)

# Plotting.
dataset.plot()
h.plot(title="b=%.3f, w=%s" % (h.b, h.w))
plot.show()
