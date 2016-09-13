

import numpy as np

from happyml.datasets import DataSet
from happyml.graphs.core import Input, Parameter
from happyml.graphs.viz import graph2dot
from happyml.graphs.optimize import fit
from happyml import plot


# Dataset
X = np.array([[-0.9375    ],
       		  [-0.828125  ],
       		  [-0.6875    ],
       		  [-0.55208333],
       		  [-0.50520833],
       		  [-0.328125  ],
       		  [-0.24479167],
       		  [-0.05729167],
       		  [ 0.04166667],
       		  [ 0.19791667]])
Y = np.array([[-0.125     ],
       		  [-0.00520833],
       		  [ 0.13020833],
       		  [ 0.25      ],
       		  [ 0.38020833],
       		  [ 0.47395833],
       		  [ 0.609375  ],
       		  [ 0.703125  ],
       		  [ 0.86979167],
       		  [ 0.98958333]])
dataset = DataSet()
dataset.X = X
dataset.Y = Y

# Computation graph.
x = Input(name="x")
w = Parameter(name="w")
b = Parameter(name="b")
h = w * x + b

# Visualize using graphviz.
g = graph2dot(h, filename="graph", format="png")
g.render()
#g.view()
img = plot.imread("graph.png")
print img.shape
plot.imshow("graph.png", title="Computational Graph")
plot.show()

# Visualize dataset and initial predictions.
x, y = plot.predict_1d_area(h.to_model())
plot.plot_line(x, y)
plot.dataset_continuous(dataset)
plot.show()

# Optimize.
fit(h, dataset)

# Visualize dataset and final predictions.
x, y = plot.predict_1d_area(h.to_model())
plot.plot_line(x, y)
plot.dataset_continuous(dataset)
plot.show()
