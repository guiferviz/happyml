
from happyml.datasets import mnist
from happyml import plot
from happyml.graphs import core


train, test = mnist.load()
fig = plot.figure()
img = plot.imshow(train.X[0])

index = 0
def update_fig(*args):
	global index
	print "Image index: %d\tLabel: %d" % (index, train.Y[index])
	img.set_data(train.X[index])
	index += 1
	return img,

plot.animation(update_fig)
