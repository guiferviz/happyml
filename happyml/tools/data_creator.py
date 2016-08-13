
import argparse
import math
import signal
import sys

import numpy as np
from matplotlib import pyplot as plt

import happyml
from happyml import plot
from happyml import datasets


class DataSetCreator(object):
    """Manage matplotlib events and store all points in a dataset array.

    """

    def __init__(self, **args):
        # Make a void plot window.
        fig, ax = plt.subplots()
        if not args['no_scaled']:
            ax.axis('scaled')
        ax.set_xlim(args['limits'][0:2])
        ax.set_ylim(args['limits'][2:4])
        ax.yaxis.grid()
        ax.xaxis.grid()
        scatters = [ax.scatter([], [], c=plot.get_class_color(i), s=50, linewidth=0.25, picker=5) \
                    for i in range(10)]
        if args['dataset']:
            dataset = datasets.load(args['dataset'])
            if dataset.get_N() > 0 and dataset.get_k() == 1:
                if dataset.get_d() == 2:  # Classification dataset.
                    for i in range(10):
                        class_i = dataset.Y.flatten() == i
                        scatters[i].set_offsets(dataset.X[class_i, 0:2])
                elif dataset.get_d() == 1:  # Regression dataset.
                    scatters[0].set_offsets(np.hstack((dataset.X, dataset.Y)))

        # Connect matplotlib event handlers.
        fig.canvas.mpl_connect('button_press_event', self.onclick)
        fig.canvas.mpl_connect('close_event', self.onclose)
        fig.canvas.mpl_connect('key_press_event', self.onkeydown)
        fig.canvas.mpl_connect('pick_event', self.onpick)

        # Save important fields.
        self.selected_class = None
        self.scatters = scatters
        self.fig = fig
        self.ax = ax
        self.save_plot = args['save_plot'] or False
        self.no_regression = args['no_regression'] or False
        self.ones = args.get('ones') or False
        self.binary = args.get('binary') or self.ones


    def getDataSet(self):
        # Join scatters points on a list.
        data = []
        for i, s in enumerate(self.scatters):
            for x1, x2 in s.get_offsets():
                data += [[i, x1, x2]]
        data = np.array(data)
        # Construct dataset object.
        dataset = datasets.DataSet()
        if data.shape[0] > 0:
            # Check if is a classification or regression dataset.
            classes = np.unique(data[:, 0])  # The sorted unique classes
            if len(classes) == 1 and not self.no_regression and not self.binary:
                # If only one class it is regression data.
                dataset.X = data[:, 1].reshape(-1, 1)
                dataset.Y = data[:, 2].reshape(-1, 1)
            else:
                dataset.X = data[:, 1:3]
                dataset.Y = data[:, 0].reshape(-1, 1)

            if self.binary:
                if len(classes) <= 2:
                    if self.ones:
                        dataset.Y[dataset.Y == classes[0]] = -1
                        if len(classes) == 2:
                            dataset.Y[dataset.Y == classes[1]] = 1
                else:
                    raise ValueError("You are suposed to create a binary dataset "
                        "but more than 2 classed were found.")

        return dataset

    def show(self):
        # Show window plot.
        plt.show()

    def onkeydown(self, event):
        key = ord(event.key[0]) - ord('0')
        if 0 <= key <= 9:
            self.selected_class = key

    def onclick(self, event):
        button = 0 if event.button == 1 else \
                 1 if event.button == 3 else None
        x, y = event.xdata, event.ydata
        # If not wheel button (if not delete) and clicked inside the axes (coordinates are not nan).
        if button is not None and (x is not None and y is not None):
            class_number = self.selected_class if self.selected_class else button
            self._add_point(self.scatters[class_number], [x, y, class_number])
            plt.draw()

    def onclose(self, event):
        self._exit()

    def onpick(self, event):
        # If picket using wheel
        if event.mouseevent.button == 2:
            self._remove_point(event.artist, event.ind)
            plt.draw()

    def onexit(self):
        pass

    def _exit(self):
        """
            Saves an image of the dataset plot, prints the dataset to the stdout
        and close the program.
        """
        # Save image.
        if self.save_plot:
            self.fig.savefig(self.save_plot)

    def _add_point(self, scatter, point):
        """
            Add point to the given scatter plot.
        """
        points = scatter.get_offsets()
        points = np.append(points, point[0:2])
        scatter.set_offsets(points)

    def _remove_point(self, scatter, point_idx):
        """
            Remove point for the given scatter plot.
        """
        points = scatter.get_offsets()
        points = np.delete(points, point_idx, axis=0)
        scatter.set_offsets(points)
