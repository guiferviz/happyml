

import numpy as np


class Loss(object):

    def loss(self, y, h_x):
        return 0

    def loss_gradient(self, y, h_x):
        return 0


class LMS(Loss):

    def loss(self, y, h_x):
        return (y - h_x) ** 2

    def loss_gradient(self, y, h_x):
        return h_x - y


class SVM(Loss):

    def loss(self, y, h_x):
        return max(0, 1 - y * h_x)

    def loss_gradient(self, y, h_x):
        if abs(h_x) < 1:
            return -y
        return 0
