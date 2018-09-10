import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class DampedNewton(object):

    def __init__(self, feature, label, iterMax, sigma, delta):
        self.feature = feature
        self.label = label
        self.iterMax = iterMax
        self.sigma = sigma
        self.delta = delta
        self.w = None

    def get_error(self, w):
        return (self.label - self.feature*w).T * (self.label - self.feature*w)/2

    def first_derivative(self):
        m, n = np.shape(self.feature)
        g = np.mat(np.zeros((n, 1)))
        for i in range(m):
            err = self.label[i, 0] - self.feature[i, ]*self.w
            for j in range(n):
                g[j, ] -= err*self.feature[i, j]
        return g

    def second_derivative(self):
        m, n = np.shape(self.feature)
        G = np.mat(np.zeros((n, n)))
        for i in range(m):
            x_left = self.feature[i, ].T
            x_right = self.feature[i, ]
            G += x_left * x_right
        return G

    def get_min_m(self, d, g):
        m = 0
        while True:
            w_new = self.w + pow(self.sigma, m)*d
            left = self.get_error(w_new)
            right = self.get_error(self.w) + self.delta*pow(self.sigma, m)*g.T*d
            if left <= right:
                break
            else:
                m += 1
        return m

    def newton(self):
        n = np.shape(self.feature)[1]
        self.w = np.mat(np.zeros((n, 1)))
        it = 0
        while it <= self.iterMax:
            g = self.first_derivative()
            G = self.second_derivative()
            d = -G.I * g
            m = self.get_min_m(d, g)
            self.w += pow(self.sigma, m) * d
            if it % 100 == 0:
                print('it: ', it, ' error: ', self.get_error(self.w)[0, 0])
            it += 1
        return self.w



