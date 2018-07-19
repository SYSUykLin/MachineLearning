import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import sklearn.datasets as dataTool
import sklearn
from sklearn.linear_model import LogisticRegression

def generator():
    np.random.seed(0)
    x, y = dataTool.make_moons(200, noise=0.2)
    plt.scatter(x[:, 0], x[:, 1], s = 40, c = y, cmap=plt.cm.Spectral)
    plt.show()
    return x, y
    pass

def logistics_regression(X, y):
    # Train the logistic rgeression classifier
    clf = LogisticRegression()
    clf.fit(X, y)
    plot_decision_boundary(clf.predict, X, y, 'Logistics Regression')

def plot_decision_boundary(pred_func, X, y, title):
    # Set min and max values and give it some padding
    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    h = 0.01
    # Generate a grid of points with distance h between them
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    # Predict the function value for the whole gid
    Z = pred_func(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    # Plot the contour and training examples
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Spectral)
    plt.title(title)
    plt.show()

def get_colors(number):
    cnames = {
        1: 'red',
        2: 'blue',
        3: 'green',
        4: 'black',
        5: 'gray',
        6: 'orange',
        7: '#FFEBCD',
        8: '#0000FF',
        9: '#8A2BE2',
        10: '#A52A2A',
        11: '#DEB887',
        12: '#5F9EA0',
        13: '#7FFF00',
        14: '#D2691E',
        15: '#FF7F50',
        16: '#6495ED',
        17: 'orange'
}
    return cnames.get(number)


if __name__ == '__main__':
    x, y = generator()
    print(y)
    logistics_regression(x, y)