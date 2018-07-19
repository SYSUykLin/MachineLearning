import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sea
import sklearn.datasets as dataset
def get_data_message():
    data = dataset.load_iris()
    features = data.data
    target = data.target
    return features, target
class pca(object):
    def fit(self, data_features, y):
        data_mean = np.mean(data_features, axis=0)
        data_features -= data_mean
        cov = np.dot(data_features.T, data_features)
        eig_vals, eig_vecs = np.linalg.eig(cov)
        eig_pairs = [(np.abs(eig_vals[i]), eig_vecs[:, i]) for i in range(len(eig_vals))]
        a = np.matrix(eig_pairs[0][1]).T
        b = np.matrix(eig_pairs[1][1]).T
        u = np.hstack((a, b))
        data_new = np.dot(data_features, u)
        return data_new
    def show(self, data_new):
        plt.scatter(data_new[:, 0].tolist(), data_new[:, 1].tolist(), c='red')
        plt.show()
    pass

if __name__ == '__main__':
    p = pca()
    x, y = get_data_message()
    data = p.fit(x, y)
    p.show(data)