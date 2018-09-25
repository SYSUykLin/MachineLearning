import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn as sk
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA

class DataPreprocessor(object):

    @staticmethod
    def distance(vecA, vecB):
        '''calculate the distance from vectorA to vectorB
        input: coordinate A and B
        output:distance from A to B
        '''
        dist = (vecA - vecB) * (vecA - vecB).T
        return dist[0, 0]

    @staticmethod
    def randCent(data, k):
        '''initialize center point in KMeans way
        '''
        n = np.shape(data)[1]
        centeroids = np.mat(np.zeros((k, n)))
        for i in range(n):
            minJ = np.min(data[:, i])
            rangJ =np.max(data[:, i]) - minJ
            centeroids[:, i] = minJ * np.mat(np.ones((k, 1))) + np.random.rand(k, 1) * rangJ
        return centeroids

    @staticmethod
    def load_data():
        pca = PCA(n_components=2)
        dataSet = load_iris()
        data = dataSet['data']
        target = dataSet['target']
        data = np.mat(data)
        pca.fit(data)
        data = pca.transform(data)
        data = np.mat(data)
        return data

    @staticmethod
    def show(title, data, center, sub):
        plt.title(title)
        plt.scatter(np.array(data[:, 0]), np.array(data[:, 1]), c = np.array(sub[:, 0]))
        plt.scatter(np.array(center[:, 0]), np.array(center[:, 1]), marker = 'x', s = 50, c = 'red')
        plt.show()
        pass




