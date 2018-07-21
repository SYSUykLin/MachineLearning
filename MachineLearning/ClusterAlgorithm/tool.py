import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import sklearn.datasets as dataSet
import sklearn.decomposition.pca as pca

def loadDataSet():
    '''loading data......'''
    data = dataSet.load_iris()
    dataset = data.data
    target = data.target
    PCA = pca.PCA(n_components=2)
    dataset = PCA.fit_transform(dataset)
    return np.mat(dataset), np.mat(target)
    pass

def show(dataFrame):
    '''show the classification with label'''
    plt.scatter(dataFrame[0], dataFrame[1], c = dataFrame[2], cmap=plt.cm.Spectral)
    plt.show()
    pass

def distEclud(vecA, vecB):
    '''calculate the distance from vecA to vecB'''
    return np.sqrt(np.sum(np.power(vecA - vecB, 2)))
    pass

def randCent(dataSet, k):
    '''create the center'''
    n = np.shape(dataSet)[1]
    centroids = np.mat(np.zeros((k, n)))
    for j in range(n):
        minJ = np.min(dataSet[:, j])
        rangeJ = float(max(dataSet[:, j]) - minJ)
        centroids[:, j] = minJ + rangeJ * np.random.rand(k, 1)
    return centroids

if __name__ == '__main__':
    x, y = loadDataSet()
    dataFrame = pd.DataFrame(data=np.hstack((x, y.T)))
    show(dataFrame)