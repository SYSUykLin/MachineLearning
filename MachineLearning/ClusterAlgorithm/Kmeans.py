import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import MachineLearning.ClusterAlgorithm.tool as tool

def KMeans(dataSet, k, distMeas = tool.distEclud, createCent = tool.randCent):
    '''KMeans Algorithm is running......'''
    m = np.shape(dataSet)[0]
    clusterAssment = np.mat(np.zeros((m, 2)))
    centroids = createCent(dataSet, k)
    clusterChanged = True
    while clusterChanged:
        clusterChanged = False
        for i in range(m):
            minDist = np.inf
            minIndex = -1
            for j in range(k):
                distJI = distMeas(centroids[j,:], dataSet[i,:])
                if distJI < minDist:
                    minDist = distJI
                    minIndex = j
            if clusterAssment[i, 0] != minIndex:
                clusterChanged = True
            clusterAssment[i, :] = minIndex, minDist**2
        for cent in range(k):
            ptsInClust = dataSet[np.nonzero(clusterAssment[:, 0].A == cent)[0]]
            centroids[cent, :] = np.mean(ptsInClust, axis=0)
    dataFrame = pd.DataFrame(data=np.hstack((dataSet, clusterAssment[:,0])))
    return dataFrame, centroids

if __name__ == '__main__':
    x, y = tool.loadDataSet()
    dataFrame, center = KMeans(x, 3)
    tool.show(dataFrame)