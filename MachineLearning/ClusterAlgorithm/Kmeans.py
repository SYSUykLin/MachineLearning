import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from Tool import DataPreprocessor
def Kmeans(data, k, centroids):
    '''input: data(mat)
              k(int)
    output: centriods(mat)'''
    m, n = np.shape(data)
    subCenter = np.mat(np.zeros((m, 2)))
    change = True
    while change == True:
        change = False
        for i in range(m):
            minDist = np.inf
            minIndex = 0
            for j in range(k):
                dist = DataPreprocessor.distance(data[i, ], centroids[j, ])
                if dist < minDist:
                    minDist = dist
                    minIndex = j
            if subCenter[i, 0] != minIndex:
                change = True
                subCenter[i, ] = np.mat([minIndex, minDist])
        for j in range(k):
            sum_all = np.mat(np.zeros((1, n)))
            r = 0
            for i in range(m):
                if subCenter[i, 0] == j:
                    sum_all += data[i, ]
                    r += 1
            for z in range(n):
                try:
                    centroids[j, z] = sum_all[0, z]/r
                except:
                    print('r is zero!')
    return centroids, subCenter

if __name__ == '__main__':
    data = DataPreprocessor.load_data()
    centroids, subCenter = Kmeans(data, 4, DataPreprocessor.randCent(data, 4))
    DataPreprocessor.show('KMeans', data, centroids, subCenter)