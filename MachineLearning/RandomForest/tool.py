import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_breast_cancer,load_iris
from sklearn.model_selection import train_test_split

def split_data(data_train, feature):
    '''select the feature from data
    input:data, feature
    output:data, type = list
    '''
    m = np.shape(data_train)[0]
    data = []
    for i in range(m):
        data_tmp = []
        for x in feature:
            data_tmp.append(data_train[i][x])
        data_tmp.append(data_train[i][-1])
        data.append(data_tmp)
    return data

def load_data():
    '''use the boston dataset from sklearn'''
    print('loading data......')
    dataSet = load_breast_cancer()
    data = dataSet.data
    target = dataSet.target
    for i in range(len(target)):
        if target[i] == 0:
            target[i] = -1
    dataframe = pd.DataFrame(data)
    dataframe.insert(np.shape(data)[1], 'target', target)
    dataMat = np.mat(dataframe)
    X_train, X_test, y_train, y_test =  train_test_split(dataMat[:, 0:-1], dataMat[:, -1], test_size=0.3, random_state=0)
    data_train = np.hstack((X_train, y_train))
    data_train = data_train.tolist()
    X_test = X_test.tolist()

    return data_train, X_test, y_test

if __name__ == '__main__':
    load_data()

