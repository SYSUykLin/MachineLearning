import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import MachineLearning.ClusterAlgorithm.Kmeans as kMeans
import sklearn.datasets as dataset
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings("ignore")

def load_data():
    data = dataset.load_breast_cancer().data
    target = dataset.load_breast_cancer().target
    for i in range(len(target)):
        if target[i] == 0:
            target[i] = -1
    x_train, x_test, y_train, y_test = train_test_split(data, target, random_state=42, shuffle=True, test_size=0.4)
    return x_train, x_test, y_train, y_test
    pass

def rbf(x_train, center, y_train, gama = 0.001, lamda = 0.01):
    M = center.shape[0]
    N = len(x_train)
    Z = np.zeros((M, N))
    for i in range(M):
        for j in range(N):
            Z[i][j] = Gaussian(x_train[j], center[i], gama)
    I1 = np.eye(N, k = 0)
    beta = np.linalg.inv(np.dot(Z.T, Z) + lamda * I1)
    beta = np.dot(beta, Z.T)
    y_train = np.mat(y_train)
    beta = np.dot(y_train, beta)
    return beta
    pass

def predict(beta, x_train, center, gama):
    result = []
    for x in x_train:
        x = np.mat(x)
        sum = 0
        for i, vecB in enumerate(center):
            sum += beta[0,i]*Gaussian(x, vecB, gama)
        result.append(sum)
    return result
    pass

def Gaussian(vecA, vecB, gama):
    x_x = np.abs(np.sum(vecA - vecB))
    x_x_2 = np.power(x_x, 2)
    return np.exp(-1.0 * gama * x_x_2)
    pass

def RBF(y_test, x_test, y_train, x_train, gamma = 0.001, lamda = 0.01, k = 4):
    Again = True
    while Again == True:
        _,center = kMeans.KMeans(np.mat(x_train), k)
        beta = rbf(x_train, center, y_train, gama=gamma, lamda=lamda)
        Again = False
        for i in range(beta.shape[1]):
            if np.isnan(beta[0, i]):
                Again = True
    result = predict(beta, x_train, center, gamma)
    for i in range(len(result)):
        if result[i] > 0:
            result[i] = 1
        else:
            result[i] = -1
    posibility = 0
    for i in range(len(result)):
        if result[i] == y_train[i]:
            posibility += 1
    train_accuracy = posibility/len(result)
    result = predict(beta, x_test, center, gamma)

    for i in range(len(result)):
        if result[i] > 0:
            result[i] = 1
        else:
            result[i] = -1
    posibility = 0
    for i in range(len(result)):
        if result[i] == y_test[i]:
            posibility += 1
    test_accuracy = posibility/len(result)
    return train_accuracy, test_accuracy


if __name__ == '__main__':
    x_train, x_test, y_train, y_test = load_data()
    gamma = [1,0.1,0.01,0.001,0.0001]
    lamda = gamma
    train_accuracy = []
    test_accutacy = []
    c = ['red', 'blue', 'orange', 'green', 'yellow', 'black']
    for n, i in enumerate(gamma):
        for j in lamda:
            train, text = RBF(x_test=x_test, y_test=y_test, x_train=x_train, y_train=y_train, gamma=i, lamda=j)
            print('gama : ',i, ' lamda : ', j, ' train_accuracy : ', train, ' text_accuray : ', text)
            train_accuracy.append(train)
            test_accutacy.append(text)
        plt.plot(lamda, train_accuracy, c = c[n], label = 'gamma:'+str(i) + ' (train)')
        plt.plot(lamda, test_accutacy, c = c[n], linestyle='--', label = 'gamma:'+str(i) + ' (test)')
        plt.xlabel('lambda')
        plt.ylabel('accuracy')
        plt.legend(loc = 'upper right')
        train_accuracy = []
        test_accutacy = []
    plt.show()

    for n, i in enumerate(lamda):
        for j in gamma:
            train, text = RBF(x_test=x_test, y_test=y_test, x_train=x_train, y_train=y_train, gamma=j, lamda=i)
            print('lamda : ',i, ' gama : ', j, ' train_accuracy : ', train, ' text_accuray : ', text)
            train_accuracy.append(train)
            test_accutacy.append(text)
        plt.plot(gamma, train_accuracy, c = c[n], label = 'lamda:'+str(i) + ' (train)')
        plt.plot(gamma, test_accutacy, c = c[n], linestyle='--', label = 'lamda:'+str(i) + ' (test)')
        plt.xlabel('gamma')
        plt.ylabel('accuracy')
        plt.legend(loc = 'upper right')
        train_accuracy = []
        test_accutacy = []
    plt.show()

    ks = [2,3,4,5,6,7]
    train_accuracy = []
    test_accutacy = []
    for i in range(6):
        for n, i in enumerate(ks):
            train, text = RBF(x_test=x_test, y_test=y_test, x_train=x_train, y_train=y_train, gamma=0.0001, lamda=0.01, k=i)
            print('k == ' + str(i))
            train_accuracy.append(train)
            test_accutacy.append(text)
        plt.plot(ks, train_accuracy, c = c[n], label = 'train')
        plt.plot(ks, test_accutacy, c = c[n], linestyle='--', label = 'test')
        plt.xlabel('the number of k')
        plt.ylabel('accuracy')
        plt.legend(loc = 'upper left')
        plt.show()
        train_accuracy = []
        test_accutacy = []
    pass
