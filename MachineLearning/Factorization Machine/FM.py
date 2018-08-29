import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from loadDataSet import loadData

def Accuracy(preiction, classlabel):
    score = 0
    for i in range(len(preiction)):
        if preiction[i] > 0.5:
            preiction[i] = 1
        else:
            preiction[i] = -1
        if preiction[i] == classlabel[i]:
            score += 1
    print('Accuracy: ', score/len(preiction))

def initialize(n, k):
    v = np.mat(np.zeros((n, k)))
    for i in range(n):
        for j in range(k):
            v[i, j] = np.random.normal(0, 0.2)
    return v

def sigmoid(inx):
    return 1.0/(1+np.exp(-inx))

def getCost(predict, classLabels):
    m = len(predict)
    error = 0.0
    for i in range(m):
        error -= np.log(sigmoid(predict[i]*classLabels[i]))
    return error

def getPrediction(dataMatrix, w0, w, v):
    m = np.shape(dataMatrix)[0]
    result = []
    for x in range(m):
        inter_1 = dataMatrix[x] * v
        inter_2 = np.multiply(dataMatrix[x], dataMatrix[x]) * np.multiply(v, v)
        interaction = np.sum(np.multiply(inter_1, inter_1) - inter_2) / 2
        p = w0 + dataMatrix[x] * w + interaction
        pre = sigmoid(p[0, 0])
        result.append(pre)
    return result

def stocGradAscent(dataMatrix, classLabels, k, max_iter, alpha):
    #initialize parameters
    m, n = np.shape(dataMatrix)
    w = np.zeros((n, 1))
    w0 = 0
    v = initialize(n, k)
    #training
    for it in range(max_iter):
        for x in range(m):
            inter_1 = dataMatrix[x] * v
            inter_2 = np.multiply(dataMatrix[x], dataMatrix[x])*np.multiply(v, v)
            interaction = np.sum(np.multiply(inter_1, inter_1) - inter_2)/2
            p = w0 + dataMatrix[x]*w + interaction
            loss = sigmoid(classLabels[x] * p[0, 0]) - 1
            w0 = w0 - alpha*loss*classLabels[x]
            for i in range(n):
                if dataMatrix[x, i] != 0:
                    w[i, 0] = w[i, 0] - alpha*loss*classLabels[x]*dataMatrix[x, i]
                for j in range(k):
                    v[i, j] = v[i, j] - alpha*loss*classLabels[x]*(dataMatrix[x, i]*inter_1[0, j]-v[i, j]*dataMatrix[x, i]*dataMatrix[x, i])
        if it % 1000 == 0:
            print('-----iter: ', it, ', cost: ', getCost(getPrediction(np.mat(dataMatrix), w0, w, v), classLabels))
            Accuracy(getPrediction(np.mat(dataMatrix), w0, w, v), classLabels)

if __name__ == '__main__':
    dataMatrix, target = loadData('../Data/testSetRBF2.txt')
    stocGradAscent(dataMatrix, target, 5, 5000, 0.01)
