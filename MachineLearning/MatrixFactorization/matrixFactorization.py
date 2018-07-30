import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from MachineLearning.MatrixFactorization.loadData import load_Data

def gradDscent(dataMat, k, alpha, beta, maxIter):
    '''

    :param dataMat:dataSet
    :param k: params of the matrix fatorization
    :param alphs: learning rate
    :param beta: regularization params
    :param maxIter: maxiter
    :return:
    '''
    print('start training......')
    m, n = np.shape(dataMat)
    p = np.mat(np.random.random((m, k)))
    q = np.mat(np.random.random((k, n)))

    for step in range(maxIter):
        for i in range(m):
            for j in range(n):
                if dataMat[i, j] > 0:
                    error = dataMat[i, j]
                    for r in range(k):
                        error = error - p[i, r]*q[r, j]
                    for r in range(k):
                        p[i, r] = p[i, r] + alpha * (2 * error * q[r, j] - beta * p[i, r])
                        q[r, j] = q[r, j] + alpha * (2 * error * p[i, r] - beta * q[r, j])
        loss = 0.0
        for i in range(m):
            for j in range(n):
                if dataMat[i, j] > 0:
                    error = 0.0
                    for r in range(k):
                        error = error + p[i, r] * q[r, j]
                    loss = np.power((dataMat[i, j] - error), 2)
                    for r in range(k):
                        loss = loss + beta * (p[i, r]*p[i, r] + q[r, j]*q[r, j])/2
        if loss < 0.001:
            break
        print('step : ', step, ' loss : ', loss)
    return p, q

if __name__ == '__main__':
    dataMat = load_Data('movies.csv', 'ratings.csv')
    p, q = gradDscent(dataMat, 10, 0.0002, 0.02, 100000)