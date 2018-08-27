import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from LoadData import DataPrecessing

def gradientAscent(feature_data, label_data, k, maxCycle, alpha):
    '''train softmax model by gradientAscent
    input:feature_data(mat) feature
    label_data(mat) target
    k(int) number of classes
    maxCycle(int) max iterator
    alpha(float) learning rate
    '''
    Dataprecessing = DataPrecessing()
    x_train, x_target_tarin, x_test, x_target_test = Dataprecessing.loadFile()
    x_target_tarin = x_target_tarin.tolist()[0]
    x_target_test = x_target_test.tolist()[0]
    m, n = np.shape(feature_data)
    weights = np.mat(np.ones((n, k)))
    i = 0
    while i <= maxCycle:
        err = np.exp(feature_data*weights)
        if i % 100 == 0:
            print('cost score : ', cost(err, label_data))
            train_predict = Dataprecessing.predict(x_train, weights)
            test_predict = Dataprecessing.predict(x_test, weights)
            print('Train_accuracy : ', Dataprecessing.Calculate_accuracy(x_target_tarin, train_predict))
            print('Test_accuracy : ', Dataprecessing.Calculate_accuracy(x_target_test, test_predict))
        rowsum = -err.sum(axis = 1)
        rowsum = rowsum.repeat(k, axis = 1)
        err = err / rowsum
        for x in range(m):
            err[x, label_data[x]] += 1
        weights = weights + (alpha/m) * feature_data.T * err
        i += 1
    return weights

def cost(err, label_data):
    m = np.shape(err)[0]
    sum_cost = 0.0
    for i in range(m):
        if err[i, label_data[i]] / np.sum(err[i, :]) > 0:
            sum_cost -= np.log(err[i, label_data[i]] / np.sum(err[i, :]))
        else:
            sum_cost -= 0
    return sum_cost/m

if __name__ == '__main__':
    Dataprecessing = DataPrecessing()
    x_train, x_target_tarin, x_test, x_target_test = Dataprecessing.loadFile()
    x_target_tarin = x_target_tarin.tolist()[0]
    gradientAscent(x_train, x_target_tarin, 10, 100000, 0.001)
