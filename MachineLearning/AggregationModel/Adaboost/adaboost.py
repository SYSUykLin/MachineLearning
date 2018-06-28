import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
import MachineLearning.AggregationModel.Adaboost.getData as getData

def stumpClassify(dataMatrix, dimen, threshVal, threshIneq):  # just classify the data
    retArray = np.ones((np.shape(dataMatrix)[0], 1))
    if threshIneq == 'lt':
        retArray[dataMatrix[:, dimen] <= threshVal] = -1.0
    else:
        retArray[dataMatrix[:, dimen] > threshVal] = -1.0
    return retArray

def decision_stump(dataMat , Labels , u):
    dataMat = np.mat(dataMat)
    Labels = np.mat(Labels).T
    n , d = dataMat.shape
    numSteps = 60.0
    bestSump = {}
    bestClasEst = np.mat(np.zeros((n , 1)))
    minError = np.inf
    for i in range(d):
        rangeMin = dataMat[: , i].min()
        rangeMax = dataMat[: , i].max()
        stepSize = (rangeMax - rangeMin) / numSteps
        for j in range(-1 , int(numSteps) + 1):
            for inequal in ['lt' , 'gt']:
                threshVal = (rangeMin + np.float32(j) * stepSize)
                predictedVals = stumpClassify(dataMat , i , threshVal , inequal)
                errArr = np.mat(np.ones((n , 1)))
                errArr[predictedVals == Labels] = 0
                weightedError = u.T * errArr

                if weightedError < minError:
                    minError = weightedError
                    bestClasEst = predictedVals.copy()
                    bestSump['dim'] = i
                    bestSump['thresh'] = threshVal
                    bestSump['inseq'] = inequal
    return bestSump, minError, bestClasEst

def Running(Number , numIt):
    dataX , dataY = getData.get_Data(Number)
    weakClassArr = []
    n = dataX.shape[0]
    u = np.mat(np.ones((n , 1)) / n)
    aggClassEst = np.mat(np.zeros((n , 1)))
    getData.draw(dataX , dataY , Number , [])
    for i in range(numIt):
        bestSump, error, classEst = decision_stump(dataX, dataY, u)
        alpha = float(0.5 * np.log((1.0 - error) / max(error, 1e-16)))
        bestSump['alpha'] = alpha
        weakClassArr.append(bestSump)
        expon = np.multiply(-1 * alpha * np.mat(dataY).T, classEst)
        u = np.multiply(u , np.exp(expon)) / np.sum(u) #if miss the normalization,the u will be bigger than bigger , and the thresh is unusable.
        aggClassEst += alpha * classEst
        aggErrors = np.multiply(np.sign(aggClassEst) != np.mat(dataY).T, np.ones((n, 1)))
        errorRate = aggErrors.sum() / n
        precision = np.multiply(np.sign(aggClassEst) == np.mat(dataY).T, np.ones((n, 1)))
        precision = (sum(precision) / n) * 100

        if i % 10 == 0:
            if precision == 100.0:
                break
            print('precision : ',precision)
            getData.draw(dataX , np.sign(aggClassEst) ,200, weakClassArr)
    return weakClassArr, aggClassEst

if __name__ == '__main__':
    Running(200 , 100000)