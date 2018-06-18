'''
load data and define some tool function
'''
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import random

def loadDataSet(filename):
    '''
    :param filename:
    :return dataset and label:
    '''

    dataset = []
    label = []
    fr = open(filename)
    for line in fr.readlines():
        lineArr = line.strip().split('\t')
        dataset.append( [np.float32(lineArr[0]) , np.float32(lineArr[1])] )
        label.append(np.float32(lineArr[2]))
    return dataset , label
    pass

'''
select alpha2 randomly
'''
def selectAlphaTwo(i , m):
    '''
    :param i:
    :param m:
    :return:
    '''
    j = i
    while(j == i):
        j = int(random.uniform(0 , m))
    return j

def rangeSelectionForAlpha(aj , H , L):
    if aj > H:
        aj = H
    if L > aj:
        aj = L
    return aj
    pass

'''
calculate Ei
'''
def calculateEi(os , k):
    fxk = float(np.multiply(os.alphas, os.labels).T * (os.x * os.x[k, :].T)) + os.b
    Ek = fxk - float(os.labels[k])
    return Ek

'''
put the Ei into the cache when calculate Ei 
'''
def selectj(i , os , Ei):
    maxk = -1
    maxDeltaE = 0
    Ej = 0
    os.eCache[i] = [1 , Ei]
    validEachlist = np.nonzero(os.eCache[: , 0].A)[0]
    if (len(validEachlist) > 1):
        for k in validEachlist:
            if k == i:
                continue
            Ek = calculateEi(os , k)
            deltaE = np.abs(Ei - Ek)
            if deltaE > maxDeltaE:
                maxk = k
                maxDeltaE = deltaE
                Ej = Ek
        return maxk , Ej
        pass
    else:
        j = selectAlphaTwo(i , os.m)
        Ej = calculateEi(os , j)
    return j , Ej
    pass

'''
draw picture
'''
def drawDataset(data , label , x = None , y = None , line = True , alphas = None , kernel = True):
    index_one = []
    index_negative_one = []
    for i in range(100):
        if label[i] == 1:
            index_one.append(data[i])
        else:
            index_negative_one.append(data[i])
    index_one = np.matrix(index_one)
    index_negative_one = np.matrix(index_negative_one)
    plt.scatter(index_one[ : , 0].tolist() , index_one[: , 1].tolist() , c = 'r' , marker='<' , label = 'class equal one')
    plt.scatter(index_negative_one[: , 0].tolist() , index_negative_one[: , 1].tolist() , c = 'b' , marker='x' , label = 'class equal negative one')
    if line == True:
        plt.plot(x , y)
        pass

    '''
    draw the support vector,the point which the α not equal zero
    '''
    if line == True or kernel == True:
        a1 = []
        for i in range(len(alphas)):
            a = alphas[i]
            if a != 0:
               a1.append(data[i])
        a1 =  np.matrix(a1)
        print('The number of the support vector : ' , len(a1))
        plt.scatter(a1[: , 0].tolist(),a1[: , 1].tolist(), s=100, c="y", marker="v", label="support_v")

    plt.legend()
    plt.xlabel('X axis')
    plt.ylabel('Y axis')
    plt.show()

def updateEk(os,k):
    Ek = calculateEi(os,k)
    os.eCache[k]=[1,Ek]

if __name__ == '__main__':
    data , label = loadDataSet('../Data/testSetRBF.txt')
    drawDataset(data , label , line=False ,kernel=False)

