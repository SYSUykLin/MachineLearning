import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import Tool
def kernelTrans(X,A,kTup):
    m,n = np.shape(X)
    K = np.mat(np.zeros((m,1)))
    if kTup[0]=='lin':
        K = X*A.T
    elif kTup[0] =='rbf':
        for j in range(m):
            deltRow = X[j,:]-A
            K[j] = deltRow*deltRow.T
        K = np.exp(K/(-1*kTup[1]**2))
    return K

'''
update the innel function
'''
def innerL(i ,os):
    Ei = calculateEi(os , i)
    if ((os.labels[i]*Ei < -os.toler) and
        (os.alphas[i] < os.C)) or ((os.labels[i]*Ei > os.toler) and
                                   (os.alphas[i] > 0)):
        j , Ej = Tool.selectj(i , os , Ei)
        alphaIold = os.alphas[i].copy()
        alphaJold = os.alphas[j].copy()
        if (os.labels[i] != os.labels[j]):
            L = max(0 , os.alphas[j] - os.alphas[i])
            H = min(os.C , os.C + np.array(os.alphas)[j] - np.array(os.alphas)[i])
        else:
            L = max(0 , os.alphas[j] + os.alphas[i] - os.C)
            H = min(os.C , np.array(os.alphas)[j] + np.array(os.alphas)[i])
        if L == H:
            return 0
        eta = 2.0 * os.K[i, j] - os.K[i, i] - os.K[j, j]
        if eta >= 0:
            print('η> 0，the kernel matrix is not semi-positive definite')
            return 0
        os.alphas[j] -= os.labels[j]*(Ei - Ej)/eta
        os.alphas[j] = Tool.rangeSelectionForAlpha(os.alphas[j] , H , L)
        updateEk(os , j)

        if (abs(os.alphas[j] - alphaJold) < 0.00001):
            print("j not moving enough")
            return 0
        os.alphas[i] += os.labels[j] * os.labels[i] * (alphaJold - os.alphas[j])
        updateEk(os , i)
        b1 = os.b - Ei - os.labels[i] * (os.alphas[i] - alphaIold) * \
             os.K[i , i] - os.labels[j] * \
             (os.alphas[j] - alphaJold) *  os.K[i , j]
        b2 = os.b - Ej - os.labels[i] * (os.alphas[i] - alphaIold) * \
             os.K[i , j] - os.labels[j] * \
             (os.alphas[j] - alphaJold) * os.K[j , j]
        if (0 < os.alphas[i]) and (os.C > os.alphas[i]):
            os.b = b1
        elif (0 < os.alphas[j]) and (os.C > os.alphas[j]):
            os.b = b2
        else:
            os.b = (b1 + b2) / 2.0
        return 1
    else:
        return 0

'''
updata the Ei
'''
def calculateEi(os , k):
    fxk = float(np.multiply(os.alphas, os.labels).T * os.K[:, k] + os.b)
    Ek = fxk - float(os.labels[k])
    return Ek
def updateEk(os,k):
    Ek = calculateEi(os,k)
    os.eCache[k]=[1,Ek]