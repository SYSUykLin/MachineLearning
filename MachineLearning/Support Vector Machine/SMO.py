import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import random
import Tool
import smo_class
import KernelTransform
def innerL(i ,os):
    Ei = Tool.calculateEi(os , i)
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
        eta = 2.0*os.x[i,:]*os.x[j,:].T - os.x[i,:]*os.x[i,:].T - os.x[j,:]*os.x[j,:].T
        if eta >= 0:
            print('η> 0，the kernel matrix is not semi-positive definite')
            return 0
        os.alphas[j] -= os.labels[j]*(Ei - Ej)/eta
        os.alphas[j] = Tool.rangeSelectionForAlpha(os.alphas[j] , H , L)
        Tool.updateEk(os , j)

        if (abs(os.alphas[j] - alphaJold) < 0.00001):
            print("j not moving enough")
            return 0
        os.alphas[i] += os.labels[j] * os.labels[i] * (alphaJold - os.alphas[j])
        Tool.updateEk(os , i)
        b1 = os.b - Ei - os.labels[i] * (os.alphas[i] - alphaIold) * \
             os.x[i, :] * os.x[i, :].T - os.labels[j] * \
             (os.alphas[j] - alphaJold) * os.x[i, :] * os.x[j, :].T
        b2 = os.b - Ej - os.labels[i] * (os.alphas[i] - alphaIold) * \
             os.x[i, :] * os.x[j, :].T - os.labels[j] * \
             (os.alphas[j] - alphaJold) * os.x[j, :] * os.x[j, :].T
        if (0 < os.alphas[i]) and (os.C > os.alphas[i]):
            os.b = b1
        elif (0 < os.alphas[j]) and (os.C > os.alphas[j]):
            os.b = b2
        else:
            os.b = (b1 + b2) / 2.0
        return 1
    else:
        return 0

def smo(data,labels,C = 0.6,toler = 0.001,maxIter = 40 , kernel = True):
    oS = smo_class.optStruct(np.mat(data),np.mat(labels).transpose(),C,toler)
    iter =0
    entireSet  = True
    alphaPairsChanged = 0
    while(iter < maxIter) and ((alphaPairsChanged >0) or (entireSet)):
        alphaPairsChanged = 0
        if entireSet:
            for i in range(oS.m):
                if kernel == True:
                    alphaPairsChanged += KernelTransform.innerL(i,oS)
                else:
                    alphaPairsChanged += innerL(i, oS)
            print("fullSet,iter: %d i: %d,pairs changed %d" %\
                (iter,i,alphaPairsChanged))
            iter +=1
        else:
            # 两个元素乘积非零，每两个元素做乘法[0,1,1,0,0]*[1,1,0,1,0]=[0,1,0,0,0]
            nonBoundIs = np.nonzero((oS.alphas.A > 0)*(oS.alphas.A < C))[0]
            for i in nonBoundIs:
                alphaPairsChanged += innerL(i,oS)
                print("nou-bound,iter: %d i:%d,pairs changed %d" % (iter,i,alphaPairsChanged))
            iter +=1
        # entireSet 控制交替的策略选择
        if entireSet:
            entireSet = False
        # 必须有alpha对进行更新
        elif(alphaPairsChanged == 0):
            entireSet = True
        print("iteration number：%d" % iter)
    return oS.b,oS.alphas
