import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import random
import KernelTransform
class optStruct:
    def __init__(self , dataMat , labels , C , toler):
        self.x = dataMat
        self.labels = labels
        self.C = C
        self.toler = toler
        self.m = np.shape(dataMat)[0]
        self.alphas = np.mat(np.zeros((self.m , 1)))
        self.b = 0
        self.eCache = np.mat(np.zeros((self.m , 2)))
        self.K = np.mat(np.zeros((self.m , self.m)))
        for i in range(self.m):
            self.K[: , i] = KernelTransform.kernelTrans(self.x , self.x[i , :] , kTup=('rbf' , 1.2))
        pass

if __name__ == '__main__':
    os = optStruct([1,2] , [3,4] , 1,1)
    a = os.alphas.tolist()[0][0] -  os.alphas.tolist()[1][0]
    print(max(1.0 , a))

