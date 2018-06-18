import Tool
import SMO
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import KernelTransform
'''
calculate w and draw the picture,
the variable which the α not equal zero , 
we call support vector
'''
def calculateW(alphas , data , labels):
    x = np.mat(data)
    label = np.mat(labels).transpose()
    m , n = np.shape(x)
    w = np.zeros((n , 1))
    for i in range(m):
        w += np.multiply(alphas[i] * label[i] , x[i , :].T)
    return w
    pass

if __name__ == '__main__':
    data, label = Tool.loadDataSet('../Data/testSet.txt')
    b,alphas = SMO.smo(data , label , kernel=False)
    w = calculateW(alphas , data , label)
    x = np.arange(0 , 11)
    print(w)
    y = (-b - w[0]*x)/w[1]
    Tool.drawDataset(data , label , x , y.tolist()[0] , line=True , alphas=alphas)

    data, label = Tool.loadDataSet('../Data/testSetRBF.txt')
    b, alphas = SMO.smo(data, label,kernel=True ,maxIter=100)
    svInd = np.nonzero(alphas.A > 0)[0]
    Tool.drawDataset(data, label,  line=False, alphas=alphas)






