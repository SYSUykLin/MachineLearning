import numpy as np
import random
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
def get_Data(Number):
    x = np.zeros((2*Number , 2),dtype = np.float32)
    y = np.zeros(2*Number , dtype = np.float32)
    t = np.linspace(0 , np.pi * 2 , Number)
    for i in range(Number):
        x[i] = np.c_[t[i] , np.sin(t[i]) - np.random.random() * 3]
        y[i] = 1
        pass
    for i in range(Number):
        x[Number+i] = np.c_[t[i] , np.sin(t[i]) + np.random.random() * 3]
        y[Number+i] = -1
    return x , y
    pass

def draw(X, Y, Number , line):
    for i in range(2*Number):
        if Y[i] == 1:
            plt.scatter(X[i,0], X[i, 1], color='r', marker='x')
        else:
            plt.scatter(X[i, 0], X[i, 1], color='b', marker='<')

        pass

    for l in line:
        if l['dim'] == 0:
            plt.plot(8*[l['thresh']] , np.linspace(0, 8, 8) , color = 'gray' , alpha = 0.5)
        else:
            plt.plot(np.linspace(0, 6.5, 6), 6 * [l['thresh']], color = 'gray', alpha=0.5)
        pass
    plt.title('DataSet')
    plt.xlabel('First Dimension')
    plt.ylabel('Second Dimension')
    plt.show()
    pass

if __name__ == '__main__':
    x , y = get_Data(200)
    draw(x, y, 200 , [])