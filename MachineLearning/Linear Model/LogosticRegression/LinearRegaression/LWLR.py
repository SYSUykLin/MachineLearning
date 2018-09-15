import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from DataProcesser import DataProcesser
def lwlr(feature, label, k):
    m = np.shape(feature)[0]
    predict = np.zeros(m)
    weights = np.mat(np.eye(m))
    for i in range(m):
        for j in range(m):
            different = feature[i, ] -feature[j, ]
            weights[j, j] = np.exp(different*different.T/(-2.0*k**2))
        xTx = feature.T * (weights * feature)
        ws = xTx.I * (feature.T * (weights * label.T))
        predict[i] = feature[i, ] * ws
    return predict

def show(features, target, predict):
    plt.title('Distribution of the data')
    plt.scatter(np.array(features[:, 1]), np.array(target[:, 0]), c = 'green')
    plt.xlabel('y')
    plt.ylabel('x')
    x = features[:, 1]
    y = predict
    dataFrame = pd.DataFrame(x)
    dataFrame.insert(1, 'y', predict)
    dataFrame.sort_values(by=0, ascending=True, inplace=True)
    x = dataFrame.iloc[:, 0]
    y = dataFrame.iloc[:, 1]
    plt.plot(np.array(x.tolist()), np.array(y.tolist()), c = 'red')
    plt.show()

if __name__ == '__main__':
    Processer = DataProcesser()
    features, target = Processer.get_dataset_from_file('data.txt')
    predict = lwlr(features, target, 0.002)
    show(features, target.T, predict)