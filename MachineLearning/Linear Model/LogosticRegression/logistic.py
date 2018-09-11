import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

def predict(w, x):
    h = x * w
    h = sigmoid(h)
    if h > 0.5:
        return int(1)
    else:
        return int(0)

def sigmoid(x):
    return np.longfloat(1.0/(1+np.exp(-x)))

def error_rate(h, label):
    m = np.shape(h)[0]
    sum_err = 0.0
    for i in range(m):
        if h[i, 0] > 0 and (1-h[i, 0]) > 0:
            sum_err += (label[i, 0] * np.log(h[i, 0]) + (1-label[i, 0])*np.log(1-h[i, 0]))
        else:
            sum_err += 0
    return sum_err/m

def lr_train_bgd(feature, label, maxCycle, alpha, df):
    n = np.shape(feature)[1]
    m = np.shape(feature)[0]
    w = np.mat(np.random.rand(n,1))
    i = 0
    while True:
        i += 1
        h = sigmoid(feature * w)
        err = label - h
        if i % 100== 0:
            print('error : ', error_rate(h, label))
            d = 0
            scores = []
            for i in range(m):
                score = predict(w, feature[i])
                scores.append(score)
                if score == label[i]:
                    d += 1
            print('train accuracy : ', (d/m)*100, '%')
            if (d/m)*100 >= 90:
                for i in range(m):
                    if df.iloc[i, 2] == 1:
                        c = 'red'
                    else:
                        c = 'blue'
                    plt.scatter(df.iloc[i, 0], df.iloc[i, 1], c=c)
                x = [i for i in range(0, 10)]
                plt.plot(np.mat(x).T, np.mat((-w[0]-w[1]*x)/w[2]).T, c = 'blue')
                plt.show()
                return

        w += alpha * feature.T * err

def loadData(filename):
    df = pd.read_csv(filename, sep='	', names=['a', 'b', 'label'])
    n, m = df.shape
    features = []
    labels = []
    for i in range(n):
        feature = []
        feature.append(int(1))
        for j in range(0,m-1):
            feature.append(df.iloc[i, j])
        if df.iloc[i, m-1] == -1:
            labels.append(0)
        else:
            labels.append(1)
        features.append(feature)

    for i in range(n):
        if df.iloc[i, 2] == 1:
            c = 'red'
        else:
            c = 'blue'
        plt.scatter(df.iloc[i, 0], df.iloc[i, 1], c = c)
    plt.show()
    print(features)
    return np.mat(features), np.mat(labels).T, df

if __name__ == '__main__':
    f, t, df = loadData('../../Data/testSet.txt')
    lr_train_bgd(f, t, 10000, 0.001, df)




