import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

'''余弦相似度'''
def cos_sim(x, y):
    numerator = x * y.T
    denominator = np.sqrt(x * x.T)
    return (numerator / denominator)[0, 0]

def similarity(data):
    m = np.shape(data)[0]
    w = np.mat(np.zeros((m, m)))
    for i in range(m):
        for j in range(i, m):
            if j != i:
                w[i, j] = cos_sim(data[i, ], data[j, ])
                w[j, i] = w[i, j]
            else:
                w[i, j] = 0
    return w

def load_data(file_path):
    f = open(file_path)
    data = []
    for line in f.readlines():
        lines = line.strip().split("\t")
        tmp = []
        for x in lines:
            if x != "-":
                tmp.append(float(x))
            else:
                tmp.append(0)
        data.append(tmp)
    f.close()
    return np.mat(data)

def top_k(predict, k):
    top_recom = []
    len_result = len(predict)
    if k >= len_result:
        top_recom = predict
    else:
        for i in range(k):
            top_recom.append(predict[i])
    return top_recom

if __name__ == '__main__':
    print('Tool')