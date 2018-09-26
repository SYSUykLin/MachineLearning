import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from Tool_recommender import similarity,load_data,top_k

def item_based_recommend(data, w, user):
    m, n = np.shape(data)
    interation = data[:, user].T
    not_inter = []
    for i in range(n):
        if interation[0, i] == 0:
            not_inter.append(i)

    predict = {}
    for x in not_inter:
        item = np.copy(interation)
        for j in range(m):
            if item[0, j] != 0:
                if x not in predict:
                    predict[x] = w[x, j] * item[0, j]
                else:
                    predict[x] = predict[x] + w[x, j] * item[0, j]
    return sorted(predict.items(), key=lambda d:d[1], reverse=True)

if __name__ == '__main__':
    data = load_data('../../Data/recommenderData.txt')
    data = data.T
    w = similarity(data)
    print('similarity matrix: \n', w)
    for i in range(np.shape(data)[0]):
        predict = item_based_recommend(data, w, i)
        top =top_k(predict, 2)
        print(top)