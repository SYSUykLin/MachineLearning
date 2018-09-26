import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from Tool_recommender import similarity,load_data,top_k

def user_based_recommend(data, w, user):
    m, n = np.shape(data)
    interaction = data[user, ]
    not_inter = []
    for i in range(n):
        if interaction[0, i] == 0:
            not_inter.append(i)
    predict = {}
    for x in not_inter:
        item = np.copy(data[:, x])
        for i in range(m):
            if item[i, 0] != 0:
                if x not in predict:
                    predict[x] = w[user, i] * item[i, 0]
                else:
                    predict[x] = predict[x] + w[user, i] * item[i, 0]
    return sorted(predict.items(), key=lambda  d:d[1], reverse=True)

if __name__ == '__main__':
    data = load_data('../../Data/recommenderData.txt')
    w = similarity(data)
    print('similarity matrix: \n', w)
    for i in range(np.shape(data)[0]):
        predict = user_based_recommend(data, w, i)
        top =top_k(predict, 2)
        print(top)