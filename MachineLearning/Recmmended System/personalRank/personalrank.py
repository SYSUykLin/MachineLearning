import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from Tool_recommender import load_data

def PersonalRank(data_dict, alpha, user, maxCycles):
    rank = {}
    for x in data_dict.keys():
        rank[x] = 0
    rank[user] = 1
    step = 0
    while step < maxCycles:
        tmp = {}
        for x in data_dict.keys():
            tmp[x] = 0
        for i, ri in data_dict.items():
            for j in ri.keys():
                if j not in tmp:
                    tmp[j] = 0
                tmp[j] += alpha * rank[i] / (1.0 * len(ri))
                if j == user:
                    tmp[j] += (1-alpha)
        check = []
        for k in tmp.keys():
            check.append(tmp[k] - rank[k])
        if sum(check) <= 0.0001:
            break
        rank = tmp
        if step % 20 == 0:
            print('NO: ', step)
        step += 1
    return rank

def generate_dict(dataTmp):
    m, n = np.shape(dataTmp)
    data_dict = {}
    for i in range(m):
        tmp_dict = {}
        for j in range(n):
            if dataTmp[i, j] != 0:
                tmp_dict["D_" + str(j)] = dataTmp[i, j]
        data_dict["U_" + str(i)] = tmp_dict
    for j in range(n):
        tmp_dict = {}
        for i in range(m):
            if dataTmp[i, j] != 0:
                tmp_dict["U_" + str(i)] = dataTmp[i, j]
        data_dict["D_" + str(j)] = tmp_dict
    return data_dict

if __name__ == '__main__':
    data = load_data('../../Data/recommenderData.txt')
    data_dict = generate_dict(data)
    print(PersonalRank(data_dict, 0.85, "U_0", 500))
