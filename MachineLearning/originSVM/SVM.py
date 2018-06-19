import numpy as np
import matplotlib.pyplot as plt
import random
import seaborn as sea
import pandas as pd
import dataSet
import Tools

'''
初始化参数
:param data_mat: 输入数据
:param class_label: 标签数据
:param C: 惩罚力度
:param tolar: 容错率
:param max_iter:最大迭代次数
:return:返回参数
'''
def SVM(data_mat , class_label , C , tolar , max_iter):

    data_mat = np.mat(data_mat)
    label_mat = np.mat(class_label)
    b = 0
    m , n = np.shape(data_mat)
    alphas = np.zeros((m , 1))
    iter = 0

    while iter < max_iter:
        #作为迭代变化量
        alpha_pairs_changed = 0
        #作为第一个a
        for i in range(m):
            WT_i = np.dot(np.multiply(alphas , label_mat).T , data_mat)
            f_xi = float(np.dot(WT_i , data_mat[i , :].T)) + b
            Ei = f_xi - float(label_mat[i])
            if ((label_mat[i]*Ei < -tolar) and (alphas[i] < C)) or ((label_mat[i]*Ei > tolar) and (alphas[i] > 0)):
                j = Tools.select_jrand(i , m)
                WT_j = np.dot(np.multiply(alphas , label_mat).T , data_mat)
                f_xj  = float(np.dot(WT_j , data_mat[j , :].T)) + b
                Ej = f_xj - float(label_mat[j])
                alpha_iold = alphas[i].copy()
                alpha_jold = alphas[j].copy()

                if (label_mat[i] != label_mat[j]):
                    L = max(0 , alphas[j] - alphas[i])
                    H = min(C , C + alphas[j] - alphas[i])
                else:
                    L = max(0 , alphas[j] + alphas[i] - C)
                    H = min(C , alphas[j] + alphas[i])
                if H == L:
                    continue

                eta = 2.0 * data_mat[i, :] * data_mat[j, :].T - data_mat[i, :] * data_mat[i, :].T - data_mat[j, :] * data_mat[j, :].T
                if eta >= 0: continue
                alphas[j] = (alphas[j] - label_mat[j]*(Ei - Ej))/eta
                alphas[j] = Tools.clip_alpha(alphas[j], H, L)
                if (abs(alphas[j] - alpha_jold) < 0.00001):
                    continue
                alphas[i] = alphas[i] + label_mat[j]*label_mat[i]*(alpha_jold - alphas[j])


                b1 = b - Ei + label_mat[i]*(alpha_iold - alphas[i])*np.dot(data_mat[i,:], data_mat[i,:].T) +\
                label_mat[j]*(alpha_jold - alphas[j])*np.dot(data_mat[i,:], data_mat[j,:].T)
                b2 = b - Ej + label_mat[i]*(alpha_iold - alphas[i])*np.dot(data_mat[i,:], data_mat[j,:].T) +\
                label_mat[j]*(alpha_jold - alphas[j])*np.dot(data_mat[j,:], data_mat[j,:].T)
                if (0 < alphas[i]) and (C > alphas[i]):
                    b = b1
                elif (0 < alphas[j]) and (C > alphas[j]):
                    b = b2
                else:
                    b = (b1 + b2)/2.0
                print(b)
                alpha_pairs_changed += 1
                pass
        if alpha_pairs_changed == 0:
            iter += 1
        else:
            iter = 0

    support_x = []
    support_y = []
    class1_x = []
    class1_y = []
    class01_x = []
    class01_y = []
    for i in range(m):
        if alphas[i] > 0.0:
            support_x.append(data_mat[i, 0])
            support_y.append(data_mat[i, 1])
    for i in range(m):
        if label_mat[i] == 1:
            class1_x.append(data_mat[i, 0])
            class1_y.append(data_mat[i, 1])
        else:
            class01_x.append(data_mat[i, 0])
            class01_y.append(data_mat[i, 1])
    w_best = np.dot(np.multiply(alphas, label_mat).T, data_mat)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.scatter(support_x, support_y, s=100, c="y", marker="v", label="support_v")
    ax.scatter(class1_x, class1_y, s=30, c="b", marker="o", label="class 1")
    ax.scatter(class01_x, class01_y, s=30, c="r", marker="x", label="class -1")
    lin_x = np.linspace(0, 100)
    lin_y = (-float(b) - w_best[0, 0] * lin_x) / w_best[0, 1]
    plt.plot(lin_x, lin_y, color="black")
    ax.legend()
    ax.set_xlabel("factor1")
    ax.set_ylabel("factor2")
    plt.show()
    return b , alphas
datamat , labelmat = dataSet.load_data_set()
b, alphas = SVM(datamat , labelmat , 0.6 , 0.001 , 10)
print(b , alphas)

