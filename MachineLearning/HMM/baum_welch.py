import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from MachineLearning.HMM.tool import Tool
import random
def cal_alpha(pi, A, B, o, alpha):
    print('start calculating alpha......')
    for i in range(4):
        alpha[0][i] = pi[i] + B[i][ord(o[0])]
    T = len(o)
    temp = [0 for i in range(4)]
    del i
    for t in range(1, T):
        for i in range(4):
            for j in range(4):
                temp[j] = (alpha[t-1][j] + A[j][i])
            alpha[t][i] = Tool.log_sum(temp)
            alpha[t][i] += B[i][ord(o[t])]
    print('The calculation of alpha have been finished......')

def cal_beta(pi, A, B, o, beta):
    print('start calculating beta......')
    T = len(o)
    for i in range(4):
        beta[T-1][i] = 1
    temp = [0 for i in range(4)]
    del i
    for t in range(T-2, -1, -1):
        for i in range(4):
            beta[t][i] = 0
            for j in range(4):
                temp[j] = A[i][j] + B[j][ord(o[t + 1])] + beta[t + 1][j]
            beta[t][i] += Tool.log_sum(temp)
    print('The calculation of beta have been finished......')

def cal_gamma(alpha, beta, gamma):
    print('start calculating gamma......')
    for t in range(len(alpha)):
        for i in range(4):
            gamma[t][i] = alpha[t][i] + beta[t][i]
        s = Tool.log_sum(gamma[t])
        for i in range(4):
            gamma[t][i] -= s
    print('The calculation of gamma have been finished......')

def cal_kesi(alpha, beta, A, B, o, ksi):
    print('start calculating ksi......')
    T = len(o)
    temp = [0 for i in range(16)]
    for t in range(T - 1):
        k = 0
        for i in range(4):
            for j in range(4):
                ksi[t][i][j] = alpha[t][i] + A[i][j] + B[j][ord(o[t+1])] + beta[t+1][j]
                temp[k] = ksi[t][i][j]
                k += 1
        s = Tool.log_sum(temp)
        for i in range(4):
            for j in range(4):
                ksi[t][i][j] -= s
    print('The calculation of kesi have been finished......')

def update(pi, A, B, alpha, beta, gamma, ksi, o):
    print('start updating......')
    T = len(o)
    for i in range(4):
        pi[i] = gamma[0][i]
    s1 = [0 for x in range(T-1)]
    s2 = [0 for x in range(T-1)]
    for i in range(4):
        for j in range(4):
            for t in range(T-1):
                s1[t] = ksi[t][i][j]
                s2[t] = gamma[t][i]
            A[i][j] = Tool.log_sum(s1) - Tool.log_sum(s2)
    s1 = [0 for x in range(T)]
    s2 = [0 for x in range(T)]
    for i in range(4):
        for k in range(65536):
            if k%5000 == 0:
                print(i, k)
            valid = 0
            for t in range(T):
                if ord(o[t]) == k:
                    s1[valid] = gamma[t][i]
                    valid += 1
                s2[t] = gamma[t][i]
            if valid == 0:
                B[i][k] = -Tool.log_sum(s2)
            else:
                B[i][k] = Tool.log_sum(s1[:valid]) - Tool.log_sum(s2)
    print('baum-welch algorithm have been finished......')


def baum_welch(pi, A, B, filename):
    f = open(filename , encoding='UTF-8')
    sentences = f.read()[3:]
    f.close()
    T = len(sentences)   # 观测序列
    alpha = [[0 for i in range(4)] for t in range(T)]
    beta = [[0 for i in range(4)] for t in range(T)]
    gamma = [[0 for i in range(4)] for t in range(T)]
    ksi = [[[0 for j in range(4)] for i in range(4)] for t in range(T-1)]
    for time in range(1000):
        print('Time : ', time)
        sentence = sentences
        cal_alpha(pi, A, B, sentence, alpha)
        cal_beta(pi, A, B, sentence, beta)
        cal_gamma(alpha, beta, gamma)
        cal_kesi(alpha, beta, A, B, sentence, ksi)
        update(pi, A, B, alpha, beta, gamma, ksi, sentence)
        Tool.saveParameter(pi, A, B, 'unsupervisedParam')
        print('Save matrix successfully!')

def inite():
    pi = [random.random() for x in range(4)]
    Tool.log_normalize(pi)
    A = [[random.random() for y in range(4)] for x in range(4)]
    A[0][0] = A[0][3] = A[1][0] = A[1][3] = A[2][1] = A[2][2] = A[3][1] = A[3][2] = 0
    B = [[random.random() for y in range(65536)] for x in range(4)]
    for i in range(4):
        A[i] = Tool.log_normalize(A[i])
        B[i] = Tool.log_normalize(B[i])
    return pi , A , B

if __name__ == '__main__':
    pi, A, B = inite()
    baum_welch(pi, A, B, '../Data/novel.txt')
    pass


