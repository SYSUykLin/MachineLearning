import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

def viterbi(pi, A, B, o):
    '''
    viterbi algorithm
    :param pi:initial matrix
    :param A:transfer matrox
    :param B:launch matrix
    :param o:observation sequence
    :return:I
    '''
    T = len(o)
    delta = [[0 for i in range(4)] for t in range(T)]
    pre = [[0 for i in range(4)] for t in range(T)]
    for i in range(4):
        #first iteration
        delta[0][i] = pi[i] + B[i][ord(o[0])]
    for t in range(1, T):
        for i in range(4):
            delta[t][i] = delta[t-1][0] + A[0][i]
            for j in range(1, 4):
                vj = delta[t-1][j] + A[0][j]
                if delta[t][i] < vj:
                    delta[t][i] = vj
                    pre[t][i] = j
            delta[t][i] += B[i][ord(o[t])]
    decode = [-1 for t in range(T)]
    q = 0
    for i in range(1, 4):
        if delta[T-1][i] > delta[T-1][q]:
            q = i
    decode[T-1] = q
    for t in range(T-2, -1, -1):
        q = pre[t+1][q]
        decode[t] = q
    return decode

def segment(sentence, decode):
    N = len(sentence)
    i = 0
    while i < N:
        if decode[i] == 0 or decode[i] == 1:
            j = i+1
            while j < N:
                if decode[j] == 2:
                    break
                j += 1
            print(sentence[i:j+1],"|",end=' ')
            i = j+1
        elif decode[i] == 3 or decode[i] == 2:  # single
            print (sentence[i:i + 1], "|", end=' ')
            i += 1
        else:
            print ('Error:', i, decode[i] , end=' ')
            i += 1

if __name__ == '__main__':
    pi = np.loadtxt('unsupervisedParam/pi.txt')
    A = np.loadtxt('unsupervisedParam/A.txt')
    B = np.loadtxt('unsupervisedParam/B.txt')
    f = open("../Data/novel.txt" , encoding='UTF-8')
    data = f.read()[3:]
    f.close()
    decode = viterbi(pi, A, B, data)
    segment(data, decode)

