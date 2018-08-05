import math
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from MachineLearning.HMM.tool import Tool
def supervised(filename):
    '''
    The number of types of lastest variable is four,0B(begin)|1M(meddle)|2E(end)|3S(sigle)
    :param filename:learning fron this file
    :return: pi A B matrix
    '''
    pi = [0]*4
    a = [[0] * 4 for x in range(4)]
    b = [[0] * 65535 for x in range(4)]
    f = open(filename,encoding='UTF-8')
    data = f.read()[3:]
    f.close()
    tokens = data.split('  ')

    #start training
    last_q = 2
    old_process = 0
    allToken = len(tokens)
    print('schedule : ')
    for k, token in enumerate(tokens):
        process = float(k) /float(allToken)
        if process > old_process + 0.1:
            print('%.3f%%' % (process * 100))
            old_process = process
        token = token.strip()
        n = len(token)
        #empty we will choose another
        if n <= 0:
            continue
        #if just only one
        if n == 1:
            pi[3] += 1
            a[last_q][3] += 1
            b[3][ord(token[0])] += 1
            last_q = 3
            continue
        #if not
        pi[0] += 1
        pi[2] += 1
        pi[1] += (n-2)
        #transfer matrix
        a[last_q][0] += 1
        last_q = 2
        if n == 2:
            a[0][2] += 1
        else:
            a[0][1] += 1
            a[1][1] += (n-3)
            a[1][2] += 1
        #launch matrix
        b[0][ord(token[0])] += 1
        b[2][ord(token[n-1])] += 1
        for i in range(1, n-1):
            b[1][ord(token[i])] += 1
    pi = Tool.log_normalize(pi)
    for i in range(4):
        a[i] = Tool.log_normalize(a[i])
        b[i] = Tool.log_normalize(b[i])
    return pi, a, b

if __name__ == '__main__':
    filename = '../Data/pku_training.utf8'
    pi, A, B = supervised(filename)
    Tool.saveParameter(pi, A, B, 'supervisedParam')
    pass

