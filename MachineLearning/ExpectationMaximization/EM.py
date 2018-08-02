#coding=UTF-8
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import copy
import seaborn as sns

def generate_data(sigma, N, mu1, mu2, mu3, mu4, alpha):
    global X
    X = np.zeros((N, 2))
    X = np.matrix(X)
    global mu
    mu = np.random.random((4, 2))
    mu = np.matrix(mu)
    global expect
    expect = np.zeros((N, 4))
    global alphas
    alphas = [0.25, 0.25, 0.25, 0.25]
    for i in range(N):
        if np.random.random(1) < 0.1:
            X[i, :] = np.random.multivariate_normal(mu1, sigma, 1)
        elif 0.1 <= np.random.random(1) < 0.3:
            X[i, :] = np.random.multivariate_normal(mu2, sigma, 1)
        elif 0.3 <= np.random.random(1) < 0.6:
            X[i, :] = np.random.multivariate_normal(mu3, sigma, 1)
        else:
            X[i, :] = np.random.multivariate_normal(mu4, sigma, 1)
    plt.title('Generator')
    plt.scatter(X[:, 0].tolist(), X[:, 1].tolist(), c = 'b')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.show()



def e_step(sigma, k, N):
    global X
    global mu
    global expect
    global alphas
    for i in range(N):
        W = 0
        for j in range(k):
            W += alphas[j] * math.exp(-(X[i, :] - mu[j, :]) * sigma.I * np.transpose(X[i, :] - mu[j, :])) / np.sqrt(np.linalg.det(sigma))
        for j in range(k):
            w = math.exp(-(X[i, :] - mu[j, :]) * sigma.I * np.transpose(X[i, :] - mu[j, :])) / np.sqrt(np.linalg.det(sigma))
            expect[i, j] = alphas[j]*w/W
            pass

def m_step(k, N):
    global expect
    global X
    global alphas
    for j in range(k):
        mathor = 0
        son = 0
        for i in range(N):
            son += expect[i, j]*X[i, :]
            mathor += expect[i, j]
        mu[j, :] = son / mathor
        alphas[j] = mathor / N

if __name__ == '__main__':
    iterNum = 1000
    N = 500
    k = 4
    probility = np.zeros(N)
    mu1 = [5, 35]
    mu2 = [30, 40]
    mu3 = [20, 20]
    mu4 = [45, 15]
    sigma = np.matrix([[30, 0], [0, 30]])
    alpha = [0.1, 0.2, 0.3, 0.4]
    generate_data(sigma, N, mu1, mu2, mu3, mu4, alpha)
    for i in range(iterNum):
        print('iterNum : ', i)
        err = 0
        err_alpha = 0
        Old_mu = copy.deepcopy(mu)
        Old_alpha = copy.deepcopy(alphas)
        e_step(sigma, k, N)
        m_step(k, N)
        for z in range(k):
            err += (abs(Old_mu[z, 0] - mu[z, 0]) + abs(Old_mu[z, 1] - mu[z, 1]))
            err_alpha += abs(Old_alpha[z] - alphas[z])
        if (err <= 0.001) and (err_alpha < 0.001):
            print(err, err_alpha)
            break
    color = ['blue', 'red', 'yellow', 'green']
    markers  = ['<', 'x', 'o', '>']
    order = np.zeros(N)
    for i in range(N):
        for j in range(k):
            if expect[i, j] == max(expect[i, ]):
                order[i] = j
        plt.scatter(X[i, 0], X[i, 1], c = color[int(order[i])], alpha=0.5, marker=markers[int(order[i])])
    plt.show()
    print('standedμ:',mu4, mu3, mu2, mu1)
    print('standedα:',alpha)
    print('new μ:', mu)
    print('new α:',alphas)

