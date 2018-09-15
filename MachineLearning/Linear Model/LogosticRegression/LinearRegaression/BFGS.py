import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from DataProcesser import DataProcesser

def bfgs(feature, label, lam, maxCycle):
    n = np.shape(feature)[1]
    w0 = np.mat(np.zeros((n, 1)))
    rho = 0.55
    sigma = 0.4
    Bk = np.eye(n)
    k = 1
    while (k < maxCycle):
        print('Iterator: ', k, ' error: ', get_error(feature, label, w0))
        gk = get_gradient(feature, label, w0, lam)
        dk = np.mat(-np.linalg.solve(Bk, gk))
        m = 0
        mk = 0
        while (m < 20):
            newf = get_result(feature, label, (w0 + rho**m*dk), lam)
            oldf = get_result(feature, label, w0, lam)
            if (newf < oldf + sigma * (rho ** m)*(gk.T*dk)[0, 0]):
                mk = m
                break
            m += 1
        #BFGS
        w = w0 + rho**mk * dk
        sk = w-w0
        yk = get_gradient(feature, label, w, lam) - gk
        if (yk.T * sk > 0):
            Bk = Bk - (Bk * sk * sk.T * Bk) / (sk.T * Bk * sk) + (yk * yk.T) / (yk.T * sk)
        k = k + 1
        w0 = w
    return w0

def get_error(feature, label, w):
    return (label - feature * w).T*(label - feature * w)/2

def get_gradient(feature, label, w, lam):
    err = (label - feature * w).T
    left = err * (-1) * feature
    return left.T + lam * w

def get_result(feature, label, w, lam):
    left = (label - feature * w).T * (label - feature * w)
    right = lam * w.T * w
    return (left + right)/2

if __name__ == '__main__':
    Processer = DataProcesser()
    features, target = Processer.get_dataset_from_file('data.txt')
    w = bfgs(features, target.T, 0.3, 1000)
    Processer.showDatasetDistribution(features, target.T, w, True)