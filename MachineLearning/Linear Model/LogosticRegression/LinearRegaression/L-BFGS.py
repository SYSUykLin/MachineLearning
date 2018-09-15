import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from DataProcesser import DataProcesser

def lbfgs(feature, label, lam, maxCycle, m = 10):
    n = np.shape(feature)[1]
    w0 = np.mat(np.zeros((n, 1)))
    rho = 0.55
    sigma = 0.4

    H0 = np.eye(n)
    s = []
    y = []

    k = 1
    gk = get_gradient(feature, label, w0, lam)
    dk = -H0 * gk

    while (k < maxCycle):
        print('iter: ', k, ' error:', get_error(feature, label, w0))
        m1 = 0
        mk = 0
        gk = get_gradient(feature, label, w0, lam)
        while (m1 < 20):
            newf = get_result(feature, label, (w0 + rho ** m1 * dk), lam)
            oldf = get_result(feature, label, w0, lam)
            if newf < oldf + sigma * (rho ** m1) * (gk.T * dk)[0, 0]:
                mk = m1
                break
            m1 = m1 + 1
        w = w0 + rho ** mk * dk
        if k > m:
            s.pop(0)
            y.pop(0)
        sk = w - w0
        qk = get_gradient(feature, label, w, lam)
        yk = qk - gk
        s.append(sk)
        y.append(yk)

        t = len(s)
        a = []

        for i in range(t):
            alpha = (s[t - i -1].T * qk) / (y[t - i - 1].T * s[t - i - 1])
            qk = qk - alpha[0, 0] * y[t - i -1]
            a.append(alpha[0, 0])
        r = H0 * qk

        for i in range(t):
            beta = (y[i].T * r) / (y[i].T * s[i])
            r = r + s[i] * (a[t - i - 1] - beta[0, 0])
        if yk.T * sk > 0:
            dk = -r
        k = k + 1
        w0 = w
    return w0


def get_gradient(feature, label, w, lam):
    err = (label - feature * w).T
    left = err * (-1) * feature
    return left.T + lam * w

def get_error(feature, label, w):
    return (label - feature * w).T*(label - feature * w)/2

def get_result(feature, label, w, lam):
    left = (label - feature * w).T * (label - feature * w)
    right = lam * w.T * w
    return (left + right)/2

if __name__ == '__main__':
    Processer = DataProcesser()
    features, target = Processer.get_dataset_from_file('data.txt')
    w = lbfgs(features, target.T, 0.3, 1000)
    Processer.showDatasetDistribution(features, target.T, w, True)