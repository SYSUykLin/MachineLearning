import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from Tool_recommender import load_data, top_k
def train(V, r, maxCycles, e):
    m, n = np.shape(V)
    W = np.mat(np.random.random((m, r)))
    H = np.mat(np.mat(np.random.random((r, n))))

    for step in range(maxCycles):
        V_pre = W * H
        E = V - V_pre
        err = 0.0
        for i in range(m):
            for j in range(n):
                err += E[i, j] * E[i, j]
        if err < e:
            break
        if step % 1000 == 0:
            print("\Interator: ", step, " Loss: ", err)
        a = W.T * V
        b = W.T * W * H
        for i in range(r):
            for j in range(n):
                if b[i, j] != 0:
                    H[i, j] = H[i, j] * a[i, j] / b[i, j]
        c = V * H.T
        d = W * H * H.T
        for i in range(m):
            for j in range(r):
                if d[i, j] != 0:
                    W[i, j] = W[i, j] * c[i, j] / d[i, j]
    return W, H

if __name__ == '__main__':
    V = load_data('../../Data/recommenderData.txt')
    W, H = train(V, 5, 20000, 1e-5)
    print(W)
    print(H)
