import matplotlib.pyplot as plt
import pandas as pd
import math
import numpy as np

class Tool(object):
    infinite = float(-2**31)
    @staticmethod
    def log_normalize(a):
        s = 0
        for x in a:
            s += x
        if s == 0:
            print('Normalize error,value equal zero')
            return
        s = np.log(s)
        for i in range(len(a)):
            if a[i] == 0:
                a[i] = Tool.infinite
            else:
                a[i] = np.log(a[i]) - s
        return a

    @staticmethod
    def log_sum(a):
        if not a:
            return Tool.infinite
        m = max(a)
        s = 0
        for t in a:
            s += np.exp(t - m)
        return m + np.log(s)


    @staticmethod
    def saveParameter(pi,A, B, catalog):
        np.savetxt(catalog + '/' + 'pi.txt', pi)
        np.savetxt(catalog + '/' + 'A.txt', A)
        np.savetxt(catalog + '/' + 'B.txt', B)

