import numpy as np
import matplotlib.pyplot as plt
import random
import seaborn as sea
import pandas as pd
'''
生成一个0到m之间与j不同的整数
'''
def select_jrand(i , m):
    j = i
    while(j == i):
        j = int(random.uniform(0 , m))
    return j
    pass

'''
把ai限制在H到L里面
'''
def clip_alpha(aj , H , L):
    if aj > H:
        aj = H
    elif aj < L:
        aj = L
    return aj
    pass