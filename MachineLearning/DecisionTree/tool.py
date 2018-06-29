import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import math
from sklearn.decomposition import PCA

def label_uniqueness(data):
    '''Counting the number of defferent labels in the dataset
    input:dataset
    output:Number of labels
    '''
    label_uniq = {}
    for x in data:
        label = x[len(x) - 1]
        if label not in label_uniq:
            label_uniq[label] = 0
        label_uniq[label] += 1
    return label_uniq
    pass

def cal_gini(data):
    '''calculate the gini index
    input:data(list)
    output:gini(float)
    '''
    total_sample = len(data)
    if total_sample == 0:
        return 0
    label_count = label_uniqueness(data)
    gini = 0
    for label in label_count:
        gini = gini + pow(label_count[label] , 2)
    gini = 1 - float(gini) / pow(total_sample , 2)
    return gini
    pass



