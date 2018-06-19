import numpy as np
import matplotlib.pyplot as plt
import random
import seaborn as sea
import pandas as pd


def get_positive_and_negative():
    dataSet = pd.read_csv('Datas/LogiReg_data.txt', names=['V1', 'V2', 'Class'])
    dataSet.Class[dataSet.Class == 0] = -1
    dataSet = dataSet[60 : 80]
    positive = dataSet[dataSet['Class'] == 1]
    negative = dataSet[dataSet['Class'] == -1]
    return positive , negative , dataSet


def show_picture(positive , negative):
    columns = ['V1', 'V2']
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.scatter(positive[columns[0]], positive[columns[1]], s=30, c="b", marker="o", label="class 1")
    ax.scatter(negative[columns[0]], negative[columns[1]], s=30, c="r", marker="x", label="class -1")
    ax.legend()
    ax.set_xlabel('V1')
    ax.set_ylabel('V3')
    plt.show()

def load_data_set():
    _ , _ , file = get_positive_and_negative()
    orig_data = file.as_matrix()
    cols = orig_data.shape[1]
    data_mat = orig_data[ : , 0 : cols-1]
    label_mat = orig_data[ : , cols-1 : cols]
    return  data_mat , label_mat

positive , negative , data = get_positive_and_negative()
show_picture(positive , negative)
print(data)



