import numpy as np
import pandas as pd

def read_file(filename):
    '''
    use to read file which you open
    :param filename:
    :return:dataset
    '''
    data = pd.read_csv(filename)
    return data
    pass

def load_data():
    print('Loading data......')
    train = read_file('../../Data/train.csv')
    test = read_file('../../Data/test.csv')
    y_train = train.iloc[: , 0]
    x_train = train.iloc[: , 1:]
    y_test = test.iloc[: , 0]
    x_test = test.iloc[: , 1:]
    return x_train , y_train , x_test , y_test
    pass

if __name__ == '__main__':
    x_train , y_train , x_test , y_test = load_data()
    print('x_train : ',x_train)
    print('y_train : ',y_train)