import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from NewTon import newton_method
from DampedNewton import DampedNewton

class DataProcesser(object):
    '''initlize paramenter
       filename:LogiReg_data.txt
    '''
    def __init__(self):
        self.base_dir = '../../../Data/'
    '''
    get x and y
    '''
    def get_param_target(self):
        np.random.seed(12)
        num_observations = 5000
        x1 = np.random.multivariate_normal([0, 0], [[1, .75], [.75, 1]], num_observations)
        x2 = np.random.multivariate_normal([1, 4], [[1, .75], [.75, 1]], num_observations)
        simulated_separableish_features = np.vstack((x1, x2)).astype(np.float32)
        simulated_labels = np.hstack((np.zeros(num_observations),
                                      np.ones(num_observations)))
        return (simulated_separableish_features[:, 0], simulated_separableish_features[:, 1], simulated_labels)

    def get_dataset_from_file(self, filename):
        self.filename = filename
        dataframe = pd.read_csv(self.base_dir + self.filename,  encoding='UTF-8', names = ['x1', 'x2', 'target'], sep='	')
        dataframe.insert(0, 'x0', len(dataframe)*[1])
        features = np.matrix(dataframe.iloc[:, 0:2])
        target = np.matrix(dataframe.iloc[:, 2])
        return features, target

    def showDatasetDistribution(self, features, target, w, flag):
        plt.title('Distribution of the data')
        plt.scatter(np.array(features[:, 1]), np.array(target[:, 0]), alpha=0.5)
        plt.xlabel('y')
        plt.ylabel('x')
        if flag == True:
            x = np.arange(0,  1, 0.1)
            x = np.mat(x).T
            print(x.shape)
            plt.plot(x, w[0] + np.mat(x)*w[1], c = 'red')
            pass
        plt.show()

    def show(self, x1, x2, y, w):
        plt.scatter(x1, x2, c = y)
        x = np.array(range(-3, 4))
        plt.plot(x, (-w[2] - w[0]*x)/w[1])
        plt.show()
        pass


if __name__ == '__main__':
    Processer = DataProcesser()
    # x1, x2, y = Processer.get_param_target()
    # w = newton_method(x1, x2, y)
    # Processer.show(x1, x2, y, w)

    features, target = Processer.get_dataset_from_file('data.txt')
    newTon = DampedNewton(features, target.T, 100, 0.1, 0.5)
    Processer.showDatasetDistribution(features, target.T, None, False)
    w = newTon.newton()
    print('weight: ', w)
    Processer.showDatasetDistribution(features, target.T, w, True)

