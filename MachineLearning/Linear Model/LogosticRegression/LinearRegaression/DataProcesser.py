import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from NewTon import newton_method

class DataProcesser(object):
    '''initlize paramenter
       filename:LogiReg_data.txt
    '''
    def __init__(self, filename):
        self.filename = filename
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

if __name__ == '__main__':
    Processer = DataProcesser('LogiReg_data.txt')
    x1, x2, y = Processer.get_param_target()
    print(newton_method(x1, x2, y))
