import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import random as rd
import MachineLearning.DecisionTree.decisionTree as tree
import MachineLearning.RandomForest.tool as tool
from MachineLearning.DecisionTree.drawTree import draw
import math

class randomForest(object):

    def choose_samples(self, data, k):
        '''choose the feature from data
        input:data, type = list
        output:k
        '''
        n, d = np.shape(data)
        feature = []
        for j in range(k):
            feature.append(rd.randint(0, d - 2))
        index = []
        for i in range(n):
            index.append(rd.randint(0, n-1))
        data_samples = []
        for i in range(n):
            data_tmp = []
            for fea in feature:
                data_tmp.append(data[i][fea])
            data_tmp.append(data[i][-1])
            data_samples.append(data_tmp)
            pass
        return data_samples, feature
        pass

    def random_forest(self, data, trees_num):
        '''create a forest
        input:data, type = list
        output:trees_result, trees_feature
        '''
        decisionTree = tree.decision_tree()
        trees_result = []
        trees_feature = []
        d = np.shape(data)[1]
        if d > 2:
            k = int(math.log(d - 1, 2)) + 1
        else:
            k = 1
        for i in range(trees_num):
            print('The ', i, ' tree. ')
            data_samples, feature = self.choose_samples(data, k)
            t = decisionTree.build_tree(data_samples)
            trees_result.append(t)
            trees_feature.append(feature)
            pass
        return trees_result, trees_feature

    def get_predict(self, trees_result, trees_feature, data_train):
        '''predict the result
        input:trees_result, trees_feature, data
        output:final_prediction
        '''
        decisionTree = tree.decision_tree()
        m_tree = len(trees_result)
        m = np.shape(data_train)[0]
        result = []
        for i in range(m_tree):
            clf = trees_result[i]
            feature = trees_feature[i]
            data = tool.split_data(data_train, feature)
            result_i = []
            for i in range(m):
                result_i.append( list((decisionTree.predict(data[i][0 : -1], clf).keys()))[0] )
            result.append(result_i)
        final_predict = np.sum(result, axis = 0)
        return final_predict

    def cal_correct_rate(self, target, final_predict):
        m = len(final_predict)
        corr = 0.0
        for i in range(m):
            if target[i] * final_predict[i] > 0:
                corr += 1
            pass
        return corr/m
        pass




