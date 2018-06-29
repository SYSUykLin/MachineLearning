import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from math import *
import MachineLearning.DecisionTree.tool as tool

class node:
    '''Tree node
    '''
    def __init__(self , fea = -1, value = None, results = None, right = None, left = None):
        '''
        initialization function
        :param fea:column index value
        :param value:split value
        :param results:The class belongs to
        :param right:right side
        :param left:left side
        '''
        self.fea = fea
        self.value = value
        self.results = results
        self.right = right
        self.left = left
        pass

class decision_tree(object):

    def build_tree(self,data):
        '''Create decision tree
        input:data
        output:root
        '''
        if len(data) == 0:
            return node()

        currentGini = tool.cal_gini(data)
        bestGain = 0.0
        bestCriterria = None # store the optimal cutting point
        bestSets = None # store two datasets which have been splited

        feature_num = len(data[0]) - 1 # Number of features
        for fea in range(0 , feature_num):
            feature_values = {}
            for sample in data:
                feature_values[sample[fea]] = 1 # store the value in the demension fea possibly
            for value in feature_values.keys():
                (set_first, set_second) = self.split_tree(data, fea, value)
                nowGini = float(len(set_first) * tool.cal_gini(set_first) + len(set_second) * tool.cal_gini(set_second)) / len(data)
                gain = currentGini - nowGini
                if gain > bestGain and len(set_first) > 0 and len(set_second) > 0:
                    bestGain = gain
                    bestCriterria = (fea , value)
                    bestSets = (set_first , set_second)
                pass
        if bestGain > 0:
            right = self.build_tree(bestSets[0])
            left = self.build_tree(bestSets[1])
            return node(fea = bestCriterria[0], value = bestCriterria[1], right = right, left = left)
        else:
            return node(results=tool.label_uniqueness(data))

    def split_tree(self , data , fea , value):
        '''split the dataset according demension and value
        input:data
        output:two data
        '''
        set_first = []
        set_second = []
        for x in data:
            if x[fea] >= value:
                set_first.append(x)
            else:
                set_second.append(x)
        return (set_first, set_second)
        pass

    def predict(self, sample, tree):
        '''prediction
        input:sample, the tree which we have been built
        output:label
        '''
        if tree.results != None:
            return tree.results

        else:
            val_sample = sample[tree.fea]
            branch = None
            if val_sample >= tree.value:
                branch = tree.right
            else:
                branch = tree.left
            return self.predict(sample, branch)

    def predcit_samples(self, samples, tree):
        predictions = []
        for sample in samples:
            predictions.append(self.predict(sample, tree))
        return predictions

    pass
