import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from collections import defaultdict
import random
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score,recall_score,accuracy_score
from sklearn.ensemble import IsolationForest
class Bagging(object):

    def __init__(self ,n_estimators , estimator , rate = 1.0):
        self.n_estimators = n_estimators
        self.estimator = estimator
        self.rate = rate
        pass


    def Voting(self , data):
        term = np.transpose(data)
        result = list()
        def Vote(df):
            store = defaultdict()
            for kw in df:
                store.setdefault(kw , 0)
                store[kw] += 1
            return max(store , key=store.get)
        result = map(Vote , term)
        return result

    def UnderSampling(self,data , number):
        data = np.array(data)
        np.random.shuffle(data)
        newdata = data[0:int(data.shape[0]*self.rate),:]
        return newdata
        pass

    def TrainPredict(self , train , test):
        clf = self.estimator.fit(train[: , 0:-1] , train[: , -1])
        result = clf.predict(test[: , 0:-1])
        return result
        pass

    def RepetitionRandomSampling(self , data , number):
        samples = []
        for i in range(int(self.rate * number)):
            samples.append(data[random.randint(0, len(data) - 1)])
            pass
        return samples
        pass

    def Metrics(self, predict_data , test):
        score = predict_data
        pre = np.matrix(test[: , -1])
        score = list(score)
        score = np.matrix(score)
        recall = recall_score(pre.T , score.T, average=None)
        precision = accuracy_score(pre.T, score.T)
        return recall , precision
        pass

    def MutModel_clf(self , train , test , sample_type = 'RepetitionRandomSampling'):
        result = list()
        sample_function = self.RepetitionRandomSampling
        num_estimators = len(self.estimator)
        if sample_type == "RepetitionRandomSampling":
            print('Sample type : ' , sample_type)
        elif sample_type == "UnderSampling":
            print('Sample type : ' , sample_type)
            sample_function = self.UnderSampling
            print ("sampling frequency : ",self.rate)
        for estimator in self.estimator:
            for i in range(int(self.n_estimators/num_estimators)):
                sample=np.array(sample_function(train,len(train)))
                clf = estimator.fit(sample[:,0:-1],sample[:,-1])
                result.append(clf.predict(test[:,0:-1]))

        score = self.Voting(result)
        recall,precosoion = self.Metrics(score,test)
        return recall,precosoion


