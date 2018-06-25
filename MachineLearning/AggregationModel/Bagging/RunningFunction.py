import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import MachineLearning.AggregationModel.Bagging.bagging as bagging
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier , ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
def loadData():
    train = pd.read_csv('../../Data/newtrain.csv')
    test = pd.read_csv('../../Data/newtest.csv')
    train = train.as_matrix()
    test = test.as_matrix()
    return train , test
    pass

def Running(sample = 'RepetitionRandomSampling'):
    train , test = loadData()
    clfs = [RandomForestClassifier(n_estimators=100, n_jobs=-1, criterion='gini'),
            RandomForestClassifier(n_estimators=100, n_jobs=-1, criterion='entropy'),
            ExtraTreesClassifier(n_estimators=100, n_jobs=-1, criterion='gini'),
            ExtraTreesClassifier(n_estimators=100, n_jobs=-1, criterion='entropy'),
            GradientBoostingClassifier(learning_rate=0.05, subsample=0.5, max_depth=6, n_estimators=50)]
    bag = bagging.Bagging(n_estimators=5 , estimator=clfs)
    recall , precision = bag.MutModel_clf(train , test , sample_type=sample)
    print(recall , precision)
    return precision
    pass

if __name__ == '__main__':
    pres1 = []
    pres2 = []

    for i in range(10):
        pre = Running(sample='UnderSampling')
        pres2.append(pre)

    for i in range(10):
        pre = Running()
        pres1.append(pre)


    plt.plot([x for x in range(10)] , pres1 , c = 'r')
    plt.plot([x for x in range(10)] , pres2 , c = 'b')
    plt.show()
