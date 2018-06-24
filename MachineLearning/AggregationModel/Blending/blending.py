import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import MachineLearning.AggregationModel.Blending.load_data as load_data
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier , ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

def calculate_accuracy(predictions , y_test):
    sum = 0
    for i in range(len(y_test)):
        if predictions[i] == y_test.tolist()[i]:
           sum += 1
    print("accuracy ：",(sum/len(y_test))*100 , '%')
    return (sum/len(y_test))*100

def draw(accuracys , indexs_name):
    fig, ax = plt.subplots()
    colors = ['#B0C4DE','#6495ED','#0000FF','#0000CD','#000080','#191970']
    ax.bar(range(len(accuracys)), accuracys, color=colors, tick_label=indexs_name)
    plt.xlabel('ModelName')
    plt.ylabel('Accuracy')
    ax.set_xticklabels(indexs_name,rotation=30)
    plt.show()
    pass
if __name__ == '__main__':

    np.random.seed(0)  # seed to shuffle the train set
    accuracys = []
    n_folds = 10
    verbose = True
    shuffle = False

    X, y, X_test , y_test = load_data.load_data()

    if shuffle:
        idx = np.random.permutation(y.size) #random a sequence
        X = X[idx]
        y = y[idx]
    kfold = StratifiedKFold(n_splits=5)
    clfs = [RandomForestClassifier(n_estimators=100, n_jobs=-1, criterion='gini'),
            RandomForestClassifier(n_estimators=100, n_jobs=-1, criterion='entropy'),
            ExtraTreesClassifier(n_estimators=100, n_jobs=-1, criterion='gini'),
            ExtraTreesClassifier(n_estimators=100, n_jobs=-1, criterion='entropy'),
            GradientBoostingClassifier(learning_rate=0.05, subsample=0.5, max_depth=6, n_estimators=50)]

    print('Building the model......')
    dataset_blend_train = np.zeros((X.shape[0], len(clfs)))
    dataset_blend_test = np.zeros((X_test.shape[0], len(clfs)))
    test_matrix = np.zeros((len(y_test) , 5))




    for j, clf in enumerate(clfs):
        print (j, clf)
        clf.fit(X[j*700 : (j+1)*700] , y[j*700 : (j+1)*700])
        predict = clf.predict_proba(X_test)[: , 1]
        predict1 = []
        for i in range(len(predict)):
            if predict[i] > 0.5:
                predict1.append(1)
            else:
                predict1.append(0)
            pass
        accuracys.append(calculate_accuracy(predict1 , y_test))
        test_matrix[: , j] = predict

    predictions = []
    for i in range(len(test_matrix)):
        if test_matrix[i].mean() > 0.5:
            predictions.append(1)
        else:
            predictions.append(0)

    accuracys.append(calculate_accuracy(predictions , y_test))
    indexs_name = ['RandomForest','RandomForest','ExtraTrees_gini','ExtraTrees_entropy' , 'GradientBoosting','Average']
    draw(accuracys , indexs_name)



