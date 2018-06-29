from sklearn.datasets import load_iris as load_data
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import MachineLearning.DecisionTree.decisionTree as decisionTree
import MachineLearning.DecisionTree.tool as tool
from MachineLearning.DecisionTree.drawTree import *
if __name__ == '__main__':
    print('load_data......')
    dataSet = load_data()
    data = dataSet.data
    target = dataSet.target
    dataframe = pd.DataFrame(data = data, dtype = np.float32)
    dataframe.insert(4, 'label', target)
    dataMat = np.mat(dataframe)

    '''test and train
    '''
    X_train, X_test, y_train, y_test = train_test_split(dataMat[:, 0:-1], dataMat[:, -1], test_size=0.3, random_state=0)
    data_train = np.hstack((X_train, y_train))
    data_train = data_train.tolist()
    X_test = X_test.tolist()
    tree = decisionTree.decision_tree()
    tree_root = tree.build_tree(data_train)
    predictions = tree.predcit_samples(X_test, tree_root)
    pres = []
    for i in predictions:
        pres.append(list(i.keys()))

    y_test = y_test.tolist()
    accuracy = 0
    for i in range(len(y_test)):
        if y_test[i] == pres[i]:
            accuracy += 1
    print('Accuracy : ', accuracy / len(y_test))
    print('Number of leaf : ' , getNumLeafs(tree_root) - 1)
    print('Depth of decision tree : ', getDepth(tree_root))
    draw(tree_root)
    plt.title('Decision Tree')
    plt.xlabel('Number of leaf : ' + str(getNumLeafs(tree_root) - 1) + '\n' + 'Depth of decision tree : ' + str(getDepth(tree_root)))
    plt.show()


