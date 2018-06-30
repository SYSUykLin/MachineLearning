from MachineLearning.RandomForest.randomForest import randomForest
from MachineLearning.RandomForest.tool import load_data
import matplotlib.pyplot as plt

def running():
    '''entrance'''
    data_train, text, target = load_data()
    forest = randomForest()
    predic = []
    for i in range(1, 20):
        trees, features = forest.random_forest(data_train, i)
        predictions = forest.get_predict(trees, features, text)
        accuracy = forest.cal_correct_rate(target, predictions)
        print('The forest has ', i, 'tree', 'Accuracy : ' , accuracy)
        predic.append(accuracy)

    plt.xlabel('Number of tree')
    plt.ylabel('Accuracy')
    plt.title('The relationship between tree number and accuracy')
    plt.plot(range(1, 20), predic, color = 'orange')
    plt.show()
    pass

if __name__ == '__main__':
    running()