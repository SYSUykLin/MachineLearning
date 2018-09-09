import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.datasets import mnist

class DataPrecessing(object):
    def loadFile(self):
        (x_train, x_target_tarin), (x_test, x_target_test) = mnist.load_data()
        x_train = x_train.astype('float32')/255.0
        x_test = x_test.astype('float32')/255.0
        x_train = x_train.reshape(len(x_train), np.prod(x_train.shape[1:]))
        x_test = x_test.reshape(len(x_test), np.prod(x_test.shape[1:]))
        x_train = np.mat(x_train)
        x_test = np.mat(x_test)
        x_target_tarin = np.mat(x_target_tarin)
        x_target_test = np.mat(x_target_test)
        return x_train, x_target_tarin, x_test, x_target_test

    def Calculate_accuracy(self, target, prediction):
        score = 0
        for i in range(len(target)):
            if target[i] == prediction[i]:
                score += 1
        return score/len(target)

    def predict(self, test, weights):
        h = test * weights
        return h.argmax(axis=1)

if __name__ == '__main__':
    dataPer = DataPrecessing()
    dataPer.loadFile()

