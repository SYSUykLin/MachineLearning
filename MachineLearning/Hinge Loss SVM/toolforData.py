import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import pickle
import platform
import seaborn as sns

def showPicture(x_train, y_train):
    classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    num_classes = len(classes)
    samples_per_classes = 7
    for y, cls in enumerate(classes):
        idxs = np.flatnonzero(y_train == y)
        idxs = np.random.choice(idxs, samples_per_classes, replace=False)
        for i, idx in enumerate(idxs):
            plt_index = i*num_classes +y + 1
            plt.subplot(samples_per_classes, num_classes, plt_index)
            plt.imshow(x_train[idx].astype('uint8'))
            plt.axis('off')
            if i == 0:
                plt.title(cls)
    plt.show()

def load_pickle(f):
    version = platform.python_version_tuple()
    if version[0] == '2':
        return pickle.load(f)
    elif version[0] == '3':
        return pickle.load(f, encoding='latin1')
    raise ValueError("invalid python version: {}".format(version))

def loadCIFAR_batch(filename):
    with open(filename, 'rb') as f:
        datadict = load_pickle(f)
        x = datadict['data']
        y = datadict['labels']
        x = x.reshape(10000, 3, 32, 32).transpose(0, 3, 2, 1).astype('float')
        y = np.array(y)
        return x, y

def loadCIFAR10(root):
    xs = []
    ys = []
    for b in range(1, 6):
        f = os.path.join(root, 'data_batch_%d' % (b, ))
        x, y = loadCIFAR_batch(f)
        xs.append(x)
        ys.append(y)
    X = np.concatenate(xs)
    Y = np.concatenate(ys)
    x_test, y_test = loadCIFAR_batch(os.path.join(root, 'test_batch'))
    return X, Y, x_test, y_test

def data_validation(x_train, y_train, x_test, y_test):
    num_training = 49000
    num_validation = 1000
    num_test = 1000
    num_dev = 500
    mean_image = np.mean(x_train, axis=0)
    x_train -= mean_image
    mask = range(num_training, num_training + num_validation)
    X_val = x_train[mask]
    Y_val = y_train[mask]
    mask = range(num_training)
    X_train = x_train[mask]
    Y_train = y_train[mask]
    mask = np.random.choice(num_training, num_dev, replace=False)
    X_dev = x_train[mask]
    Y_dev = y_train[mask]
    mask = range(num_test)
    X_test = x_test[mask]
    Y_test = y_test[mask]
    X_train = np.reshape(X_train, (X_train.shape[0], -1))
    X_val = np.reshape(X_val, (X_val.shape[0], -1))
    X_test = np.reshape(X_test, (X_test.shape[0], -1))
    X_dev = np.reshape(X_dev, (X_dev.shape[0], -1))
    X_train = np.hstack([X_train, np.ones((X_train.shape[0], 1))])
    X_val = np.hstack([X_val, np.ones((X_val.shape[0], 1))])
    X_test = np.hstack([X_test, np.ones((X_test.shape[0], 1))])
    X_dev = np.hstack([X_dev, np.ones((X_dev.shape[0], 1))])
    return X_val, Y_val, X_train, Y_train, X_dev, Y_dev, X_test, Y_test
    pass

if __name__ == '__main__':
    cifar10_name = '../Data/cifar-10-batches-py'
    x_train, y_train, x_test, y_test = loadCIFAR10(cifar10_name)
    print(x_train.shape)
    print(y_train.shape)
    print(x_test.shape)
    print(y_test.shape)
    showPicture(x_train, y_train)
    data_validation(x_train, y_train, x_test, y_test)