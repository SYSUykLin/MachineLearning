import numpy as np
from keras.datasets import mnist
import matplotlib.pyplot as plt
def get_dataSets():
    (x_train, _), (x_test, _) = mnist.load_data()
    x_train = x_train.astype('float32')/255.
    x_test = x_test.astype('float32')/255.
    x_train = x_train.reshape(len(x_train), np.prod(x_train.shape[1:]))
    x_test = x_test.reshape(len(x_test), np.prod(x_test.shape[1:]))
    return x_train, x_test
    pass

def show(orignal, x):
    n = 10  # how many digits we will display
    plt.figure(figsize=(20, 4))
    for i in range(n):
        ax = plt.subplot(2, n, i + 1)
        plt.imshow(orignal[i].reshape(28, 28))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        ax = plt.subplot(2, n, i + 1 + n)
        plt.imshow(x[i].reshape(28, 28))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    plt.show()

if __name__ == '__main__':
    get_dataSets()