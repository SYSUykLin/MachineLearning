import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time
from toolforData import loadCIFAR10, loadCIFAR_batch, data_validation

class LinearSVM(object):
    def __init__(self):
        self.W = None

    def loss(self, x, y, reg):
        loss = 0.0
        dw = np.zeros(self.W.shape)
        num_train = x.shape[0]
        scores = x.dot(self.W)
        correct_class_score = scores[range(num_train), list(y)].reshape(-1, 1)
        margin = np.maximum(0, scores - correct_class_score + 1)
        margin[range(num_train), list(y)] = 0
        loss = np.sum(margin)/num_train + 0.5 * reg * np.sum(self.W*self.W)

        num_classes = self.W.shape[1]
        inter_mat = np.zeros((num_train, num_classes))
        inter_mat[margin > 0] = 1
        inter_mat[range(num_train), list(y)] = 0
        inter_mat[range(num_train), list(y)] = -np.sum(inter_mat, axis=1)

        dW = (x.T).dot(inter_mat)
        dW = dW/num_train + reg*self.W
        return loss, dW
        pass

    def train(self, X, y, learning_rate=1e-3, reg=1e-5, num_iters=100,batch_size=200, verbose=False):
        num_train, dim = X.shape
        num_classes = np.max(y) + 1
        if self.W is None:
            self.W = 0.001 * np.random.randn(dim, num_classes)
        # Run stochastic gradient descent to optimize W
        loss_history = []
        for it in range(num_iters):
            X_batch = None
            y_batch = None
            idx_batch = np.random.choice(num_train, batch_size, replace = True)
            X_batch = X[idx_batch]
            y_batch = y[idx_batch]
            # evaluate loss and gradient
            loss, grad = self.loss(X_batch, y_batch, reg)
            loss_history.append(loss)
            self.W -=  learning_rate * grad
            if verbose and it % 100 == 0:
                print('iteration %d / %d: loss %f' % (it, num_iters, loss))
        return loss_history
        pass

    def predict(self, X):
        y_pred = np.zeros(X.shape[0])
        scores = X.dot(self.W)
        y_pred = np.argmax(scores, axis = 1)
        return y_pred

if __name__ == '__main__':
    svm = LinearSVM()
    tic = time.time()
    cifar10_name = '../Data/cifar-10-batches-py'
    x_train, y_train, x_test, y_test = loadCIFAR10(cifar10_name)
    X_val, Y_val, X_train, Y_train, X_dev, Y_dev, X_test, Y_test = data_validation(x_train, y_train, x_test, y_test)
    loss_hist = svm.train(X_train, Y_train, learning_rate=1e-7, reg=2.5e4,
                          num_iters=3000, verbose=True)
    toc = time.time()
    print('That took %fs' % (toc - tic))
    plt.plot(loss_hist)
    plt.xlabel('Iteration number')
    plt.ylabel('Loss value')
    plt.show()
    y_test_pred = svm.predict(X_test)
    test_accuracy = np.mean(Y_test == y_test_pred)
    print('accuracy: %f' % test_accuracy)
    w = svm.W[:-1, :]  # strip out the bias
    w = w.reshape(32, 32, 3, 10)
    w_min, w_max = np.min(w), np.max(w)
    classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    for i in range(10):
        plt.subplot(2, 5, i + 1)
        wimg = 255.0 * (w[:, :, :, i].squeeze() - w_min) / (w_max - w_min)
        plt.imshow(wimg.astype('uint8'))
        plt.axis('off')
        plt.title(classes[i])
    plt.show()
