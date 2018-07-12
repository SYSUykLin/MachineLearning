import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
import MachineLearning.NeuralNetwork.Generator as Tool

class Network(object):
    def __init__(self, x, y):
        '''initialize the data'''
        self.x = x
        self.num_examples = len(x)
        self.y = y
        self.n_output = 2
        self.n_input = 2
        self.epsilon = 0.01
        self.reg_lambed = 0.01
        self.model = None
        pass

    def calculate_loss(self, model):
        '''calculate the loss function'''
        w1, b1, w2, b2 = model['w1'], model['b1'], model['w2'], model['b2']
        z1 = self.x.dot(w1) + b1
        a1 = np.tanh(z1)
        z2 = a1.dot(w2) + b2
        exp_scores = np.exp(z2)
        probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
        corect_logprobs = -np.log(probs[range(self.num_examples), self.y])
        data_loss = np.sum(corect_logprobs)
        data_loss += self.reg_lambed / 2 * (np.sum(np.square(w1)) + np.sum(np.square(w2)))
        return 1.0/self.num_examples * data_loss

    def predict(self, x):
        '''according to the model,predict the consequence'''
        w1, b1, w2, b2 = self.model['w1'], self.model['b1'], self.model['w2'], self.model['b2']
        z1 = x.dot(w1) + b1
        a1 = np.tanh(z1)
        z2 = a1.dot(w2) + b2
        exp_scores = np.exp(z2)
        probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
        return np.argmax(probs, axis=1)
        pass

    def build_Network(self, n_hinden, num_pass = 20000, print_loss = True):
        loss = []
        np.random.seed(0)
        w1 = np.random.randn(self.n_input, n_hinden) / np.sqrt(self.n_input)
        b1 = np.zeros((1, n_hinden))
        w2 = np.random.randn(n_hinden, self.n_output) / np.sqrt(n_hinden)
        b2 = np.zeros((1, self.n_output))

        self.model = {}
        for i in range(num_pass):
            z1 = self.x.dot(w1) + b1
            a1 = np.tanh(z1)
            z2 = a1.dot(w2) + b2
            exp_scores = np.exp(z2)
            probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

            delta3 = probs
            delta3[range(self.num_examples), self.y] -= 1
            dw2 = (a1.T).dot(delta3)
            db2 = np.sum(delta3, axis=0, keepdims=True)
            delta2 = delta3.dot(w2.T)*(1 - np.power(a1, 2))
            dw1 = np.dot(self.x.T, delta2)
            db1 = np.sum(delta2, axis=0)
            dw2 += self.reg_lambed * w2
            dw1 += self.reg_lambed * w1

            w1 += -self.epsilon * dw1
            b1 += -self.epsilon * db1
            w2 += -self.epsilon * dw2
            b2 += -self.epsilon * db2

            self.model = {'w1':w1, 'b1':b1, 'w2':w2, 'b2':b2}
            if print_loss and i %200 == 0:
                print('Loss : ', (i, self.calculate_loss(model=self.model)))
                loss.append(self.calculate_loss(model=self.model))
        return loss

if __name__ == '__main__':
    Accuracy = []
    losses = []
    x, y = Tool.generator()
    for i in range(1, 5):
        mlp = Network(x, y)
        loss = mlp.build_Network(n_hinden=i)
        losses.append(loss)
        Tool.plot_decision_boundary(mlp.predict, x, y, 'Neutral Network when Hidden layer size is ' + str(i))
        predicstions = mlp.predict(x)
        a = sum(1*(predicstions == y)) / len(y)
        Accuracy.append(a)

    '''draw the accuracy picture'''
    plt.plot(range(len(Accuracy)), Accuracy, c = 'blue')
    plt.title('The Accuracy of the Neural Network')
    plt.xlabel('Hinden Layer Size')
    plt.ylabel('Accuracy')
    plt.show()


    '''draw the loss function picture'''
    for i, loss in enumerate(losses):
        plt.plot(range(len(loss)), loss, c = Tool.get_colors(i), label = 'the hindden layer size '+str(i))
    plt.title('Loss Function')
    plt.xlabel('time')
    plt.ylabel('loss score')
    plt.legend()
    plt.show()