import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn

'''sigmoid function, nonlinear transform
'''
def sigmoid(
        x1, x2, theta1, theta2, theta3):
    z = (theta1*x1+ theta2*x2+theta3).astype("float_")
    return 1.0 / (1.0 + np.exp(-z))

'''cross entropy loss function
'''
def cost(
        x1, x2, y, theta_1, theta_2, theta_3):
    sigmoid_probs = sigmoid(x1,x2, theta_1, theta_2,theta_3)
    return -np.mean(y * np.log(sigmoid_probs)
                  + (1 - y) * np.log(1 - sigmoid_probs))

'''one dimenssion
 '''
def gradient(
        x_1, x_2, y, theta_1, theta_2, theta_3):
    sigmoid_probs = sigmoid(x_1,x_2, theta_1, theta_2,theta_3)
    return np.array([np.sum((y - sigmoid_probs) * x_1),
                     np.sum((y - sigmoid_probs) * x_2),
                     np.sum(y - sigmoid_probs)])

'''calculate hassion matrix'''
def Hassion(
        x1, x2, y, theta1, theta2, theta3):
    sigmoid_probs = sigmoid(x1,x2, theta1,theta2,theta3)
    d1 = np.sum((sigmoid_probs * (1 - sigmoid_probs)) * x1 * x1)
    d2 = np.sum((sigmoid_probs * (1 - sigmoid_probs)) * x1 * x2)
    d3 = np.sum((sigmoid_probs * (1 - sigmoid_probs)) * x1 )
    d4 = np.sum((sigmoid_probs * (1 - sigmoid_probs)) * x2 * x1)
    d5 = np.sum((sigmoid_probs * (1 - sigmoid_probs)) * x2 * x2)
    d6 = np.sum((sigmoid_probs * (1 - sigmoid_probs)) * x2 )
    d7 = np.sum((sigmoid_probs * (1 - sigmoid_probs)) * x1 )
    d8 = np.sum((sigmoid_probs * (1 - sigmoid_probs)) * x2 )
    d9 = np.sum((sigmoid_probs * (1 - sigmoid_probs)))
    H = np.array([[d1, d2,d3],[d4, d5,d6],[d7,d8,d9]])
    return H

'''update parameter by newton method'''
def newton_method(x1, x2, y):
    #initialize parameter
    theta1 = 0.001
    theta2 = -0.4
    theta3 = 0.6
    delta_loss = np.Infinity
    loss = cost(x1, x2, y, theta1, theta2, theta3)
    kxi = 0.0000001
    max_iteration = 10000
    i = 0
    print('while i = ', i, ' Loss = ', loss, 'delta_loss = ', delta_loss)
    while np.abs(delta_loss) > kxi and i < max_iteration:
        i += 1
        g = gradient(x1, x2, y, theta1, theta2, theta3)
        hassion = Hassion(x1, x2, y, theta1, theta2, theta3)
        H_inverse = np.linalg.inv(hassion)
        delta = H_inverse @ g.T
        delta_theta_1 = delta[0]
        delta_theta_2 = delta[1]
        delta_theta_3 = delta[2]
        theta1 += delta_theta_1
        theta2 += delta_theta_2
        theta3 += delta_theta_3
        loss_new = cost(x1, x2, y, theta1, theta2, theta3)
        delta_loss = loss - loss_new
        loss = loss_new
        print('while i = ', i, ' Loss = ', loss, 'delta_loss = ', delta_loss)
    return np.array([theta1, theta2, theta3])