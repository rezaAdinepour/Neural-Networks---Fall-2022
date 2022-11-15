#           ******************************************************
#          **   course         : Neural Networks                 **
#         ***   HomeWork       : 01                              ***
#        ****   Topic          : Single Layer Perceptron         ****
#        ****   AUTHOR         : Reza Adinepour                  ****
#         ***   Student ID:    : 9814303                         ***
#          **   Github         : github.com/reza_adinepour/      **
#           ******************************************************

import numpy as np


def sigmoid(x, der=False):
    f = 1 / (1 + np.exp(-x))
    if (der == True):
        f = f * (1 -f)
    return f

def tanh(x, der=False):
    f = (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))
    if (der == True):
        f = 1 - f ** 2
    return f

def linear(x, der=False):
    f = x
    if (der == True):
        f = 1
    return f

def ReLU(x, der=False):
    if (der == True):
        f = np.heaviside(x, 1)
    else :
        f = np.maximum(x, 0)
    return f

def unitStep(x):
    outPut = 1 * (x >= 0)
    #outPut = 1.0 if (x > 0.0) else 0.0
    return outPut