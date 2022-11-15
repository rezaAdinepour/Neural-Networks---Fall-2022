#           ******************************************************
#          **   course         : Neural Networks                 **
#         ***   HomeWork       : 01                              ***
#        ****   Topic          : Single Layer Perceptron         ****
#        ****   AUTHOR         : Reza Adinepour                  ****
#         ***   Student ID:    : 9814303                         ***
#          **   Github         : github.com/reza_adinepour/      **
#           ******************************************************

import numpy as np
from unitStep import unitStep

def predict(weights, inputs, bias):
    sum = np.dot(weights, inputs) + bias
    #activation = 1.0 if(sum > 0) else 0.0
    activation = unitStep(sum)

    return activation