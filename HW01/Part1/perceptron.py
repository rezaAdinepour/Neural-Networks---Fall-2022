#           ******************************************************
#          **   course         : Neural Networks                 **
#         ***   HomeWork       : 01                              ***
#        ****   Topic          : Single Layer Perceptron         ****
#        ****   AUTHOR         : Reza Adinepour                  ****
#         ***   Student ID:    : 9814303                         ***
#          **   Github         : github.com/reza_adinepour/      **
#           ******************************************************


import numpy as np
import matplotlib.pyplot as plt
from predict import predict
import sklearn.datasets



def plot_decision_boundary():
    x_min, x_max = X[:, 0].min() - 0.25, X[:, 0].max() + 0.25
    y_min, y_max = X[:, 1].min() - 0.25, X[:, 1].max() + 0.25
    step = 0.1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, step), np.arange(y_min, y_max, step))     # partition some pieces with step 0.1
    points = np.c_[xx.ravel(), yy.ravel()]
    Z=[]
    for inputs in points:    
        Z.append(predict(weights[1:], inputs, weights[0])) 
    # Z = predict(points, w)
    Z = np.array(Z)
    Z = Z.reshape(xx.shape)
    plt.figure()
    plt.title("Decision Boundary")
    plt.contourf(xx, yy, Z)                # Coloring pieces
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors = 'gray')
    #plt.cla()
    plt.show()
    plt.pause(0.05)
    plt.waitforbuttonpress()
    plt.close()



######################################## complete perceptron ########################################
# information of network
X, y = sklearn.datasets.make_blobs(n_samples=100, n_features=2, centers=[(1, -1), (3, 3)], cluster_std=0.7, center_box=(-10.0, 10.0),
                                   shuffle=True, random_state=None, return_centers=False)
#plt.scatter(X[:, 0], X[:, 1], c=y)
numInput = 2
EPOCH = 100
lr = 0.01
iter = 0
weights = np.random.random(numInput + 1) - 0.5

#train phase
for epoch in range(EPOCH):
    failCount = 0
    for (inputs, labels) in zip(X, y):
        prediction = predict(weights[1:], inputs, weights[0])
        iter += 1
        if(labels != prediction):
            weights[1:] += lr * (labels - prediction) * inputs
            weights[0] += lr * (labels - prediction)
            failCount += 1

            plt.figure(1)
            plt.cla()
            plt.scatter(X[:, 0], X[:, 1], c=y)
            plt.xlim(-1, 5)
            plt.ylim(-3, 5)
            xTest = np.arange(-1, 5, 0.1)
            yTest = ( -weights[0] - weights[1] * xTest) / weights[2]
            plt.plot(xTest, yTest, color='black')
            plt.text(-1, 4.5, 'epoch/iter = {:2d}/{:2d}'.format(epoch, iter), fontdict={'size':12, 'color':'red'})
            plt.pause(0.05)
    if(failCount == 0):
        plt.waitforbuttonpress()
        plot_decision_boundary()
        break


plt.show()