#           ******************************************************
#          **   course         : Neural Networks                 **
#         ***   HomeWork       : 01                              ***
#        ****   Topic          : Single Layer Perceptron         ****
#        ****   AUTHOR         : Reza Adinepour                  ****
#         ***   Student ID:    : 9814303                         ***
#          **   Github         : github.com/reza_adinepour/      **
#           ******************************************************

import matplotlib.pyplot as plt
import numpy as np
from predict import*

datas = np.array([])
labels = np.array([])

def display(event):
    global datas, labels
    numInput = 2
    epochs = 100
    learningRate = 0.01
    global weights
    weights = np.random.random(numInput) - 0.5  #generate 2 random number between -0.5 and 0.5
    global bias 
    bias = np.random.random(1) - 0.5


    if event.button == 1:
        #print("Left Mouse button was clicked!")
        #print(event.xdata, event.ydata)
        plt.scatter(event.xdata, event.ydata, color='red')
        fig.canvas.draw()
        fig.canvas.flush_events()
        datas = np.append(datas, event.xdata)
        datas = np.append(datas, event.ydata)
        labels = np.append(labels, 1)

    elif event.button == 3:
        #print("Right Mouse button was clicked!")
        #print(event.xdata, event.ydata)
        plt.scatter(event.xdata, event.ydata, color='blue')
        fig.canvas.draw()
        fig.canvas.flush_events()
        datas = np.append(datas, event.xdata)
        datas = np.append(datas, event.ydata)
        labels = np.append(labels, 0)
    
    elif event.button == 2:
        for ep in range(epochs):
            failCount = 0
            i = 0
            for (data, label) in zip(datas, labels):
                i += 1
                output = predict(weights, data, bias)
                if(output != label):
                    weights += learningRate * (label - output) * data
                    bias += learningRate * (label - output)
                    failCount += 1

                    plt.cla()
                    plt.scatter(datas[:, 0], datas[:, 1], c=labels, cmap='bwr')
                    x_test = np.arange(-5, 5.1, 0.1)
                    y_test = (-1 / weights[1]) * (bias + weights[0] * x_test) # bias + W1*x_test + W2*y_test = 0 -> y_test = (-1 / W2) * (W0 + W1*x_test)
                    plt.plot(x_test, y_test, color='black')
                    plt.xlim(-5, 5)
                    plt.ylim(-5, 5)
                    plt.text(-5, 4.5, 'epoch|iter = {:2d}|{:2d}'.format(ep, i), fontdict={'size': 16, 'color': 'red'})
                    plt.pause(0.01)
            if(failCount == 0):
                plt.waitforbuttonpress()
                plot_decision_boundary()
                break


    datas = np.reshape(datas, (-1, 2))
    #print(datas)
    #print(labels)
    #print('-'*20)


def plot_decision_boundary():
    x_min, x_max = datas[:, 0].min() - 0.25, datas[:, 0].max() + 0.25
    y_min, y_max = datas[:, 1].min() - 0.25, datas[:, 1].max() + 0.25
    step = 0.1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, step), np.arange(y_min, y_max, step))     # partition some pieces with step 0.1
    points = np.c_[xx.ravel(), yy.ravel()]
    Z=[]
    for inputs in points:    
        Z.append(predict(weights, inputs, bias)) 
    # Z = predict(points, w)
    Z = np.array(Z)
    Z = Z.reshape(xx.shape)
    plt.figure(2)
    plt.title("Decision Boundary")
    plt.contourf(xx, yy, Z, cmap='jet')                # Coloring pieces
    plt.scatter(datas[:, 0], datas[:, 1], c=labels, edgecolors = 'gray', cmap='jet')
    #plt.cla()
    plt.show()
    plt.pause(0.05)
    plt.waitforbuttonpress()
    plt.close()


fig = plt.figure(1)
plt.title('add sample with left/right click mouse, for train click mouse scroll')
plt.xlim(-5, 5)
plt.ylim(-5, 5)

fig.canvas.mpl_connect("button_press_event", display)

plt.show()