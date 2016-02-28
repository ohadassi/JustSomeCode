# Machine Learning Online Class - Exercise 1: Linear Regression
#  Instructions
#  ------------
#
#  This file contains code that helps you get started on the
#  linear exercise. You will need to complete the following functions
#  in this exericse:
#
#     warmUpExercise.m
#     plotData.m
#     gradientDescent.m
#     computeCost.m
#     gradientDescentMulti.m
#     computeCostMulti.m
#     featureNormalize.m
#     normalEqn.m
#
#  For this exercise, you will not need to change any code in this file,
#  or any other files other than those mentioned above.
#
# x refers to the population size in 10,000s
# y refers to the profit in $10,000s
#
import os
os.environ['PATH'] = r'C:\Anaconda2\Library\bin;'+os.environ['PATH']

from pylab import *
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
from mpl_toolkits.mplot3d.axes3d import Axes3D
import scipy.io as io

############################
## Initialization
plt.close()
gInPause = False
def onclick(event):
    global gInPause
    gInPause = not gInPause
def figPause(fig):
    global gInPause
    buttonAxes = plt.axes([0.9, 0.9, 0.1, 0.05])
    button = Button(buttonAxes, '>>')
    button.on_clicked(onclick)
    gInPause = True
    print('Program paused in figure window...')
    while gInPause:
        plt.pause(0.2)
        plt.draw()
    button.disconnect_events()
    #fig.canvas.mpl_connect('key_press_event', onKeyPress)
    #global pause
    #pause = True
    #while pause:
    #    plt.pause(0.2)
    #    #ax.canvas.get_tk_widget().update() # process events
    #    plt.draw()
def plotData(x, y):
#PLOTDATA Plots the data points x and y into a new figure
#   PLOTDATA(x,y) plots the data points and gives the figure axes labels of
#   population and profit.

# ====================== YOUR CODE HERE ======================
# Instructions: Plot the training data into a figure using the
#               "figure" and "plot" commands. Set the axes labels using
#               the "xlabel" and "ylabel" commands. Assume the
#               population and revenue data have been passed in
#               as the x and y arguments of this function.
#
# Hint: You can use the 'rx' option with plot to have the markers
#       appear as red crosses. Furthermore, you can make the
#       markers larger by using plot(..., 'rx', 'MarkerSize', 10);
    fig, ax = plt.subplots(); # open a new figure window
    ax.plot(x, y, 'rx'); # Plot the data
    ax.set_ylabel('Profit in $10,000s'); # Set the y ?axis label
    ax.set_xlabel('Population of City in 10,000s'); # Set the x ?axis label
    return fig, ax
##=================================================
def warmUpExercise():
    # WARMUPEXERCISE Example function in octave
    #   A = WARMUPEXERCISE() is an example function that returns the 5x5 identity matrix

    # ============= YOUR CODE HERE ==============
    # Instructions: Return the 5x5 identity matrix
    #               In octave, we return values by defining which variables
    #               represent the return values (at the top of the file)
    #               and then set them accordingly.
    A=np.eye(5);
    return A
##=================================================
def computeCost(X, y, theta):
    #COMPUTECOST Compute cost for linear regression
    #   J = COMPUTECOST(X, y, theta) computes the cost of using theta as the
    #   parameter for linear regression to fit the data points in X and y

    # Initialize some useful values
    m = y.shape[0]; # number of training examples
    # You need to return the following variables correctly
    J = 0;
    # ====================== YOUR CODE HERE ======================
    # Instructions: Compute the cost of a particular choice of theta
    #               You should set J to the cost.
    diffVec = np.dot(X, theta) - y;
    J = np.dot(diffVec.T, diffVec) / (2*m);
    return J
##=================================================
def gradientDescent(X, y, theta, alpha, num_iters):
    #GRADIENTDESCENT Performs gradient descent to learn theta
    #   theta = GRADIENTDESENT(X, y, theta, alpha, num_iters) updates theta by
    #   taking num_iters gradient steps with learning rate alpha

    # Initialize some useful values
    m = size(y); # number of training examples
    J_history = np.zeros([num_iters, 1], dtype=float);

    for iter in range(1, num_iters):
        #====================== YOUR CODE HERE ======================
        # Instructions: Perform a single gradient step on the parameter vector
        #               theta.
        #
        # Hint: While debugging, it can be useful to print out the values
        #       of the cost function (computeCost) and gradient here.
        #
        diffVec = (np.dot(X, theta) - y).flatten()
        theta[0] -=  (alpha/m)*np.sum(diffVec*X[:,0])
        theta[1] -=  (alpha/m)*np.sum(diffVec*X[:,1])
        # ============================================================
        # Save the cost J in every iteration
        J_history[iter] = computeCost(X, y, theta);
    return theta, J_history
def main():
    ## ==================== Part 1: Basic Function ====================
    # Complete warmUpExercise.m
    print('Running warmUpExercise ... \n');
    print('5x5 Identity Matrix: \n');
    A = warmUpExercise()
    print A

    raw_input('Program paused. Press enter to continue.\n');



    ## ======================= Part 2: Plotting =======================
    print 'Plotting Data ...'
    #data = io.loadmat(r'C:\Users\ohadassi\xxx\ML-AndrewNG\Week2\machine-learning-ex1\ex1\ex1data1.txt');
    data = np.genfromtxt(r'C:\Users\ohadassi\xxx\ML-AndrewNG\Week2\machine-learning-ex1\ex1\ex1data1.txt', delimiter=',')
    X = data[:, 0].astype(float)
    y = data[:, 1].astype(float)
    m = size(y); # number of training examples

    # Plot Data
    # Note: You have to complete the code in plotData.m
    fig,ax = plotData(X, y);
    figPause(fig)

    ## =================== Part 3: Gradient descent ===================
    print 'Running Gradient Descent ...'
    X = np.concatenate((np.ones([m, 1], dtype=float), X[:,newaxis]), axis=1) # Add a column of ones to x
    y = y[:,newaxis]
    theta = np.zeros([2, 1], dtype=float) # initialize fitting parameters

    # Some gradient descent settings
    iterations = 1500
    alpha = 0.01

    # compute and display initial cost
    cost = computeCost(X, y, theta)
    print('Initial cost: %f' % cost)
    # run gradient descent
    theta, J_Hist = gradientDescent(X, y, theta, alpha, iterations);

    # print theta to screen
    print 'Theta found by gradient descent: ';
    print('%f %f \n' % (theta[0], theta[1]));

    # Plot the linear fit
    ax.plot(X[:,1], np.dot(X,theta), '-')
    ax.legend(['Training data', 'Linear regression'])
    figPause(fig)

    # Predict values for population sizes of 35,000 and 70,000
    predict1 = np.dot([1, 3.5], theta)*10000
    print('For population = 35,000, we predict a profit of %f\n' % predict1);
    predict2 = np.dot([1, 7], theta)*10000
    print('For population = 70,000, we predict a profit of %f\n' % predict2);

    figPause(fig)

    ## ============= Part 4: Visualizing J(theta_0, theta_1) =============
    print('Visualizing J(theta_0, theta_1) ...\n')

    # Grid over which we will calculate J
    theta0_vals = np.linspace(-10, 10, 100);
    theta1_vals = np.linspace(-1, 4, 100);

    # initialize J_vals to a matrix of 0's
    J_vals = np.zeros([size(theta0_vals), size(theta1_vals)]);

    # Fill out J_vals
    for i in range(0,size(theta0_vals)):
        for j in range (0, size(theta1_vals)):
            t = np.array([[theta0_vals[i]], [theta1_vals[j]]])
            J_vals[i,j] = computeCost(X, y, t)


    # Because of the way meshgrids work in the surf command, we need to
    # transpose J_vals before calling surf, or else the axes will be flipped
    J_vals = J_vals.T;
    # Surface plot
    fig2 = plt.figure()
    ax2 = fig2.add_subplot(1,1,1,projection='3d')
    ax2.plot_surface(theta0_vals, theta1_vals, J_vals)
    ax2.set_xlabel('\theta_0'); ax2.set_ylabel('\theta_1');

    # Contour plot
    fig3, ax3 = plt.subplots();
    # Plot J_vals as 15 contours spaced logarithmically between 0.01 and 100
    ax3.contour(theta0_vals, theta1_vals, J_vals, logspace(-2, 3, 20))
    ax3.set_xlabel('\theta_0'); ax3.set_ylabel('\theta_1');
    ax3.plot(theta[0], theta[1], 'rx');
    figPause(fig3)
##################################
main()