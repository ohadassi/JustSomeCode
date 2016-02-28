# Machine Learning Online Class - Exercise 2: Logistic Regression
#
#  Instructions
#  ------------
# 
#  This file contains code that helps you get started on the logistic
#  regression exercise. You will need to complete the following functions 
#  in this exericse:
#
#     sigmoid.m
#     costFunction.m
#     predict.m
#     costFunctionReg.m
#
#  For this exercise, you will not need to change any code in this file,
#  or any other files other than those mentioned above.
#

import os
os.environ['PATH'] = r'C:\Anaconda2\Library\bin;'+os.environ['PATH']

from pylab import *
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
from mpl_toolkits.mplot3d.axes3d import Axes3D
import scipy.optimize as opt
import sklearn as skl
from sklearn import linear_model
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
        plt.pause(0.1)
        plt.draw()
    button.disconnect_events()
    plt.delaxes(buttonAxes)
    plt.draw()
    #fig.canvas.mpl_connect('key_press_event', onKeyPress)
    #global pause
    #pause = True
    #while pause:
    #    plt.pause(0.2)
    #    #ax.canvas.get_tk_widget().update() # process events
    #    plt.draw()
################## plotData ########################
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

    posInd = np.where(y==1);
    ax.plot(x[posInd,0], x[posInd,1], 'k+')
    negInd = np.where(y==0);
    ax.plot(x[negInd,0], x[negInd,1], 'ko');
    return fig, ax
############# sigmoid ###################
def sigmoid(z):
    #SIGMOID Compute sigmoid functoon
    #   J = SIGMOID(z) computes the sigmoid of z.

    # You need to return the following variables correctly
    g = np.zeros(z.shape);

    # ====================== YOUR CODE HERE ======================
    # Instructions: Compute the sigmoid of each value of z (z can be a matrix,
    #               vector or scalar).

    g = np.ones(z.shape)/(1+np.exp(-z));
    return g
#################### costFunction #############################
def costFunction(theta, X, y):
    #COSTFUNCTION Compute cost and gradient for logistic regression
    #   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
    #   parameter for logistic regression and the gradient of the cost
    #   w.r.t. to the parameters.

    # Initialize some useful values
    m = size(y) # number of training examples

    # You need to return the following variables correctly
    J = 0
    theta = theta.reshape([size(theta), 1])

    #====================== YOUR CODE HERE ======================
    # Instructions: Compute the cost of a particular choice of theta.
    #               You should set J to the cost.
    #               Compute the partial derivatives and set grad to the partial
    #               derivatives of the cost w.r.t. each parameter in theta
    #
    # Note: grad should have the same dimensions as theta
    #
    sm = sigmoid(np.dot(X, theta))
    sm[sm==1] = .9999
    J = 1./m*np.sum(-np.dot(y.T, np.log(sm)) - np.dot((1-y.T), (np.log(1-sm))))

    return J
def derFunction(theta, X, y):
    #COSTFUNCTION Compute cost and gradient for logistic regression
    #   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
    #   parameter for logistic regression and the gradient of the cost
    #   w.r.t. to the parameters.

    # Initialize some useful values
    m = size(y) # number of training examples

    # You need to return the following variables correctly
    J = 0
    theta = theta.reshape([size(theta), 1])

    #====================== YOUR CODE HERE ======================
    # Instructions: Compute the cost of a particular choice of theta.
    #               You should set J to the cost.
    #               Compute the partial derivatives and set grad to the partial
    #               derivatives of the cost w.r.t. each parameter in theta
    #
    # Note: grad should have the same dimensions as theta
    #

    grad = (1./m)*np.sum((sigmoid(np.dot(X, theta))-y)*X, axis=0)
    return grad

def fixedXyCost(X, y):
    def decoratedCost(theta):
        return costFunction(theta, X, y)
    return decoratedCost
def fixedXyDer(X, y):
    def decoratedDer(theta):
        return derFunction(theta, X, y)
    return decoratedDer


################# plotDecisionBoundary ###############
def mapFeature(X1, X2):
    # MAPFEATURE Feature mapping function to polynomial features
    #
    #   MAPFEATURE(X1, X2) maps the two input features
    #   to quadratic features used in the regularization exercise.
    #
    #   Returns a new feature array with more features, comprising of
    #   X1, X2, X1.^2, X2.^2, X1*X2, X1*X2.^2, etc..
    #
    #   Inputs X1, X2 must be the same size
    #
    X1 = X1.reshape([size(X1), 1])
    X2 = X2.reshape([size(X2), 1])
    degree = 6
    out = np.ones(X1.shape)
    for i in range(1, degree+1):
        for j in range(0, i+1):
            out = hstack([out, (X1**(i-j))*(X2**j)]);
    return out
################# plotDecisionBoundary ###############
def plotDecisionBoundary(theta, X, y):
    #PLOTDECISIONBOUNDARY Plots the data points X and y into a new figure with
    #the decision boundary defined by theta
    #   PLOTDECISIONBOUNDARY(theta, X,y) plots the data points with + for the
    #   positive examples and o for the negative examples. X is assumed to be
    #   a either
    #   1) Mx3 matrix, where the first column is an all-ones column for the
    #      intercept.
    #   2) MxN, N>3 matrix, where the first column is all-ones

    # Plot Data
    fig, ax = plotData(X[:,1:3], y)

    if X.shape[1] <= 3:
        # Only need 2 points to define a line, so choose two endpoints
        plot_x = np.array([np.min(X[:,1])-2,  np.max(X[:,1])+2]);
        # Calculate the decision boundary line
        plot_y = (-1./theta[2])*(theta[1]*plot_x + theta[0]);
        # Plot, and adjust axes for better viewing
        ax.plot(plot_x, plot_y, 'b-')

        # Legend, specific for the exercise
        ax.legend(['Admitted', 'Not admitted', 'Decision Boundary'])
        ax.axis([30, 100, 30, 100])
    else:
        # Here is the grid range
        u = linspace(-1, 1.5, 50)
        v = linspace(-1, 1.5, 50)

        z = np.zeros([size(u), size(v)])
        # Evaluate z = theta*x over the grid
        for i in range(0, size(u)):
            for j in range(0, size(v)):
                z[i,j] = np.dot(mapFeature(u[i], v[j]), theta)
        z = z.T # important to transpose z before calling contour

        # Plot z = 0
        # Notice you need to specify the range [0, 0]
        ax.contour(u, v, z, [0, 0])
    return fig, ax
################ predict ####################
def  predict(theta, X):
    #PREDICT Predict whether the label is 0 or 1 using learned logistic
    #regression parameters theta
    #   p = PREDICT(theta, X) computes the predictions for X using a
    #   threshold at 0.5 (i.e., if sigmoid(theta'*x) >= 0.5, predict 1)

    m = X.shape[0]# Number of training examples

    # You need to return the following variables correctly
    p = np.zeros([m, 1])

    # ====================== YOUR CODE HERE ======================
    # Instructions: Complete the following code to make predictions using
    #               your learned logistic regression parameters.
    #               You should set p to a vector of 0's and 1's
    #
    p = sigmoid(np.dot(X, theta)) > .5
    return p.reshape((m,1))
################## costFunctionReg ####################
def costFunctionReg(theta, X, y, iLambda):
    #COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
    #   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
    #   theta as the parameter for regularized logistic regression and the
    #   gradient of the cost w.r.t. to the parameters.

    # Initialize some useful values
    m = y.shape[0] # number of training examples
    theta = theta.reshape([size(theta), 1])
    # You need to return the following variables correctly
    J = 0
    grad = np.zeros(theta.shape)

    # ====================== YOUR CODE HERE ======================
    # Instructions: Compute the cost of a particular choice of theta.
    #               You should set J to the cost.
    #               Compute the partial derivatives and set grad to the partial
    #               derivatives of the cost w.r.t. each parameter in theta


    penalizedThetas = np.hstack([np.zeros([1,1]), theta[1:].T])
    sm = sigmoid(np.dot(X, theta))
    J = (1./m)*np.sum(-np.dot(y.T, np.log(sm)) - np.dot((1-y.T), (np.log(1-sm)))) + \
        (iLambda/(2*m))*np.sum(np.dot(penalizedThetas, penalizedThetas.T))

    return J
def costFunctionRegDer(theta, X, y, iLambda):
    m = y.shape[0] # number of training examples
    theta = theta.reshape([size(theta), 1])

    penalizedThetas = np.hstack([np.zeros([1,1]), theta[1:].T])
    grad = (1./m)*np.sum((sigmoid(np.dot(X, theta))-y)*X, axis=0) + (iLambda/m)*penalizedThetas
    grad = grad.flatten()
    return grad
def fixedXylrCostReg(X, y, iLambda):
    def decoratedCost(theta):
        return costFunctionReg(theta, X, y, iLambda)
    return decoratedCost
def fixedXylrDerReg(X, y, iLambda):
    def decoratedDer(theta):
        return costFunctionRegDer(theta, X, y, iLambda)
    return decoratedDer
#############################################################
def ex2():
    ##==================================================###
    #  Load Data
    #  The first two columns contains the exam scores and the third column
    #  contains the label.

    data = np.genfromtxt(r'C:\Users\ohadassi\xxx\ML-AndrewNG\Week3\machine-learning-ex2\ex2\ex2data1.txt', delimiter=',')
    X = data[:, [0, 1]]
    y = data[:, 2]
    y = y[:,newaxis]

    # ==================== Part 1: Plotting ====================
    #  We start the exercise by first plotting the data to understand the
    #  the problem we are working with.

    print('Plotting data with + indicating (y = 1) examples and o indicating (y = 0) examples.')

    fig1, ax1 = plotData(X, y);

    # Put some labels
    # Labels and Legend
    ax1.set_xlabel('Exam 1 score')
    ax1.set_ylabel('Exam 2 score')

    # Specified in plot order
    ax1.legend()
    figPause(fig1)

    # ============ Part 2: Compute Cost and Gradient ============
    #  In this part of the exercise, you will implement the cost and gradient
    #  for logistic regression. You neeed to complete the code in
    #  costFunction.m

    #  Setup the data matrix appropriately, and add ones for the intercept term
    m, n = X.shape

    # Add intercept term to x and X_test
    X = np.hstack([np.ones([m, 1]), X])

    # Initialize fitting parameters
    initial_theta = np.zeros([n + 1, 1])

    # Compute and display initial cost and gradient
    cost = costFunction(initial_theta, X, y)
    grad = derFunction(initial_theta, X, y)

    print('Cost at initial theta (zeros): %f' % cost)
    print('Gradient at initial theta (zeros): ')
    print(grad)

    figPause(fig1)

    # ============= Part 3: Optimizing using fminunc  =============
    #  In this exercise, you will use a built-in function (fminunc) to find the
    #  optimal parameters theta.

    opts = {'maxiter':400, 'disp': True}
    #optRes = opt.minimize(fixedXyCost(X,y), initial_theta, method='nelder-mead', options=opts)
    optRes = opt.minimize(fixedXyCost(X,y), initial_theta, jac=fixedXyDer(X, y), method='BFGS', options=opts)
    #optRes = opt.fmin_bfgs(fixedXyCost(X,y), initial_theta, maxiter=400)
    # Print theta to screen
    theta = optRes.x
    cost = optRes.fun
    print('Cost at theta found by fminunc: %f\n' % optRes.fun)
    print('theta: \n')
    print(optRes.x)

    # Plot Boundary
    plotDecisionBoundary(theta, X, y);

    # Put some labels
    ## Labels and Legend
    ax1.set_xlabel('Exam 1 score')
    ax1.set_ylabel('Exam 2 score')

    # Specified in plot order
    ax1.legend(['Admitted', 'Not admitted'])

    figPause(fig1)

    # ============== Part 4: Predict and Accuracies ==============
    #  After learning the parameters, you'll like to use it to predict the outcomes
    #  on unseen data. In this part, you will use the logistic regression model
    #  to predict the probability that a student with score 45 on exam 1 and
    #  score 85 on exam 2 will be admitted.
    #
    #  Furthermore, you will compute the training and test set accuracies of
    #  our model.
    #
    #  Your task is to complete the code in predict.m

    #  Predict probability for a student with score 45 on exam 1
    #  and score 85 on exam 2

    prob = sigmoid(np.dot([1, 45, 85], theta))
    print('For a student with scores 45 and 85, we predict an admission probability of %f' % prob);
    ### Compute accuracy on our training set
    p = predict(theta, X);
    #
    acc = np.mean((p == y).astype(float)) * 100
    print('Train Accuracy: %f\n' % acc );

    raw_input('\nProgram paused. Press enter to continue.\n');

#######################################################
def ex2_scikitlearn():
    ## Machine Learning Online Class - Exercise 2: Logistic Regression
    #
    #  Instructions
    #  ------------
    #
    #  This file contains code that helps you get started on the second part
    #  of the exercise which covers regularization with logistic regression.
    #
    #  You will need to complete the following functions in this exericse:
    #
    #     sigmoid.m
    #     costFunction.m
    #     predict.m
    #     costFunctionReg.m
    #
    #  For this exercise, you will not need to change any code in this file,
    #  or any other files other than those mentioned above.
    #

    ## Initialization
    ## Load Data
    #  The first two columns contains the X values and the third column
    #  contains the label (y).

    data = np.genfromtxt(r'C:\Users\ohadassi\xxx\ML-AndrewNG\Week3\machine-learning-ex2\ex2\ex2data2.txt', delimiter=',')
    X = data[:, [0, 1]]
    y = data[:, 2]
    y = y[:,newaxis]

    fig1, ax1 = plotData(X, y)

    # Put some labels
    # Labels and Legend
    ax1.set_xlabel('Microchip Test 1')
    ax1.set_ylabel('Microchip Test 2')

    # Specified in plot order
    ax1.legend(['y = 1', 'y = 0'])


    # =========== Part 1: Regularized Logistic Regression ============
    #  In this part, you are given a dataset with data points that are not
    #  linearly separable. However, you would still like to use logistic
    #  regression to classify the data points.
    #
    #  To do so, you introduce more features to use -- in particular, you add
    #  polynomial features to our data matrix (similar to polynomial
    #  regression).
    #

    # Add Polynomial Features

    # Note that mapFeature also adds a column of ones for us, so the intercept
    # term is handled
    X = mapFeature(X[:,0], X[:,1])

    # Initialize fitting parameters
    initial_theta = np.zeros([X.shape[1], 1])

    # Set regularization parameter lambda to 1
    Lambda = 1

    # Compute and display initial cost and gradient for regularized logistic
    # regression
    cost = costFunctionReg(initial_theta, X, y, Lambda)

    print('Cost at initial theta (zeros): %f\n' % cost)

    figPause(fig1)

    ## ============= Part 2: Regularization and Accuracies =============
    #  Optional Exercise:
    #  In this part, you will get to try different values of lambda and
    #  see how regularization affects the decision coundart
    #
    #  Try the following values of lambda (0, 1, 10, 100).
    #
    #  How does the decision boundary change when you vary lambda? How does
    #  the training set accuracy vary?
    #

    # Initialize fitting parameters
    initial_theta = np.zeros([X.shape[1], 1])

    # Set regularization parameter lambda to 1 (you should vary this)
    Lambda = 1

    clf = linear_model.LogisticRegression(penalty='l2', C=1/Lambda, max_iter=400, solver='lbfgs', verbose=1, fit_intercept=False)
    clf_i = linear_model.LogisticRegression(penalty='l2', C=1/Lambda, max_iter=400, solver='lbfgs', verbose=1, fit_intercept=True)
    clf.fit(X, y.flatten())
    clf_i.fit(X, y.flatten())
    # Set Options
    opts = {'maxiter':40, 'disp': False}
    # Optimize
    optRes = opt.minimize(fixedXylrCostReg(X, y, Lambda), initial_theta, method='BFGS', jac=fixedXylrDerReg(X, y, Lambda), options=opts)
    theta = optRes.x
    skl_theta = clf.coef_[0]
    print('skl result: {0}'.format(skl_theta))
    print('skl intercept: {0}'.format(clf.intercept_))
    skl_theta_i = clf_i.coef_
    print('skl_i result: {0}'.format(skl_theta_i))
    print('skl_i intercept: {0}'.format(clf_i.intercept_))
    #print('skl iterations: {0}'.format(clf.n_iter_))
    print('minimize result: {0}'.format(theta))

    # Plot Boundary
    fig2, ax2 = plotDecisionBoundary(skl_theta, X, y)
    fig3, ax3 = plotDecisionBoundary(theta, X, y)
    ax2.set_title('lambda = ' + str(Lambda))

    # Labels and Legend
    ax2.set_xlabel('Microchip Test 1')
    ax2.set_ylabel('Microchip Test 2')

    ax2.legend(['y = 1', 'y = 0', 'Decision boundary'])

    # Compute accuracy on our training set
    p = predict(theta, X)
    p_skl = predict(skl_theta, X)


    acc = np.mean((p == y).astype(float)) * 100
    acc_skl = np.mean((p_skl == y).astype(float)) * 100
    interceptL = np.array([clf_i.intercept_])
    skl_theta_i = np.hstack([interceptL, skl_theta_i])
    skl_theta_i = skl_theta_i.reshape([len(skl_theta)+1, 1])
    X_i = np.hstack([np.ones([len(X),1]), X])

    p_skl_i = predict(skl_theta_i, X_i) #clf_i.predict(X_i)
    acc_skl_i = np.mean((p_skl_i == y).astype(float)) * 100
    p_skl_dblchk = clf.predict(X)
    acc_skl_dblchk = np.mean((p_skl_dblchk[:,newaxis] == y).astype(float)) * 100
    print('Train Accuracy: {}\n', acc)
    print('Train Accuracy (skl): {}\n', acc_skl)
    print('Train Accuracy (skl_dblchk): {}\n', acc_skl_dblchk)
    print('Train Accuracy (skl_i): {}\n', acc_skl_i)
    figPause(fig2)
def ex2_reg():
    ## Machine Learning Online Class - Exercise 2: Logistic Regression
    #
    #  Instructions
    #  ------------
    #
    #  This file contains code that helps you get started on the second part
    #  of the exercise which covers regularization with logistic regression.
    #
    #  You will need to complete the following functions in this exericse:
    #
    #     sigmoid.m
    #     costFunction.m
    #     predict.m
    #     costFunctionReg.m
    #
    #  For this exercise, you will not need to change any code in this file,
    #  or any other files other than those mentioned above.
    #

    ## Initialization
    ## Load Data
    #  The first two columns contains the X values and the third column
    #  contains the label (y).

    data = np.genfromtxt(r'C:\Users\ohadassi\xxx\ML-AndrewNG\Week3\machine-learning-ex2\ex2\ex2data2.txt', delimiter=',')
    X = data[:, [0, 1]]
    y = data[:, 2]
    y = y[:,newaxis]

    fig1, ax1 = plotData(X, y)

    # Put some labels
    # Labels and Legend
    ax1.set_xlabel('Microchip Test 1')
    ax1.set_ylabel('Microchip Test 2')

    # Specified in plot order
    ax1.legend(['y = 1', 'y = 0'])


    # =========== Part 1: Regularized Logistic Regression ============
    #  In this part, you are given a dataset with data points that are not
    #  linearly separable. However, you would still like to use logistic
    #  regression to classify the data points.
    #
    #  To do so, you introduce more features to use -- in particular, you add
    #  polynomial features to our data matrix (similar to polynomial
    #  regression).
    #

    # Add Polynomial Features

    # Note that mapFeature also adds a column of ones for us, so the intercept
    # term is handled
    X = mapFeature(X[:,0], X[:,1])

    # Initialize fitting parameters
    initial_theta = np.zeros([X.shape[1], 1])

    # Set regularization parameter lambda to 1
    Lambda = 1

    # Compute and display initial cost and gradient for regularized logistic
    # regression
    cost = costFunctionReg(initial_theta, X, y, Lambda)

    print('Cost at initial theta (zeros): %f\n' % cost)

    figPause(fig1)

    ## ============= Part 2: Regularization and Accuracies =============
    #  Optional Exercise:
    #  In this part, you will get to try different values of lambda and
    #  see how regularization affects the decision coundart
    #
    #  Try the following values of lambda (0, 1, 10, 100).
    #
    #  How does the decision boundary change when you vary lambda? How does
    #  the training set accuracy vary?
    #

    # Initialize fitting parameters
    initial_theta = np.zeros([X.shape[1], 1])

    # Set regularization parameter lambda to 1 (you should vary this)
    Lambda = 1;

    # Set Options
    opts = {'maxiter':4000, 'disp': False}
    # Optimize
    optRes = opt.minimize(fixedXylrCostReg(X, y, Lambda), initial_theta, method='BFGS', jac=fixedXylrDerReg(X, y, Lambda), options=opts)
    theta = optRes.x

    # Plot Boundary
    fig2, ax2 = plotDecisionBoundary(theta, X, y)
    ax2.set_title('lambda = ' + str(Lambda))

    # Labels and Legend
    ax2.set_xlabel('Microchip Test 1')
    ax2.set_ylabel('Microchip Test 2')

    ax2.legend(['y = 1', 'y = 0', 'Decision boundary'])

    # Compute accuracy on our training set
    p = predict(theta, X)

    acc = np.mean((p == y).astype(float)) * 100
    print('Train Accuracy: %f\n' % acc)
    figPause(fig2)
#####################################
#ex2()
#ex2_reg()
ex2_scikitlearn()