# Machine Learning Online Class - Exercise 3 | Part 1: One-vs-all

#  Instructions
#  ------------
# 
#  This file contains code that helps you get started on the
#  linear exercise. You will need to complete the following functions 
#  in this exericse:
#
#     lrCostFunction.m (logistic regression cost function)
#     oneVsAll.m
#     predictOneVsAll.m
#     predict.m
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
        plt.pause(0.0001)
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
################## displayData ########################
def displayData(X, example_width=None):
    #DISPLAYDATA Display 2D data in a nice grid
    #   [h, display_array] = DISPLAYDATA(X, example_width) displays 2D data
    #   stored in X in a nice grid. It returns the figure handle h and the
    #   displayed array if requested.

    fig, ax = plt.subplots() # open a new figure window
    # Set example_width automatically if not passed in
    if example_width==None:
        example_width = round(np.sqrt(X.shape[1]))

    # Gray Image
    #colormap(gray)

    # Compute rows, cols
    m, n = X.shape
    example_height = (n / example_width)

    # Compute number of items to display
    display_rows = np.floor(np.sqrt(m))
    display_cols = np.ceil(m / display_rows)

    # Between images padding
    pad = 1
    # Setup blank display
    display_array = - np.ones([pad + display_rows * (example_height + pad), pad + display_cols * (example_width + pad)])

    # Copy each example into a patch on the display array
    curr_ex = 0
    for j in range(int(display_rows)):
        for i in range(int(display_cols)):
            if curr_ex >= m:
                break
            # Copy the patch
		    # Get the max value of the patch
            max_val = np.max(np.abs(X[curr_ex, :]))
            ind1 = pad + j  * (example_height + pad)
            ind2 = pad + i  * (example_width + pad)
            img = np.reshape(X[curr_ex, :], [example_height, example_width]) / max_val
            display_array[ind1:ind1+example_height, ind2:ind2+example_width] = img.T
            curr_ex = curr_ex + 1
        if curr_ex >= m:
            break

    # Display Image
    h = ax.imshow(display_array, extent = [-1, 1, -1, 1], cmap='gray')

    # Do not show axis
    ax.set_frame_on(False)
    return fig, ax
############# sigmoid ###################
def sigmoid(z):
    #SIGMOID Compute sigmoid functoon
    #   J = SIGMOID(z) computes the sigmoid of z.

    # You need to return the following variables correctly
    g = np.zeros(z.shape)

    # ====================== YOUR CODE HERE ======================
    # Instructions: Compute the sigmoid of each value of z (z can be a matrix,
    #               vector or scalar).

    g = np.ones(z.shape)/(1+np.exp(-z))
    return g
################# lrCostFunction ##################################
def lrCostFunction(theta, X, y, iLambda):
    #LRCOSTFUNCTION Compute cost and gradient for logistic regression with
    #regularization
    #   J = LRCOSTFUNCTION(theta, X, y, lambda) computes the cost of using
    #   theta as the parameter for regularized logistic regression and the
    #   gradient of the cost w.r.t. to the parameters.

    # Initialize some useful values
    m = y.shape[0] # number of training examples
    theta = theta.reshape([size(theta), 1])

    # You need to return the following variables correctly
    J = 0

    # ====================== YOUR CODE HERE ======================
    # Instructions: Compute the cost of a particular choice of theta.
    #               You should set J to the cost.
    #               Compute the partial derivatives and set grad to the partial
    #               derivatives of the cost w.r.t. each parameter in theta
    #
    # Hint: The computation of the cost function and gradients can be
    #       efficiently vectorized. For example, consider the computation
    #
    #           sigmoid(X * theta)
    #
    #       Each row of the resulting matrix will contain the value of the
    #       prediction for that example. You can make use of this to vectorize
    #       the cost function and gradient computations.
    #
    # Hint: When computing the gradient of the regularized cost function,
    #       there're many possible vectorized solutions, but one solution
    #       looks like:
    #           grad = (unregularized gradient for logistic regression)
    #           temp = theta;
    #           temp(1) = 0;   % because we don't add anything for j = 0
    #           grad = grad + YOUR_CODE_HERE (using the temp variable)
    #
    penalizedThetas = np.hstack([np.zeros([1,1]), theta[1:].T])

    sm = sigmoid(np.dot(X, theta))
    sm[sm==1] = 0.9999
    J = (1./m)*np.sum(-np.dot(y.T, np.log(sm)) - np.dot((1-y.T), (np.log(1-sm)))) + \
        (iLambda/(2*m))*np.sum(np.dot(penalizedThetas, penalizedThetas.T))
    return J

def lrDer(theta, X, y, iLambda):
    m = y.shape[0] # number of training examples
    theta = theta.reshape([size(theta), 1])

    penalizedThetas = np.hstack([np.zeros([1,1]), theta[1:].T])
    grad = (1./m)*np.sum((sigmoid(np.dot(X, theta))-y)*X, axis=0) + (iLambda/m)*penalizedThetas
    grad = grad.flatten()
    return grad

def fixedXylrCost(X, y, iLambda):
    def decoratedCost(theta):
        return lrCostFunction(theta, X, y, iLambda)
    return decoratedCost
def fixedXylrDer(X, y, iLambda):
    def decoratedDer(theta):
        return lrDer(theta, X, y, iLambda)
    return decoratedDer
################# oneVsAll ########################################
def oneVsAll(X, y, num_labels, iLambda):
    #ONEVSALL trains multiple logistic regression classifiers and returns all
    #the classifiers in a matrix all_theta, where the i-th row of all_theta
    #corresponds to the classifier for label i
    #   [all_theta] = ONEVSALL(X, y, num_labels, lambda) trains num_labels
    #   logisitc regression classifiers and returns each of these classifiers
    #   in a matrix all_theta, where the i-th row of all_theta corresponds
    #   to the classifier for label i

    # Some useful variables
    m ,n = X.shape

    # You need to return the following variables correctly
    all_theta = np.zeros([num_labels, n + 1])

    # Add ones to the X data matrix
    X = np.hstack([np.ones([m, 1]), X])

    # ====================== YOUR CODE HERE ======================
    # Instructions: You should complete the following code to train num_labels
    #               logistic regression classifiers with regularization
    #               parameter lambda.
    #
    # Hint: theta(:) will return a column vector.
    #
    # Hint: You can use y == c to obtain a vector of 1's and 0's that tell use
    #       whether the ground truth is true/false for this class.
    #
    # Note: For this assignment, we recommend using fmincg to optimize the cost
    #       function. It is okay to use a for-loop (for c = 1:num_labels) to
    #       loop over the different classes.
    #
    #       fmincg works similarly to fminunc, but is more efficient when we
    #       are dealing with large number of parameters.
    #
    # Example Code for fmincg:
    #
    #     % Set Initial theta
    #     initial_theta = zeros(n + 1, 1);
    #
    #     % Set options for fminunc
    #     options = optimset('GradObj', 'on', 'MaxIter', 50);
    #
    #     % Run fmincg to obtain the optimal theta
    #     % This function will return theta and the cost
    #     [theta] = ...
    #         fmincg (@(t)(lrCostFunction(t, X, (y == c), lambda)), ...
    #                 initial_theta, options);
    #
    for c in range(num_labels):
        initial_theta = np.zeros([n + 1, 1]);
        # Set options for fminunc
        #options = optimset('GradObj', 'on', 'MaxIter', 50);
        # Run fmincg to obtain the optimal theta
        # This function will return theta and the cost
        #[theta] = fmincg (@(t)(lrCostFunction(t, X, (y == c), lambda)), initial_theta, options);
        opts = {'maxiter':100, 'disp': False}
        currY = (y==(c+1)).astype(float)
        optRes = opt.minimize(fixedXylrCost(X,currY, iLambda), initial_theta, method='TNC', jac=fixedXylrDer(X, currY, iLambda), options=opts)
        #optRes = opt.minimize(fixedXylrCost(X,currY, iLambda), initial_theta, method='nelder-mead', options=opts)
        print optRes.fun
        all_theta[c,:] = optRes.x
    return all_theta

###################### predictOneVsAll ############################
def predictOneVsAll(all_theta, X):
    #PREDICT Predict the label for a trained one-vs-all classifier. The labels
    #are in the range 1..K, where K = size(all_theta, 1).
    #  p = PREDICTONEVSALL(all_theta, X) will return a vector of predictions
    #  for each example in the matrix X. Note that X contains the examples in
    #  rows. all_theta is a matrix where the i-th row is a trained logistic
    #  regression theta vector for the i-th class. You should set p to a vector
    #  of values from 1..K (e.g., p = [1; 3; 1; 2] predicts classes 1, 3, 1, 2
    #  for 4 examples)

    m = X.shape[0]
    num_labels = all_theta.shape[0]

    # You need to return the following variables correctly
    p = np.zeros([m, 1])

    # Add ones to the X data matrix
    X = np.hstack([np.ones([m, 1]), X])

    # ====================== YOUR CODE HERE ======================
    # Instructions: Complete the following code to make predictions using
    #               your learned logistic regression parameters (one-vs-all).
    #               You should set p to a vector of predictions (from 1 to
    #               num_labels).
    #
    # Hint: This code can be done all vectorized using the max function.
    #       In particular, the max function can also return the index of the
    #       max element, for more information see 'help max'. If your examples
    #       are in rows, then, you can use max(A, [], 2) to obtain the max
    #       for each row.
    #
    p = np.argmax(sigmoid(np.dot(all_theta,X.T)), axis=0)+1
    p = p.T
    return p
################ predict NN #####################
def predict(Theta1, Theta2, X):
    #PREDICT Predict the label of an input given a trained neural network
    #   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
    #   trained weights of a neural network (Theta1, Theta2)

    # Useful values
    m = X.shape[0]
    num_labels = Theta2.shape[0]

    # You need to return the following variables correctly
    p = np.zeros([m, 1])

    # ====================== YOUR CODE HERE ======================
    # Instructions: Complete the following code to make predictions using
    #               your learned neural network. You should set p to a
    #               vector containing labels between 1 to num_labels.
    #
    # Hint: The max function might come in useful. In particular, the max
    #       function can also return the index of the max element, for more
    #       information see 'help max'. If your examples are in rows, then, you
    #       can use max(A, [], 2) to obtain the max for each row.
    #
    a1 = np.hstack([np.ones([m, 1]), X])
    z2 = np.dot(Theta1, a1.T)
    a2 = np.vstack([np.ones([1,m]), sigmoid(z2)])
    z3 = np.dot(Theta2, a2)
    a3 = sigmoid(z3)
    p = np.argmax(a3, axis=0)+1
    return p.reshape([m,1])
###################### main ########################################
# Setup the parameters you will use for this part of the exercise
def ex3():
    input_layer_size  = 400  # 20x20 Input Images of Digits
    num_labels = 10          # 10 labels, from 1 to 10
                             # (note that we have mapped "0" to label 10)

    # =========== Part 1: Loading and Visualizing Data =============
    #  We start the exercise by first loading and visualizing the dataset.
    #  You will be working with a dataset that contains handwritten digits.
    #

    # Load Training Data
    print('Loading and Visualizing Data ...\n')
    datadict = io.loadmat(r'C:\Users\ohadassi\xxx\ML-AndrewNG\Week4\machine-learning-ex3\ex3\ex3data1.mat');
    X = datadict['X']
    y = datadict['y']
    m = X.shape[0]

    # Randomly select 100 data points to display
    rand_indices = np.random.permutation(range(m))
    sel = X[rand_indices[1:100], :]

    fig1, ax1 = displayData(sel)

    figPause(fig1)

    # ============ Part 2: Vectorize Logistic Regression ============
    #  In this part of the exercise, you will reuse your logistic regression
    #  code from the last exercise. You task here is to make sure that your
    #  regularized logistic regression implementation is vectorized. After
    #  that, you will implement one-vs-all classification for the handwritten
    #  digit dataset.
    #

    print('\nTraining One-vs-All Logistic Regression...\n')

    Lambda = 0.1
    all_theta = oneVsAll(X, y, num_labels, Lambda)

    raw_input('Program paused. Press enter to continue.\n')

    # ================ Part 3: Predict for One-Vs-All ================
    #  After ...
    pred = predictOneVsAll(all_theta, X);

    acc = np.mean((pred[:,newaxis] == y).astype(float)) * 100
    print('\nTraining Set Accuracy: %f\n' % acc)
###################### main ########################################
# Setup the parameters you will use for this part of the exercise
def ex3_nn():
    ## Setup the parameters you will use for this exercise
    input_layer_size  = 400  # 20x20 Input Images of Digits
    hidden_layer_size = 25   # 25 hidden units
    num_labels = 10#         # 10 labels, from 1 to 10
                             # (note that we have mapped "0" to label 10)

    ## =========== Part 1: Loading and Visualizing Data =============
    #  We start the exercise by first loading and visualizing the dataset.
    #  You will be working with a dataset that contains handwritten digits.
    #

    # Load Training Data
    print('Loading and Visualizing Data ...\n')
    datadict = io.loadmat(r'C:\Users\ohadassi\xxx\ML-AndrewNG\Week4\machine-learning-ex3\ex3\ex3data1.mat');
    X = datadict['X']
    y = datadict['y']
    m = X.shape[0]
    n = X.shape[1]

    # Randomly select 100 data points to display
    rand_indices = np.random.permutation(range(m))
    sel = X[rand_indices[1:100], :]

    fig1, ax1 = displayData(sel)

    figPause(fig1)

    ## ================ Part 2: Loading Pameters ================
    # In this part of the exercise, we load some pre-initialized
    # neural network parameters.

    print('\nLoading Saved Neural Network Parameters ...\n')

    # Load the weights into variables Theta1 and Theta2
    datadict = io.loadmat(r'C:\Users\ohadassi\xxx\ML-AndrewNG\Week4\machine-learning-ex3\ex3\ex3weights.mat');
    Theta1 = datadict['Theta1']
    Theta2 = datadict['Theta2']

    ## ================= Part 3: Implement Predict =================
    #  After training the neural network, we would like to use it to predict
    #  the labels. You will now implement the "predict" function to use the
    #  neural network to predict the labels of the training set. This lets
    #  you compute the training set accuracy.

    pred = predict(Theta1, Theta2, X)

    acc = mean((pred == y).astype(float)) * 100
    print('\nTraining Set Accuracy: %f\n' % acc)
    raw_input('Program paused. Press enter to continue.\n');

    #  To give you an idea of the network's output, you can also run
    #  through the examples one at the a time to see what it is predicting.

    #  Randomly permute examples
    rp = np.random.permutation(range(m))

    for i in rp:
        # Display
        print('\nDisplaying Example Image\n')
        singleDigit = X[i,:].reshape([1,n])
        fig2, ax2 = displayData(singleDigit)

        pred = predict(Theta1, Theta2, singleDigit)[0]
        print('\nNeural Network Prediction: %d (digit %d)\n' % (pred, mod(pred, 10)))
        ax2.set_title('NN Prediction: ' + str(pred) + ' (digit ' + str(mod(pred, 10)) +')')

        # Pause
        figPause(fig2)
        plt.close(fig2)
############################################################
#ex3()
ex3_nn()
