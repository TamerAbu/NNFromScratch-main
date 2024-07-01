# Import necessary libraries for the project


from matplotlib import pyplot as plt
import numpy as np
from activation_functions import compute_loss_for_testing, softmax
from data_handling import pickRandomBach
from data_handling import data_training_X,data_training_C



def softmax_regression_gradient_by_W(X ,W, C):
    """
    Compute the gradient of the loss with respect to weights W for softmax regression.

    Parameters:
        X (numpy.ndarray): Input features, where each column is a feature vector.
        W (numpy.ndarray): Weights matrix.
        C (numpy.ndarray): True labels in one-hot encoded form.

    Returns:
        numpy.ndarray: Gradient of the loss with respect to W.
    """
    Z = W.dot(X)
    m = Z.shape[1]
    probabilities = softmax(Z)
    probabilities -= C
    gradient = np.dot(probabilities, X.T) / m
    return gradient

def softmax_regression_gradient_by_X(X, W, C):
    """
    Compute the gradient of the loss with respect to input features X for softmax regression.

    Parameters:
        X (numpy.ndarray): Input features, where each column is a feature vector.
        W (numpy.ndarray): Weights matrix.
        C (numpy.ndarray): True labels in one-hot encoded form.

    Returns:
        numpy.ndarray: Gradient of the loss with respect to X.
    """
    m = X.shape[1]
    A = np.dot(W, X)
    prob = softmax(A)
    P = prob - C
    grad_X = np.dot(W.T, P) / m
    return grad_X

def least_squares_loss(X, W, C):
    """
    Calculate the least squares loss for regression models.

    Parameters:
        X (numpy.ndarray): Input features, where each column is a feature vector.
        W (numpy.ndarray): Weights matrix.
        C (numpy.ndarray): True labels or target values.

    Returns:
        float: Computed least squares loss.
    """

    # Calculate the least squares loss
    N = len(C)
    loss = (1/N) * np.sum((np.dot(W, X) - C)**2)
    return loss

def least_squares_gradient(X, W, C):
    """
    Calculate the gradient of the least squares loss with respect to the weights matrix W.

    Parameters:
        X (numpy.ndarray): Input features.
        W (numpy.ndarray): Weights matrix.
        C (numpy.ndarray): True labels or target values.

    Returns:
        numpy.ndarray: Gradient of the least squares loss with respect to W.
    """
        
    # Calculate the gradient for least squares regression
    N = len(C)
    gradient = (2/N) * np.dot((np.dot(W, X) - C),X.T)
    return gradient




#                                                                   Gradient tests :
# to test each derivative we use a G-test 

def gradient_test_byW(function, gradient):
    """
    Perform a gradient check for the softmax regression with respect to weights W.

    Parameters:
        function (callable): Loss function that takes logits and true labels and computes the loss.
        gradient (callable): Function to compute the gradient of the loss with respect to W.

    Outputs:
        Plots the gradient checking results, comparing numerical and analytical gradients.
    """

    # Gradient Check Setup
    batchSize = 2
    EPSILON = 0.1
    outputnum = 2

    X , C =pickRandomBach(data_training_X,data_training_C,batchSize)
    X = np.vstack([X, np.ones((1, X.shape[1]))])

    W = np.random.randn(outputnum, X.shape[0])


    d = np.random.randn(*W.shape)

    def F(W):
        Z = W.dot(X)
        return function(Z, C)

    def g_F(W):
        return gradient(X, W, C)


    F0 = F(W)
    g0 = g_F(W)
    y0 = np.zeros(8)
    y1 = np.zeros(8)

    print("k\terror order 1\t\terror order 2")

    for k in range(1, 9):
        epsk = EPSILON * (0.5 ** k)
        Fk = F(W + epsk * d)
        F1 = F0 + epsk * np.sum(g0 * d)
        y0[k-1] = abs(Fk - F0)
        y1[k-1] = abs(Fk - F1)
        print(f"{k}\t{y0[k-1]}\t{y1[k-1]}")

    # Plotting
    plt.semilogy(range(1, 9), y0, label="Zero order approx")
    plt.semilogy(range(1, 9), y1, label="First order approx")
    plt.legend()
    plt.title("Gradient Test for Softmax Regression By W")
    plt.xlabel("k")
    plt.ylabel("Error")
    plt.show()

def gradient_test_byX(function, gradient):
    """
    Perform a gradient check for the softmax regression with respect to input features X.

    Parameters:
        function (callable): Loss function that takes logits and true labels and computes the loss.
        gradient (callable): Function to compute the gradient of the loss with respect to X.

    Outputs:
        Plots the gradient checking results, comparing numerical and analytical gradients.
    """
    # Gradient Check Setup
    batchSize = 2
    EPSILON = 0.1
    outputnum = 2

    X , C =pickRandomBach(data_training_X,data_training_C,batchSize)

    W1 = np.random.randn(2, 2)

    d = np.random.randn(*X.shape)

    def F(X):
        Z = W1.dot(X)
        return function(Z , C)

    def g_F(X):
        return gradient( X , W1 , C)


    F0 = F(X)
    g0 = g_F(X)
    y0 = np.zeros(8)
    y1 = np.zeros(8)

    print("k\terror order 1\t\terror order 2")

    for k in range(1, 9):
        epsk = EPSILON * (0.5 ** k)
        Fk = F(X + epsk * d)
        F1 = F0 + epsk * np.sum(g0 * d)
        y0[k-1] = abs(Fk - F0)
        y1[k-1] = abs(Fk - F1)
        print(f"{k}\t{y0[k-1]}\t{y1[k-1]}")

    # Plotting
    plt.semilogy(range(1, 9), y0, label="Zero order approx")
    plt.semilogy(range(1, 9), y1, label="First order approx")
    plt.legend()
    plt.title("Gradient Test for Softmax Regression By X")
    plt.xlabel("k")
    plt.ylabel("Error")
    plt.show()

def gradient_test_LS(function, gradient):
    """
    Perform a gradient check for the least squares loss function using finite differences.

    Parameters:
        function (callable): The least squares loss function.
        gradient (callable): The gradient function for the least squares loss.

    Outputs:
        Plots the gradient checking results, comparing numerical and analytical gradients.
    """

    # Gradient Check Setup
    batchSize = 2
    EPSILON = 0.1
    outputnum = 2

    X , C =pickRandomBach(data_training_X,data_training_C,batchSize)
    X = np.vstack([X, np.ones((1, X.shape[1]))])
    W = np.random.randn(outputnum, X.shape[0])
    d = np.random.randn(*W.shape)

    def F(W):
        return function(X, W, C)

    def g_F(W):
        return gradient(X, W, C)

    F0 = F(W)
    g0 = g_F(W)
    y0 = np.zeros(8)
    y1 = np.zeros(8)

    print("k\terror order 1\t\terror order 2")

    for k in range(1, 9):
        epsk = EPSILON * (0.5 ** k)
        Fk = F(W + epsk * d)
        F1 = F0 + epsk * np.sum(g0 * d)
        y0[k-1] = abs(Fk - F0)
        y1[k-1] = abs(Fk - F1)
        print(f"{k}\t{y0[k-1]}\t{y1[k-1]}")

    # Plotting
    plt.semilogy(range(1, 9), y0, label="Zero order approx")
    plt.semilogy(range(1, 9), y1, label="First order approx")
    plt.legend()
    plt.title("Gradient Test for Least Squares")
    plt.xlabel("k")
    plt.ylabel("Error")
    plt.show()



#                                                                              testing Gradients 
# To test the gradient function, please remove the '#' before the one you wish to test :
# gradient_test_byW(compute_loss_for_testing,softmax_regression_gradient_by_W)
# gradient_test_byX(compute_loss_for_testing,softmax_regression_gradient_by_X)
# gradient_test_LS(least_squares_loss, least_squares_gradient)
