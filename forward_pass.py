# Import necessary libraries for the project
import numpy as np
from activation_functions import relu, softmax, tanh


def single_step_forward_class(X, W, activation='relu'):
    """
    Perform a single forward step in a neural network layer with specified activation function.

    Parameters:
        X (numpy.ndarray): Input features, where each column represents a data point.
        W (numpy.ndarray): Weights matrix for the layer.
        activation (str): Activation function to use ('relu' or 'tanh'). Default is 'relu'.

    Returns:
        numpy.ndarray: The output of the layer after applying the activation function.
    """

    activation_func = relu if activation == 'relu' else tanh

    Z = W.dot(X)
    A = activation_func(Z)
    return A

def single_step_forward_res(X, W1, W2, activation='relu'):
    """
    Perform a single forward step in a neural network layer with residual connections.

    Parameters:
        X (numpy.ndarray): Input features, where each column represents a data point.
        W1 (numpy.ndarray): Weights matrix for the first layer in the residual block.
        W2 (numpy.ndarray): Weights matrix for the second layer in the residual block.
        activation (str): Activation function to use ('relu' or 'tanh'). Default is 'relu'.

    Returns:
        numpy.ndarray: The output of the layer after applying the residual connection.
    """

    activation_func = relu if activation == 'relu' else tanh

    #step 1 :
    Z_res1 = W1.dot(X)
    A_res1 = activation_func(Z_res1)

    #step 2 :
    Z_res2 = W2.dot(A_res1)

    A = X[:-1, :]

    Z_res2 += A

    return Z_res2

def forward_pass(network, X, activation='relu', res_places=[]):
    """
    Perform a forward pass through a neural network with standard and ResNet layers.

    Parameters:
        network (dict): Dictionary containing the network's weight matrices.
        X (numpy.ndarray): Input features, where each column represents a data point.
        activation (str): Activation function to use ('relu' or 'tanh'). Default is 'relu'.
        res_places (list of int): List specifying the indices of layers where residual connections are placed.

    Returns:
        tuple: The output of the network after the forward pass and a cache containing the activations for each layer.
    """

    activation_func = relu if activation == 'relu' else tanh
    cache = {'A0': X}
    A = X
    L = len(network) - (len(res_places))
    for i in range(1, L):
        # bias
        A_aug = np.vstack([A, np.ones((1, A.shape[1]))])
        # resNet
        if i-1 in res_places:
            W1 = network['W' + str(i)+'_res1']
            W2 = network['W' + str(i)+'_res2']
            A = single_step_forward_res(A_aug, W1, W2, activation)

        # classic
        else:
            W = network['W' + str(i)]
            A = single_step_forward_class(A_aug, W , activation)

        # caching
        cache['A' + str(i)] = A

    #last layer
    A_aug = np.vstack([A, np.ones((1, A.shape[1]))])
    A = softmax(network['W' + str(L)].dot(A_aug))
    cache['A' + str(L)] = A

    return A, cache
