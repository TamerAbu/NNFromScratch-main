# Import necessary libraries for the project
import numpy as np
from activation_functions import relu
from gradient_computations import softmax_regression_gradient_by_W, softmax_regression_gradient_by_X
from jacobian_computations import JacobianT_W1_Res, JacobianT_W2_Res, JacobianT_X_Res, JacobianTByW, JacobianTByX


def backpropagation(gradients,network, cache, Y, res_places = [], activation='relu'):
    """
    Perform backpropagation through a neural network with standard and ResNet layers.

    Parameters:
        gradients (dict): Dictionary to store the gradients of the network's weights.
        network (dict): Dictionary containing the network's weight matrices.
        cache (dict): Dictionary containing the cached activations from the forward pass.
        Y (numpy.ndarray): True labels in a one-hot encoded format.
        res_places (list of int): List specifying the indices of layers where residual connections are placed.
        activation (str): Activation function to use ('relu' or 'tanh'). Default is 'relu'.

    Returns:
        dict: Updated dictionary of gradients for the network's weights.
    """

    activation_func = relu if activation == 'relu' else 'tanh'

    L = len(network) - (len(res_places))
    A = cache['A' + str(L - 1)]
    A = np.vstack([A, np.ones((1, A.shape[1]))])
    W_prime = softmax_regression_gradient_by_W(A ,network['W'+str(L)], Y)
    gradients['W'+str(L)] = W_prime
    dA =  softmax_regression_gradient_by_X(A ,network['W'+str(L)], Y)
    dA = dA[:-1,:]

    for i in reversed(range(1, L)):
        A = cache['A'+str(i-1)]
        A_aug = np.vstack([A, np.ones((1, A.shape[1]))])

        #resBack
        if i - 1 in res_places:
            W1 = network['W'+str(i)+'_res1']
            W2 = network['W'+str(i)+'_res2']

            W1_prime = JacobianT_W1_Res(A_aug, W1, W2, dA, activation)
            gradients['W'+str(i)+'_res1'] = W1_prime

            W2_prime = JacobianT_W2_Res(A_aug, W1, W2, dA, activation)
            gradients['W'+str(i)+'_res2'] = W2_prime

            dA = JacobianT_X_Res(A_aug, W1, W2, dA, activation)

        #clasicBack
        else:
            W = network['W'+str(i)]
            W_prime = JacobianTByW(A_aug, W, dA, activation)
            gradients['W'+str(i)] = W_prime
            dA = JacobianTByX(A_aug, W, dA, activation)
            dA = dA[:-1,:]

    return gradients