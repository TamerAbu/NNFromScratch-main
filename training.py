# Import necessary libraries for the project
from matplotlib import pyplot as plt
import numpy as np
from activation_functions import calculate_success_percentage, compute_loss
from backpropagation import backpropagation
from data_handling import pickRandomBach, shuffle_and_batch, data_training_X, data_training_C
from evaluation import data_validation_test, plot_training_results_extended
from forward_pass import forward_pass
from network_utils import initialize_network, res_indices
from optimization_methods import SGD_with_momentum

def Gradient_test_for_complete_network():
    """
    Perform a gradient check for the entire neural network, including both standard and ResNet layers.

    Outputs:
        Plots the gradient checking results, comparing numerical and analytical gradients.
    """

    # Gradient test SetUp
    layer_sizes = [2,4,4,4,4,4,4,2]
    res_places = [1,2,3,4,5]
    batchSize = 3
    gradients = {}
    EPSILON = 0.1
    network = initialize_network(layer_sizes, res_places)
    X , C =pickRandomBach(data_training_X,data_training_C,batchSize)

    # Generate a random vector d for each layer
    d = {layer: np.random.randn(network[layer].shape[0], network[layer].shape[1]) for layer in network}

    def F(network):
      A , cache = forward_pass(network, X, 'tanh', res_places)
      return  compute_loss(A,C) , cache

    def g_F(network , cache):
      return backpropagation(gradients, network, cache, C, res_places, 'tanh')

    F0, cache = F(network)
    g0 = g_F(network , cache)
    y0 = np.zeros(8)
    y1 = np.zeros(8)

    for k in range(1, 9):
        epsk = EPSILON * (0.5 ** k)
        # Perturb weights by epsk * d
        W_original = {layer: np.copy(network[layer]) for layer in network}
        for layer in network:
            network[layer] += epsk * d[layer]

        Fk,_ = F(network)
        temp = 0
        for key in g0:
            temp += np.sum(g0[key] * (epsk * d[key]))
        F1 = F0 + temp
        y0[k-1] = abs(Fk - F0)
        y1[k-1] = abs(Fk - F1)
        # Reset weights
        for layer in network:
            network[layer] = W_original[layer]

        print(f"{k}\t{y0[k-1]}\t{y1[k-1]}")
    # Plotting
    plt.semilogy(range(1, 9), y0, label="Zero order approx")
    plt.semilogy(range(1, 9), y1, label="First order approx")
    plt.legend()
    plt.title("Gradient Test for complete network")
    plt.xlabel("k")
    plt.ylabel("Error")
    plt.show()

#                                                                              testing all  
# Gradient_test_for_complete_network()

def train_network(X, Y, print_train_process, batchSize, network, res_places, num_epochs=1, learning_rate=0.05, momentum=0.9, activation='relu'):
    """
    Train a neural network with specified parameters, including ResNet layers and SGD with momentum.

    Parameters:
        X (numpy.ndarray): Input features for training.
        Y (numpy.ndarray): True labels for training.
        print_train_process (bool): Whether to print training progress.
        batchSize (int): Size of each mini-batch for training.
        network (dict): Dictionary containing the network's weight matrices.
        res_places (list of int): List specifying the indices of layers where residual connections are placed.
        num_epochs (int): Number of epochs for training.
        learning_rate (float): Learning rate for the optimizer.
        momentum (float): Momentum factor for the optimizer.
        activation (str): Activation function to use ('relu' or 'tanh'). Default is 'relu'.

    Returns:
        tuple: Updated network and lists of losses and success percentages per batch and per epoch.
    """

    lossesList = []
    successList = []
    avg_lossesList = []
    avg_successList = []
    for epoch in range(num_epochs):
        avg_lossesList_it = []
        avg_successList_it = []
        if print_train_process:
            print(f'Epoch {epoch + 1}')

        for X_batch, Y_batch in shuffle_and_batch(X, Y, batchSize):
            velocity = {key: np.zeros_like(val) for key, val in network.items()}
            gradients = {}
            A, cache = forward_pass(network, X_batch, activation, res_places)
            gradients = backpropagation(gradients, network, cache, Y_batch, res_places, activation)
            SGD_with_momentum(network, gradients, velocity, learning_rate, momentum)
            avg_lossesList_it.append(compute_loss(A, Y_batch))
            avg_successList_it.append(calculate_success_percentage(A ,Y_batch))

        lossesList.append(compute_loss(A, Y_batch))
        successList.append(calculate_success_percentage(A ,Y_batch))
        avg_lossesList.append(np.mean(avg_lossesList_it))
        avg_successList.append(np.mean(avg_successList_it))

    return network, lossesList, successList, avg_lossesList,avg_successList

def Set_ENV_train_network(X, C, layer_sizes, res_places ,batchSize, num_epochs, learning_rate, momentum, activation='relu', print_train_prosses=False):
    """
    Set up the environment and train a neural network with specified parameters.

    Parameters:
        X (numpy.ndarray): Input features for training.
        C (numpy.ndarray): True labels for training.
        layer_sizes (list of int): List specifying the number of units in each layer of the network.
        res_places (list of int): List specifying the indices of layers where residual connections are placed.
        batchSize (int): Size of each mini-batch for training.
        num_epochs (int): Number of epochs for training.
        learning_rate (float): Learning rate for the optimizer.
        momentum (float): Momentum factor for the optimizer.
        activation (str): Activation function to use ('relu' or 'tanh'). Default is 'relu'.
        print_train_process (bool): Whether to print training progress. Default is False.

    Returns:
        tuple: Updated network and lists of losses and success percentages per batch and per epoch.
    """

    network = initialize_network(layer_sizes, res_places)
    network, lossesList, successList , avg_lossesList,avg_successList = train_network(X, C, print_train_prosses, batchSize, network, res_places, num_epochs, learning_rate , momentum , activation)
    return network ,lossesList, successList, avg_lossesList,avg_successList


#                                                           run and train the network!
layer_sizes = [2,12,12,12,2]
res_places = res_indices(layer_sizes)
batchSize = 64
num_epochs = 100
learning_rate = 0.09
momentum = 0.2
activation = 'relu'
print_train_prosses = False
X = data_training_X
C = data_training_C

network_fin, lossesList, successList , avg_lossesList,avg_successList = Set_ENV_train_network(X, C, layer_sizes, res_places ,batchSize, num_epochs, learning_rate, momentum, activation, print_train_prosses)

print(data_validation_test(network_fin, res_places))
plot_training_results_extended(lossesList, successList, avg_lossesList, avg_successList)