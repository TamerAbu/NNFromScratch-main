# Import necessary libraries for the project
import numpy as np


def initialize_network(layer_sizes, res_places):

    """
    Initialize the weights of a neural network, including support for ResNet-style residual connections.

    Parameters:
        layer_sizes (list of int): List specifying the number of units in each layer of the network.
        res_places (list of int): List specifying the indices of layers where residual connections are placed.

    Returns:
        dict: Dictionary containing initialized weight matrices for each layer. If the layer sizes in residual places do not match, returns an empty dictionary.
    """
    
    network = {}
    for i in res_places:
      if (layer_sizes[i] != layer_sizes[i+ 1]):
        return {}
    for i in range(1, len(layer_sizes)):

        if i-1 in res_places:
            # For ResNet: W1 and W2 weights
            network['W' + str(i) + '_res1'] = np.random.randn(layer_sizes[i-1], layer_sizes[i] + 1) / layer_sizes[i-1]
            network['W' + str(i) + '_res2'] = np.random.randn(layer_sizes[i] , layer_sizes[i]) / layer_sizes[i-1]
        else:
          network['W' + str(i)] = np.random.randn(layer_sizes[i], layer_sizes[i-1] + 1) / (layer_sizes[i])


    return network

def res_indices(layer_sizes):
    """
    Identify the indices of layers with repeated sizes in a neural network, excluding the input layer.

    Parameters:
        layer_sizes (list of int): List specifying the number of units in each layer of the network.

    Returns:
        list of int: Indices of layers with repeated sizes, excluding the input layer.
    """

    index_dict = {}
    for idx, value in enumerate(layer_sizes):
        if value in index_dict:
            index_dict[value].append(idx)
        else:
            index_dict[value] = [idx]
    repeating_indices = []

    for indices in index_dict.values():
        if len(indices) > 1:
            repeating_indices.extend(indices[:-1])

    if 0 in repeating_indices:
        repeating_indices.remove(0)

    return repeating_indices