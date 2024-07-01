# Import necessary libraries for the project


from matplotlib import pyplot as plt
import numpy as np
from activation_functions import calculate_success_percentage
from data_handling import pickRandomBach , data_training_X ,data_training_C
from gradient_computations import least_squares_gradient, least_squares_loss


def SGD_with_momentum(network, gradients, velocity, learning_rate, momentum):
    for key in network.keys():
        velocity[key] = momentum * velocity[key] - learning_rate * gradients[key]
        network[key] += velocity[key]

def testSGD(lossFunction , gradientLossFunction):
    """
    Test the SGD with momentum optimizer on a simple least squares problem.

    Parameters:
        lossFunction (callable): Function to compute the loss of the model.
        gradientLossFunction (callable): Function to compute the gradient of the loss.
    """
    batchSize = 2
    EPSILON = 0.1
    outputnum = 2
    network = {}

    X , C =pickRandomBach(data_training_X,data_training_C,batchSize)
    X = np.vstack([X, np.ones((1, X.shape[1]))])
    W1 = np.random.randn(outputnum, X.shape[0])
    network = {'W1' : W1}
    velocity = {'W1' :np.zeros_like(W1)}

    success_percentages_list = []
    success_percentages = 0
    losses = []

    # Set hyperparameters
    learning_rate = 0.05
    momentum = 0.8
    epochs = 80
    batch_Size = 20

    for i in range(epochs):
        W1 = network['W1']
        X , C =pickRandomBach(data_training_X,data_training_C,batchSize)
        X = np.vstack([X, np.ones((1, X.shape[1]))])
        gradients = {'W1': least_squares_gradient(X, W1, C)}

        SGD_with_momentum(network, gradients, velocity, learning_rate, momentum)

        losses.append(least_squares_loss(X, W1, C))

        success_percentages_list.append(calculate_success_percentage(W1.dot(X), C))


    # Plot the loss over epochs
    plt.plot(range(epochs), losses)
    plt.title("Losses over Epochs (SGD with Momentum) with LS")
    plt.xlabel("Epoch")
    plt.ylabel("losses")
    plt.show()

    # Plot the success percentages over epochs
    plt.plot(range(epochs), success_percentages_list)
    plt.title("success percentages over Epochs (SGD with Momentum) with LS")
    plt.xlabel("Epoch")
    plt.ylabel("success")
    plt.show()


# test the sgd with mumentum :
# testSGD(least_squares_loss,least_squares_gradient)