# Import necessary libraries for the project
from matplotlib import pyplot as plt
from activation_functions import calculate_success_percentage, compute_loss
from forward_pass import forward_pass
from data_handling import data_validation_X, data_validation_C

def data_validation_test(network, res_places):
    """
    Perform a validation test on the given network using validation data.

    Parameters:
        network (dict): Dictionary containing the network's weight matrices.
        res_places (list of int): List specifying the indices of layers where residual connections are placed.

    Returns:
        tuple: The computed loss and success percentage on the validation data.
    """

    A, _ = forward_pass(network, data_validation_X, 'relu', res_places)
    loss = compute_loss(A ,data_validation_C)
    success = calculate_success_percentage(A,data_validation_C)
    return loss, success

def plot_training_results_extended(lossesList, successList, avg_lossesList, avg_successList):
    """
    Plot the training results, including losses and success percentages per batch and per epoch.

    Parameters:
        lossesList (list of float): List of loss values per batch.
        successList (list of float): List of success percentages per batch.
        avg_lossesList (list of float): List of average loss values per epoch.
        avg_successList (list of float): List of average success percentages per epoch.

    Returns:
        None
    """
    plt.figure(figsize=(14, 7))

    # Plotting the losses per batch
    plt.subplot(2, 2, 1)
    plt.plot(lossesList, label='Loss per Batch', color='blue')
    plt.title('Loss per Batch')
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.legend()

    # Plotting the success percentage per batch
    plt.subplot(2, 2, 2)
    plt.plot(successList, label='Success % per Batch', color='green')
    plt.title('Success % per Batch')
    plt.xlabel('Iterations')
    plt.ylabel('Success %')
    plt.grid(True)
    plt.legend()

    # Plotting the average losses per epoch
    plt.subplot(2, 2, 3)
    plt.plot(avg_lossesList, label='Average Loss per Epoch', color='red')
    plt.title('Average Loss per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Average Loss')
    plt.grid(True)
    plt.legend()

    # Plotting the average success percentage per epoch
    plt.subplot(2, 2, 4)
    plt.plot(avg_successList, label='Average Success % per Epoch', color='purple')
    plt.title('Average Success % per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Success %')
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.show()