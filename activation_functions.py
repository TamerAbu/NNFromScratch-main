# Import necessary libraries for the project
import numpy as np

def tanh(z):
    """
    Compute the hyperbolic tangent of the input array, element-wise.

    Parameters:
        z (numpy.ndarray): Input data array.

    Returns:
        numpy.ndarray: The hyperbolic tangent of each element in z.
    """
    return np.tanh(z)

def tanh_derivative(z):
    """
    Compute the derivative of the hyperbolic tangent function, element-wise.

    Parameters:
        z (numpy.ndarray): Input data array.

    Returns:
        numpy.ndarray: The derivative of the hyperbolic tangent of each element in z.
    """
    return 1 - np.tanh(z) ** 2

def relu(z):
    """
    Apply the rectified linear unit function element-wise.

    Parameters:
        z (numpy.ndarray): Input data array.

    Returns:
        numpy.ndarray: Input data where each element is replaced by the maximum of that element and 0.
    """
    return np.maximum(0, z)

def relu_derivative(z):
    """
    Compute the derivative of the rectified linear unit function, element-wise.

    Parameters:
        z (numpy.ndarray): Input data array.

    Returns:
        numpy.ndarray: Derivative of ReLU, where each element is 1 if the input element is positive, else 0.
    """
    return (z > 0).astype(float)

def softmax(Z):
    """
    Compute the softmax of the input array Z in a numerically stable way.

    Parameters:
        Z (numpy.ndarray): Input data array.

    Returns:
        numpy.ndarray: Softmax of Z computed over the first axis.
    """
    expZ = np.exp(Z - np.max(Z, axis=0, keepdims=True))
    return expZ / np.sum(expZ, axis=0, keepdims=True)

def compute_loss_for_testing(Z, C):
    """
    Compute the cross-entropy loss for testing, where softmax is applied to logits before calculating the loss.

    Parameters:
        Z (numpy.ndarray): Logits output by the model, before activation function is applied.
        C (numpy.ndarray): True labels in one-hot encoded form.

    Returns:
        float: The cross-entropy loss averaged over all samples.
    """

    m = C.shape[1]
    Z = softmax(Z)
    loss = -np.sum(C * np.log(Z + 1e-8)) / m
    return loss

def compute_loss(Z, C):
    """
    Compute the cross-entropy loss where the input is expected to be probabilities (after softmax).

    Parameters:
        Z (numpy.ndarray): Probabilities output by the model.
        C (numpy.ndarray): True labels in one-hot encoded form.

    Returns:
        float: The cross-entropy loss averaged over all samples.
    """

    m = C.shape[1]
    loss = -np.sum(C * np.log(Z + 1e-8)) / m
    return loss

def calculate_success_percentage(Z, C):
    """
    Calculate the percentage of correct predictions made by the model.

    Parameters:
        Z (numpy.ndarray): Output logits or probabilities from the model, where each column corresponds to a prediction for a data point.
        C (numpy.ndarray): True labels in a one-hot encoded format or as class indices.

    Returns:
        float: The percentage of correctly predicted labels.
    """
    predicted_labels = np.argmax(Z, axis=0)  # Get the predicted class as the one with the highest probability
    true_labels = np.argmax(C, axis=0)  # Directly use C as it is already class indices
    correct_predictions = np.sum(predicted_labels == true_labels)  # Count how many predictions match the true labels
    success_percentage = correct_predictions / Z.shape[1] * 100.0  # Calculate percentage and convert to a percentage
    return success_percentage