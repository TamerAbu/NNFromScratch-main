# Import necessary libraries for the project


# Paths to the data files
import numpy as np
import scipy


file_path = './data/SwissRollData.mat'
file_path2 = './data/GMMData.mat'
file_path3 = './data/PeaksData.mat'

# Load the data from the Swiss Roll dataset using SciPy's loadmat function
# To test different datasets, simply change the file_path variable to file_path2 or file_path3.
# For additional modifications and detailed descriptions of changes, please refer to the README file included in this repository.
mat_data = scipy.io.loadmat(file_path)

# Extract training data from the dataset
data_training_X = mat_data['Yt']
data_training_C = mat_data['Ct']

# Extract validation data from the dataset
data_validation_X = mat_data['Yv']
data_validation_C = mat_data['Cv']


def pickRandomBach(X,C,batchSize):
    """
    Selects a random batch of data points and corresponding labels from the input datasets.

    Parameters:
    X (numpy.ndarray): The input features dataset, where each column represents a data point.
    C (numpy.ndarray): The labels dataset, where each column corresponds to the label of a data point.
    batchSize (int): The number of data points to include in the batch.

    Returns:
    tuple: Two numpy arrays containing the batch of input features and labels respectively.
    """
    # Generate random indices for the batch, without replacement
    random_indexes = np.random.choice(X.shape[1],size=batchSize,replace=False)
    X_batch = X[:,random_indexes]
    C_batch = C[:,random_indexes]
    return X_batch , C_batch


def shuffle_and_batch(X, C, batchSize):
    """
    Generates batches of data points and corresponding labels from the input datasets, after shuffling.

    Parameters:
        X (numpy.ndarray): The input features dataset, where each column represents a data point.
        C (numpy.ndarray): The labels dataset, where each column corresponds to the label of a data point.
        batchSize (int): The number of data points to include in each batch.

    Yields:
        tuple: Two numpy arrays containing the batch of input features and labels respectively, iterated over all data.
    """
    # Number of data points in the dataset
    num_data = X.shape[1]

    # Generate a shuffled array of indices
    indices = np.arange(num_data)
    np.random.shuffle(indices)

    # Yield batches according to the shuffled indices
    for start_idx in range(0, num_data, batchSize):
        end_idx = min(start_idx + batchSize, num_data)
        batch_indices = indices[start_idx:end_idx]

        X_batch = X[:, batch_indices]
        C_batch = C[:, batch_indices]

        yield X_batch, C_batch