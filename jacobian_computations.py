# Import necessary libraries for the project


from matplotlib import pyplot as plt
import numpy as np
from activation_functions import relu, relu_derivative, tanh, tanh_derivative
from forward_pass import single_step_forward_class, single_step_forward_res


def JacobianTByW(X, W, u, activation='relu'):
    """
    Compute the Jacobian transpose with respect to weights W for a given layer.

    Parameters:
        X (numpy.ndarray): Input features, where each column represents a data point.
        W (numpy.ndarray): Weights matrix for the layer.
        u (numpy.ndarray): Vector used in Jacobian computation, same shape as the layer output.
        activation (str): Activation function to use ('relu' or 'tanh'). Default is 'relu'.

    Returns:
        numpy.ndarray: The computed Jacobian transpose.
    """
    activationD_func = relu_derivative if activation == 'relu' else tanh_derivative

    Z = W.dot(X)
    A = activationD_func(Z)

    return np.dot(np.multiply(A,u),X.T)

def JacobianTByX(X, W, u, activation='relu'):
    """
    Compute the Jacobian transpose with respect to input features X for a given layer.

    Parameters:
        X (numpy.ndarray): Input features, where each column represents a data point.
        W (numpy.ndarray): Weights matrix for the layer.
        u (numpy.ndarray): Vector used in Jacobian computation, same shape as the layer output.
        activation (str): Activation function to use ('relu' or 'tanh'). Default is 'relu'.

    Returns:
        numpy.ndarray: The computed Jacobian transpose with respect to X.
    """
    activationD_func = relu_derivative if activation == 'relu' else tanh_derivative

    Z = W.dot(X)
    A = activationD_func(Z)

    return np.dot(W.T, np.multiply(A,u))

def JacobianT_W1_Res(X, W1, W2, u, activation='relu'):
    """
    Compute the Jacobian transpose with respect to weights W1 for a ResNet block.

    Parameters:
        X (numpy.ndarray): Input features, where each column represents a data point.
        W1 (numpy.ndarray): Weights matrix for the first layer in the residual block.
        W2 (numpy.ndarray): Weights matrix for the second layer in the residual block.
        u (numpy.ndarray): Vector used in Jacobian computation, same shape as the layer output.
        activation (str): Activation function to use ('relu' or 'tanh'). Default is 'relu'.

    Returns:
        numpy.ndarray: The computed Jacobian transpose with respect to W1.
    """

    activationD_func = relu_derivative if activation == 'relu' else tanh_derivative

    Z_res1 = W1.dot(X)
    A_res1 = activationD_func(Z_res1)

    Z_u = W2.T.dot(u)
    Z_res2 = np.multiply(A_res1,Z_u)

    return Z_res2.dot(X.T)

def JacobianT_W2_Res(X, W1, W2, u, activation='relu'):
    """
    Compute the Jacobian transpose with respect to weights W2 for a ResNet block.

    Parameters:
        X (numpy.ndarray): Input features, where each column represents a data point.
        W1 (numpy.ndarray): Weights matrix for the first layer in the residual block.
        W2 (numpy.ndarray): Weights matrix for the second layer in the residual block.
        u (numpy.ndarray): Vector used in Jacobian computation, same shape as the layer output.
        activation (str): Activation function to use ('relu' or 'tanh'). Default is 'relu'.

    Returns:
        numpy.ndarray: The computed Jacobian transpose with respect to W2.
    """

    activation_func = relu if activation == 'relu' else tanh

    Z_res1 = W1.dot(X)
    A = activation_func(Z_res1)
    A = A.T

    return u.dot(A)

def JacobianT_X_Res(X, W1, W2, u, activation='relu'):
    """
    Compute the Jacobian transpose with respect to input features X for a ResNet block.

    Parameters:
        X (numpy.ndarray): Input features, where each column represents a data point.
        W1 (numpy.ndarray): Weights matrix for the first layer in the residual block.
        W2 (numpy.ndarray): Weights matrix for the second layer in the residual block.
        u (numpy.ndarray): Vector used in Jacobian computation, same shape as the layer output.
        activation (str): Activation function to use ('relu' or 'tanh'). Default is 'relu'.

    Returns:
        numpy.ndarray: The computed Jacobian transpose with respect to X.
    """

    activationD_func = relu_derivative if activation == 'relu' else tanh_derivative

    #step 1 :
    Z_res1 = W1.dot(X)
    A_res1 = activationD_func(Z_res1)


    #step 2 :
    Z_2 = np.dot(W2.T,u)
    A_res2 = np.multiply(A_res1,Z_2)

    #step 3 :
    Jac = np.dot(W1.T,A_res2)
    Jac = Jac[:-1,:]
    Jac += u

    return Jac

#                                                                   Jacobian tests :
# to test each derivative we use a J-test

def jacobian_test_by_W_for_classic():
    """
    Perform a Jacobian check for a classic neural network layer with respect to weights W.

    Outputs:
        Plots the Jacobian checking results, comparing numerical and analytical Jacobian.
    """

    # Jacobian Check Setup
    num_classes = 3
    input_size = 2
    num_features = 5
    EPSILON = 0.1
    W = np.random.randn(num_classes, num_features + 1)
    X = np.random.randn(num_features, input_size)
    X = np.vstack([X, np.ones((1, X.shape[1]))])
    d = np.random.randn(num_classes, num_features + 1)
    u = np.random.randn(num_classes, input_size)

    def F(W):
        return single_step_forward_class(X, W, tanh)

    def g_F(W):
        return np.dot(u.flatten(), F0.flatten())

    F0 = F(W)
    g0 = g_F(W)
    grad_g0 = JacobianTByW(X, W, u, tanh)

    y0 = np.zeros(8)
    y1 = np.zeros(8)


    print("k\terror order 1 \t\t\t error order 2")
    for k in range(1, 9):
        epsk = EPSILON * (0.5 ** k)
        fk = F(W + (epsk * d))
        gk = np.dot(fk.flatten(), u.flatten())
        g1 = g0 + (epsk * (np.dot(d.flatten(),grad_g0.flatten())))
        y0[k - 1] = abs(gk - g0)
        y1[k - 1] = abs(gk - g1)
        print(f"{k}\t{y0[k-1]}\t{y1[k-1]}")

    # Plotting
    plt.semilogy(range(1, 9), y0, label="Zero order approx")
    plt.semilogy(range(1, 9), y1, label="First order approx")
    plt.legend()
    plt.title("Jacobian test By W for classic")
    plt.xlabel("k")
    plt.ylabel("Error")
    plt.show()

def jacobian_test_by_X_for_classic():
    """
    Perform a Jacobian check for a classic neural network layer with respect to input features X.

    Outputs:
        Plots the Jacobian checking results, comparing numerical and analytical Jacobian.
    """

    # Jacobian Check Setup
    num_classes = 3
    input_size = 2
    num_features = 5
    EPSILON = 0.1
    W = np.random.randn(num_classes, num_features + 1)
    X = np.random.randn(num_features, input_size)
    X = np.vstack([X, np.ones((1, X.shape[1]))])
    d = np.random.randn(num_features + 1, input_size)
    u = np.random.randn(num_classes, input_size)

    def F(X):
        return single_step_forward_class(X, W, tanh)

    def g_F(X):
        return np.dot(u.flatten(), F0.flatten())

    F0 = F(X)
    g0 = g_F(X)
    grad_g0 = JacobianTByX(X, W, u, tanh)

    y0 = np.zeros(8)
    y1 = np.zeros(8)


    print("k\terror order 1 \t\t\t error order 2")
    for k in range(1, 9):
        epsk = EPSILON * (0.5 ** k)
        fk = F(X + (epsk * d))
        gk = np.dot(fk.flatten(), u.flatten())
        g1 = g0 + (epsk * (np.dot(d.flatten(),grad_g0.flatten())))
        y0[k - 1] = abs(gk - g0)
        y1[k - 1] = abs(gk - g1)
        print(f"{k}\t{y0[k-1]}\t{y1[k-1]}")

    # Plotting
    plt.semilogy(range(1, 9), y0, label="Zero order approx")
    plt.semilogy(range(1, 9), y1, label="First order approx")
    plt.legend()
    plt.title("Jacobian test By X for classic")
    plt.xlabel("k")
    plt.ylabel("Error")
    plt.show()

def jacobian_test_res_w1():
    """
    Perform a Jacobian check for a ResNet block with respect to weights W1.

    Outputs:
        Plots the Jacobian checking results, comparing numerical and analytical Jacobian.
    """

    # Jacobian Check Setup
    num_classes = 3
    input_size = 2
    num_features = num_classes
    EPSILON = 0.1
    W1 = np.random.randn(num_classes, num_features + 1)
    W2 = np.random.randn(num_classes, num_features )
    X = np.random.randn(num_features, input_size)
    X = np.vstack([X, np.ones((1, X.shape[1]))])
    d = np.random.randn(num_classes, num_features + 1)
    u = np.random.randn(num_classes, input_size)

    def F(W1):
        return single_step_forward_res(X, W1, W2, tanh)

    def g_F(W1):
        return np.dot(u.flatten(), F0.flatten())

    F0 = F(W1)
    g0 = g_F(W1)
    grad_g0 = JacobianT_W1_Res(X, W1, W2, u, tanh)

    y0 = np.zeros(8)
    y1 = np.zeros(8)

    print("k\terror order 1 \t\t\t error order 2")
    for k in range(1, 9):
        epsk = EPSILON * (0.5 ** k)
        fk = F(W1 + (epsk * d))
        gk = np.dot(fk.flatten(), u.flatten())
        g1 = g0 + (epsk * (np.dot(d.flatten(),grad_g0.flatten())))
        y0[k - 1] = abs(gk - g0)
        y1[k - 1] = abs(gk - g1)
        print(f"{k}\t{y0[k-1]}\t{y1[k-1]}")

    # Plotting
    plt.semilogy(range(1, 9), y0, label="Zero order approx")
    plt.semilogy(range(1, 9), y1, label="First order approx")
    plt.legend()
    plt.title("Jacobian test By W1 for ResNet")
    plt.xlabel("k")
    plt.ylabel("Error")
    plt.show()

def jacobian_test_res_w2():
    """
    Perform a Jacobian check for a ResNet block with respect to weights W2.

    Outputs:
        Plots the Jacobian checking results, comparing numerical and analytical Jacobian.
    """

    # Jacobian Check Setup
    num_classes = 3
    input_size = 2
    num_features = num_classes
    EPSILON = 0.1
    X = np.random.randn(num_features, input_size)
    X = np.vstack([X, np.ones((1, X.shape[1]))])
    W1 = np.random.randn(num_features, num_classes + 1)
    W2 = np.random.randn(num_features, num_classes)
    d = np.random.randn(num_features , num_classes)
    u = np.random.randn(num_features, input_size)


    def F(W2):
        return single_step_forward_res(X, W1, W2, tanh)

    def g_F(W2):
        return np.dot(u.flatten(), F0.flatten())

    F0 = F(W2)
    g0 = g_F(W2)
    grad_g0 = JacobianT_W2_Res(X, W1, W2, u, tanh)

    y0 = np.zeros(8)
    y1 = np.zeros(8)


    print("k\terror order 1 \t\t\t error order 2")
    for k in range(1, 9):
        epsk = EPSILON * (0.5 ** k)
        fk = F(W2 + (epsk * d))
        gk = np.dot(fk.flatten(), u.flatten())
        g1 = g0 + (epsk * (np.dot(d.flatten(),grad_g0.flatten())))
        y0[k - 1] = abs(gk - g0)
        y1[k - 1] = abs(gk - g1)
        print(f"{k}\t{y0[k-1]}\t{y1[k-1]}")

    # Plotting
    plt.semilogy(range(1, 9), y0, label="Zero order approx")
    plt.semilogy(range(1, 9), y1, label="First order approx")
    plt.legend()
    plt.title("Jacobian test By W2 for ResNet")
    plt.xlabel("k")
    plt.ylabel("Error")
    plt.show()

def jacobian_test_res_X():
    """
    Perform a Jacobian check for a ResNet block with respect to input features X.

    Outputs:
        Plots the Jacobian checking results, comparing numerical and analytical Jacobian.
    """

    # Jacobian Check Setup
    num_classes = 3
    input_size = 2
    num_features = num_classes
    EPSILON = 0.1
    W1 = np.random.randn(num_classes, num_features + 1)
    W2 = np.random.randn(num_classes, num_features )
    X = np.random.randn(num_features, input_size)
    d = np.random.randn(num_features, input_size)
    u = np.random.randn(num_classes, input_size)


    def F(X):
        A = np.vstack([X, np.ones((1, X.shape[1]))])
        return single_step_forward_res(A, W1, W2, tanh)

    def g_F(X):
        return np.dot(u.flatten(), F0.flatten())

    def jaco(X):
        A = np.vstack([X, np.ones((1, X.shape[1]))])
        return JacobianT_X_Res(A, W1, W2, u, tanh)

    F0 = F(X)
    g0 = g_F(X)
    grad_g0 = jaco(X)


    y0 = np.zeros(8)
    y1 = np.zeros(8)


    print("k\terror order 1 \t\t\t error order 2")
    for k in range(1, 9):
        epsk = EPSILON * (0.5 ** k)
        fk = F(X + (epsk * d))
        gk = np.dot(fk.flatten(), u.flatten())
        g1 = g0 + (epsk * (np.dot(d.flatten(),grad_g0.flatten())))
        y0[k - 1] = abs(gk - g0)
        y1[k - 1] = abs(gk - g1)
        print(f"{k}\t{y0[k-1]}\t{y1[k-1]}")

    # Plotting
    plt.semilogy(range(1, 9), y0, label="Zero order approx")
    plt.semilogy(range(1, 9), y1, label="First order approx")
    plt.legend()
    plt.title("Jacobian test By X for ResNet")
    plt.xlabel("k")
    plt.ylabel("Error")
    plt.show()




#                                                                              testing Jacobian 
# To test the Jacobian function, please remove the '#' before the one you wish to test :
# jacobian_test_by_W_for_classic()
# jacobian_test_by_X_for_classic()
# jacobian_test_res_w1()
# jacobian_test_res_w2()
# jacobian_test_res_X()