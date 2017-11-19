import numpy as np
import pandas as pd

def optimize(weights, bias, input_matrix, targets, num_iterations, learning_rate, log_cost = False, storage_frequency=100):
    """
    This function optimizes w and b by running a gradient descent algorithm

    Arguments:
    w -- weights, a numpy array of size (num_px * num_px * 3, 1)
    b -- bias, a scalar
    X -- data of shape (num_px * num_px * 3, number of examples)
    Y -- true "label" vector (containing 0 if non-cat, 1 if cat), of shape (1, number of examples)
    num_iterations -- number of iterations of the optimization loop
    learning_rate -- learning rate of the gradient descent update rule
    print_cost -- True to print the loss every 100 steps

    Returns:
    params -- dictionary containing the weights w and bias b
    grads -- dictionary containing the gradients of the weights and bias with respect to the cost function
    costs -- list of all the costs computed during the optimization, this will be used to plot the learning curve.

    Tips:
    You basically need to write down two steps and iterate through them:
        1) Calculate the cost and the gradient for the current parameters. Use propagate().
        2) Update the parameters using gradient descent rule for w and b.
    """
    costs = list([])
    dw = None
    db = None

    for epoch in range(num_iterations):
        gradients, cost = propagate_forward_and_back(weights=weights, bias=bias, input_matrix=input_matrix, targets=targets)

        # Retrieve derivatives from gradients
        dw = gradients['dw']
        db = gradients['db']

        # Update weights and bias
        weights = weights-learning_rate*dw
        bias = bias-learning_rate*db

        # Store the cost in an array for later analysis
        if epoch % storage_frequency == 0:
            costs.append(costs)

    parameters = {
        'weights' : weights,
        'bias' : bias
    }

    gradients = {
        'dw' : dw,
        'db' : db
    }

    return parameters, gradients, costs

def predict(weights, bias, input_matrix):
    '''
    Predict whether the label is 0 or 1 using learned logistic regression parameters (w, b)

    Arguments:
    w -- weights, a numpy array of size (num_px * num_px * 3, 1)
    b -- bias, a scalar
    X -- data of size (num_px * num_px * 3, number of examples)

    Returns:
    Y_prediction -- a numpy array (vector) containing all predictions (0/1) for the examples in X
    '''

    sample_size = input_matrix.shape[1]
    Y_prediction = np.zeros((1,sample_size))
    weights = weights.reshape(input_matrix.shape[0], 1)

    # Compute activation
    activation = get_activation(weights=weights, input_matrix=input_matrix, bias=bias)

    Y_prediction = np.greater_equal(activation,0.5).astype(float)

    assert(Y_prediction.shape == (1,sample_size))

    return Y_prediction

def propagate_forward_and_back(weights, bias, input_matrix, targets):
    """
    Implement the cost function and its gradient for the propagation explained above
    Arguments:
    weights -- weights, a numpy array of size (num_px * num_px * 3, 1)
    bias -- bias, a scalar
    input_matrix -- data of size (num_px * num_px * 3, number of examples)
    targets -- true "label" vector (containing 0 if non-cat, 1 if cat) of size (1, number of examples)
    Return:
    cost -- negative log-likelihood cost for logistic regression
    dw -- gradient of the loss with respect to w, thus same shape as w
    db -- gradient of the loss with respect to b, thus same shape as b

    Tips:
    - Write your code step by step for the propagation. np.log(), np.dot()
    """
    activation, cost = propagate_forward(weights=weights, bias=bias, input_matrix=input_matrix, targets=targets)
    dw, db = propagate_back(activation=activation, input_matrix=input_matrix, targets=targets)

    gradients = {
        "dw" : dw,
        "db" : db
    }

    return gradients, cost

def propagate_forward(weights, bias, input_matrix, targets):
    """
    Performs propagate forward step for a single neural network node

    :param weights: weights, a numpy array of size (num_px * num_px * 3, 1)
    :param bias: bias, a scalar
    :param input_matrix: data of size (num_px * num_px * 3, number of examples)
    :param targets: targets
    :return:
            cost -- negative log-likelihood cost for logistic regression
    """
    activation = get_activation(weights, input_matrix, bias)
    cost = calculate_cost_function(targets, activation)
    return activation, cost

def propagate_back(activation, input_matrix, targets):
    """
    Performs propagate forward step for a single neural network node

    :param activation:
    :param input_matrix:
    :param targets:
    :return:
        dw -- gradient of the loss with respect to w, thus same shape as w
        db -- gradient of the loss with respect to b, thus same shape as b
    """

    sample_size = targets.shape[1]

    dw = 1/sample_size * np.dot(input_matrix, (activation-targets))
    db = 1/sample_size * np.sum(activation - targets)

    return dw, db

def sigmoid(z):
    """
    Compute the sigmoid of z

    :param z: A scalar or numpy array of any size
    :return: s -- sigmoid(z)
    """

    s = 1/(1+np.exp(-z))

    return s

def get_activation(weights, input_matrix, bias):
    """
    Calculate activation matrix

    :param w: weights
    :param X: input parameters
    :param b: bias
    :return: activation list
    """
    
    input_sigmoid = weights.T.dot(input_matrix)
    
    return sigmoid(weights.T.dot(input_matrix)+bias)

def calculate_cost_function(targets, activation):
    """
    Forward propagation step

    :param Y: Targets
    :param A: Activation
    :return: cost
    """
    m = targets.shape[1]

    return -1/m*np.sum(np.multiply(targets,np.log(activation)) + np.multiply((1-targets), np.log(1-activation)))

def initialize_weights_and_bias(number_of_features):
    """
    This function creates a vector of random values of shape (number_of_features, 1) for w and initializes b to a random value.

    :param number_of_features: size of the w vector we want (or number of parameters in this case)
    :return:
        weights -- initialized vector of shape (number_of_features, 1)
        bias -- initialized scalar (corresponds to the bias)    """

    weights = np.random.randn(number_of_features, 1) / np.sqrt(number_of_features)
    bias = 0

    return weights.astype(float), bias

def parse_fer2013_pixels(pixel_data):
    result = None
    count = 0
    
    for row in pixel_data:

        try:
            row_list = np.array(list(map(float, row.split(' '))))
        except AttributeError:
            row_list = np.array(list(map(float, row)))
        pixel_count = row_list.shape[0]
        row_list = row_list.reshape((pixel_count, 1))
        

        if result == None:
            result = row_list
        else:
            result = np.append(result, row_list, axis=1)

        count += 1
        
        if count % 100 == 0:
            print(result.shape, count, len(pixel_data))
            count = 0

    print(result, result.shape, 'result')
    
    return result
    
def load_fer2013_data(file_location):
    X_train, X_test = [], []
    Y_train, Y_test = [], []
    first = True
    
    for index, line in enumerate(open(file_location)):
        if first: first = False
        else:
            row = line.split(',')
            sample_type = row[2]
            if sample_type == "Training\n":
                # first column is a label
                Y_train.append(int(row[0]))
                # second column is space separated integers
                X_train.append([int(p) for p in row[1].split()])
            else:
                # first column is a label
                Y_test.append(int(row[0]))
                # second column is space separated integers
                X_test.append([int(p) for p in row[1].split()])
#            if index % 100 == 0:
#                print(row, index)

    train_input_matrix, train_targets = np.array(X_train) / 255.0, np.array(Y_train)    
    test_input_matrix, test_targets = np.array(X_test) / 255.0, np.array(Y_test)

    train_targets = train_targets.reshape((train_input_matrix.shape[0],1))
    test_targets = test_targets.reshape((test_input_matrix.shape[0],1))
 
    print(train_input_matrix, train_input_matrix.shape, "train_input_matrix")
    print(train_targets, train_targets.shape, "train_targets")
    print(test_input_matrix, test_input_matrix.shape, "test_input_matrix")
    print(test_targets, test_targets.shape, "test_targets")

    return train_input_matrix, train_targets, test_input_matrix, test_targets