

import numpy as np
import matplotlib.pyplot as plt
from testCases_v2 import *
import sklearn
import sklearn.datasets
import sklearn.linear_model
from planar_utils import plot_decision_boundary, sigmoid, load_planar_dataset, load_extra_datasets

class ShallowNeuralNetwork(object):
    def __init__(self, train_input_matrix, train_targets, num_iterations=15000, hidden_layer_size=4):
        self.train_input_matrix = train_input_matrix
        self.train_targets = train_targets
        self.num_iterations = num_iterations
        self.hidden_layer_size = hidden_layer_size
        self.parameters = {
            "W1" : None,
            "b1" : None,
            "W2" : None,
            "b2" : None
        }
        self.cache = {
            "Z1" : None,
            "A1" : None,
            "Z2" : None,
            "A2" : None
        }

    def __del__(self):
        self.close()

    def close(self):
        del self.train_input_matrix
        del self.train_targets
        del self.num_iterations
        del self.hidden_layer_size

        del self.parameters
        del self.cache

    def train_model(self):
        (n_x, n_h, n_y) = self.get_layer_sizes(self.train_input_matrix, self.train_targets)

        self.initialize_parameters(n_x, n_h, n_y)

        for i in range(self.num_iterations):
            A2, self.cache = self.propagate_forward(self.train_input_matrix, self.parameters)

            cost = self.compute_cost(A2, self.train_targets, self.parameters)

            grads = self.backward_propagation(self.parameters, self.cache, self.train_input_matrix, self.train_targets)

            self.parameters = self.update_parameters(self.parameters, grads)

        return self.parameters


    def predict(self, parameters, X):
        """
        Using the learned parameters, predicts a class for each example in X

        Arguments:
        parameters -- python dictionary containing your parameters
        X -- input data of size (n_x, m)

        Returns
        predictions -- vector of predictions of our model (red: 0 / blue: 1)
        """

        # Computes probabilities using forward propagation, and classifies to 0/1 using 0.5 as the threshold.
        A2, cache = self.propagate_forward(X, parameters)
        predictions = (A2 > 0.5)

        return predictions


    def get_layer_sizes(self, input_matrix, target_matrix):
        """
        Arguments:
        input_matrix -- input dataset of shape (input size, number of examples)
        target_matrix -- labels of shape (output size, number of examples)

        Returns:
        n_x -- the size of the input layer
        n_h -- the size of the hidden layer
        n_y -- the size of the output layer
        """
        n_x = input_matrix.shape[0]
        n_h = self.hidden_layer_size
        n_y = target_matrix.shape[0]

        return (n_x, n_h, n_y)

    def compute_cost(self, A2, Y, parameters):
        """
        Computes the cross-entropy cost given in equation (13)

        Arguments:
        A2 -- The sigmoid output of the second activation, of shape (1, number of examples)
        Y -- "true" labels vector of shape (1, number of examples)
        parameters -- python dictionary containing your parameters W1, b1, W2 and b2

        Returns:
        cost -- cross-entropy cost given equation (13)
        """

        m = float(Y.shape[1])  # number of example

        # Compute the cross-entropy cost
        logprobs = np.multiply(np.log(A2), Y) + np.multiply(np.log(1 - A2), 1 - Y)
        cost = -1 / m * np.sum(logprobs)

        cost = np.squeeze(cost)  # makes sure cost is the dimension we expect.
        # E.g., turns [[17]] into 17
        assert (isinstance(cost, float))

        return cost


    def backward_propagation(self, parameters, cache, X, Y):
        """
        Implement the backward propagation using the instructions above.

        Arguments:
        parameters -- python dictionary containing our parameters
        cache -- a dictionary containing "Z1", "A1", "Z2" and "A2".
        X -- input data of shape (2, number of examples)
        Y -- "true" labels vector of shape (1, number of examples)

        Returns:
        grads -- python dictionary containing your gradients with respect to different parameters
        """
        m = float(X.shape[1])

        # First, retrieve W1 and W2 from the dictionary "parameters".
        W1 = parameters['W1']
        W2 = parameters['W2']

        # Retrieve also A1 and A2 from dictionary "cache".
        A1 = cache['A1']
        A2 = cache['A2']

        # Backward propagation: calculate dW1, db1, dW2, db2.
        dZ2 = A2 - Y
        dW2 = 1 / m * dZ2.dot(A1.T)
        db2 = 1 / m * np.sum(dZ2, axis=1, keepdims=True)
        dZ1 = W2.T.dot(dZ2) * (1 - np.power(A1, 2))
        dW1 = 1 / m * dZ1.dot(X.T)
        db1 = 1 / m * np.sum(dZ1, axis=1, keepdims=True)

        grads = {"dW1": dW1,
                 "db1": db1,
                 "dW2": dW2,
                 "db2": db2}

        return grads

    def update_parameters(self, parameters, grads, learning_rate = 0.01):
        """
        Updates parameters using the gradient descent update rule given above

        Arguments:
        parameters -- python dictionary containing your parameters
        grads -- python dictionary containing your gradients

        Returns:
        parameters -- python dictionary containing your updated parameters
        """
        # Retrieve each parameter from the dictionary "parameters"
        W1 = parameters['W1']
        b1 = parameters['b1']
        W2 = parameters['W2']
        b2 = parameters['b2']

        # Retrieve each gradient from the dictionary "grads"
        dW1 = grads['dW1']
        db1 = grads['db1']
        dW2 = grads['dW2']
        db2 = grads['db2']

        # Update rule for each parameter
        W1 = W1 - learning_rate * dW1
        b1 = b1 - learning_rate * db1
        W2 = W2 - learning_rate * dW2
        b2 = b2 - learning_rate * db2

        parameters = {"W1": W1,
                      "b1": b1,
                      "W2": W2,
                      "b2": b2}

        return parameters


    def initialize_parameters(self, n_x, n_h, n_y):
        """
        Argument:
        n_x -- size of the input layer
        n_h -- size of the hidden layer
        n_y -- size of the output layer

        Returns:
        params -- python dictionary containing your parameters:
                        W1 -- weight matrix of shape (n_h, n_x)
                        b1 -- bias vector of shape (n_h, 1)
                        W2 -- weight matrix of shape (n_y, n_h)
                        b2 -- bias vector of shape (n_y, 1)
        """
        W1 = np.random.randn(n_h, n_x) * 0.01
        b1 = np.zeros((n_h, 1))
        W2 = np.random.randn(n_y, n_h) * 0.01
        b2 = np.zeros((n_y, 1))

        assert (W1.shape == (n_h, n_x))
        assert (b1.shape == (n_h, 1))
        assert (W2.shape == (n_y, n_h))
        assert (b2.shape == (n_y, 1))

        self.parameters = {
            "W1" : W1,
            "b1" : b1,
            "W2" : W2,
            'b2' : b2
        }

    def propagate_forward(self, input_matrix, parameters):
        """
        Argument:
        X -- input data of size (n_x, m)
        parameters -- python dictionary containing your parameters (output of initialization function)

        Returns:
        A2 -- The sigmoid output of the second activation
        cache -- a dictionary containing "Z1", "A1", "Z2" and "A2"
        """
        # Retrieve each parameter from the dictionary "parameters"
        W1 = parameters['W1']
        b1 = parameters['b1']
        W2 = parameters['W2']
        b2 = parameters['b2']

        # Implement Forward Propagation to calculate A2 (probabilities)
        Z1 = W1.dot(input_matrix) + b1
        A1 = np.tanh(Z1)
        Z2 = W2.dot(A1) + b2
        A2 = sigmoid(Z2)

        assert (A2.shape == (1, input_matrix.shape[1]))

        cache = {"Z1": Z1,
                 "A1": A1,
                 "Z2": Z2,
                 "A2": A2}

        return A2, cache


def main():
    X, Y = load_planar_dataset()

    hidden_layer_sizes = list([1,2,3,4,5,10,15,20,25])
    plt.figure(figsize=(16,32))

    # Visualize the data:
    plt.scatter(X[0, :], X[1, :], c=Y, s=40, cmap=plt.cm.Spectral)

    plt.show()


    accuracies = list([])

    for hidden_layer_size in hidden_layer_sizes:
        model = ShallowNeuralNetwork(X, Y, hidden_layer_size=hidden_layer_size)
        parameters = model.train_model()

        plot_decision_boundary(lambda x: model.predict(parameters, x.T), X, Y)

        predictions = model.predict(parameters, X)
        accuracy = float((np.dot(Y,predictions.T) + np.dot(1-Y,1-predictions.T))/float(Y.size)*100)
        print ("Accuracy for {} hidden units: {} %".format(hidden_layer_size, accuracy))
        accuracies.append(accuracy)

    print(accuracies)




if __name__ == '__main__':
    main()
