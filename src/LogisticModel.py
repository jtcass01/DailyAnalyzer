from utilities import initialize_weights_and_bias, optimize, predict

class LogisticModel(object):
    def __init__(self, train_parameter_matrix, train_targets, test_parameter_matrix, test_targets, num_iterations, learning_rate = 0.1, log_cost = False):
        self.train_parameter_matrix = train_parameter_matrix
        self.train_targets = train_targets
        self.test_parameter_matrix = test_parameter_matrix
        self.test_targets = test_targets
        self.num_iterations = num_iterations
        self.learning_rate = learning_rate
        self.log_cost = log_cost
        self.model_info = {
            "costs": None,
            "test_target_predictions": None,
            "train_target_predictions": None,
            "weights": None,
            "bias": None,
            "learning_rate": self.learning_rate,
            "num_iterations": self.num_iterations
        }

    def __del__(self):
        self.close()

    def close(self):
        # Probably should log the model here.
        del self.train_parameter_matrix
        del self.train_targets
        del self.test_parameter_matrix
        del self.test_targets
        del self.num_iterations
        del self.learning_rate
        del self.log_cost
        del self.model_info
        del self

    def build_model(self):
        print("Initializing weights and bias")
        weights, bias = initialize_weights_and_bias(number_of_features=self.train_parameter_matrix.shape[0])
        print("Weights and bias initialized.")

        print("Optimizing parameters and gradients")
        parameters, gradients, costs = optimize(weights=weights, bias=bias, input_matrix=self.train_parameter_matrix, targets=self.train_targets, num_iterations=self.num_iterations, learning_rate=self.learning_rate)
        print("Finished optimizing.")

        # Retrieve parameters
        weights = parameters['weights']
        bias = parameters['bias']

        print("Making predictions...")
        test_target_predictions = predict(weights=weights, bias=bias, input_matrix=self.test_parameter_matrix)
        train_target_predictions = predict(weights=weights, bias=bias, input_matrix=self.train_parameter_matrix)
        print("Finished making predictions.")


        self.model_info = {
            "costs": costs,
            "test_target_predictions": test_target_predictions,
            "train_target_predictions": train_target_predictions,
            "weights": weights,
            "bias": bias,
            "learning_rate": self.learning_rate,
            "num_iterations": self.num_iterations
        }

def main():
    print('test')

if __name__ == '__main__':
    main()