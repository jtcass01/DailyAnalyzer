from LogisticModel import LogisticModel
from utilities import load_data

class NeuralNetworkNode(object):
    def __init__(self, train_parameter_matrix, train_targets, test_parameter_matrix, test_targets, num_iterations, learning_rate = 0.1, log_cost = False):
        self.train_parameter_matrix = train_parameter_matrix
        self.train_targets = train_targets
        self.test_parameter_matrix = test_parameter_matrix
        self.test_targets = test_targets
        self.num_iterations = num_iterations
        self.learning_rate = learning_rate
        self.log_cost = log_cost
        self.model = LogisticModel(train_parameter_matrix, train_targets, test_parameter_matrix, test_targets, num_iterations, 0.1, False)
        self.model.build_model()

    def __del__(self):
        self.close()

    def close(self):
        del self.train_parameter_matrix
        del self.train_targets
        del self.test_parameter_matrix
        del self.test_targets
        del self.num_iterations
        del self.learning_rate
        del self.log_cost
        del self.model

def main():
    load_data('/home/durzo/Datasets/fer2013/fer2013.csv')

if __name__ == '__main__':
    main()
