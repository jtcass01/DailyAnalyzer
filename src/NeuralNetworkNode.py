from LogisticModel import LogisticModel
from utilities import load_fer2013_data

class NeuralNetworkNode(object):
    def __init__(self, train_parameter_matrix, train_targets, test_parameter_matrix, test_targets, num_iterations, learning_rate = 0.1, log_cost = False):
        self.model = LogisticModel(train_parameter_matrix, train_targets, test_parameter_matrix, test_targets, num_iterations, 0.1, False)
        self.model.build_model()

    def __del__(self):
        self.close()

    def close(self):
        del self.model

def run():
    print("Loading data...")
    train_X, train_Y, test_X, test_Y = load_fer2013_data('../fer2013/fer2013.csv')
    print("Data successfully loaded.")

    node = NeuralNetworkNode(train_X, train_Y, test_X, test_Y, num_iterations=1000, learning_rate=0.1)
    
if __name__ == '__main__':
    run()
