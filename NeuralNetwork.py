import hyperparmeter_optimization
from skopt.space import Real, Integer
import supervised
import numpy as np
import NN

class NeuralNetworkRegr(supervised.MachineLearningModel):
    # nimmt ...
    # gibt ... aus
    # ergebniss ...
    def __init__(self, params, dim_features=None, dim_labels=None) -> None:
        if params[1] == 0:
            self.model = NN.cont_feedforward_nn(dim_features, [params[0]], NN.ReLU, NN.ReLUDeriv, NN.output, NN.MSE_out_deriv, dim_labels)
        self.model = NN.cont_feedforward_nn(dim_features, [params[0], params[1]], NN.ReLU, NN.ReLUDeriv, NN.output, NN.MSE_out_deriv, dim_labels)
        self.untrained_weights = self.model.retrieve_weights()
        self.norm = [0, 0]
        self.alpha = params[2]
        self._lambda = params[3]
        self.dropout =  params[4]

    # nimmt ...
    # gibt ... aus
    # ergebniss ...
    def train(self, features, labels):
        self.model.assign_weights(self.untrained_weights)
        std = np.std(features, axis = 0)
        std[std == 0] = 0.001
        self.norm[0], self.norm[1] = np.mean(features, axis = 0), std
        features_train = (features - self.norm[0]) / self.norm[1]
        for i in range(300):
            self.model.stochastic_gradient_descent(self.alpha, self._lambda, features_train, labels, NN.MSE, [self.dropout])
    # nimmt ...
    # gibt ... aus
    # ergebniss ...
    def predict(self, features):
        features_pred = (features - self.norm[0]) / self.norm[1]
        self.model.forward_propagation(features_pred, np.ones((features_pred.shape[0], self.model.len_output)), self.MSE)
        return self.model.output_layer()
    
class NeuralNetworkAutoRegr(supervised.OptimizerRegr):
    # nimmt ...
    # gibt ... aus
    # ergebniss ...
    def optim(self, features, labels):
        super().optim(features, labels)
        self.opt_hyperparams = hyperparmeter_optimization.find_opt_hyperparameters([Integer(10, 1024), Integer(0, 1024), Real(0.0000001, 0.1), Real(0.0, 10), Real(0, 1), Integer(1, 100)], NeuralNetworkRegr, is_classifier=False, n_calls=10, features=features, labels=labels)
        self.model = NeuralNetworkRegr(self.opt_hyperparams, dim_features = features.shape[1], dim_labels = labels.shape[1])
        self.train(features, labels)

class NeuralNetworkClass(supervised.MachineLearningModel):
    # nimmt ...
    # gibt ... aus
    # ergebniss ...
    def __init__(self, params, dim_features=None, dim_labels=None) -> None:
        if params[1] == 0:
            self.model = NN.cont_feedforward_nn(dim_features, [params[0]], NN.ReLU, NN.ReLUDeriv, NN.sigmoid, NN.Sigmoid_out_deriv, dim_labels)
        self.model = NN.cont_feedforward_nn(dim_features, [params[0], params[1]], NN.ReLU, NN.ReLUDeriv, NN.sigmoid, NN.Sigmoid_out_deriv, dim_labels)
        self.untrained_weights = self.model.retrieve_weights()
        self.norm = [0, 0]
        self.alpha = params[2]
        self._lambda = params[3]
        self.dropout =  params[4]
    # nimmt ...
    # gibt ... aus
    # ergebniss ...
    def train(self, features, labels):
        self.model.assign_weights(self.untrained_weights)
        std = np.std(features, axis = 0)
        std[std == 0] = 0.001
        self.norm[0], self.norm[1] = np.mean(features, axis = 0), std
        features_train = (features - self.norm[0]) / self.norm[1]
        for i in range(50):
            self.model.stochastic_gradient_descent(self.alpha, self._lambda, features_train, labels, NN.logistic_cost, [self.dropout])
    # nimmt ...
    # gibt ... aus
    # ergebniss ...
    def predict(self, features):
        features_pred = (features - self.norm[0]) / self.norm[1]
        self.model.forward_propagatioen(features_pred, np.ones((features_pred.shape[0], self.model.len_output)), NN.logistic_cost)
        return self.model.output_layer()
    
class NeuralNetworkAutoClass(supervised.OptimizerClass):
    # nimmt ...
    # gibt ... aus
    # ergebniss ...
    def optim(self, features, labels):
        super().optim(features, labels)
        self.opt_hyperparams = hyperparmeter_optimization.find_opt_hyperparameters([Integer(10, 1024), Integer(0, 1024), Real(0.0000001, 0.1), Real(0.0, 10), Real(0, 1), Integer(1, 100)], NeuralNetworkClass, is_classifier=True, n_calls=10, features=features, labels=labels)
        self.model = NeuralNetworkClass(self.opt_hyperparams, dim_features = features.shape[1], dim_labels = labels.shape[1])
        self.train(features, labels)

def unit_test():
    f, l = np.zeros((5, 3)), np.zeros((5, 1))
    _class = NeuralNetworkAutoClass(f, l)
    print(_class.pred(f), _class.cost(f, l))
    regr = NeuralNetworkAutoRegr(f, l)
    print(regr.pred(f), _class.cost(f, l))

if __name__ == '__main__':
    unit_test()