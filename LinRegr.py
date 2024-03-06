import hyperparmeter_optimization
from skopt.space import Real, Integer
import numpy as np

"""
Parameters: params[0] = degree, params[1] = lambda

PolynomialRegr: Base-class for facilitating the model itself
PolynomialRegrAuto: Class providing a higher-level mask that facilitates the model search and the prediction
"""

def MSE(pred, Y):
    return np.sum((np.array(pred) - np.array(Y).reshape(np.array(pred).shape))**2) / (2 * np.array(pred).size)

class linear_regression:
    def __init__(self, dim_features, dim_labels = 1, _lambda = 0) -> None:
        self.theta = np.zeros((dim_features + 1, dim_labels))
        self._lambda = _lambda
    
    def normal_eq(self, features, labels):
        features_bias = np.hstack((features, np.ones((features.shape[0], 1))))
        self.theta = np.linalg.pinv(features_bias.T @ features_bias)@ features_bias.T@labels

    def ridge_normal_eq(self, features, labels):
        features_bias = np.hstack((features, np.ones((features.shape[0], 1))))
        reg_matrix = np.identity(features_bias.shape[1])
        #We need to make sure to not regularize the bias weight, which is done by setting its entry to 0
        reg_matrix[features_bias.shape[1] - 1, features_bias.shape[1] - 1] = 0
        self.theta = np.linalg.pinv(features_bias.T @ features_bias + self._lambda * reg_matrix) @ features_bias.T @labels

    def predict(self, features):
        features_bias = np.hstack((features, np.ones((features.shape[0], 1))))
        return features_bias @ self.theta
    
    def MSE(self, features, labels):
        return np.sum(np.square(self.predict(features) - labels)) / (2 * np.array(labels).size)


#In polynomial regression, it is usually good to normalize the data
#In this case, we will save the mean and std used to normalize for training
#And then apply it when predicting
class PolynomialRegr(linear_regression):
    def __init__(self, params, dim_features, dim_labels=1) -> None:
        self.degree = params[0]
        self._lambda = params[1]
        super().__init__(dim_features * self.degree, dim_labels, self._lambda)

    def polynomialize(self, features):
        return np.hstack(features ** np.arange(1, self.degree + 1)[:, None, None])
    
    def normal_eq(self, features, labels):
        self.mean = np.mean(features, axis = 0)
        self.std = np.std(features, axis = 0)
        features = (features - self.mean) / self.std

        return super().normal_eq(self.polynomialize(features), labels)
    
    def ridge_normal_eq(self, features, labels):
        features = self.polynomialize(features)
        self.mean = np.mean(features, axis = 0)
        self.std = np.std(features, axis = 0)
        self.std[self.std == 0] = 0.001
        features = (features - self.mean) / self.std
        return super().ridge_normal_eq(features, labels)
    
    def MSE(self, features, labels):
        return np.sum(np.square(self.predict(features) - labels)) / (2 * np.array(labels).size)

    def train(self, features, labels):
        self.ridge_normal_eq(features, labels)

    def predict(self, features):
        features = self.polynomialize(features)
        features = (features - self.mean) / self.std
        return super().predict(features)

class PolynomialRegrAuto:
    def __init__(self, features, labels) -> None:
        self.train_features = features
        self.train_labels = labels
        self.optimize_polyregr(features, labels)
        
    def train_polyregr(self, features, labels):
        self.model.ridge_normal_eq(features, labels)

    def optimize_polyregr(self, features, labels):
        self.train_features, self.train_labels = features, labels
        self.opt_hyperparams = hyperparmeter_optimization.find_opt_hyperparameters([Integer(1, 20), Real(0, 40)], PolynomialRegr, MSE, n_calls=100, features=features, labels=labels)
        self.model = PolynomialRegr(self.opt_hyperparams, dim_features = features.shape[1], dim_labels = labels.shape[1])
        self.train_polyregr(features, labels)

    def pred(self, new_features):
        return self.model.predict(new_features)

    def cost(self, features, labels):
        return MSE(self.pred(features), labels)