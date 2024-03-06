import hyperparmeter_optimization
from skopt.space import Real, Integer
import numpy as np
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier

def MSE(pred, Y):
    return np.sum((np.array(pred) - np.array(Y).reshape(np.array(pred).shape))**2) / (2 * np.array(pred).size)

class DecisionTreeRegr:
    def __init__(self, params, dim_features=None, dim_labels=None):
        self.tree_regr = DecisionTreeRegressor(max_depth=params[0], min_samples_split=params[1], min_samples_leaf=params[2], max_features=params[3], random_state=42)

    def train(self, features, labels):
        self.tree_regr.fit(features, labels)

    def predict(self, features):
        return self.tree_regr.predict(features)

class DecisionTreeAutoRegr:
    def __init__(self, features, labels):
        self.train_features = features
        self.train_labels = labels
        self.optimize_tree_regr(features, labels)

    def train_tree_regr(self, features, labels):
        self.model.train(features, labels)

    def optimize_tree_regr(self, features, labels):
        self.train_features, self.train_labels = features, labels
        self.opt_hyperparams = hyperparmeter_optimization.find_opt_hyperparameters([Integer(1, 50), Integer(2, 20), Integer(1, 16), Real(0.1, 1.0)], DecisionTreeRegr, MSE, n_calls=100, features=features, labels=labels)
        self.model = DecisionTreeRegr(self.opt_hyperparams)
        self.train_tree_regr(features, labels)

    def predict_tree_regr(self, new_features, params):
        return self.model.predict(new_features)

    def cost(self, features, labels):
        return MSE(self.predict_tree_regr(features, self.opt_hyperparams), labels)

def logistic_cost(pred, Y):
    return np.sum(-pred * np.log2(Y + (Y == 0)*0.0001) - (1-pred) * np.log2(1 - Y + (Y == 1)*0.0001)) / pred.shape[0]

class DecisionTreeClass:
    def __init__(self, params, dim_features=None, dim_labels=None):
        self.tree_class = DecisionTreeClassifier(max_depth=params[0], min_samples_split=params[1], min_samples_leaf=params[2], max_features=params[3], random_state=42)

    def train(self, features, labels):
        self.tree_class.fit(features, labels)

    def predict(self, features):
        return self.tree_class.predict(features)

class DecisionTreeAutoClsfr:
    def __init__(self, features, labels):
        self.train_features = features
        self.train_labels = labels
        self.optimize_decision_tree(features, labels)

    def train_decision_tree(self, features, labels):
        self.model.train(features, labels)

    def optimize_decision_tree(self, features, labels):
        self.train_features, self.train_labels = features, labels
        self.opt_hyperparams = hyperparmeter_optimization.find_opt_hyperparameters([Integer(1, 50), Integer(2, 20), Integer(1, 16), Real(0.1, 1.0)], DecisionTreeClass, MSE, n_calls=100, features=features, labels=labels)
        self.model = DecisionTreeClass(self.opt_hyperparams)
        self.train_decision_tree(features, labels)

    def predict_decision_tree(self, new_features, params):
        return self.model.predict(new_features)
    
    def cost(self, features, labels):
        return logistic_cost(self.model.predict(features), labels)