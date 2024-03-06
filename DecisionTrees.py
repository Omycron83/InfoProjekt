import hyperparmeter_optimization
from skopt.space import Real, Integer
import supervised
import numpy as np
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier

class DecisionTreeRegr(supervised.MachineLearningModel):
    def __init__(self, params, dim_features=None, dim_labels=None):
        self.tree_regr = DecisionTreeRegressor(max_depth=params[0], min_samples_split=params[1], min_samples_leaf=params[2], max_features=params[3], random_state=42)

    def train(self, features, labels):
        self.tree_regr.fit(features, labels)

    def predict(self, features):
        return self.tree_regr.predict(features)

class DecisionTreeAutoRegr(supervised.OptimizerRegr):
    def optim(self, features, labels):
        super().optim(features, labels)
        self.opt_hyperparams = hyperparmeter_optimization.find_opt_hyperparameters([Integer(1, 50), Integer(2, 20), Integer(1, 16), Real(0.1, 1.0)], DecisionTreeRegr, self.cost, n_calls=100, features=features, labels=labels)
        self.model = DecisionTreeRegr(self.opt_hyperparams)
        self.train(features, labels)

class DecisionTreeClass(supervised.MachineLearningModel):
    def __init__(self, params, dim_features=None, dim_labels=None):
        self.tree_class = DecisionTreeClassifier(max_depth=params[0], min_samples_split=params[1], min_samples_leaf=params[2], max_features=params[3], random_state=42)

    def train(self, features, labels):
        self.tree_class.fit(features, labels)

    def predict(self, features):
        return self.tree_class.predict(features)

class DecisionTreeAutoClass(supervised.OptimizerRegr):
    def optim(self, features, labels):
        super().optim(features, labels)
        self.opt_hyperparams = hyperparmeter_optimization.find_opt_hyperparameters([Integer(1, 50), Integer(2, 20), Integer(1, 16), Real(0.1, 1.0)], DecisionTreeClass, self.cost, n_calls=100, features=features, labels=labels)
        self.model = DecisionTreeClass(self.opt_hyperparams)
        self.train(features, labels)

x = np.ones((100, 2))
y = np.zeros((100, 1))

g = DecisionTreeAutoRegr(x, y)
