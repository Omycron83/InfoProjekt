import hyperparmeter_optimization
from skopt.space import Real, Integer
import supervised
import numpy as np
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

class RandomForestRegr(supervised.MachineLearningModel):
    def __init__(self, params, dim_features=None, dim_labels=None) -> None:
        self.regr_forest = RandomForestRegressor(n_estimators=params[0], max_depth=params[1], min_samples_split=params[2], min_samples_leaf=params[3], random_state=42)
    
    def train(self, features, labels):
        self.regr_forest.fit(features, labels)

    def predict(self, features):
        return self.regr_forest.predict(features)

class RandomForestAutoRegr(supervised.OptimizerRegr):
    def optim(self, features, labels):
        super().optim(features, labels)
        self.opt_hyperparams = hyperparmeter_optimization.find_opt_hyperparameters([Integer(10, 1000), Integer(1, 50), Integer(2, 100), Integer(1, 100)], RandomForestRegr, self.cost, n_calls=100, features=features, labels=labels)
        self.model = RandomForestRegr(self.opt_hyperparams)
        self.train(features, labels)
    
class RandomForestClass(supervised.MachineLearningModel):
    def __init__(self, params, dim_features=None, dim_labels=None) -> None:
        self.regr_forest = RandomForestClassifier(n_estimators=params[0], max_depth=params[1], min_samples_split=params[2], min_samples_leaf=params[3], random_state=42)
    
    def train(self, features, labels):
        self.regr_forest.fit(features, labels)

    def predict(self, features):
        return self.regr_forest.predict(features)

class RandomForestAutoClass(supervised.OptimizerClass):
    def optim(self, features, labels):
        super().optim(features, labels)
        self.opt_hyperparams = hyperparmeter_optimization.find_opt_hyperparameters([Integer(10, 1000), Integer(1, 50), Integer(2, 100), Integer(1, 100)], RandomForestClassifier, self.cost, n_calls=100, features=features, labels=labels)
        self.model = RandomForestClassifier(self.opt_hyperparams)
        self.train(features, labels)

def unit_test():
    f, l = np.zeros(5, 3), np.zeros(5, 1)
    _class = DecisionTreeAutoClass(f, l)
    print(_class.pred(f), _class.cost(f, l))
    regr = DecisionTreeAutoRegr(f, l)
    print(regr.pred(f), _class.cost(f, l))

if __name__ == '__main__':
    unit_test()