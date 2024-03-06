import hyperparmeter_optimization
from skopt.space import Real, Integer
import supervised
import numpy as np
from sklearn.linear_model import LogisticRegression

class LogisticClass(supervised.MachineLearningModel):
    def __init__(self, params, dim_features=None, dim_labels=None) -> None:
        self.log_class = LogisticRegression(C = params[0], random_state=42)
    
    def train(self, features, labels):
        self.log_class.fit(features, labels)

    def predict(self, features):
        return self.log_class.predict(features)

class LogisticAutoClass(supervised.OptimizerClass):
    def optim(self, features, labels):
        super().optim(features, labels)
        self.opt_hyperparams = hyperparmeter_optimization.find_opt_hyperparameters([Integer(1, 50), Integer(2, 20), Integer(1, 16), Real(0.1, 1.0)], LogisticClass, self.cost, n_calls=100, features=features, labels=labels)
        self.model = LogisticClass(self.opt_hyperparams)
        self.train(features, labels)

def unit_test():
    f, l = np.zeros(5, 3), np.zeros(5, 1)
    _class = DecisionTreeAutoClass(f, l)
    print(_class.pred(f), _class.cost(f, l))
    regr = DecisionTreeAutoRegr(f, l)
    print(regr.pred(f), _class.cost(f, l))

if __name__ == '__main__':
    unit_test()