import hyperparmeter_optimization
from skopt.space import Real, Integer
import xgboost
import supervised
import numpy as np

"""
Parameters: params[0] = gamma, params[1] = learning_rate, params[2] = max_depth, params[3] = n_estimators, params[4] = sub_sample, params[5] = min_child_weight, params[6] = reg_alpha, params[7] = reg_lambda

XGBoostRegr: Base-class for facilitating the model itself
XGBoostAutoRegr: Class providing a higher-level mask that facilitates the model search and the prediction
"""

class XGBoostRegr(supervised.MachineLearningModel):
    def __init__(self, params, dim_features = None, dim_labels = None) -> None:
        self.xgboost_reg = xgboost.XGBRegressor(gamma = params[0], learning_rate = params[1], max_depth = params[2], n_estimators = params[3], n_jobs = 16, objective = 'reg:squarederror', subsample = params[4], scale_pos_weight = 0, reg_alpha = params[6], reg_lambda = params[7], min_child_weight = params[5])

    def train(self, features, labels):
        self.xgboost_reg.fit(features, labels)

    def predict(self, features):
        return self.xgboost_reg.predict(features)

class XGBoostAutoRegr(supervised.OptimizerRegr):
    def optim(self, features, labels):
        super().optim(features, labels)
        self.opt_hyperparams = hyperparmeter_optimization.find_opt_hyperparameters([Real(0, 20), Real(0.01, 0.5), Integer(3, 10), Integer(100, 1100), Real(0.5, 1), Integer(1, 10), Real(0, 5), Real(0, 5)], XGBoostRegr, is_classifier=False, n_calls=20, features=features, labels=labels)
        self.model = XGBoostRegr(self.opt_hyperparams)
        self.train(features, labels)

"""
Parameters: params[0] = gamma, params[1] = learning_rate, params[2] = max_depth, params[3] = n_estimators, params[4] = sub_sample, params[5] = min_child_weight, params[6] = reg_alpha, params[7] = reg_lambda

XGBoostClass: Base-class for facilitating the model itself
XGBoostAutoClass: Class providing a higher-level mask that facilitates the model search and the prediction
"""

class XGBoostClass(supervised.MachineLearningModel):
    def __init__(self, params, dim_features = None, dim_labels = None) -> None:
        self.xgboost_class = xgboost.XGBClassifier(gamma = params[0], learning_rate = params[1], max_depth = params[2], n_estimators = params[3], n_jobs = 16, objective = 'binary:logistic', subsample = params[4], scale_pos_weight = 0, reg_alpha = params[6], reg_lambda = params[7], min_child_weight = params[5])

    def train(self, features, labels):
        self.xgboost_class.fit(features, labels)

    def predict(self, features):
        return self.xgboost_class.predict(features)

class XGBoostAutoClass(supervised.OptimizerClass):
    def optim(self, features, labels):
        super().optim(features, labels)
        self.opt_hyperparams = hyperparmeter_optimization.find_opt_hyperparameters([Real(0, 20), Real(0.01, 0.5), Integer(3, 10), Integer(100, 1100), Real(0.5, 1), Integer(1, 10), Real(0, 5), Real(0, 5)], XGBoostRegr, is_classifier=True, n_calls=20, features=features, labels=labels)
        self.model = XGBoostRegr(self.opt_hyperparams)
        self.train(features, labels)

def unit_test():
    f, l = np.zeros((20, 3)), np.zeros((20, 1))
    _class = XGBoostAutoClass(f, l)
    print(_class.pred(f), _class.cost(f, l))
    regr = XGBoostAutoRegr(f, l)
    print(regr.pred(f), _class.cost(f, l))

if __name__ == '__main__':
    unit_test()