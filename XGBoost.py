import hyperparmeter_optimization
from skopt.space import Real, Integer
import xgboost
import numpy as np

"""
Parameters: params[0] = gamma, params[1] = learning_rate, params[2] = max_depth, params[3] = n_estimators, params[4] = sub_sample, params[5] = min_child_weight, params[6] = reg_alpha, params[7] = reg_lambda

XGBoostRegr: Base-class for facilitating the model itself
XGBoostAutoRegr: Class providing a higher-level mask that facilitates the model search and the prediction
"""

def MSE(pred, Y):
    return np.sum((np.array(pred) - np.array(Y).reshape(np.array(pred).shape))**2) / (2 * np.array(pred).size)

class XGBoostRegr:
    def __init__(self, params, dim_features = None, dim_labels = None) -> None:
        self.xgboost_reg = xgboost.XGBRegressor(gamma = params[0], learning_rate = params[1], max_depth = params[2], n_estimators = params[3], n_jobs = 16, objective = 'reg:squarederror', subsample = params[4], scale_pos_weight = 0, reg_alpha = params[6], reg_lambda = params[7], min_child_weight = params[5])

    def train(self, features, labels):
        self.xgboost_reg.fit(features, labels)
    def predict(self, features):
        self.xgboost_reg.predict(features)

class XGBoostAutoRegr:
    def __init__(self, features, labels) -> None:
        self.train_features = features
        self.train_labels = labels
        self.optimize_xgboost(features, labels)
        
    def train_xgboost(self, features, labels):
        self.model.train(features, labels)

    def optimize_xgboost(self, features, labels):
        self.opt_hyperparams = hyperparmeter_optimization.find_opt_hyperparameters([Integer(1, 8192), Real(0, 0.9999), Integer(1, 128), Real(0.0001, 0.1), Real(0, 20)], XGBoostRegr, MSE, n_calls=100, features=features, labels=labels)
        self.model = XGBoostRegr(self.opt_hyperparams)
        self.train_xgboost(features, labels)

    def pred(self, new_features):
        return self.model.predict(new_features)
    
    def cost(self, features, labels):
        return MSE(self.pred(features), labels)

"""
Parameters: params[0] = gamma, params[1] = learning_rate, params[2] = max_depth, params[3] = n_estimators, params[4] = sub_sample, params[5] = min_child_weight, params[6] = reg_alpha, params[7] = reg_lambda

XGBoostClass: Base-class for facilitating the model itself
XGBoostAutoClass: Class providing a higher-level mask that facilitates the model search and the prediction
"""

def MSE(pred, Y):
    return np.sum(-pred * np.log2(Y + (Y == 0)*0.0001) - (1-pred) * np.log2(1 - Y + (Y == 1)*0.0001)) / pred.shape[0]

class XGBoostClass:
    def __init__(self, params, dim_features = None, dim_labels = None) -> None:
        self.xgboost_reg = xgboost.XGBClassifier(gamma = params[0], learning_rate = params[1], max_depth = params[2], n_estimators = params[3], n_jobs = 16, objective = 'binary:logistic', subsample = params[4], scale_pos_weight = 0, reg_alpha = params[6], reg_lambda = params[7], min_child_weight = params[5])

    def train(self, features, labels):
        self.xgboost_class.fit(features, labels)
    def predict(self, features):
        self.xgboost_class.predict(features)

class XGBoostAutoClass:
    def __init__(self, features, labels) -> None:
        self.train_features = features
        self.train_labels = labels
        self.optimize_xgboost(features, labels)
        
    def train_xgboost(self, features, labels):
        self.model.train(features, labels)

    def optimize_xgboost(self, features, labels):
        self.opt_hyperparams = hyperparmeter_optimization.find_opt_hyperparameters([Integer(1, 8192), Real(0, 0.9999), Integer(1, 128), Real(0.0001, 0.1), Real(0, 20)], XGBoostRegr, MSE, n_calls=100, features=features, labels=labels)
        self.model = XGBoostRegr(self.opt_hyperparams)
        self.train_xgboost(features, labels)

    def pred(self, new_features):
        return self.model.predict(new_features)
    
    def cost(self, features, labels):
        return MSE(self.pred(features), labels)
