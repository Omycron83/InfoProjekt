import hyperparmeter_optimization
from skopt.space import Real, Integer
import xgboost

class XGBoostRegr:
    def __init__(self) -> None:
        pass

class XGBoostAutoRegr:
    def __init__(self, features, labels) -> None:
        self.train_features = features
        self.train_labels = labels
        self.opt_hyperparams = self.optimize_xgboost(features, labels)
        self.model = XGBoostRegr(self.opt_hyperparams)


    def train_xboost(self, features, labels):
        pass

    def optimize_xgboost(self, features, labels):
        self.opt_hyperparams = hyperparmeter_optimization.find_opt_hyperparameters([Integer(1, 8192), Real(0, 0.9999), Integer(1, 128), Real(0.0001, 0.1), Real(0, 20)], xgboost.XGBRegressor, xgboost.MSE)
    
    def pred(self, new_features):
        return self.model.predict(new_features)