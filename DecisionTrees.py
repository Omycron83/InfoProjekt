import hyperparmeter_optimization
from skopt.space import Real, Integer
import xgboost

class XGBoostAutoRegr:
    def __init__(self, params, features_dim_1, labels_dim_1) -> None:
        self.opt_hyperparams = []

    def train_xboost(self, )
    def optimize_xgboost(self, features, labels):
        self.opt_hyperparams = hyperparmeter_optimization.find_opt_hyperparameters([Integer(1, 8192), Real(0, 0.9999), Integer(1, 128), Real(0.0001, 0.1), Real(0, 20)], xgboost.XGBRegressor, xgboost.MSE)
        