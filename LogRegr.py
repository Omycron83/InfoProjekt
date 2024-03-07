import hyperparmeter_optimization
from skopt.space import Real, Integer
import supervised
import numpy as np
from sklearn.linear_model import LogisticRegression

class LogisticClass(supervised.MachineLearningModel):
    def __init__(self, params, dim_features=None, dim_labels=None) -> None:
        self.theta = np.random.rand(dim_features + 1, dim_labels)
        self.alpha = params[0]
        self._lambda = params[1]

    def sigmoid(self, X):
        sig = 1 / (1 + np.exp(-X))
        sig = np.minimum(sig, 0.9999)
        sig = np.maximum(sig, 0.0001)
        return sig
    
    def train(self, features, labels):
        features = np.hstack((np.ones((features.shape[0], 1)), features))
        grad = np.ones(self.theta.size)
        grad[0] = np.sum((self.sigmoid(features@self.theta) - labels)*features[:,0])/len(labels)
        for i in range(1,len(grad)):
            grad[i] = np.sum((self.sigmoid(features@self.theta) - labels)*features[:,i])/len(labels) + self._lambda * self.theta[i]/len(labels)
        for i in range(5000):
            self.theta -= self.alpha*grad.reshape(self.theta.shape)

    def predict(self, features):
        features = np.hstack((np.ones((features.shape[0], 1)), features))
        return self.sigmoid(features @ self.theta)

class LogisticAutoClass(supervised.OptimizerClass):
    def optim(self, features, labels):
        super().optim(features, labels)
        self.opt_hyperparams = hyperparmeter_optimization.find_opt_hyperparameters([Real(0.0, 1.0), Real(0, 50)], LogisticClass, is_classifier=True, n_calls=20, features=features, labels=labels)
        self.model = LogisticClass(self.opt_hyperparams, dim_features=features.shape[1], dim_labels=labels.shape[1])
        self.train(features, labels)

def unit_test():
    f, l = np.zeros((50, 3)), np.zeros((50, 1))
    l[0:25] = 1
    _class = LogisticAutoClass(f, l)
    print(_class.pred(f), _class.cost(f, l))

if __name__ == '__main__':
    unit_test()