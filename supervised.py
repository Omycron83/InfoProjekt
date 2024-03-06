import numpy as np
from abc import ABC, abstractmethod

#Basic frameworks for all of the base-models to inherit from 
class MachineLearningModel(ABC):
    @abstractmethod
    def __init__(self, params, dim_features=None, dim_labels=None) -> None:
        pass
    @abstractmethod
    def train(self, features, labels):
        pass
    @abstractmethod
    def predict(self, features):
        pass

    def MSE(self, features, labels):
        pred, Y = self.predict(features), labels
        return np.sum((np.array(pred) - np.array(Y).reshape(np.array(pred).shape))**2) / (2 * np.array(pred).size)
    def Log(self, features, labels):
        pred, Y = self.predict(features), labels
        return np.sum(-pred * np.log2(Y + (Y == 0)*0.0001) - (1-pred) * np.log2(1 - Y + (Y == 1)*0.0001)) / pred.shape[0]

#Basic framework for all of the optimizer-models to inherit from
class OptimizerRegr(ABC):
    def __init__(self, features, labels):
        self.train_features = features
        self.train_labels = labels
        self.optim(features, labels)

    def train(self, features, labels):
        self.model.train(features, labels)
    
    @abstractmethod
    def optim(self, features, labels):
        self.train_features, self.train_labels = features, labels

    def pred(self, new_features):
        return self.model.predict(new_features)

    def cost(self, features, labels):
        return self.model.MSE(features, labels)

class OptimizerClass(ABC):
    def __init__(self, features, labels):
        self.train_features = features
        self.train_labels = labels
        self.optim(features, labels)

    def train(self, features, labels):
        self.model.train(features, labels)
    
    @abstractmethod
    def optim(self, features, labels):
        self.train_features, self.train_labels = features, labels

    def pred(self, new_features):
        return self.model.predict(new_features)

    def cost(self, features, labels):
        return self.model.Log(features, labels)