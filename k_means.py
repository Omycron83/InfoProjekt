import numpy as np
from random import choice

class KMeansClustering:
    def __init__(self, k, data) -> None:
        self.threshhold = 1e-6
        self.data = data
        self.means = data[np.random.choice(len(data), k, replace=False), :]
        self.get_means(data)

    def get_labels(self, data):
        distances = np.linalg.norm(data[:, np.newaxis] - self.means, axis=2)
        labels = np.argmin(distances, axis=1)
        return labels

    def get_means(self, data, iterations = 200):
        for i in range(iterations):
            labels = self.get_labels(data)
            new_means = np.array([data[labels == j].mean(axis=0) for j in range(len(self.means))])
            if np.linalg.norm(self.means - new_means) < self.threshhold:
                break
            else:
                self.means = new_means
                
def unit_test():
    f, l = np.zeros(5, 3), np.zeros(5, 1)
    _class = DecisionTreeAutoClass(f, l)
    print(_class.pred(f), _class.cost(f, l))
    regr = DecisionTreeAutoRegr(f, l)
    print(regr.pred(f), _class.cost(f, l))

if __name__ == '__main__':
    unit_test()