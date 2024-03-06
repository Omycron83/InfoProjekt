import numpy as np
from random import choice

class KMeansClustering:
    def __init__(self, k, data) -> None:
        self.threshhold = 1e-6
        self.means = data[np.random.choice(len(X), k, replace=False), :]
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