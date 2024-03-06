import numpy as np

class PCA:
    def __init__(self, num_components) -> None:
        self.num_components = num_components
    
    def pca(self, data):
        data = (data - np.mean(data, axis = 0)) / np.std(data, axis = 0)
        cov = np.cov(data, rowvar=False)
        eigenvalues, eigenvectors = np.linalg.eigh(cov)

        sorted_indices = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[sorted_indices]
        eigenvectors = eigenvectors[:, sorted_indices]

        top_eigenvectors = eigenvectors[:, :self.num_components]
        return data @ top_eigenvectors

def unit_test():
    f, l = np.zeros(5, 3), np.zeros(5, 1)
    _class = DecisionTreeAutoClass(f, l)
    print(_class.pred(f), _class.cost(f, l))
    regr = DecisionTreeAutoRegr(f, l)
    print(regr.pred(f), _class.cost(f, l))

if __name__ == '__main__':
    unit_test()