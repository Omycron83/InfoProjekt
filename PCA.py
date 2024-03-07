import numpy as np

class PCA:
    def __init__(self, num_components=5) -> None:
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
    z = np.random.rand(40, 40)
    g = PCA(10)
    x = g.pca(z)

if __name__ == '__main__':
    unit_test()