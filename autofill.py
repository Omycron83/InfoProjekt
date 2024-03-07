import numpy as np
from sklearn.impute import KNNImputer

class AutoFill:
    def __init__(self) -> None:
        pass

    def mean_imputation(self, data):
        mean_values = np.nanmean(data, axis=0)

        nan_indices = np.isnan(data)

        data[nan_indices] = np.take(mean_values, np.where(nan_indices)[1])

    def k_nn_imputation(self, data, k=2):
        imputer = KNNImputer(n_neighbors=k)
        return imputer.fit_transform(data)

def unit_test():
    z = np.random.rand(40, 40)
    g = AutoFill()
    x, y = g.mean_imputation(z), g.k_nn_imputation(z)

if __name__ == '__main__':
    unit_test()