import numpy as np
from sklearn.impute import KNNImputer

def mean_imputation(data):
    mean_values = np.nanmean(data, axis=0)

    nan_indices = np.isnan(data)

    data[nan_indices] = np.take(mean_values, np.where(nan_indices)[1])

def k_nn_imputation(data, k=2):
    imputer = KNNImputer(n_neighbors=k)
    return imputer.fit_transform(data)

def unit_test():
    z = np.random.rand(40, 40)
    x, y = mean_imputation(z), k_nn_imputation(z)

if __name__ == '__main__':
    unit_test()