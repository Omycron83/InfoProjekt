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
    f, l = np.zeros(5, 3), np.zeros(5, 1)
    _class = DecisionTreeAutoClass(f, l)
    print(_class.pred(f), _class.cost(f, l))
    regr = DecisionTreeAutoRegr(f, l)
    print(regr.pred(f), _class.cost(f, l))

if __name__ == '__main__':
    unit_test()