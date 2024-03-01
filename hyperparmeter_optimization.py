import numpy as np
from skopt import gp_minimize

def k_fold_cross_val(k, features, labels, train_func, pred_func, cost_func, seed = 0):
    #Shuffling
    np.random.seed(seed)
    p = np.random.permutation(features.shape[0])
    shuffled_features = features.copy()[p]
    shuffled_labels = labels.copy()[p]
    error = 0
    for l in range(k - 1):
        #The test data of the current fold
        test_features = shuffled_features[features.shape[0] // k * l:features.shape[0] // k * (l+1), :]
        test_labels = shuffled_labels[features.shape[0] // k * l:features.shape[0] // k * (l+1), :]
        #The remaining training data of the current fold
        train_features = np.vstack((shuffled_features[:features.shape[0] // k * l, :], shuffled_features[features.shape[0] // k * (l+1):, :]))
        train_labels = np.vstack((shuffled_labels[:features.shape[0] // k * l, :], shuffled_labels[features.shape[0] // k * (l+1):, :])) 
        #Now, train the model on the current fold
        train_func(train_features, train_labels)
        error += cost_func(pred_func(test_features), test_labels) / k

    #For the last fold, we dont really know the size of the holdout-set (we dont know about the divisibility of the amount of datapoints by k) so we do this seperately
    #The test data of the last fold 
    test_features = shuffled_features[features.shape[0] // k * (l + 1):, :]
    test_labels = shuffled_labels[features.shape[0] // k * (l + 1):, :]
    #The remaining training data of the last fold
    train_features = shuffled_features[:features.shape[0] // k * (l + 1), :]
    train_labels = shuffled_labels[:features.shape[0] // k * (l + 1), :]
    #Now, train the model on the current fold
    train_func(train_features, train_labels)
    error += cost_func(pred_func(test_features), test_labels) / k
    return error

def find_opt_hyperparameters(parameter_ranges, model_class, cost, n_calls, features, labels):
    def model_eval(params):
        model = model_class(params)
        return k_fold_cross_val(10, features, labels, model.train, model.predict, cost)
    return gp_minimize(model_eval, parameter_ranges, n_calls = n_calls)