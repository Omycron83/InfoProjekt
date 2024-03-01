import numpy as np
class linear_regression:
    def __init__(self, dim, dim_2 = 1, _lambda = 0) -> None:
        self.theta = np.zeros((dim + 1, dim_2))
        self._lambda = _lambda
    
    def normal_eq(self, features, labels):
        features_bias = np.hstack((features, np.ones((features.shape[0], 1))))
        self.theta = np.linalg.pinv(features_bias.T @ features_bias)@ features_bias.T@labels

    def ridge_normal_eq(self, features, labels):
        features_bias = np.hstack((features, np.ones((features.shape[0], 1))))
        reg_matrix = np.identity(features_bias.shape[1])
        #We need to make sure to not regularize the bias weight, which is done by setting its entry to 0
        reg_matrix[features_bias.shape[1] - 1, features_bias.shape[1] - 1] = 0
        self.theta = np.linalg.pinv(features_bias.T @ features_bias + self._lambda * reg_matrix) @ features_bias.T @labels

    def predict(self, features):
        features_bias = np.hstack((features, np.ones((features.shape[0], 1))))
        return features_bias @ self.theta
    
    def MSE(self, features, labels):
        return np.sum(np.square(self.predict(features) - labels)) / (2 * np.array(labels).size)


#In polynomial regression, it is usually good to normalize the data
#In this case, we will save the mean and std used to normalize for training
#And then apply it when predicting
class polynomial_regression(linear_regression):
    def __init__(self, dim, dim_2=1, _lambda=0, degree = 1) -> None:
        self.degree = degree
        super().__init__(dim * degree, dim_2, _lambda)

    def polynomialize(self, features):
        return np.hstack(features ** np.arange(1, self.degree + 1)[:, None, None])
    
    def normal_eq(self, features, labels):
        self.mean = np.mean(features, axis = 0)
        self.std = np.std(features, axis = 0)
        features = (features - self.mean) / self.std

        return super().normal_eq(self.polynomialize(features), labels)
    
    def ridge_normal_eq(self, features, labels):
        features = self.polynomialize(features)
        self.mean = np.mean(features, axis = 0)
        self.std = np.std(features, axis = 0)
        self.std[self.std == 0] = 0.001
        features = (features - self.mean) / self.std
        return super().ridge_normal_eq(features, labels)
    
    def MSE(self, features, labels):
        return np.sum(np.square(self.predict(features) - labels)) / (2 * np.array(labels).size)

    def predict(self, features):
        features = self.polynomialize(features)
        features = (features - self.mean) / self.std
        return super().predict(features)
        
    def optimize_polyregr(features, labels):
