import my_adaboost
import numpy as np

class FastSimpleClassifier(my_adaboost.ClassifierBase):
    def fit(self, X, y, sample_weight=None):
        n_samples = len(y)
        if sample_weight is None:
            sample_weight = (1.0 / n_samples) * np.ones(n_samples)
        n_features = X.shape[1]
        self.acc_score = 0.0
        for i in range(n_features):
            xi = X[:, i]
            train_y_pred = (xi > 0.5).astype(y.dtype)
            curr_acc = my_adaboost.accuracy(y, train_y_pred, sample_weight=sample_weight)
            curr_sig = 1
            if curr_acc < 0.5:
                train_y_pred = 1 - train_y_pred
                curr_acc = 1 - curr_acc
                curr_sig = -1
            if curr_acc > self.acc_score:
                self.train_y_pred = train_y_pred
                self.acc_score = curr_acc
                self.sig = curr_sig
                self.feature_index = i
        return self
    def predict(self, X):
        if self.sig == 1:
            return (X[:, self.feature_index] > 0.5).astype(np.int32)
        else:
            return (X[:, self.feature_index] <= 0.5).astype(np.int32)

