# 参考
# https://zhuanlan.zhihu.com/p/343196025

import numpy as np
from sklearn.preprocessing import OneHotEncoder

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

def accuracy(y_true, y_pred, *, sample_weight=None):
    if sample_weight is None:
        return np.mean(y_true == y_pred)
    else:
        return np.sum(sample_weight * (y_true == y_pred))

class ClassifierBase(object):
    def get_params(self, deep=True):
        return self.__dict__
    def predict(self, X):
        pass
    def score(self, X, y, sample_weight=None):
        return accuracy(y, self.predict(X), sample_weight=sample_weight)

class SimpleClassifier(ClassifierBase):
    def fit(self, X, y, sample_weight=None):
        n_samples = len(y)
        if sample_weight is None:
            sample_weight = (1.0 / n_samples) * np.ones(n_samples)
        n_features = X.shape[1]
        self.acc_score = 0.0
        for i in range(n_features):
            xi = X[:, i]
            uxi = np.unique(xi)
            threshold_list = 0.5 * (uxi[:-1] + uxi[1:])
            for threshold in threshold_list:
                train_y_pred = (xi > threshold).astype(y.dtype)
                curr_acc = accuracy(y, train_y_pred, sample_weight=sample_weight)
                curr_sig = 1
                if curr_acc < 0.5:
                    train_y_pred = 1 - train_y_pred
                    curr_acc = 1 - curr_acc
                    curr_sig = -1
                if curr_acc > self.acc_score:
                    self.train_y_pred = train_y_pred
                    self.acc_score = curr_acc
                    self.sig = curr_sig
                    self.threshold = threshold
                    self.feature_index = i
        return self
    def predict(self, X):
        if self.sig == 1:
            return (X[:, self.feature_index] > self.threshold).astype(np.int32)
        else:
            return (X[:, self.feature_index] <= self.threshold).astype(np.int32)

class AdaBoostSolver(ClassifierBase):
    def __init__(self, base_estimator, *, n_estimators=50):
        self.base_estimator = base_estimator
        self.n_estimators = n_estimators
    def fit(self, X, y, sample_weight=None):
        n_samples = len(y)
        if sample_weight is None:
            sample_weight = (1.0 / n_samples) * np.ones(n_samples)
        self.models = []
        for i in range(self.n_estimators):
            base_estimator_obj = self.base_estimator.__class__(**self.base_estimator.get_params())
            base_estimator_obj.fit(X, y, sample_weight=sample_weight)
            y_pred = base_estimator_obj.train_y_pred
            acc = base_estimator_obj.acc_score
            del base_estimator_obj.train_y_pred
            del base_estimator_obj.acc_score
            alpha = 0.5 * np.log(acc / (1 - acc))
            # print("alpha acc sig thr", alpha, acc, base_estimator_obj.sig, base_estimator_obj.threshold)
            print("alpha acc", alpha, acc)
            self.models.append((alpha, base_estimator_obj))
            sample_weight *= np.exp(-alpha * np.where(y == y_pred, 1, -1))
            sample_weight /= sample_weight.sum()
        return self
    def predict(self, X):
        y_pred = 0
        for alpha, model in self.models:
            y_pred += alpha * (2 * model.predict(X) - 1)
        return (y_pred > 0).astype(np.int32)

class AdaBoostMulticlassSolver(ClassifierBase):
    def __init__(self, base_estimator, *, n_estimators=50):
        self.base_estimator = base_estimator
        self.n_estimators = n_estimators
    def fit(self, X, y, sample_weight=None):
        n_samples = len(y)
        oobj = OneHotEncoder(sparse=False, categories='auto')
        yhot = oobj.fit_transform(y.reshape((-1, 1)))
        n_labels = yhot.shape[1]
        self.oobj = oobj
        self.n_labels = n_labels
        if sample_weight is None:
            sample_weight = (1.0 / n_samples) * np.ones((n_samples, n_labels))
        self.models = []
        for i in range(self.n_estimators):
            curr_model = []
            for k in range(n_labels):
                base_estimator_obj = self.base_estimator.__class__(**self.base_estimator.get_params())
                base_estimator_obj.fit(X, yhot[:, k], sample_weight=sample_weight[:, k])
                y_pred = base_estimator_obj.train_y_pred
                acc = base_estimator_obj.acc_score
                del base_estimator_obj.train_y_pred
                del base_estimator_obj.acc_score
                alpha = 0.5 * np.log(acc / (1 - acc))
                # print("alpha acc sig thr", alpha, acc, base_estimator_obj.sig, base_estimator_obj.threshold)
                print(i, k, "alpha acc", alpha, acc)
                curr_model.append((alpha, base_estimator_obj))
                sample_weight[:, k] *= np.exp(-alpha * np.where(yhot[:, k] == y_pred, 1, -1))
                sample_weight[:, k] /= sample_weight[:, k].sum()
            self.models.append(curr_model)
            print(i, "current acc", self.score(X, y))
        return self
    def predict(self, X):
        y_pred_hot = np.zeros((X.shape[0], self.n_labels))
        for curr_list in self.models:
            for k, (alpha, model) in enumerate(curr_list):
                y_pred_hot[:, k] += alpha * (2 * model.predict(X) - 1)
        # return self.oobj.inverse_transform(y_pred_hot > 0)
        # return (y_pred_hot > 0).argmax(axis=1)
        return self.oobj.categories_[0][y_pred_hot.argmax(axis=1)]

def main():
    import my_get_data
    import matplotlib.pyplot as plt
    dobj = my_get_data.DataCreater(my_get_data.yfunc2)
    x, y = dobj.gen_rand_data(size=500)
    # aobj = AdaBoostSolver(SimpleClassifier(), n_estimators=50)
    aobj = AdaBoostMulticlassSolver(SimpleClassifier(), n_estimators=100)
    aobj.fit(x, y)
    print("准确率", aobj.score(x, y))
    my_get_data.plot_full_region(dobj, aobj.predict, x=x, y_true=y, size=200, show_scatter=True, print_accuracy=True)

if __name__ == '__main__':
    main()
