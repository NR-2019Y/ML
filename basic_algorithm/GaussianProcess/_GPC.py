import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

# 参考 https://zhuanlan.zhihu.com/p/99436309

class GPC(object):
    def fit(self, X, y):
        # n_samples, n_features = X.shape
        label_obj = LabelEncoder()
        y_labels = label_obj.fit_transform(y)
        n_labels = len(label_obj.classes_)
        x_each = [X[y_labels == i, :] for i in range(n_labels)]
        x_mean_each = [xi.mean(axis=0) for xi in x_each]
        x_cov_each = [np.cov(xi.T) for xi in x_each]
        x_covinv_each = [np.linalg.inv(xi_cov) for xi_cov in x_cov_each]
        x_covdet_each = np.array([np.linalg.det(xi_cov) for xi_cov in x_cov_each])
        self.label_obj = label_obj
        self.x_mean = x_mean_each
        self.x_covinv_each = x_covinv_each
        self.x_covdet_each = x_covdet_each
        return self
    def _get_p(self, X):
        n_samples = X.shape[0]
        n_labels = len(self.label_obj.classes_)
        proba = np.empty((n_samples, n_labels))
        for i in range(n_samples):
            for j in range(n_labels):
                xi_scalej = X[i] - self.x_mean[j]
                proba[i, j] = np.exp(-0.5 * xi_scalej @ self.x_covinv_each[j] @ xi_scalej)
        proba /= np.sqrt(self.x_covdet_each)
        return proba
    def predict_proba(self, X):
        proba = self._get_p(X)
        proba = proba / proba.sum(axis=1, keepdims=True)
        return proba
    def predict(self, X):
        return self.label_obj.inverse_transform(self._get_p(X).argmax(axis=1))
    def score(self, X, y):
        return accuracy_score(y, self.predict(X))
