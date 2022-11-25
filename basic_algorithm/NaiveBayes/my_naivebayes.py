import numpy as np
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder, OneHotEncoder
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import accuracy_score
from sklearn.utils.extmath import safe_sparse_dot
from abc import abstractmethod

# 参考
# sklearn源码
# github.com/carefree0910

class ClassifierBase(object):
    @abstractmethod
    def _predict_p(self, X):
        pass
    def predict(self, X):
        return self.label_encoder.inverse_transform(self._predict_p(X).argmax(axis=1))
    def predict_proba(self, X):
        pori = self._predict_p(X)
        return pori / np.sum(pori, axis=1, keepdims=True)
    def score(self, X, y):
        return accuracy_score(y, self.predict(X))

class MyNaiveBayes1(ClassifierBase):
    # 相当于 CategoricalNB
    # 对先验概率也进行了拉普拉斯平滑处理
    def __init__(self, *, alpha=0.0):
        self.alpha = alpha
    def fit(self, X : np.ndarray, y : np.ndarray):
        # X : 各特征是离散的
        _, n_features = X.shape
        self.feature_encoder = OrdinalEncoder(dtype=np.int64)
        X = self.feature_encoder.fit_transform(X)
        self.label_encoder = LabelEncoder()
        y = self.label_encoder.fit_transform(y)
        n_classes = len(self.label_encoder.classes_)

        # 各特征的频数
        feature_category_count = [None] * n_features
        for i in range(n_features):
            curr_feature_classes = np.max(X[:, i]).astype(np.int64) + 1
            curr_feature_category_count = np.empty((n_classes, curr_feature_classes), dtype=np.float64)
            for j in range(n_classes):
                curr_feature_category_count[j] = np.bincount(X[y == j, i], minlength=curr_feature_classes).astype(np.float64)
            curr_feature_category_count += self.alpha
            feature_category_count[i] = curr_feature_category_count

        # 先验概率
        class_count = np.bincount(y) + self.alpha
        self.log_class_prior = np.log(class_count / (len(y) + self.alpha * n_classes))

        self.log_feature_category_freq = [
            np.log(count / np.sum(count, axis=1, keepdims=True)) for count in feature_category_count
        ]

        return self

    def _predict_p(self, X):
        X = self.feature_encoder.transform(X)
        n_samples = len(X)
        n_classes = len(self.label_encoder.classes_)
        log_class_prior = self.log_class_prior

        log_pori = np.full((n_samples, n_classes), log_class_prior)
        for i in range(n_samples):
            for curr_x, curr_fq in zip(X[i], self.log_feature_category_freq):
                log_pori[i] += curr_fq[:, curr_x]

        return np.exp(log_pori)

# 独热处理，代码更简洁，功能同MyNaiveBayes1
class MyNaiveBayes2(ClassifierBase):
    def __init__(self, *, alpha=0.0):
        self.alpha = alpha
    def fit(self, X : np.ndarray, y : np.ndarray):
        # n_features, n_samples = X.shape
        self.feature_encoder = OneHotEncoder(sparse=False, categories='auto')
        X = self.feature_encoder.fit_transform(X)

        self.label_encoder = LabelEncoder()
        y = self.label_encoder.fit_transform(y)
        n_classes = len(self.label_encoder.classes_)
        y = np.eye(n_classes)[y]

        feature_category_fqcalc = np.dot(y.T, X)
        feature_category_fqcalc += self.alpha
        n_feature_index = np.r_[0, np.cumsum( [len(v) for v in self.feature_encoder.categories_] )]
        for ifirst, ilast in zip(n_feature_index[:-1], n_feature_index[1:]):
            feature_category_fqcalc[:, ifirst:ilast] /= np.sum(
                feature_category_fqcalc[:, ifirst:ilast], axis=1, keepdims=True
            )
        n_classes = y.shape[1]
        class_count = np.sum(y, axis=0) + self.alpha
        self.log_class_prior = np.log(class_count / (len(y) + self.alpha * n_classes))
        self.log_feature_category_freq = np.log(feature_category_fqcalc)
        return self
    def _predict_p(self, X):
        X = self.feature_encoder.transform(X)
        log_pori = np.dot(X, self.log_feature_category_freq.T) + self.log_class_prior
        return np.exp(log_pori)

# 多项分布
class MyMultinomialNaiveBayes(ClassifierBase):
    def __init__(self, *, alpha=0.0):
        self.alpha = alpha
    def fit(self, X : np.ndarray, y : np.ndarray):
        self.label_encoder = LabelEncoder()
        y = self.label_encoder.fit_transform(y)
        n_classes = len(self.label_encoder.classes_)
        y = np.eye(n_classes, dtype=np.float64)[y]

        class_count = np.sum(y, axis=0) + self.alpha
        self.log_class_prior = np.log( class_count / (len(y) + self.alpha * n_classes) )

        # pandas : groupby + sum 也能实现此功能
        feature_count = np.dot(y.T, X) + self.alpha
        feature_freq = feature_count / np.sum(feature_count, axis=1, keepdims=True)
        self.log_feature_prob = np.log(feature_freq)
        return self
    def _predict_p(self, X):
        log_pori = np.dot(X, self.log_feature_prob.T) + self.log_class_prior
        return np.exp(log_pori)

# 高斯分布
class MyGaussianNaiveBayes(ClassifierBase):
    def fit(self, X, y):
        _, n_features = X.shape
        self.label_encoder = LabelEncoder()
        y = self.label_encoder.fit_transform(y)
        n_classes = len(self.label_encoder.classes_)
        var_mat = np.empty((n_classes, n_features), dtype=np.float64)
        mu_mat =  np.empty_like(var_mat)
        for i in range(n_classes):
            imask = y == i
            var_mat[i] = np.var(X[imask, :], axis=0)
            mu_mat[i]  = np.mean(X[imask, :], axis=0)
        self.var_mat = var_mat
        self.mu_mat = mu_mat
        self.log_class_prior = np.log(np.bincount(y) / len(y))
        return self
    def _predict_p(self, X):
        n_samples = X.shape[0]
        n_classes = len(self.label_encoder.classes_)
        log_pori = np.full((n_samples, n_classes), self.log_class_prior)
        for i in range(n_classes):
            curr_var, curr_mu = self.var_mat[i], self.mu_mat[i]
            log_feature_proba = - 0.5 * np.sum((X - curr_mu)**2 / curr_var, axis=1) - 0.5 * np.sum(np.log(curr_var))
            log_pori[:, i] += log_feature_proba
        return np.exp(log_pori)

# 伯努利分布
class MyBernoulliNaiveBayes(ClassifierBase):
    def __init__(self, *, alpha=0.0, threshold=0.0):
        self.alpha = alpha
        self.threshold = 0.0
    def fit(self, X, y):
        self.feature_encoder = VarianceThreshold()
        self.feature_encoder.fit(X)

        X = (X > self.threshold).astype(np.float64)
        X = self.feature_encoder.transform(X)

        self.label_encoder = LabelEncoder()
        y = self.label_encoder.fit_transform(y)
        n_classes = len(self.label_encoder.classes_)
        y = np.eye(n_classes)[y]

        class_count = np.sum(y, axis=0) + self.alpha
        self.log_class_prior = np.log( class_count / (len(y) + self.alpha * n_classes) )  # (n_classes, )

        feature_count = np.dot(y.T, X) + self.alpha
        feature_proba = feature_count / (class_count.reshape((-1, 1)) + 2 * self.alpha)
        log_feature_proba = np.log(feature_proba) # (n_classes, n_features)
        log_neg_feature_proba = np.log(1 - feature_proba)

        # log(p) - log(1 - p)
        self.log_feature_probadiff = log_feature_proba - log_neg_feature_proba
        self.log_neg_feature_proba = log_neg_feature_proba

        return self
    def _predict_p(self, X):
        X = (X > self.threshold).astype(np.float64)
        X = self.feature_encoder.transform(X)
        # x * log(p) + (1 - x) * log(1 - p)
        # x * (log(p) - log(1 - p)) + ones * log(1 - p)
        # X : (n_samples, n_features)
        # self.log_feature_probadiff, self.log_neg_feature_proba : (n_classes, n_features)
        log_pori = (
                self.log_class_prior +                       # (n_classes, )
                np.sum(self.log_neg_feature_proba, axis=1) + # (n_classes, )
                np.dot(X, self.log_feature_probadiff.T)      # (n_samples, n_classes)
        )
        return np.exp(log_pori)
