import matplotlib.pyplot as plt
from sklearn.base import ClassifierMixin
import numpy as np
from mnist.adaboost.my_get_data import *


def sigmoid(h):
    return 1.0 / (1.0 + np.exp(-h))


# 因子分解机，目前只支持二分类
class FMClassifier(ClassifierMixin):
    def __init__(self, *, lr=0.01, max_iter=100, kval=10):
        self.lr = lr
        self.max_iter = max_iter
        self.kval = kval

    def fit(self, X, y):
        n_samples, n_features = X.shape
        lr, max_iter, kval = self.lr, self.max_iter, self.kval

        b = np.random.random()
        w = np.random.random(n_features)
        v = np.random.random((n_features, kval))

        for i in range(max_iter):
            for _index in np.random.permutation(n_samples):
                xi, yi = X[_index], y[_index]
                xmat = np.dot(xi.reshape((-1, 1)), xi.reshape((1, -1)))
                vmat = np.dot(v, v.T)
                xvmat = xmat * vmat
                hi = sigmoid(b + np.dot(xi, w) + 0.5 * (np.sum(xvmat) - np.sum(np.diag(xvmat))))
                err = hi - yi
                db = err
                dw = xi * err
                dv = err * (
                        xi.reshape((-1, 1)) * np.dot(v.T, xi).reshape((1, -1))
                        - v * np.power(xi.reshape((-1, 1)), 2)
                )
                b -= lr * db
                w -= lr * dw
                v -= lr * dv
        self.b, self.w, self.v = b, w, v
        return self

    def predict_proba(self, X):
        b, w, v = self.b, self.w, self.v
        n_samples = X.shape[0]
        y_proba = np.empty(n_samples)
        for _index in range(n_samples):
            xi = X[_index]
            xmat = np.dot(xi.reshape((-1, 1)), xi.reshape((1, -1)))
            vmat = np.dot(v, v.T)
            xvmat = xmat * vmat
            y_proba[_index] = sigmoid(b + np.dot(xi, w) + 0.5 * (np.sum(xvmat) - np.sum(np.diag(xvmat))))
        return y_proba

    def predict(self, X):
        return np.where(self.predict_proba(X) > 0.5, 1, 0)


def yfunc(x):
    x1, x2 = x[:, 0], x[:, 1]
    return ((x1 + 10) * (x2 + 10) > 64).astype(np.int32)


def main():
    dobj = DataCreater(yfunc)
    x, y = dobj.gen_rand_data(size=1000)
    sc = plt.scatter(x[:, 0], x[:, 1], c=y)
    plt.legend(*sc.legend_elements())
    plt.show()
    fobj = FMClassifier(kval=100)
    fobj.fit(x, y)
    print("准确率", fobj.score(x, y))
    # plot_full_region(dobj, fobj.predict, show_scatter=False, print_accuracy=True)
    plot_full_region(dobj, fobj.predict_proba, show_scatter=False, print_accuracy=False)


if __name__ == '__main__':
    main()
