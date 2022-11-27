import numpy as np
import pandas as pd
import operator
from sklearn.base import ClassifierMixin
from common import my_get_data


def calc_gini_ori(x: np.ndarray):
    pval: np.ndarray = pd.Series(x).value_counts(normalize=True).to_numpy()
    return 1 - np.sum(pval ** 2)


def calc_gini_fea_thr(x_current: np.ndarray, y: np.ndarray, x_threshold):
    left_mask = x_current <= x_threshold
    yleft = y[left_mask]
    yright = y[~left_mask]
    return (len(yleft) * calc_gini_ori(yleft) + len(yright) * calc_gini_ori(yright)) / len(y)


def calc_gini(x_current: np.ndarray, y: np.ndarray):
    x_uniq = np.unique(x_current)
    if len(x_uniq) == 1:
        return None, np.inf
    x_threshold = 0.5 * (x_uniq[:-1] + x_uniq[1:])
    return min(
        ((xi, calc_gini_fea_thr(x_current, y, xi)) for xi in x_threshold),
        key=operator.itemgetter(1)
    )


def select_bestfeature_cart(X: np.ndarray, y: np.ndarray):
    n_features = X.shape[1]
    feature, (threshold, _) = min(
        ((i, calc_gini(X[:, i], y)) for i in range(n_features)),
        key=lambda v: v[1][1]
    )
    return feature, threshold


class CartTree2(object):
    def __init__(self, val, feature=None, left=None, right=None):
        self.feature = feature
        self.val = val
        self.left = left
        self.right = right

    def __str__(self):
        if self.feature is None:
            return f'[{self.val}]'
        return f'CartTree2({self.feature}:{self.val}, {self.left}, {self.right})'

    __repr__ = __str__

    @staticmethod
    def _print_tree_internal(tree, fstr1, fstr2):
        su1 = "├────"
        su2 = "└────"
        sm1 = "│    "
        sm2 = "     "
        if tree.feature is None:
            print(fstr1, str(tree), sep='')
        else:
            print(fstr1, f"{tree.feature}:{tree.val}", sep='')
            s1 = fstr2 + su1
            s2 = fstr2 + sm1
            CartTree2._print_tree_internal(tree.left, s1, s2)
            s1 = fstr2 + su2
            s2 = fstr2 + sm2
            CartTree2._print_tree_internal(tree.right, s1, s2)

    def print_tree(self):
        self._print_tree_internal(self, '', '')


def create_cart_tree(X: np.ndarray, y: np.ndarray):
    y_value_count: pd.Series = pd.Series(y).value_counts()
    if len(y_value_count) == 1:
        return CartTree2(y_value_count.index[0])
    feature, val = select_bestfeature_cart(X, y)
    if val is None:
        return CartTree2(y_value_counts.index[0])
    mask1 = X[:, feature] <= val
    mask2 = ~mask1
    return CartTree2(val, feature,
                     create_cart_tree(X[mask1], y[mask1]),
                     create_cart_tree(X[mask2], y[mask2]))


class CARTClassifier2(ClassifierMixin):
    def fit(self, X: np.ndarray, y: np.ndarray):
        self.y_dtype = y.dtype
        self.tree = create_cart_tree(X, y)
        return self

    def predict(self, X: np.ndarray):
        n_samples = X.shape[0]
        y_pred = np.empty(n_samples, dtype=self.y_dtype)
        for i in range(n_samples):
            xi = X[i]
            tree = self.tree
            while tree.feature is not None:
                if xi[tree.feature] <= tree.val:
                    tree = tree.left
                else:
                    tree = tree.right
            y_pred[i] = tree.val
        return y_pred


def main():
    dobj = my_get_data.DataCreater(my_get_data.yfunc2)
    x, y = dobj.gen_rand_data(size=500)
    cobj = CARTClassifier2()
    cobj.fit(x, y)
    print(cobj.score(x, y))
    cobj.tree.print_tree()
    my_get_data.plot_full_region(dobj, cobj.predict, x=x, y_true=y, show_scatter=True)


if __name__ == '__main__':
    main()
