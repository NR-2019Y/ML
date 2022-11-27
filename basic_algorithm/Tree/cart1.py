import numpy as np
import pandas as pd
from sklearn.base import ClassifierMixin
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder
import common


class CartTree1(object):
    def __init__(self, val, feature=None, left=None, right=None, *, feature_names, encoder_x, encoder_y):
        self.feature = feature
        self.val = val
        self.left = left
        self.right = right
        self.feature_names = feature_names
        self.encoder_x = encoder_x
        self.encoder_y = encoder_y

    def __str__(self):
        if self.feature is None:
            label = self.encoder_y.classes_[self.val]
            return f'[{label}]'
        return f'CartTree1({self.feature_names[self.feature]}:{self.encoder_x.categories_[self.feature][self.val]}, {self.left}, {self.right})'

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
            s_feature = tree.feature_names[tree.feature]
            s_val = tree.encoder_x.categories_[tree.feature][tree.val]
            print(fstr1, f"{s_feature}:{s_val}", sep='')
            s1 = fstr2 + su1
            s2 = fstr2 + sm1
            CartTree1._print_tree_internal(tree.left, s1, s2)
            s1 = fstr2 + su2
            s2 = fstr2 + sm2
            CartTree1._print_tree_internal(tree.right, s1, s2)

    def print_tree(self):
        self._print_tree_internal(self, '', '')


def calc_gini_ori(x: np.ndarray):
    pval: np.ndarray = pd.Series(x).value_counts(normalize=True).to_numpy()
    return 1 - np.sum(pval ** 2)


def calc_gini(x_current: np.ndarray, y: np.ndarray):
    x_fq = pd.Series(x_current).value_counts(normalize=True)
    if len(x_fq) == 1:
        return None, np.inf

    def fgini(xi):
        return x_fq[xi] * calc_gini_ori(y[x_current == xi]) + (1 - x_fq[xi]) * calc_gini_ori(y[x_current != xi])

    if len(x_fq) == 2:
        return x_fq.index[0], fgini(x_fq.index[0])
    return min(((xi, fgini(xi)) for xi in x_fq.index), key=lambda v: v[1])


def select_bestfeature_cart(X: np.ndarray, y: np.ndarray):
    n_features = X.shape[1]
    feature, (val, _) = min((
        (i, calc_gini(X[:, i], y)) for i in range(n_features)
    ), key=lambda v: v[1][1])
    return feature, val


def create_cart_tree(X: np.ndarray, y: np.ndarray, **kwargs):
    y_value_counts: pd.Series = pd.Series(y).value_counts()
    if len(y_value_counts) == 1:
        return CartTree1(y_value_counts.index[0], **kwargs)
    feature, val = select_bestfeature_cart(X, y)
    if val is None:
        return CartTree1(y_value_counts.index[0], **kwargs)
    mask1 = X[:, feature] == val
    mask2 = ~mask1
    return CartTree1(val, feature,
                     create_cart_tree(X[mask1], y[mask1], **kwargs),
                     create_cart_tree(X[mask2], y[mask2], **kwargs),
                     **kwargs)


class CARTClassifier1(ClassifierMixin):
    # 处理离散特征
    def fit(self, X: np.ndarray, y: np.ndarray, *, feature_names=None):
        self.feature_encoder = OrdinalEncoder(dtype=np.int64)
        X = self.feature_encoder.fit_transform(X)
        self.label_encoder = LabelEncoder()
        y = self.label_encoder.fit_transform(y)
        self.tree = create_cart_tree(X, y,
                                     feature_names=feature_names,
                                     encoder_x=self.feature_encoder,
                                     encoder_y=self.label_encoder)
        return self

    def predict(self, X):
        X = self.feature_encoder.transform(X)
        n_samples, n_features = X.shape
        y_pred = np.empty(n_samples, dtype=np.int64)
        for i in range(n_samples):
            xi = X[i]
            tree = self.tree
            while tree.feature is not None:
                if xi[tree.feature] == tree.val:
                    tree = tree.left
                else:
                    tree = tree.right
            y_pred[i] = tree.val
        return self.label_encoder.inverse_transform(y_pred)


if __name__ == '__main__':
    df = common.load_xigua_df()
    print(df)
    X = df.iloc[:, :-1].to_numpy(dtype=object)
    y = df.iloc[:, -1].to_numpy(dtype=object)
    cobj = CARTClassifier1()
    cobj.fit(X, y, feature_names=df.columns[:-1])
    print(cobj.tree)
    cobj.tree.print_tree()
    print(cobj.score(X, y))
