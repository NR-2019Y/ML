import numpy as np
import pandas as pd
import wcwidth
import common
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
from sklearn.base import ClassifierMixin


def calc_ent(x: np.ndarray):
    pval = pd.Series(x).value_counts(normalize=True).to_numpy()
    return - np.sum(pval * np.log2(pval))


def calc_cond_ent(x_current: np.ndarray, y: np.ndarray):
    n_xtype = x_current.max() + 1
    y_eachxval = [y[x_current == i] for i in range(n_xtype)]
    return sum(len(yi) * calc_ent(yi) for yi in y_eachxval) / len(y)


# 信息增益比
def gain_ratio(x_current: np.ndarray, y: np.ndarray):
    ent_y = calc_ent(y)
    ent_x = calc_ent(x_current)
    eps = 1e-10
    return (ent_y - calc_cond_ent(x_current, y)) / (ent_x + eps)


# 参考：统计学习方法(李航)
# g(D, A) = H(D) - H(D|A) => g(D|A)最大即要求H(D|A)最小
def select_feature_id3(X_y: np.ndarray, feature_set: set):
    return min(feature_set, key=lambda i: calc_cond_ent(X_y[:, i], X_y[:, -1]))


def select_feature_c45(X_y: np.ndarray, feature_set: set):
    return max(feature_set, key=lambda i: gain_ratio(X_y[:, i], X_y[:, -1]))


class DTree(object):
    def __init__(self, val, *, feature=None, feature_names, encoder_x, encoder_y):
        assert feature_names is not None
        assert encoder_x is not None
        assert encoder_y is not None
        self.feature_names = feature_names
        self.encoder_x = encoder_x
        self.encoder_y = encoder_y
        if feature is not None:
            assert isinstance(val, dict)
        self.feature = feature
        self.val = val

    def __str__(self):
        if self.feature is None:
            label = self.encoder_y.classes_[self.val]
            return f'[{label}]'
        assert isinstance(self.val, dict)
        return f'''DTree{(self.feature_names[self.feature], {
            self.encoder_x.categories_[self.feature][k]: v for k, v in self.val.items()
        })}'''

    __repr__ = __str__

    @staticmethod
    def _print_tree_internal(tree, fstr1, fstr2):
        su1 = "├──"
        su2 = "└──"
        sm1 = "│  "
        sm2 = "   "
        if tree.feature is None:
            print(fstr1, str(tree), sep='')
        else:
            print(fstr1, str(tree.feature_names[tree.feature]), sep='')
            for i, (k, v) in enumerate(tree.val.items()):
                s1add = '──' + str(tree.encoder_x.categories_[tree.feature][k]) + '──'
                s2add = ' ' * wcwidth.wcswidth(s1add)
                if i != len(tree.val) - 1:
                    s1 = fstr2 + su1 + s1add
                    s2 = fstr2 + sm1 + s2add
                    DTree._print_tree_internal(v, s1, s2)
                else:
                    s1 = fstr2 + su2 + s1add
                    s2 = fstr2 + sm2 + s2add
                    DTree._print_tree_internal(v, s1, s2)

    def print_tree(self):
        DTree._print_tree_internal(self, '', '')


def gen_tree(X_y: np.ndarray, feature_set: set, func_select_feature, **kwargs):
    y = X_y[:, -1]
    y_value_counts: pd.Series = pd.Series(y).value_counts()
    if not feature_set or len(y_value_counts) == 1:
        curr_val = y_value_counts.index[0]
        return DTree(curr_val, **kwargs)
    best_feature = func_select_feature(X_y, feature_set)
    new_feature_set = feature_set.copy()
    new_feature_set.remove(best_feature)
    return DTree({
        xival: gen_tree(X_y[X_y[:, best_feature] == xival], new_feature_set, func_select_feature, **kwargs) for xival in
        np.unique(X_y[:, best_feature])
    }, feature=best_feature, **kwargs)


class DTreeClassifier(ClassifierMixin):
    # id3 or c4.5
    def __init__(self, *, method='id3'):
        func_dic = {
            'id3': select_feature_id3,
            'c4.5': select_feature_c45
        }
        assert method in func_dic.keys()
        self.func_select_feature = func_dic[method]

    def fit(self, X: np.ndarray, y: np.ndarray, *, feature_names=None):
        n_samples, n_features = X.shape
        if feature_names is None:
            feature_names = np.arange(n_features).astype(str)
        assert len(feature_names) == n_features
        self.encoder_x = OrdinalEncoder(dtype=np.int64)
        self.encoder_y = LabelEncoder()
        X = self.encoder_x.fit_transform(X)
        y = self.encoder_y.fit_transform(y)
        X_y = np.c_[X, y]
        self.tree = gen_tree(X_y, set(range(n_features)),
                             self.func_select_feature,
                             feature_names=feature_names,
                             encoder_x=self.encoder_x,
                             encoder_y=self.encoder_y)
        return self

    def predict(self, X: np.ndarray):
        X = self.encoder_x.transform(X)
        n_samples, n_features = X.shape
        if self.tree.feature is None:
            return self.encoder_y.inverse_transform(np.full(n_samples, self.tree.val))
        y_pred = np.empty(n_samples, dtype=np.int64)
        for i in range(n_samples):
            xi = X[i]
            tree = self.tree
            while tree.feature is not None:
                tree = tree.val[xi[tree.feature]]
            y_pred[i] = tree.val
        return self.encoder_y.inverse_transform(y_pred)


if __name__ == '__main__':
    df = common.load_xigua_df()
    print(df)
    X = df.iloc[:, :-1].to_numpy(dtype=object)
    y = df.iloc[:, -1].to_numpy(dtype=object)
    id3obj = DTreeClassifier(method='id3')
    id3obj.fit(X, y, feature_names=df.columns[:-1])
    print(id3obj.tree)
    id3obj.tree.print_tree()
    print(id3obj.score(X, y))
