import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

def yfunc1(x):
    x1, x2 = x[:, 0], x[:, 1]
    return (x1 ** 2 + x2 ** 2 > 64).astype(np.int32)

def yfunc2(x):
    m = x.shape[0]
    x1, x2 = x[:, 0], x[:, 1]
    y = np.zeros(m)
    fval = x1 ** 2 + x2 ** 2
    y[ fval > 9  ] = 1
    y[ fval > 36 ] = 2
    y[ fval > 81 ] = 3
    return y

class DataCreater(object):
    _x1min, _x1max, _x2min, _x2max = -10, 10, -10, 10
    def __init__(self, yfunc):
        self.yfunc = yfunc
    def gen_rand_data(self, *, size=100):
        np.random.seed(10)
        x1 = np.random.uniform(DataCreater._x1min, DataCreater._x1max, size)
        x2 = np.random.uniform(DataCreater._x2min, DataCreater._x2max, size)
        x = np.c_[x1, x2]
        y = self.yfunc(x)
        return x, y
    def gen_total_data(self, *, size=100):
        mc = size * 1j
        x1t, x2t = np.mgrid[
            DataCreater._x1min:DataCreater._x1max:mc,
            DataCreater._x2min:DataCreater._x2max:mc
        ]
        xt = np.c_[x1t.ravel(), x2t.ravel()]
        yt = self.yfunc(xt)
        return xt, x1t, x2t, yt

def plot_full_region(dobj:DataCreater, func_pred, *, x=None, y_true=None, size=200, show_scatter=True, print_accuracy=True):
    xt, x1t, x2t, yt = dobj.gen_total_data(size=size)
    # func_pred 接受不含偏置的数据，返回：概率值（此时准确率计算无效）或具体类别 : (m, ) array
    yt_pred = func_pred(xt)
    if print_accuracy:
        print("全集准确率", accuracy_score(yt, yt_pred))
    if show_scatter:
        assert (x is not None) and (y_true is not None)
        # x: (m, 2) array
        # y_true : (m, ) array
        y_true = y_true.ravel()
        sc = plt.scatter(x[:, 0], x[:, 1], c=y_true, s=10, zorder=10, edgecolors='r')
        plt.legend(*sc.legend_elements())
    plt.pcolormesh(x1t, x2t, yt_pred.reshape(x1t.shape))
    plt.show()
