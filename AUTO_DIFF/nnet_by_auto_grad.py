import numpy as np
import time
import math
from my_auto_grad_v0 import *
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score

def checkOpArgs(f):
    def nfunc(*args, **kwargs):
        assert not kwargs
        for e in args:
            assert isinstance(e, Op)
        return f(*args)
    return nfunc

@checkOpArgs
def SigmoidOp(z:Op):
    return C(1.0) / (C(1.0) + Exp(-z))

@checkOpArgs
def SoftMaxOp(z:Op):
    z_exp = Exp(z)
    return z_exp / SumAxis(z_exp, axis=1)

# # 前向传播将梯度全部置0
# @checkOpArgs
# def ForwardOp(x:Op, w1:Op, w2:Op):
#     z2 = x @ w1
#     a2 = SigmoidOp(z2)
#     z3 = a2 @ w2
#     a3 = SoftMaxOp(z3)
#     return a3

# 使用RELU激活函数
@checkOpArgs
def ForwardOp(x:Op, w1:Op, w2:Op):
    z2 = x @ w1
    a2 = Relu(z2)
    z3 = a2 @ w2
    a3 = SoftMaxOp(z3)
    return a3

@checkOpArgs
def LossOp(x:Op, y:Op, w1:Op, w2:Op):
    a3 = ForwardOp(x, w1, w2)
    loss = - Sum(y * Log(a3))
    return loss

class NNet(object):
    def fit_bgd(self, x, y, lr=0.1, max_iter=1000, hidden=8):
        xm, xn = x.shape
        assert y.ndim == 2
        assert (y.shape[0] == xm) and (y.shape[1] == 1)
        self.oobj = OneHotEncoder(sparse=False, categories='auto')
        y = self.oobj.fit_transform(y)
        ym, yn = y.shape
        x = C(x)
        y = C(y)
        w1 = C(np.random.random((xn, hidden)), requires_grad=True)
        w2 = C(np.random.random((hidden, yn)), requires_grad=True)
        loss_list = []
        for i in range(max_iter):
            loss = LossOp(x, y, w1, w2)
            loss_list.append(loss._v)
            loss.backward()
            w1._v -= (lr / xm) * w1._d
            w2._v -= (lr / xm) * w2._d
        print("||梯度||", np.sqrt(np.sum(w1._v ** 2) + np.sum(w2._v ** 2)))
        self.w1, self.w2 = w1, w2
        self.loss_list = loss_list

    def fit_mbgd(self, x, y, *, lr=0.1, num_epochs=100, batch_size=32, hidden=8, nshow=50):
        xm, xn = x.shape
        assert y.ndim == 2
        assert (y.shape[0] == xm) and (y.shape[1] == 1)
        self.oobj = OneHotEncoder(sparse=False, categories='auto')
        yhot = self.oobj.fit_transform(y)
        ym, yn = yhot.shape
        assert batch_size <= xm
        w1 = C(np.random.random((xn, hidden)), requires_grad=True)
        w2 = C(np.random.random((hidden, yn)), requires_grad=True)
        loss_list = []
        acc_list = []

        # AdaGrad
        eps = 1e-10
        s1, s2 = np.zeros_like(w1._v) + eps, np.zeros_like(w2._v) + eps
        niter = math.ceil(num_epochs * xm / batch_size)
        for k in range(niter):
            i = np.random.choice(xm, batch_size, replace=False)
            xi = C(x[i])
            yi = C(yhot[i])
            loss_curr = LossOp(xi, yi, w1, w2)
            loss_curr.backward()
            s1 += w1._d ** 2
            s2 += w2._d ** 2
            w1._v -= (lr / batch_size / np.sqrt(s1)) * w1._d
            w2._v -= (lr / batch_size / np.sqrt(s2)) * w2._d
            if k % nshow == 0:
                loss = LossOp(C(x), C(yhot), w1, w2)
                loss_list.append(loss._v)
                self.w1, self.w2 = w1, w2
                acc = self.score(x, y)
                acc_list.append(acc)
                print(f"[{k//batch_size+1}, {k%batch_size}]\tLOSS:{loss._v}\tACC:{acc}")
        self.w1, self.w2 = w1, w2
        self.loss_list = loss_list
        self.acc_list = acc_list
    def predict(self, x):
        x = C(x)
        y_pval_op = ForwardOp(x, self.w1, self.w2)
        return self.oobj.inverse_transform(y_pval_op._v)
    def score(self, x, y):
        xm, xn = x.shape
        assert y.ndim == 2
        ym, yn = y.shape
        assert (xm == ym) and (yn == 1)
        return accuracy_score(y, self.predict(x))

def my_test():
    import my_get_data
    dobj = my_get_data.DataCreater(my_get_data.yfunc2)
    x, y = dobj.gen_rand_data(size=1000)
    tic = time.time()
    nobj = NNet()
    nobj.fit_mbgd(x, y, lr=0.1, num_epochs=1000, batch_size=32, hidden=50, nshow=1)
    plt.subplots_adjust(wspace=0.2)
    plt.subplot(121)
    plt.plot(nobj.loss_list, label = "LOSS")
    plt.legend()
    plt.subplot(122)
    plt.plot(nobj.acc_list, label = "ACC")
    plt.legend()
    plt.show()
    print("训练时间", time.time() - tic)
    print("准确率", nobj.score(x, y))
    my_get_data.plot_full_region(dobj, nobj.predict, x=x, y_true=y, size=200, show_scatter=True, print_accuracy=True)

if __name__ == '__main__':
    my_test()
