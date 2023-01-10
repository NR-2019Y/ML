import numpy as np
import matplotlib.pyplot as plt
from AUTO_DIFF import data_iter
from AUTO_DIFF import my_auto_grad_v0 as op
from AUTO_DIFF import optimizer

n_samples = 5000
np.random.seed(20)
x0 = np.random.uniform(-1., 1., n_samples)
x1 = np.random.uniform(-1., 1., n_samples)
train_x = np.c_[x0, x1]
w_true, b_true = np.array([-5.4, 3.265]), -1.32
train_y = (np.dot(train_x, w_true) + b_true > 0.).astype(np.float64)
train_y = 2. * train_y - 1.  # {+1, -1}


class SVM:
    def __init__(self, *, n_features, lam=0.0):
        self.lam = lam
        W = op.C(np.random.normal(size=n_features, scale=0.01), requires_grad=True)
        b = op.C(0.0, requires_grad=True)
        self.params = W, b

    def init_grad(self):
        for param in self.params:
            param._d = 0.0

    def __call__(self, X: op.Op):
        self.init_grad()
        W, b = self.params
        return op.AddBiasND(op.MatMul(X, W), b)

    # loss: hinge
    def calc_loss(self, logits: op.Op, ysig: op.Op):
        n_samples = len(ysig)
        return (
                op.C(1.0 / n_samples) * op.Sum(op.Relu(op.C(1.0) - ysig * logits)) +
                op.C(self.lam / n_samples) * op.Sum(op.Square(self.params[0]))
        )


n_features = train_x.shape[1]
svm = SVM(n_features=2, lam=0.0)

n_epochs = 1000
batch_size = 64
train_iter = data_iter.DataIter(train_x, train_y, batch_size=batch_size)
updater = optimizer.SGD(learning_rate=0.3, trainable_nodes=svm.params)
cost_list, acc_list = [], []
for epoch in range(1, n_epochs + 1):
    for xi, yi in train_iter:
        xi, yi = op.C(xi), op.C(yi)
        logits = svm(xi)
        cost = svm.calc_loss(logits, yi)
        cost.backward()
        updater()
    train_y_logits = svm(op.C(train_x))
    train_cost = svm.calc_loss(train_y_logits, op.C(train_y))
    train_acc = np.mean(np.where(train_y_logits._v > 0., 1., -1.) == train_y)
    cost_list.append(train_cost._v)
    acc_list.append(train_acc)
    print(f"{epoch}\tCOST:{train_cost._v}\tACC:{train_acc}")
    W_op, b_op = svm.params
    print(f"W = {W_op._v}, b = {b_op._v}")
    r = np.r_[W_op._v, b_op._v] / np.r_[w_true, b_true]
    print(r / r[0])

plt.subplots_adjust(wspace=0.6)
plt.subplot(121)
plt.plot(cost_list, label='COST')
plt.legend()
plt.subplot(122)
plt.plot(acc_list, label='ACC')
plt.legend()
plt.show()
