import numpy as np
import matplotlib.pyplot as plt
from AUTO_DIFF import data_iter
from AUTO_DIFF import my_auto_grad_v0 as op
from AUTO_DIFF import optimizer
from AUTO_DIFF import layers

n_samples = 5000
np.random.seed(20)
x0 = np.random.uniform(-1., 1., n_samples)
x1 = np.random.uniform(-1., 1., n_samples)
train_x = np.c_[x0, x1]
w_true, b_true = np.array([-5.4, 3.265]).reshape((-1, 1)), np.array([-1.32])
train_y = (np.dot(train_x, w_true) + b_true > 0.).astype(np.float64)
train_y = 2. * train_y - 1.  # {+1, -1}


class SVM:
    def __init__(self, *, n_features, lam=0.0):
        self.lam = lam
        self.dense = layers.Dense(units=1, input_dim=n_features, use_bias=True, l2_reg=lam)
        self.params = self.dense.params
        self.regularizer_l2_lam_and_params = self.dense.regularizer_l2_lam_and_params

    def init_grad(self):
        for param in self.params:
            param._d = 0.0

    def __call__(self, X: op.Op):
        self.init_grad()
        return self.dense(X)

    def calc_cost(self, logits: op.Op, ysig: op.Op) -> op.Op:
        return op.RegularizerL2Wrap(op.HingleLoss(logits, ysig),
                                    svm.regularizer_l2_lam_and_params,
                                    calc_regularizer_loss=True)


n_features = train_x.shape[1]
svm = SVM(n_features=2, lam=0.001)

n_epochs = 100
batch_size = 64
train_iter = data_iter.DataIter(train_x, train_y, batch_size=batch_size)
updater = optimizer.SGD(learning_rate=0.3, trainable_nodes=svm.params)
cost_list, acc_list = [], []
for epoch in range(1, n_epochs + 1):
    for xi, yi in train_iter:
        xi, yi = op.C(xi), op.C(yi)
        logits = svm(xi)
        cost = svm.calc_cost(logits, yi)
        cost.backward()
        updater()
    train_y_logits = svm(op.C(train_x))
    train_cost = svm.calc_cost(train_y_logits, op.C(train_y))
    train_acc = np.mean(np.where(train_y_logits._v > 0., 1., -1.) == train_y)
    cost_list.append(train_cost._v)
    acc_list.append(train_acc)
    print(f"{epoch}\tCOST:{train_cost._v}\tACC:{train_acc}")
    W_op, b_op = svm.params
    print(f"W = {W_op._v.ravel()}, b = {b_op._v}")
    r = np.r_[W_op._v.ravel(), b_op._v] / np.r_[w_true.ravel(), b_true]
    print(r / r[0])

plt.subplots_adjust(wspace=0.6)
plt.subplot(121)
plt.plot(cost_list, label='COST')
plt.legend()
plt.subplot(122)
plt.plot(acc_list, label='ACC')
plt.legend()
plt.show()
