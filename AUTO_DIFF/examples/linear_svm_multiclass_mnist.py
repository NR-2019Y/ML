import matplotlib.pyplot as plt
import numpy as np
from AUTO_DIFF import my_auto_grad_v0 as op
from AUTO_DIFF import optimizer
from AUTO_DIFF import data_iter
from datasets.load_img import load_mnist


class SVM:
    # OVR实现多分类
    def __init__(self, *, n_features, n_classes, lam=0.0):
        self.lam = lam
        W = op.C(np.random.normal(size=(n_features, n_classes)))
        b = op.C(np.zeros(n_classes))
        self.params = W, b

    def init_grad(self):
        for param in self.params:
            param._d = 0.0

    def __call__(self, X):
        self.init_grad()
        W, b = self.params
        return op.AddBiasND(op.MatMul(X, W), b)

    # loss: hinge
    def calc_cost(self, logits: op.Op, ysig: op.Op):
        n_samples = len(ysig)
        return (
                op.C(1.0 / n_samples) * op.Sum(op.Relu(op.C(1.0) - ysig * logits)) +
                op.C(self.lam / n_samples) * op.Sum(op.Square(self.params[0]))
        )


train_x, test_x, train_y, test_y = load_mnist()
train_y_sig = 2. * np.eye(10)[train_y] - 1.
test_y_sig = 2. * np.eye(10)[test_y] - 1.

n_features = train_x.shape[1]
n_classes = 10
batch_size = 64
n_epochs = 100

train_iter = data_iter.DataIter(train_x, train_y_sig, batch_size=batch_size)
svm = SVM(n_features=n_features, n_classes=n_classes, lam=0.01)
updater = optimizer.Adam(learning_rate=0.001, trainable_nodes=svm.params)

train_cost_list, test_cost_list, train_acc_list, test_acc_list = [], [], [], []
for epoch in range(1, n_epochs + 1):
    for xi, yi in train_iter:
        xi, yi = op.C(xi), op.C(yi)
        logits = svm(xi)
        cost = svm.calc_cost(logits, yi)
        cost.backward()
        updater()
    train_logits = svm(op.C(train_x))
    train_cost = svm.calc_cost(train_logits, op.C(train_y_sig))._v
    train_acc = np.mean(np.argmax(train_logits._v, -1) == train_y)
    test_logits = svm(op.C(test_x))
    test_cost = svm.calc_cost(test_logits, op.C(test_y_sig))._v
    test_acc = np.mean(np.argmax(test_logits._v, -1) == test_y)
    train_cost_list.append(train_cost)
    test_cost_list.append(test_cost)
    train_acc_list.append(train_acc)
    test_acc_list.append(test_acc)
    print(f"epoch:{epoch} TRAIN_COST:{train_cost} TRAIN_ACC:{train_acc} TEST_COST:{test_cost} TEST_ACC:{test_acc}")

plt.subplot(121)
plt.plot(train_cost_list, label='TRAIN COST')
plt.plot(test_cost_list, label='TEST COST')
plt.legend()
plt.subplot(122)
plt.plot(train_acc_list, label='TRAIN ACC')
plt.plot(test_acc_list, label='TEST ACC')
plt.legend()
plt.show()
