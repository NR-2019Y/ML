import numpy as np

from datasets.mnist import load_mnist
from AUTO_DIFF import my_auto_grad_v0 as op


class DataIter(object):
    def __init__(self, X, y, *, batch_size):
        assert len(X) == len(y)
        self.X = X
        self.y = y
        self.batch_size = batch_size

    def __len__(self):
        return len(self.y)

    def __iter__(self):
        size = len(self)
        rand_perm = np.random.permutation(size)
        # print(rand_perm)
        for i in range(0, size, self.batch_size):
            idx = rand_perm[i: i + self.batch_size]
            yield self.X[idx], self.y[idx]


class NNET(object):
    def __init__(self, n_features, n_hiddens, n_classes):
        W1 = op.C(np.random.normal(scale=0.01, size=(n_features, n_hiddens)).astype(np.float32), requires_grad=True)
        b1 = op.C(np.zeros(n_hiddens, dtype=np.float32), requires_grad=True)
        W2 = op.C(np.random.normal(scale=0.01, size=(n_hiddens, n_classes)).astype(np.float32), requires_grad=True)
        b2 = op.C(np.zeros(n_classes, dtype=np.float32), requires_grad=True)
        self.params = W1, b1, W2, b2

    def init_grad(self):
        for param in self.params:
            param._d = 0.0

    def __call__(self, X):
        self.init_grad()
        W1, b1, W2, b2 = self.params
        z1 = op.LinearLayer(X, W1, b1)
        a1 = op.Relu(z1)
        logits = op.LinearLayer(a1, W2, b2)
        return logits

    def update(self, *, learning_rate):
        for param in self.params:
            param._v -= learning_rate * param._d


def loss_func(logits, y):
    return op.CrossEntropyLossLayer(logits, y)


def calc_acc(logits: np.ndarray, yori: np.ndarray):
    return np.mean(logits.argmax(axis=1) == yori)


train_x, test_x, train_y_ori, test_y_ori = load_mnist.load_mnist()
train_y = np.eye(10, dtype=np.float32)[train_y_ori]
test_y = np.eye(10, dtype=np.float32)[test_y_ori]

n_epochs = 5000
batch_size = 64
learning_rate = 0.01
train_iter = DataIter(train_x, train_y, batch_size=batch_size)

c_train_x, c_test_x, c_train_y, c_test_y = map(op.C, (train_x, test_x, train_y, test_y))
net = NNET(n_features=784, n_hiddens=128, n_classes=10)
for epoch in range(1, n_epochs + 1):
    for xi, yi in train_iter:
        # print("xi, yi", xi.shape, yi.shape)
        xi, yi = op.C(xi), op.C(yi)
        logits = net(xi)
        loss = loss_func(logits, yi)
        loss.backward()
        net.update(learning_rate=learning_rate)

    c_train_logits = net(c_train_x)
    c_test_logits = net(c_test_x)
    train_loss = loss_func(c_train_logits, c_train_y)._v
    train_acc = calc_acc(c_train_logits._v, train_y_ori)
    test_acc = calc_acc(c_test_logits._v, test_y_ori)
    print(f"epoch:{epoch}\tLOSS:{train_loss}\tTRAIN_ACC:{train_acc}\tTEST_ACC:{test_acc}")
