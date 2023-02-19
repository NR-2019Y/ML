import numpy as np
import datetime
from AUTO_DIFF import my_auto_grad_v0 as op
from AUTO_DIFF import fast_conv_op_v0 as cnn_op
from AUTO_DIFF import data_iter
from AUTO_DIFF import optimizer
from datasets.load_img import load_mnist


class CNN:
    def __init__(self):
        W1 = op.C(np.random.normal(loc=0., scale=0.01, size=[5, 5, 1, 6]), requires_grad=True)
        b1 = op.C(np.zeros(6), requires_grad=True)
        W2 = op.C(np.random.normal(loc=0., scale=0.01, size=[5, 5, 6, 16]), requires_grad=True)
        b2 = op.C(np.zeros(16), requires_grad=True)
        n_flatten_features = 5 * 5 * 16
        n_classes = 10
        W3 = op.C(np.random.normal(loc=0., scale=0.01, size=[n_flatten_features, n_classes]), requires_grad=True)
        b3 = op.C(np.zeros(n_classes), requires_grad=True)
        self.params = W1, b1, W2, b2, W3, b3

    def init_grad(self):
        for param in self.params:
            param._d = 0.0

    def __call__(self, X):
        self.init_grad()
        W1, b1, W2, b2, W3, b3 = self.params
        L1 = op.BroadcastAdd(cnn_op.Conv2D(X, W1), b1)
        L1 = op.Relu(L1)
        L1 = cnn_op.MaxPool2D(L1, ksize=[2, 2], strides=[2, 2])
        L2 = op.BroadcastAdd(cnn_op.Conv2D(L1, W2), b2)
        L2 = op.Relu(L2)
        L2 = cnn_op.MaxPool2D(L2, ksize=[2, 2], strides=[2, 2])
        L_FLATTEN = op.Reshape(L2, (-1, W3._v.shape[0]))
        logits = op.LinearLayer(L_FLATTEN, W3, b3)
        return logits


def loss_func(logits, y):
    return op.CrossEntropyLossLayer(logits, y)


def calc_acc(logits: np.ndarray, y: np.ndarray):
    return np.mean(logits.argmax(axis=1) == y)


train_x, test_x, train_y, test_y = load_mnist()
train_x = train_x.reshape([-1, 28, 28, 1])
test_x = test_x.reshape([-1, 28, 28, 1])
train_x = np.pad(train_x, pad_width=[[0, 0], [2, 2], [2, 2], [0, 0]])
test_x = np.pad(test_x, pad_width=[[0, 0], [2, 2], [2, 2], [0, 0]])

n_epochs = 50
batch_size = 128
train_iter = data_iter.DataIter(train_x, train_y, batch_size=batch_size)
test_iter = data_iter.DataIter(test_x, test_y, batch_size=1000)

c_train_x, c_test_x, c_train_y, c_test_y = map(op.C, (train_x, test_x, train_y, test_y))
net = CNN()
trainer = optimizer.Adam(learning_rate=0.001, trainable_nodes=net.params)
for epoch in range(1, n_epochs + 1):

    total_loss = 0.
    nacc = 0.
    for i, (xi, yi) in enumerate(train_iter):
        sz = len(yi)
        xi, yi = op.C(xi), op.C(yi)
        logits = net(xi)
        loss = loss_func(logits, yi)
        loss.backward()
        trainer()

        total_loss += loss._v * sz
        curr_acc = calc_acc(logits._v, yi._v)
        nacc += curr_acc * sz
        print(i, loss._v, curr_acc)

    train_loss = total_loss / len(train_y)
    train_acc = nacc / len(train_y)

    nacc = 0
    for xi, yi in test_iter:
        sz = len(yi)
        xi, yi = op.C(xi), op.C(yi)
        logits = net(xi)
        nacc += calc_acc(logits._v, yi._v) * sz
    test_acc = nacc / len(test_y)
    sdate = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f"{sdate}\tepoch:{epoch}\tLOSS:{train_loss}\tTRAIN_ACC:{train_acc}\tTEST_ACC:{test_acc}", flush=True)
