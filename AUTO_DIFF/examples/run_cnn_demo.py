import numpy as np
from AUTO_DIFF import my_auto_grad_v0 as op
from AUTO_DIFF import cnn_op_v0 as cnn_op
from AUTO_DIFF import data_iter
from AUTO_DIFF import optimizer
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split


class CNN:
    def __init__(self):
        # input_shape: (28, 28, 1)
        W1 = op.C(np.random.normal(loc=0., scale=0.01, size=[3, 3, 1, 32]), requires_grad=True)
        b1 = op.C(np.zeros(32), requires_grad=True)
        W2 = op.C(np.random.normal(loc=0., scale=0.01, size=[3, 3, 32, 64]), requires_grad=True)
        b2 = op.C(np.zeros(64), requires_grad=True)
        n_flatten_features = 2 * 2 * 64
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
        L1 = op.BroadcastAdd(cnn_op.conv2d_padding_same_op(X, W1), b1)
        L1 = op.Relu(L1)
        L1 = cnn_op.max_pool2d_padding_same_op(L1, ksize=[2, 2], stride=[2, 2])
        L2 = op.BroadcastAdd(cnn_op.conv2d_padding_same_op(L1, W2), b2)
        L2 = op.Relu(L2)
        L2 = cnn_op.max_pool2d_padding_same_op(L2, ksize=[2, 2], stride=[2, 2])
        L_FLATTEN = op.Reshape(L2, (-1, W3._v.shape[0]))
        logits = op.LinearLayer(L_FLATTEN, W3, b3)
        return logits


def loss_func(logits, y):
    return op.CrossEntropyLossLayer(logits, y)


def calc_acc(logits: np.ndarray, y: np.ndarray):
    return np.mean(logits.argmax(axis=1) == y)


all_data = load_digits()
all_x, all_y = (all_data.data / 16.).astype(np.float32).reshape((-1, 8, 8, 1)), all_data.target.astype(np.int64)
train_x, test_x, train_y, test_y = train_test_split(all_x, all_y, test_size=0.3)

n_epochs = 200
batch_size = 64
train_iter = data_iter.DataIter(train_x, train_y, batch_size=batch_size)

# train_iter2 = data_iter.DataIter(train_x, train_y, batch_size=100)
test_iter2 = data_iter.DataIter(test_x, test_y, batch_size=100)

c_train_x, c_test_x, c_train_y, c_test_y = map(op.C, (train_x, test_x, train_y, test_y))
net = CNN()
trainer = optimizer.Adam(learning_rate=0.01, trainable_nodes=net.params)
for epoch in range(1, n_epochs + 1):

    total_loss = 0.
    nacc = 0.
    for xi, yi in train_iter:
        sz = len(yi)
        xi, yi = op.C(xi), op.C(yi)
        logits = net(xi)
        loss = loss_func(logits, yi)
        loss.backward()
        trainer()

        total_loss += loss._v * sz
        nacc += calc_acc(logits._v, yi._v) * sz

    train_loss = total_loss / len(train_y)
    train_acc = nacc / len(train_y)

    nacc = 0
    for xi, yi in test_iter2:
        sz = len(yi)
        xi, yi = op.C(xi), op.C(yi)
        logits = net(xi)
        nacc += calc_acc(logits._v, yi._v) * sz
    test_acc = nacc / len(test_y)
    print(f"epoch:{epoch}\tLOSS:{train_loss}\tTRAIN_ACC:{train_acc}\tTEST_ACC:{test_acc}")
