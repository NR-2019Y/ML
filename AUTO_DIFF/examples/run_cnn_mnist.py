import numpy as np
import datetime, time
from AUTO_DIFF import my_auto_grad_v0 as op
from AUTO_DIFF import fast_conv_op_v0 as cnn_op
from AUTO_DIFF import data_iter
from AUTO_DIFF import optimizer
from datasets.load_img import load_mnist


class CNN:
    def __init__(self):
        self.conv1 = cnn_op.Conv2DLayer(out_channels=6, ksize=(5, 5), batch_input_size=(None, 32, 32, 1))
        self.pool1 = cnn_op.MaxPool2DLayer(ksize=2, batch_input_size=(None, 28, 28, 6), strides=2)
        self.conv2 = cnn_op.Conv2DLayer(out_channels=16, ksize=(5, 5), batch_input_size=(None, 14, 14, 6))
        self.pool2 = cnn_op.MaxPool2DLayer(ksize=2, batch_input_size=(None, 10, 10, 16), strides=2)
        n_flatten_features = 5 * 5 * 16
        n_classes = 10
        W3 = op.C(np.random.normal(loc=0., scale=0.01, size=[n_flatten_features, n_classes]), requires_grad=True)
        b3 = op.C(np.zeros(n_classes), requires_grad=True)
        self.params = self.conv1.params + self.conv2.params + (W3, b3)

    def init_grad(self):
        for param in self.params:
            param._d = 0.0

    def __call__(self, X):
        self.init_grad()
        W1, b1, W2, b2, W3, b3 = self.params
        L1 = self.conv1(X)
        L1 = op.Relu(L1)
        L1 = self.pool1(L1)
        L2 = self.conv2(L1)
        L2 = op.Relu(L2)
        L2 = self.pool2(L2)
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
    tic = time.time()
    ts = tic
    total_loss = 0.
    nacc = 0.
    for i, (xi, yi) in enumerate(train_iter, start=1):
        sz = len(yi)
        xi, yi = op.C(xi), op.C(yi)
        logits = net(xi)
        loss = loss_func(logits, yi)
        loss.backward()
        trainer()

        total_loss += loss._v * sz
        curr_acc = calc_acc(logits._v, yi._v)
        nacc += curr_acc * sz
        if i % 5 == 0:
            te = time.time()
            tdiff = te - ts
            ts = te
            print(f"[{i}] {tdiff}s loss:{loss._v} acc:{curr_acc}")

    train_loss = total_loss / len(train_y)
    train_acc = nacc / len(train_y)

    nacc = 0
    for xi, yi in test_iter:
        sz = len(yi)
        xi, yi = op.C(xi), op.C(yi)
        logits = net(xi)
        nacc += calc_acc(logits._v, yi._v) * sz
    test_acc = nacc / len(test_y)
    stime = time.time() - tic
    print(f"epoch:[{epoch}]:{stime}s\tLOSS:{train_loss}\tTRAIN_ACC:{train_acc}\tTEST_ACC:{test_acc}", flush=True)
