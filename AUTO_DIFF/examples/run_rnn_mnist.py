import numpy as np
from AUTO_DIFF import my_auto_grad_v0 as op, layers, data_iter, optimizer
from datasets.load_img import load_mnist
import time
import numbers


class Md(object):
    def __init__(self, input_dim, n_classes, n_hiddens):
        # self.rnn = layers.LSTM(units=n_hiddens, input_dim=input_dim, return_sequences=False)
        if isinstance(n_hiddens, numbers.Number):
            n_hiddens = (n_hiddens,) * 2
        h1, h2 = n_hiddens
        self.rnn1 = layers.GRU(units=h1, input_dim=input_dim, return_sequences=True)
        self.rnn2 = layers.GRU(units=h2, input_dim=h1, return_sequences=False)
        self.dense = layers.Dense(units=n_classes, input_dim=h2)
        self.params = self.rnn1.params + self.rnn2.params + self.dense.params

    def init_grad(self):
        for param in self.params:
            param._d = 0.0

    def __call__(self, X: op.Op):
        self.init_grad()
        return self.dense(self.rnn2(self.rnn1(X)))


train_x, test_x, train_y, test_y = load_mnist()
train_x = train_x.reshape([-1, 28, 28])
test_x = test_x.reshape([-1, 28, 28])
n_epochs = 50
batch_size = 128
train_iter = data_iter.DataIter(train_x, train_y, batch_size=batch_size)
test_iter = data_iter.DataIter(test_x, test_y, batch_size=1000)
c_train_x, c_test_x, c_train_y, c_test_y = map(op.C, (train_x, test_x, train_y, test_y))

md = Md(input_dim=28, n_classes=10, n_hiddens=[32, 16])
trainer = optimizer.Adam(learning_rate=0.01, trainable_nodes=md.params)

tic = time.time()
for epoch in range(1, n_epochs + 1):
    ts = time.time()
    total_train_loss = 0.
    train_nacc = 0.
    for xi, yi in train_iter:
        xi, yi = op.C(xi), op.C(yi)
        logits = md(xi)
        loss = op.CrossEntropyLossLayer(logits, yi)
        loss.backward()
        trainer()
        total_train_loss += loss._v * len(yi)
        train_nacc += np.sum(logits._v.argmax(-1) == yi._v)
    train_loss = total_train_loss / len(train_y)
    train_acc = train_nacc / len(train_y)

    total_test_loss = 0.
    test_nacc = 0.
    for xi, yi in test_iter:
        sz = len(yi)
        xi, yi = op.C(xi), op.C(yi)
        logits = md(xi)
        loss = op.CrossEntropyLossLayer(logits, yi)
        total_test_loss += loss._v * len(yi)
        test_nacc += np.sum(logits._v.argmax(-1) == yi._v)
    test_loss = total_test_loss / len(test_y)
    test_acc = test_nacc / len(test_y)
    time_used = time.time() - ts
    print(
        f"[{epoch}]{time_used}s\tLOSS:{train_loss}\tTRAIN_ACC:{train_acc}\tTEST_LOSS:{test_loss}\tTEST_ACC:{test_acc}",
        flush=True)
print("用时", time.time() - tic)
