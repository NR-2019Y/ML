import numpy as np
from AUTO_DIFF import my_auto_grad_v0 as op
from AUTO_DIFF import optimizer, layers
from AUTO_DIFF import metrics
import matplotlib.pyplot as plt


class RNN(object):
    def __init__(self, dic_len, n_hiddens):
        self.rnn1 = layers.LSTM(units=n_hiddens, input_dim=dic_len)
        self.rnn2 = layers.LSTM(units=n_hiddens, input_dim=n_hiddens)
        self.dense = layers.Dense(units=dic_len, input_dim=n_hiddens)
        self.params = self.rnn1.params + self.rnn2.params + self.dense.params

    def init_grad(self):
        for param in self.params:
            param._d = 0.0

    def __call__(self, X: np.ndarray):  # (batch_size, n_steps, dic_len)
        self.init_grad()
        return self.dense(self.rnn2(self.rnn1(op.C(X))))


s_ori = ['hello world', 'how are you', 'he is happy']
# s_ori = ['hihello']
index2word = list(set(str.join('', s_ori)))
word2index = {word: index for index, word in enumerate(index2word)}
s_ori_index = [[word2index[c] for c in si] for si in s_ori]
train_x = np.array([v[:-1] for v in s_ori_index])
train_y = np.array([v[1:] for v in s_ori_index])
train_x_hot = np.eye(len(index2word))[train_x]

rnn = RNN(dic_len=len(index2word), n_hiddens=32)
trainer = optimizer.Adam(learning_rate=0.03, trainable_nodes=rnn.params)

cost_list, acc_list = [], []
for i in range(100):
    logits = rnn(train_x_hot)
    loss = op.CrossEntropyLossLayer(logits, op.C(train_y))
    loss.backward()
    trainer()
    train_acc = metrics.sparse_categorical_accuracy(train_y, logits._v, reduction=np.mean)
    cost_list.append(loss._v)
    acc_list.append(train_acc)

    train_y_pred = logits._v.argmax(axis=-1)
    print(f"{i} LOSS:{loss._v} TRAIN_ACC:{train_acc}")
    print("训练集预测", [''.join(index2word[i] for i in yi) for yi in train_y_pred])
    if train_acc >= 1.0 - 1e-5:
        break

test_str = ['hell', 'word']
test_str_index = [[word2index[c] for c in si] for si in test_str]
test_x = np.array([vi[:-1] for vi in test_str_index])
test_x_hot = np.eye(len(index2word))[test_x]
test_y_pred = rnn(test_x_hot)._v.argmax(axis=-1)
print("测试数据预测", [''.join(index2word[i] for i in yi) for yi in test_y_pred])

plt.subplot(121)
plt.plot(cost_list, label="COST")
plt.legend()
plt.subplot(122)
plt.plot(acc_list, label="ACC")
plt.legend()
plt.show()
