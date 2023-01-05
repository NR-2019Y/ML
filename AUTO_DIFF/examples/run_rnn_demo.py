import numpy as np
from AUTO_DIFF import my_auto_grad_v0 as op


class RNN(object):
    def __init__(self, dic_len, n_hiddens):
        n_inputs = n_classes = dic_len
        W_xh = op.C(np.random.normal(scale=0.01, size=(n_inputs, n_hiddens)), requires_grad=True)
        W_hh = op.C(np.random.normal(scale=0.01, size=(n_hiddens, n_hiddens)), requires_grad=True)
        b_h = op.C(np.zeros(n_hiddens), requires_grad=True)
        W_q = op.C(np.random.normal(scale=0.01, size=(n_hiddens, n_classes)), requires_grad=True)
        b_q = op.C(np.zeros(n_classes), requires_grad=True)
        self.params = W_xh, W_hh, b_h, W_q, b_q
        self.dic_len, self.n_hiddens = dic_len, n_hiddens

    def init_grad(self):
        for param in self.params:
            param._d = 0.0

    def __call__(self, X: np.ndarray):
        self.init_grad()
        W_xh, W_hh, b_h, W_q, b_q = self.params

        batch_size, n_steps = X.shape
        # X: (batch_size, n_steps)
        # x_one_hot: (batch_size, n_steps, dic_len)
        x_one_hot: np.ndarray = np.eye(self.dic_len)[X]

        H = op.C(np.zeros([batch_size, self.n_hiddens]))
        L = []
        for i in range(n_steps):
            xi = op.C(x_one_hot[:, i, :])
            H = op.Tanh(op.AddBias2D(xi @ W_xh + H @ W_hh, b_h))
            Y = op.LinearLayer(H, W_q, b_q)
            L.append(Y)
        # print("L", [li._v for li in L])
        logits_t = op.ListToTensor(L)
        logits = op.Transpose(logits_t, [1, 0, 2])  # (batch_size, seq_len, dic_len)
        return logits

    def update(self, *, learning_rate):
        # for nm, param in zip(("W_xh", "W_hh", "b_h", "W_q", "b_q"), self.params):
        #     print(nm, param._v.shape, param._d.shape)
        for param in self.params:
            param._v -= learning_rate * param._d


def loss_func(logits: op.Op, y: np.ndarray):
    dic_len = logits._v.shape[-1]
    y_flatten = op.C(np.reshape(y, [-1]))
    logits_flatten = op.Reshape(logits, [-1, dic_len])
    return op.CrossEntropyLossLayer(logits_flatten, y_flatten)


def calc_acc(y_pred: np.ndarray, y_true: np.ndarray):
    return np.mean(y_pred == y_true)


s_ori = ['hello world', 'how are you', 'he is happy']
# s_ori = ['hihello']
index2word = list(set(str.join('', s_ori)))
word2index = {word: index for index, word in enumerate(index2word)}
s_ori_index = [[word2index[c] for c in si] for si in s_ori]
train_x = np.array([v[:-1] for v in s_ori_index])
train_y = np.array([v[1:] for v in s_ori_index])

rnn = RNN(dic_len=len(index2word), n_hiddens=32)
for i in range(1000):
    logits = rnn(train_x)
    loss = loss_func(logits, train_y)
    loss.backward()
    rnn.update(learning_rate=0.1)
    train_y_pred = logits._v.argmax(axis=-1)
    train_acc = calc_acc(train_y_pred, train_y)
    print(f"{i} LOSS:{loss._v} TRAIN_ACC:{train_acc}")
    print("训练集预测", [''.join(index2word[i] for i in yi) for yi in train_y_pred])
    if train_acc >= 1.0 - 1e-5:
        break

test_str = ['hell', 'word']
test_str_index = [[word2index[c] for c in si] for si in test_str]
test_x = np.array([vi[:-1] for vi in test_str_index])
test_y_pred = rnn(test_x)._v.argmax(axis=-1)
print("测试数据预测", [''.join(index2word[i] for i in yi) for yi in test_y_pred])
