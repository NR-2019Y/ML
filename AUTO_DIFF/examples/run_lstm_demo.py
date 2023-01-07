import numpy as np
import matplotlib.pyplot as plt
from AUTO_DIFF import my_auto_grad_v0 as op
from AUTO_DIFF import optimizer


class LSTM(object):
    @staticmethod
    def gen_rand_w_and_b(n_inputs, n_hiddens):
        return (
            op.C(np.random.normal(scale=0.01, size=(n_inputs, n_hiddens)), requires_grad=True),
            op.C(np.random.normal(scale=0.01, size=(n_hiddens, n_hiddens)), requires_grad=True),
            op.C(np.zeros(shape=(n_hiddens,)), requires_grad=True)
        )

    def __init__(self, *, dic_len, n_hiddens):
        n_inputs = n_classes = dic_len
        W_xi, W_hi, b_i = LSTM.gen_rand_w_and_b(n_inputs, n_hiddens)
        W_xf, W_hf, b_f = LSTM.gen_rand_w_and_b(n_inputs, n_hiddens)
        W_xo, W_ho, b_o = LSTM.gen_rand_w_and_b(n_inputs, n_hiddens)
        W_xc, W_hc, b_c = LSTM.gen_rand_w_and_b(n_inputs, n_hiddens)
        W_hq = op.C(np.random.normal(scale=0.01, size=(n_hiddens, n_classes)), requires_grad=True)
        b_q = op.C(np.zeros(shape=(n_classes,)), requires_grad=True)
        self.dic_len, self.n_hiddens = dic_len, n_hiddens
        self.params = W_xi, W_hi, b_i, W_xf, W_hf, b_f, W_xo, W_ho, b_o, W_xc, W_hc, b_c, W_hq, b_q

    def init_grad(self):
        for param in self.params:
            param._d = 0.0

    def __call__(self, X: np.ndarray):
        self.init_grad()
        W_xi, W_hi, b_i, W_xf, W_hf, b_f, W_xo, W_ho, b_o, W_xc, W_hc, b_c, W_hq, b_q = self.params

        batch_size, n_steps = X.shape
        x_one_hot = np.eye(self.dic_len)[X]

        H = op.C(np.zeros(shape=(batch_size, self.n_hiddens)))
        C = op.C(np.zeros(shape=(batch_size, self.n_hiddens)))
        L = []
        for i in range(n_steps):
            xi = op.C(x_one_hot[:, i, :])
            I = op.Sigmoid(op.AddBias2D(op.MatMul(xi, W_xi) + op.MatMul(H, W_hi), b_i))
            F = op.Sigmoid(op.AddBias2D(op.MatMul(xi, W_xf) + op.MatMul(H, W_hf), b_f))
            O = op.Sigmoid(op.AddBias2D(op.MatMul(xi, W_xo) + op.MatMul(H, W_ho), b_o))
            C_d = op.Tanh(op.AddBias2D(op.MatMul(xi, W_xc) + op.MatMul(H, W_hc), b_c))
            C = F * C + I * C_d
            H = O * op.Tanh(C)
            Y = op.LinearLayer(H, W_hq, b_q)
            L.append(Y)
        logits_t = op.ListToTensor(L)
        logits = op.Transpose(logits_t, axes=[1, 0, 2])
        return logits


def cost_func(logits: op.Op, y_true: np.ndarray):
    dic_len = logits._v.shape[-1]
    y_flatten = op.C(np.reshape(y_true, [-1]))
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

rnn = LSTM(dic_len=len(index2word), n_hiddens=32)
# trainer = optimizer.SGD(learning_rate=0.3, trainable_nodes=rnn.params)
# trainer = optimizer.AdaGrad(learning_rate=0.1, trainable_nodes=rnn.params)
trainer = optimizer.Adam(learning_rate=0.1, trainable_nodes=rnn.params)

cost_list, acc_list = [], []

for i in range(50):
    logits = rnn(train_x)
    cost = cost_func(logits, train_y)
    cost.backward()
    # rnn.update(learning_rate=0.1)
    # print([e._d for e in rnn.params])
    trainer()
    train_y_pred = logits._v.argmax(axis=-1)
    train_acc = calc_acc(train_y_pred, train_y)
    cost_list.append(cost._v)
    acc_list.append(train_acc)
    print(f"{i} COST:{cost._v} TRAIN_ACC:{train_acc}")
    print("训练集预测", [''.join(index2word[i] for i in yi) for yi in train_y_pred])
    # if train_acc >= 1.0 - 1e-5:
    #     break

test_str = ['hell', 'word']
test_str_index = [[word2index[c] for c in si] for si in test_str]
test_x = np.array([vi[:-1] for vi in test_str_index])
test_y_pred = rnn(test_x)._v.argmax(axis=-1)
print("测试数据预测", [''.join(index2word[i] for i in yi) for yi in test_y_pred])

plt.subplot(121)
plt.plot(cost_list, label="COST")
plt.legend()
plt.subplot(122)
plt.plot(acc_list, label="ACC")
plt.legend()
plt.show()
