import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import time

tf.enable_eager_execution()


def gen_rand_w_and_b(n_inputs, n_hiddens):
    return (
        tf.Variable(tf.random_normal([n_inputs, n_hiddens], stddev=0.01, mean=0.0)),
        tf.Variable(tf.random_normal([n_hiddens, n_hiddens], stddev=0.01, mean=0.0)),
        tf.Variable(tf.zeros(n_hiddens, dtype=tf.float32))
    )


# eager 模式
class LSTM:
    def __init__(self, *, dic_len, n_hiddens):
        n_inputs = n_classes = dic_len
        W_xi, W_hi, b_i = gen_rand_w_and_b(n_inputs, n_hiddens)
        W_xf, W_hf, b_f = gen_rand_w_and_b(n_inputs, n_hiddens)
        W_xo, W_ho, b_o = gen_rand_w_and_b(n_inputs, n_hiddens)
        W_xc, W_hc, b_c = gen_rand_w_and_b(n_inputs, n_hiddens)
        W_hq = tf.Variable(tf.random_normal([n_hiddens, n_classes], stddev=0.01, mean=0.0))
        b_q = tf.Variable(tf.zeros(n_classes, dtype=tf.float32))
        self.dic_len, self.n_hiddens = dic_len, n_hiddens
        self.params = W_xi, W_hi, b_i, W_xf, W_hf, b_f, W_xo, W_ho, b_o, W_xc, W_hc, b_c, W_hq, b_q

    def __call__(self, X):
        x_one_hot = tf.one_hot(X, depth=self.dic_len, dtype=tf.float32)
        batch_size = x_one_hot.shape[0]
        n_steps = x_one_hot.shape[1]

        W_xi, W_hi, b_i, W_xf, W_hf, b_f, W_xo, W_ho, b_o, W_xc, W_hc, b_c, W_hq, b_q = self.params
        H = tf.zeros([batch_size, self.n_hiddens], dtype=tf.float32)
        C = tf.zeros([batch_size, self.n_hiddens], dtype=tf.float32)
        L = []
        for i in range(n_steps):
            xi = x_one_hot[:, i, :]
            I = tf.sigmoid(tf.matmul(xi, W_xi) + tf.matmul(H, W_hi) + b_i)
            F = tf.sigmoid(tf.matmul(xi, W_xf) + tf.matmul(H, W_hf) + b_f)
            O = tf.sigmoid(tf.matmul(xi, W_xo) + tf.matmul(H, W_ho) + b_o)
            C_d = tf.tanh(tf.matmul(xi, W_xc) + tf.matmul(H, W_hc) + b_c)
            C = F * C + I * C_d
            H = O * tf.tanh(C)
            Y = tf.matmul(H, W_hq) + b_q
            L.append(Y)
        logits = tf.transpose(tf.convert_to_tensor(L), perm=[1, 0, 2])
        return logits


def cost_func(logits, targets):
    return tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=targets))


def sgd(grads_and_params, *, learning_rate):
    for g, v in grads_and_params:
        # tf.assign_sub(v, learning_rate * g)
        v.assign_sub(learning_rate * g)


s_ori = ['hello world', 'how are you', 'he is happy']
# s_ori = ['hihello']
index2word = list(set(str.join('', s_ori)))
word2index = {word: index for index, word in enumerate(index2word)}
s_ori_index = [[word2index[c] for c in si] for si in s_ori]
train_x = np.array([v[:-1] for v in s_ori_index])
train_y = np.array([v[1:] for v in s_ori_index])

lstm = LSTM(dic_len=len(index2word), n_hiddens=32)

tic = time.time()
cost_list, acc_list = [], []
for i in range(2000):
    with tf.GradientTape() as tape:
        logits = lstm(train_x)
        cost = cost_func(logits, train_y)
    grads = tape.gradient(cost, lstm.params)
    sgd(zip(grads, lstm.params), learning_rate=0.3)
    train_y_pred = logits.numpy().argmax(axis=-1)
    train_acc = np.mean(train_y_pred == train_y)
    cost_list.append(cost.numpy())
    acc_list.append(train_acc)
    print(f"{i} COST:{cost.numpy()} TRAIN_ACC:{train_acc}")
    print("训练集预测", [''.join(index2word[i] for i in yi) for yi in train_y_pred])
print("运行时间", time.time() - tic)

plt.subplot(121)
plt.plot(cost_list, label="COST")
plt.legend()
plt.subplot(122)
plt.plot(acc_list, label="ACC")
plt.legend()
plt.show()
