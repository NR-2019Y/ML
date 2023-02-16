import numpy as np
import tensorflow.compat.v1 as tf
import matplotlib.pyplot as plt
import time
from typing import Tuple

tf.disable_eager_execution()


def gen_rand_w_and_b(n_inputs, n_hiddens):
    return (
        tf.Variable(tf.random_normal([n_inputs, n_hiddens], stddev=0.01, mean=0.0)),
        tf.Variable(tf.random_normal([n_hiddens, n_hiddens], stddev=0.01, mean=0.0)),
        tf.Variable(tf.zeros(n_hiddens, dtype=tf.float32))
    )


# 占位符模式

class Layer:
    params: Tuple[tf.Tensor, ...]

    def __call__(self, X):
        raise NotImplementedError


class LSTM(Layer):
    def __init__(self, units, input_dim, return_sequenses=True):
        W_xi, W_hi, b_i = gen_rand_w_and_b(input_dim, units)
        W_xf, W_hf, b_f = gen_rand_w_and_b(input_dim, units)
        W_xo, W_ho, b_o = gen_rand_w_and_b(input_dim, units)
        W_xc, W_hc, b_c = gen_rand_w_and_b(input_dim, units)
        self.n_hiddens = units
        self._return_sequenses = return_sequenses
        self.params = W_xi, W_hi, b_i, W_xf, W_hf, b_f, W_xo, W_ho, b_o, W_xc, W_hc, b_c

    def __call__(self, X):
        W_xi, W_hi, b_i, W_xf, W_hf, b_f, W_xo, W_ho, b_o, W_xc, W_hc, b_c = self.params
        batch_size = tf.shape(X)[0]
        n_steps = tf.shape(X)[1]
        cond = lambda i, *args: tf.less(i, n_steps)

        def next_step(i, hi, ci, Li):
            xi = X[:, i, :]
            I = tf.sigmoid(tf.matmul(xi, W_xi) + tf.matmul(hi, W_hi) + b_i)
            F = tf.sigmoid(tf.matmul(xi, W_xf) + tf.matmul(hi, W_hf) + b_f)
            O = tf.sigmoid(tf.matmul(xi, W_xo) + tf.matmul(hi, W_ho) + b_o)
            C_d = tf.tanh(tf.matmul(xi, W_xc) + tf.matmul(hi, W_hc) + b_c)
            ci = F * ci + I * C_d
            hi = O * tf.tanh(ci)
            Li = tf.concat([Li, hi[:, None, :]], axis=1)
            i = i + 1
            return i, hi, ci, Li

        n_hiddens = self.n_hiddens
        i = tf.constant(0, dtype=np.int32)
        H = tf.zeros([batch_size, n_hiddens], dtype=tf.float32)
        C = tf.zeros([batch_size, n_hiddens], dtype=tf.float32)
        out = tf.zeros([batch_size, 0, n_hiddens])
        i, H, C, out = tf.while_loop(
            cond, next_step, [i, H, C, out],
            shape_invariants=[
                i.shape,
                H.shape,
                C.shape,
                tf.TensorShape([None, None, n_hiddens])
            ]
        )
        if self._return_sequenses:
            return out
        else:
            return H


class Dense(Layer):
    def __init__(self, units, input_dim):
        W = tf.Variable(tf.random_normal([input_dim, units], stddev=0.01, mean=0.))
        b = tf.Variable(tf.zeros(units))
        self.params = W, b

    def __call__(self, X):
        W, b = self.params
        return tf.matmul(X, W) + b


class LSTMModel(Layer):
    def __init__(self, input_dim, n_classes):
        h1, h2 = 64, 64
        self.lstm1 = LSTM(units=h1, input_dim=input_dim)
        self.lstm2 = LSTM(units=h2, input_dim=h1)
        self.dense = Dense(units=n_classes, input_dim=h2)
        self.params = self.lstm1.params + self.lstm2.params + self.dense.params

    def __call__(self, X):
        return self.dense(self.lstm2(self.lstm1(X)))


s_ori = ['hello world', 'how are you', 'he is happy']
# s_ori = ['hihello']
index2word = list(set(str.join('', s_ori)))
word2index = {word: index for index, word in enumerate(index2word)}
s_ori_index = [[word2index[c] for c in si] for si in s_ori]
train_x = np.array([v[:-1] for v in s_ori_index])
train_x_onehot = np.eye(len(index2word), dtype=np.float32)[train_x]
train_y = np.array([v[1:] for v in s_ori_index])

test_str = ['hell', 'word']
test_str_index = [[word2index[c] for c in si] for si in test_str]
test_x = np.array([vi[:-1] for vi in test_str_index])
test_x_hot = np.eye(len(index2word))[test_x]

X_ = tf.placeholder(tf.float32, shape=[None, None, len(index2word)])
Y_ = tf.placeholder(tf.int64, shape=[None, None])
lstm = LSTMModel(input_dim=len(index2word), n_classes=len(index2word))
logits = lstm(X_)
cost_ = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=Y_))
y_pred_ = tf.argmax(logits, -1)
acc_ = tf.reduce_mean(tf.cast(tf.equal(y_pred_, Y_), tf.float32))

grads = tf.gradients(cost_, lstm.params)
learning_rate = 0.01
# trainer = [tf.assign_sub(param, learning_rate * grad) for grad, param in zip(grads, lstm.params)]
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
trainer = optimizer.apply_gradients(zip(grads, lstm.params))

tic = time.time()
cost_list, acc_list = [], []
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(200):
        cost_val, acc_val, train_y_pred, _ = sess.run([
            cost_, acc_, y_pred_, trainer
        ], feed_dict={X_: train_x_onehot, Y_: train_y})
        cost_list.append(cost_val)
        acc_list.append(acc_val)
        print(f"{i} COST:{cost_val} TRAIN_ACC:{acc_val}")
        print("训练集预测", [''.join(index2word[i] for i in yi) for yi in train_y_pred])
    test_y_pred = sess.run(y_pred_, feed_dict={X_: test_x_hot})
    print("测试数据预测", [''.join(index2word[i] for i in yi) for yi in test_y_pred])

print("运行时间", time.time() - tic)

plt.subplot(121)
plt.plot(cost_list, label="COST")
plt.legend()
plt.subplot(122)
plt.plot(acc_list, label="ACC")
plt.legend()
plt.show()
