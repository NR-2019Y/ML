import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


# 占位符模式
def gen_rand_w_and_b(n_inputs, n_hiddens):
    return (
        tf.Variable(tf.random_normal([n_inputs, n_hiddens], stddev=0.01)),
        tf.Variable(tf.random_normal([n_hiddens, n_hiddens], stddev=0.01)),
        tf.Variable(tf.zeros([n_hiddens], dtype=tf.float32))
    )


def gru(X, *, dic_len, n_hiddens):
    x_one_hot = tf.one_hot(X, depth=dic_len, dtype=tf.float32)
    n_inputs = n_classes = dic_len
    W_xr, W_hr, b_r = gen_rand_w_and_b(n_inputs, n_hiddens)
    W_xz, W_hz, b_z = gen_rand_w_and_b(n_inputs, n_hiddens)
    W_xh, W_hh, b_h = gen_rand_w_and_b(n_inputs, n_hiddens)
    W_hq = tf.Variable(tf.random_normal([n_hiddens, n_classes], stddev=0.01))
    b_q = tf.Variable(tf.zeros([n_classes]), dtype=tf.float32)
    params = W_xr, W_hr, b_r, W_xz, W_hz, b_z, W_xh, W_hh, b_h, W_hq, b_q

    batch_size = tf.shape(x_one_hot)[0]
    n_steps = tf.shape(x_one_hot)[1]
    cond = lambda i, *args: tf.less(i, n_steps)

    def next_step(i, hi, Li):
        xi = x_one_hot[:, i, :]
        R = tf.sigmoid(tf.matmul(xi, W_xr) + tf.matmul(hi, W_hr) + b_r)
        Z = tf.sigmoid(tf.matmul(xi, W_xz) + tf.matmul(hi, W_hz) + b_z)
        H_d = tf.tanh(tf.matmul(xi, W_xh) + tf.matmul(R * hi, W_hh) + b_h)
        hi = Z * hi + (1 - Z) * H_d
        yi = tf.matmul(hi, W_hq) + b_q
        Li = tf.concat([Li, yi[None, ...]], axis=0)
        i = i + 1
        return i, hi, Li

    i = tf.constant(0, dtype=tf.int32)
    H = tf.zeros([batch_size, n_hiddens], dtype=tf.float32)
    logits_t = tf.zeros([0, batch_size, n_classes], dtype=tf.float32)
    i, H, logits_t = tf.while_loop(cond, next_step, [i, H, logits_t],
                                   shape_invariants=[i.get_shape(), tf.TensorShape([None, n_hiddens]),
                                                     tf.TensorShape([None, None, n_classes])])
    logits = tf.transpose(logits_t, perm=[1, 0, 2])
    return params, logits


def cost_func(logits, targets):
    return tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=targets))


s_ori = ['hello world', 'how are you', 'he is happy']
# s_ori = ['hihello']
index2word = list(set(str.join('', s_ori)))
word2index = {word: index for index, word in enumerate(index2word)}
s_ori_index = [[word2index[c] for c in si] for si in s_ori]
train_x = np.array([v[:-1] for v in s_ori_index])
train_y = np.array([v[1:] for v in s_ori_index])

X_ = tf.placeholder(tf.int64, shape=[None, None])
Y_ = tf.placeholder(tf.int64, shape=[None, None])
params, logits = gru(X_, dic_len=len(index2word), n_hiddens=32)
cost_ = cost_func(logits, Y_)
y_pred_ = tf.argmax(logits, -1)
acc_ = tf.reduce_mean(tf.cast(tf.equal(y_pred_, Y_), tf.float32))

grads = tf.gradients(cost_, params)
learning_rate = 0.1
trainer = [tf.assign_sub(param, learning_rate * grad) for grad, param in zip(grads, params)]

cost_list, acc_list = [], []

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(5000):
        cost_val, acc_val, train_y_pred, _ = sess.run([
            cost_, acc_, y_pred_, trainer
        ], feed_dict={X_: train_x, Y_: train_y})
        cost_list.append(cost_val)
        acc_list.append(acc_val)
        print(f"{i} COST:{cost_val} TRAIN_ACC:{acc_val}")
        print("训练集预测", [''.join(index2word[i] for i in yi) for yi in train_y_pred])

plt.subplot(121)
plt.plot(cost_list, label="COST")
plt.legend()
plt.subplot(122)
plt.plot(acc_list, label="ACC")
plt.legend()
plt.show()
